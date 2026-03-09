import io
import queue
import threading
import time
import requests
import sounddevice as sd
import soundfile as sf
from openai import OpenAI
from obsws_python import ReqClient

# =========================
# OpenAI
# =========================
client = OpenAI()

TRANS_MODEL = "gpt-4o-mini"
TRANS_TEMPERATURE = 0.2

# =========================
# OBS websocket
# =========================
OBS_HOST = "127.0.0.1"
OBS_PORT = 4455
OBS_PASS = "yourpass"
TEXT_SOURCE_JP = "Caption"
TEXT_SOURCE_EN = "Caption_EN"
obs = ReqClient(host=OBS_HOST, port=OBS_PORT, password=OBS_PASS)

# =========================
# VOICEVOX
# =========================
VOICEVOX_BASE = "https://your.voicevox.api.server/voicevox_engine"
VOICEVOX_SPEAKER = 0
SPEAK_JP = True  # True: 日本語読み上げ / False: 英訳を読み上げ

# =========================
# VB-CABLE playback
# =========================
VB_PLAY_DEVICE_NAME = "CABLE Input"
VB_PLAY_DEVICE_INDEX = None  # 自動検出できない時だけ番号固定

# =========================
# 表示設定
# =========================
WRAP_JP_WIDTH = 0
WRAP_EN_WIDTH = 0

# =========================
# TTS制御
# =========================
MIN_TTS_INTERVAL_SEC = 0.2

# =========================
# 非同期ワーカー用
# =========================
work_q: queue.Queue[str] = queue.Queue(maxsize=50)
stop_event = threading.Event()


def wrap_fixed(s: str, width: int) -> str:
    s = (s or "").strip()
    if not s or width <= 0:
        return s
    return "\n".join(s[i:i+width] for i in range(0, len(s), width))


def translate_to_english_openai(jp_text: str) -> str:
    jp_text = (jp_text or "").strip()
    if not jp_text:
        return ""
    resp = client.chat.completions.create(
        model=TRANS_MODEL,
        temperature=TRANS_TEMPERATURE,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a live-stream subtitle translator.\n"
                    "Translate Japanese to natural, concise spoken English.\n"
                    "Rules:\n"
                    "- Keep it short and readable as subtitles.\n"
                    "- Do NOT add explanations.\n"
                    "- Do NOT add extra information.\n"
                    "- Keep names as-is when unsure.\n"
                ),
            },
            {"role": "user", "content": jp_text},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def obs_set_text(source_name: str, text: str):
    obs.set_input_settings(
        name=source_name,
        settings={"text": text},
        overlay=True
    )


def voicevox_tts_wav_bytes(text: str, speaker: int) -> bytes:
    r = requests.post(
        f"{VOICEVOX_BASE}/audio_query",
        params={"text": text, "speaker": speaker},
        timeout=10,
    )
    r.raise_for_status()
    query = r.json()

    r = requests.post(
        f"{VOICEVOX_BASE}/synthesis",
        params={"speaker": speaker},
        json=query,
        timeout=30,
    )
    r.raise_for_status()

    return r.content


def find_output_device_index_by_name(name_substr: str) -> int | None:
    name_substr = (name_substr or "").lower()
    try:
        devices = sd.query_devices()
    except Exception:
        return None

    for i, d in enumerate(devices):
        if d.get("max_output_channels", 0) <= 0:
            continue
        nm = (d.get("name") or "").lower()
        if name_substr in nm:
            return i
    return None


def play_wav_bytes_to_device(wav_bytes: bytes, device_index: int):
    bio = io.BytesIO(wav_bytes)
    data, fs = sf.read(bio, dtype="float32")

    # 必要なら音量調整
    gain = 0.5
    data = data * gain

    sd.play(data, fs, device=device_index)
    sd.wait()


def worker_translate_tts_play(dev_index: int):
    last_tts = 0.0

    while not stop_event.is_set():
        try:
            jp = work_q.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            # 英訳
            en = translate_to_english_openai(jp)
            obs_set_text(TEXT_SOURCE_EN, wrap_fixed(en, WRAP_EN_WIDTH) if en else "")

            # 読み上げ対象
            speak_text = jp if SPEAK_JP else (en or jp)

            # 連打抑制
            now = time.time()
            wait = MIN_TTS_INTERVAL_SEC - (now - last_tts)
            if wait > 0:
                time.sleep(wait)

            # VOICEVOX → VB-CABLE
            tts_wav = voicevox_tts_wav_bytes(speak_text, VOICEVOX_SPEAKER)
            last_tts = time.time()
            play_wav_bytes_to_device(tts_wav, dev_index)

        except Exception as e:
            print("[Worker] error:", e)
        finally:
            try:
                work_q.task_done()
            except Exception:
                pass


def submit_text(jp: str):
    jp = (jp or "").strip()
    if not jp:
        return

    obs_set_text(TEXT_SOURCE_JP, wrap_fixed(jp, WRAP_JP_WIDTH))
    print("JP:", jp)

    try:
        work_q.put_nowait(jp)
    except queue.Full:
        try:
            _ = work_q.get_nowait()
            work_q.task_done()
        except Exception:
            pass
        try:
            work_q.put_nowait(jp)
        except Exception:
            pass


def main():
    # VB-CABLE 出力先決定
    dev_index = VB_PLAY_DEVICE_INDEX
    if dev_index is None:
        dev_index = find_output_device_index_by_name(VB_PLAY_DEVICE_NAME)

    if dev_index is None:
        print("[VB-CABLE] 出力デバイスを自動検出できませんでした。")
        print("sd.query_devices() で 'CABLE Input' の番号を確認し、VB_PLAY_DEVICE_INDEX に設定してください。")
        print(sd.query_devices())
        return

    print(f"[VB-CABLE] playback device index = {dev_index} (name contains '{VB_PLAY_DEVICE_NAME}')")

    # 非同期ワーカー起動
    t = threading.Thread(target=worker_translate_tts_play, args=(dev_index,), daemon=True)
    t.start()

    print("Ready.")
    print("日本語を入力してください。")
    print("・空行 Enter 1回目: まだ送信しない")
    print("・空行 Enter 2回連続: 送信確定")
    print("・/send  : その場で送信")
    print("・/clear : 入力中テキスト破棄")
    print("・/quit  : 終了")
    print("-" * 50)

    lines: list[str] = []
    blank_count = 0

    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            print("\nStopping...")
            break

        cmd = line.strip()

        if cmd == "/quit":
            print("Stopping...")
            break

        if cmd == "/clear":
            lines.clear()
            blank_count = 0
            print("[cleared]")
            continue

        if cmd == "/send":
            text = "\n".join(lines).strip()
            if text:
                submit_text(text)
            else:
                print("[empty: nothing sent]")
            lines.clear()
            blank_count = 0
            continue

        if cmd == "":
            blank_count += 1

            if blank_count >= 2:
                text = "\n".join(lines).strip()
                if text:
                    submit_text(text)
                else:
                    print("[empty: nothing sent]")
                lines.clear()
                blank_count = 0
                print("-" * 50)
            else:
                # 1回目の空行は「段落内改行」として保持したいなら追加
                # ただし末尾空行が増えすぎるので、ここでは何もしない
                print("[blank once: press Enter once more to send]")
            continue

        # 通常行
        blank_count = 0
        lines.append(line)

    stop_event.set()
    time.sleep(0.2)
    print("Stopped.")


if __name__ == "__main__":
    main()