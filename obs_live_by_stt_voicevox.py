import io, queue, threading, time
import numpy as np
import sounddevice as sd
import soundfile as sf
import requests
from openai import OpenAI
from obsws_python import ReqClient

# =========================
# OpenAI
# =========================
client = OpenAI()

STT_MODEL = "gpt-4o-mini-transcribe"
STT_LANG  = "ja"

TRANS_MODEL = "gpt-4o-mini"
TRANS_TEMPERATURE = 0.2

# =========================
# OBS websocket
# =========================
OBS_HOST="127.0.0.1"; OBS_PORT=4455; OBS_PASS="yourpass"
TEXT_SOURCE_JP="Caption"
TEXT_SOURCE_EN="Caption_EN"
obs = ReqClient(host=OBS_HOST, port=OBS_PORT, password=OBS_PASS)

# =========================
# VOICEVOX
# =========================
VOICEVOX_BASE = "https://your.voicevox.api.server/voicevox_engine"
VOICEVOX_SPEAKER = 0
SPEAK_JP = True  # True: 日本語を読み上げ / False: 英訳を読み上げ（英語読みは変になりがち）

# =========================
# Audio capture (mic)
# =========================
SR = 16000
CHANNELS = 1

BLOCK_SEC = 0.25
SILENCE_SEC = 1.1
MIN_SPEECH_SEC = 2.2
MAX_UTTER_SEC = 25.0
SILENCE_THRESHOLD = 0.012

# =========================
# VB-CABLE playback
# =========================
VB_PLAY_DEVICE_NAME = "CABLE Input"  # ここは通常これでOK
VB_PLAY_DEVICE_INDEX = None          # 自動検出できない時だけ番号固定

# （任意）折り返し保険（OBSでうまく改行できるなら 0 のまま）
WRAP_JP_WIDTH = 0
WRAP_EN_WIDTH = 0

# 連打対策（外部VOICEVOXを守る・音声渋滞を抑える）
MIN_TTS_INTERVAL_SEC = 0.2  # ほぼ無視できる程度の間隔


def wrap_fixed(s: str, width: int) -> str:
    s = (s or "").strip()
    if not s or width <= 0:
        return s
    return "\n".join(s[i:i+width] for i in range(0, len(s), width))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def stt_openai(wav_bytes: bytes) -> str:
    bio = io.BytesIO(wav_bytes)
    bio.name = "audio.wav"
    t = client.audio.transcriptions.create(
        model=STT_MODEL,
        file=bio,
        language=STT_LANG,
    )
    return (t.text or "").strip()


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
    sd.play(data, fs, device=device_index)
    sd.wait()


# =========================
# 非同期ワーカー（翻訳 + TTS + 再生）
# =========================
work_q: queue.Queue[str] = queue.Queue(maxsize=50)
stop_event = threading.Event()

def worker_translate_tts_play(dev_index: int):
    last_tts = 0.0
    while not stop_event.is_set():
        try:
            jp = work_q.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            # 翻訳（EN字幕は“後追い”でOK）
            en = translate_to_english_openai(jp)
            obs_set_text(TEXT_SOURCE_EN, wrap_fixed(en, WRAP_EN_WIDTH) if en else "")

            # 読み上げ（JP or EN）
            speak_text = jp if SPEAK_JP else (en or jp)

            # 外部サーバへの連打を少し抑える
            now = time.time()
            wait = MIN_TTS_INTERVAL_SEC - (now - last_tts)
            if wait > 0:
                time.sleep(wait)

            tts_wav = voicevox_tts_wav_bytes(speak_text, VOICEVOX_SPEAKER)
            last_tts = time.time()

            # VB-CABLEへ再生（OBSがCABLE Outputを拾う）
            play_wav_bytes_to_device(tts_wav, dev_index)

        except Exception as e:
            print("[Worker] error:", e)
        finally:
            try:
                work_q.task_done()
            except Exception:
                pass


def main():
    # VB-CABLE出力デバイス決定
    dev_index = VB_PLAY_DEVICE_INDEX
    if dev_index is None:
        dev_index = find_output_device_index_by_name(VB_PLAY_DEVICE_NAME)
    if dev_index is None:
        print("[VB-CABLE] 出力デバイスを自動検出できませんでした。")
        print("sd.query_devices() で 'CABLE Input' の番号を見て VB_PLAY_DEVICE_INDEX に設定してください。")
        print(sd.query_devices())
        return

    print(f"[VB-CABLE] playback device index = {dev_index} (name contains '{VB_PLAY_DEVICE_NAME}')")

    # ワーカー起動
    t = threading.Thread(target=worker_translate_tts_play, args=(dev_index,), daemon=True)
    t.start()

    # マイク入力（無音区切り）
    blocksize = int(SR * BLOCK_SEC)
    q_audio = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print("Audio status:", status)
        q_audio.put(indata.copy())

    silence_blocks_needed = int(SILENCE_SEC / BLOCK_SEC)
    max_blocks = int(MAX_UTTER_SEC / BLOCK_SEC)

    print("Ready. Speak. (Ctrl+C to stop)")
    with sd.InputStream(
        samplerate=SR,
        channels=CHANNELS,
        dtype="float32",
        blocksize=blocksize,
        callback=callback,
    ):
        buf = []
        silent = 0
        started = False

        while True:
            try:
                data = q_audio.get(timeout=0.5)
            except queue.Empty:
                continue

            x = data.reshape(-1)
            level = rms(x)

            if level > SILENCE_THRESHOLD:
                started = True
                silent = 0
                buf.append(x)
            else:
                if started:
                    silent += 1
                    buf.append(x)

            if started and len(buf) >= max_blocks:
                silent = silence_blocks_needed

            if started and silent >= silence_blocks_needed:
                audio = np.concatenate(buf) if buf else np.array([], dtype=np.float32)
                buf, silent, started = [], 0, False

                duration = len(audio) / SR
                if duration < MIN_SPEECH_SEC:
                    continue

                # WAV化（メモリ上）
                bio = io.BytesIO()
                sf.write(bio, audio, SR, format="WAV")
                wav_bytes = bio.getvalue()

                # STT（これだけは必須）
                jp = stt_openai(wav_bytes)
                if not jp:
                    continue

                # JP字幕は即表示（体感を最速にする肝）
                obs_set_text(TEXT_SOURCE_JP, wrap_fixed(jp, WRAP_JP_WIDTH))
                print("JP:", jp)

                # 非同期ワーカーへ（翻訳＋音声合成＋再生は裏で）
                try:
                    work_q.put_nowait(jp)
                except queue.Full:
                    # 渋滞時は古いものを捨てて最新優先
                    try:
                        _ = work_q.get_nowait()
                        work_q.task_done()
                    except Exception:
                        pass
                    try:
                        work_q.put_nowait(jp)
                    except Exception:
                        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()
        # ちょい待つ（daemonなので無くても止まるが、ログが綺麗）
        time.sleep(0.2)
        print("Stopped.")