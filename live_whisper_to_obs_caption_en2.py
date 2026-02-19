import io, queue
import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import OpenAI
from obsws_python import ReqClient

# === OpenAI ===
client = OpenAI()  # OPENAI_API_KEY を環境変数に入れておく

# STT
STT_MODEL = "gpt-4o-mini-transcribe"
STT_LANG  = "ja"

# Translation
TRANS_MODEL = "gpt-4o-mini"   # 翻訳はこれで十分速く安い
TRANS_TEMPERATURE = 0.2

# === OBS websocket ===
OBS_HOST="127.0.0.1"; OBS_PORT=4455; OBS_PASS="yourpass"
TEXT_SOURCE_JP="Caption"
TEXT_SOURCE_EN="Caption_EN"
obs = ReqClient(host=OBS_HOST, port=OBS_PORT, password=OBS_PASS)

# === Audio ===
SR = 16000
CHANNELS = 1

BLOCK_SEC = 0.25
SILENCE_SEC = 1.1
MIN_SPEECH_SEC = 2.2
MAX_UTTER_SEC = 25.0
SILENCE_THRESHOLD = 0.012

# （任意）OBS側折り返しが効かない場合の保険。不要なら 0 のままでOK
WRAP_JP_WIDTH = 0
WRAP_EN_WIDTH = 0


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
        # prompt="配信字幕です。固有名詞はできるだけ正確に。"  # 任意
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


def main():
    blocksize = int(SR * BLOCK_SEC)
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print("Audio status:", status)
        q.put(indata.copy())

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
                data = q.get(timeout=0.5)
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
                    buf.append(x)  # 少し無音を入れると自然

            # バッファが長くなりすぎたら強制区切り
            if started and len(buf) >= max_blocks:
                silent = silence_blocks_needed

            # 無音が一定続いたら「確定」
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

                jp = stt_openai(wav_bytes)
                if not jp:
                    continue

                # 日本語字幕
                obs_set_text(TEXT_SOURCE_JP, wrap_fixed(jp, WRAP_JP_WIDTH))
                print("JP:", jp)

                # 英訳字幕（OpenAI）
                try:
                    en = translate_to_english_openai(jp)
                except Exception as e:
                    print("[Translate] error:", e)
                    en = ""

                obs_set_text(TEXT_SOURCE_EN, wrap_fixed(en, WRAP_EN_WIDTH) if en else "")
                if en:
                    print("EN:", en)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
