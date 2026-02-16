import io, time, queue
import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import OpenAI
from obsws_python import ReqClient

# === OpenAI ===
client = OpenAI()  # OPENAI_API_KEY を環境変数に入れておく
MODEL = "gpt-4o-mini-transcribe"
LANG  = "ja"

# === OBS websocket ===
OBS_HOST="127.0.0.1"; OBS_PORT=4455; OBS_PASS="yourpass"
TEXT_SOURCE="Caption"
obs = ReqClient(host=OBS_HOST, port=OBS_PORT, password=OBS_PASS)

# === Audio ===
SR = 16000
CHANNELS = 1

BLOCK_SEC = 0.25               # 0.25秒ごとに音声を受け取る
SILENCE_SEC = 1.1              # これだけ無音が続いたら「区切り」
MIN_SPEECH_SEC = 2.2           # これ未満の短い発話は送らない（切れ防止）
MAX_UTTER_SEC = 25.0           # 1発の最大長（無限に溜めない）
SILENCE_THRESHOLD = 0.012      # 無音判定（環境で調整）

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))

def stt_openai(wav_bytes: bytes) -> str:
    bio = io.BytesIO(wav_bytes)
    bio.name = "audio.wav"
    t = client.audio.transcriptions.create(
        model=MODEL,
        file=bio,
        language=LANG,
        # prompt="駅名・路線名・地名が出ます。固有名詞はカタカナ/漢字で。"  # 任意
    )
    return (t.text or "").strip()

def obs_set_text(text: str):
    obs.set_input_settings(
        name=TEXT_SOURCE,
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
                    # 短すぎるので捨てる（ブツ切れ防止）
                    continue

                # WAV化（メモリ上）
                bio = io.BytesIO()
                sf.write(bio, audio, SR, format="WAV")
                wav_bytes = bio.getvalue()

                text = stt_openai(wav_bytes)
                if text:
                    print("STT:", text)
                    obs_set_text(text)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
