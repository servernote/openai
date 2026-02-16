import io, time, queue
import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import OpenAI
from obsws_python import ReqClient

import argostranslate.package
import argostranslate.translate

# === OpenAI ===
client = OpenAI()  # OPENAI_API_KEY を環境変数に入れておく
MODEL = "gpt-4o-mini-transcribe"
LANG  = "ja"

# === OBS websocket ===
OBS_HOST="127.0.0.1"; OBS_PORT=4455; OBS_PASS="yourpass"
TEXT_SOURCE_JP="Caption"
TEXT_SOURCE_EN="Caption_EN"   # ← OBSに追加した英語字幕ソース名
obs = ReqClient(host=OBS_HOST, port=OBS_PORT, password=OBS_PASS)

# === Audio ===
SR = 16000
CHANNELS = 1

BLOCK_SEC = 0.25               # 0.25秒ごとに音声を受け取る
SILENCE_SEC = 1.1              # これだけ無音が続いたら「区切り」
MIN_SPEECH_SEC = 2.2           # これ未満の短い発話は送らない（切れ防止）
MAX_UTTER_SEC = 25.0           # 1発の最大長（無限に溜めない）
SILENCE_THRESHOLD = 0.012      # 無音判定（環境で調整）

# （任意）OBS側の折り返しが効かない場合の保険
WRAP_JP_WIDTH = 0   # 0=折り返ししない / 例: 28 などで強制改行
WRAP_EN_WIDTH = 0   # 英語も同様（0推奨、必要なら 42 とか）


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
        model=MODEL,
        file=bio,
        language=LANG,
        # prompt="配信の字幕です。固有名詞はなるべく正確に。"  # 任意
    )
    return (t.text or "").strip()


def obs_set_text(source_name: str, text: str):
    obs.set_input_settings(
        name=source_name,
        settings={"text": text},
        overlay=True
    )


# ========= Argos Translate (ja->en) =========

def ensure_argos_ja_en_installed():
    """
    Argosの ja->en 翻訳モデルが無ければ、初回だけ自動でダウンロード＆インストールする。
    （無料。ネット接続は初回のみ必要）
    """
    installed = argostranslate.translate.get_installed_languages()
    if installed:
        has_ja = any(l.code == "ja" for l in installed)
        has_en = any(l.code == "en" for l in installed)
        if has_ja and has_en:
            ja = next(l for l in installed if l.code == "ja")
            en = next(l for l in installed if l.code == "en")
            if ja.get_translation(en) is not None:
                return

    print("[Argos] ja->en model not found. Downloading & installing (first time only)...")
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()

    pkg = None
    for p in available:
        if p.from_code == "ja" and p.to_code == "en":
            pkg = p
            break
    if pkg is None:
        raise RuntimeError("[Argos] Could not find ja->en package in Argos index.")

    pkg_path = pkg.download()
    argostranslate.package.install_from_path(pkg_path)
    print("[Argos] Installed ja->en model.")


def translate_ja_to_en_argos(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    installed = argostranslate.translate.get_installed_languages()
    ja = next(l for l in installed if l.code == "ja")
    en = next(l for l in installed if l.code == "en")
    tr = ja.get_translation(en)
    if tr is None:
        raise RuntimeError("[Argos] ja->en translation not available (model missing).")
    return tr.translate(text).strip()


# ========= Main =========

def main():
    ensure_argos_ja_en_installed()

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
                jp_disp = wrap_fixed(jp, WRAP_JP_WIDTH)
                obs_set_text(TEXT_SOURCE_JP, jp_disp)
                print("JP:", jp)

                # 英語字幕（Argos：無料）
                try:
                    en = translate_ja_to_en_argos(jp)
                except Exception as e:
                    print("[Argos] Translate error:", e)
                    en = ""

                if en:
                    en_disp = wrap_fixed(en, WRAP_EN_WIDTH)
                    obs_set_text(TEXT_SOURCE_EN, en_disp)
                    print("EN:", en)
                else:
                    # 翻訳失敗時は英語字幕を空に（好みでJPを出すでもOK）
                    obs_set_text(TEXT_SOURCE_EN, "")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
