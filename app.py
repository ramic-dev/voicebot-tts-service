"""TTS Service — XTTS v2 text-to-speech API."""

import io
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

# torchaudio monkey-patch
def _sf_load(fp, **kw):
    y, sr = sf.read(str(fp), dtype="float32")
    if y.ndim == 2:
        y = y.mean(axis=1)
    return torch.FloatTensor(y).unsqueeze(0), sr

torchaudio.load = _sf_load
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

os.environ["COQUI_TOS_AGREED"] = "1"

from flask import Flask, request, jsonify, send_file
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)

_host_speaker = Path("/host/.speaker.pt")
_default_speaker = Path("/app/default_speaker.pt")
SPEAKER_FILE = os.environ.get("SPEAKER_FILE", "") or str(_host_speaker if _host_speaker.exists() else _default_speaker)
LANGUAGE = os.environ.get("LANGUAGE", "it")
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")

# Load model
log.info("Loading XTTS v2...")
mm = ModelManager()
model_dir, config_path, _ = mm.download_model("tts_models/multilingual/multi-dataset/xtts_v2")

config = XttsConfig()
config.load_json(str(config_path))
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=str(model_dir))
model.eval()

# Load speaker
log.info("Loading speaker from %s...", SPEAKER_FILE)
speaker_data = torch.load(SPEAKER_FILE, weights_only=True)
GPT_COND = speaker_data["gpt_cond_latent"]
SPEAKER_EMB = speaker_data["speaker_embedding"]
REF_RMS = float(speaker_data["ref_rms"])
log.info("TTS ready.")


@app.route("/synthesize", methods=["POST"])
def synthesize():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    fmt = data.get("format", "ogg")
    log.info("Synthesizing: %s", text[:80])

    try:
        out = model.inference(
            text=text, language=LANGUAGE,
            gpt_cond_latent=GPT_COND, speaker_embedding=SPEAKER_EMB,
            temperature=0.88, top_p=0.90, top_k=50,
            repetition_penalty=2.0, speed=1.0, enable_text_splitting=False,
        )
        wav = out["wav"]
        if hasattr(wav, "cpu"):
            wav = wav.cpu().numpy()
        wav = np.asarray(wav, dtype=np.float32).squeeze()

        # Loudness normalize
        wav_rms = float(np.sqrt(np.mean(wav**2)))
        if wav_rms > 0:
            wav = wav * (REF_RMS / wav_rms)

        # Write WAV
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp_wav.name, wav, 24000)
        tmp_wav.close()

        if fmt == "ogg":
            tmp_ogg = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
            tmp_ogg.close()
            subprocess.run(
                [FFMPEG_BIN, "-y", "-i", tmp_wav.name, "-c:a", "libopus", tmp_ogg.name],
                capture_output=True, timeout=30,
            )
            os.remove(tmp_wav.name)
            return send_file(tmp_ogg.name, mimetype="audio/ogg", as_attachment=True,
                           download_name="speech.ogg")
        else:
            return send_file(tmp_wav.name, mimetype="audio/wav", as_attachment=True,
                           download_name="speech.wav")

    except Exception as e:
        log.error("TTS failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "speaker": os.path.basename(SPEAKER_FILE)})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port)
