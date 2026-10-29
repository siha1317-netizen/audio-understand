"""Shared helpers: ffmpeg lookup, audio loading, model loading, output language."""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

AST_ID = 'MIT/ast-finetuned-audioset-10-10-0.4593'
CLAP_ID = 'laion/clap-htsat-unfused'

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')


def find_ffmpeg():
    """ffmpeg: $AUDIO_UNDERSTAND_FFMPEG > PATH > imageio-ffmpeg (pip) > error."""
    p = os.environ.get('AUDIO_UNDERSTAND_FFMPEG')
    if p and os.path.exists(p):
        return p
    p = shutil.which('ffmpeg')
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        sys.exit('ffmpeg not found. Install ffmpeg, or `pip install imageio-ffmpeg`, '
                 'or set AUDIO_UNDERSTAND_FFMPEG to the ffmpeg path.')


def load(path, rates=(16000, 48000), no_audio_msg='No audio track in this file'):
    """Decode any audio/video file to mono float32 at each sample rate. Exits cleanly if there is no audio."""
    import soundfile as sf
    ff = find_ffmpeg()
    tmp = tempfile.mkdtemp()
    out = {}
    for sr in rates:
        w = os.path.join(tmp, f'a{sr}.wav')
        r = subprocess.run([ff, '-v', 'error', '-y', '-i', path, '-vn', '-ac', '1', '-ar', str(sr), w], capture_output=True)
        if r.returncode != 0 or not os.path.exists(w) or os.path.getsize(w) < 1000:
            print(f'{no_audio_msg}: {path}')
            sys.exit(0)
        y, _ = sf.read(w, dtype='float32')
        out[sr] = y
    return out


def ast_model():
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    return ASTFeatureExtractor.from_pretrained(AST_ID), ASTForAudioClassification.from_pretrained(AST_ID).eval()


def clap_model():
    from transformers import ClapModel, ClapProcessor
    return ClapProcessor.from_pretrained(CLAP_ID), ClapModel.from_pretrained(CLAP_ID).eval()


def feat(o):
    """CLAP feature output: tensor on older transformers, output object (pooler_output = projected 512-d) on newer."""
    return o if hasattr(o, 'norm') else o.pooler_output


def db(x):
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9))


def pick(lang, ko, en):
    return ko if lang == 'ko' else en
