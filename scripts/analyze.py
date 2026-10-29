"""Listen to a file: speech / music / sound-effect / silence per segment, top sound tags, mood, tempo, brightness, loudness, key.

Usage:
  python scripts/analyze.py <audio-or-video> [--seg 5] [--describe "cute ukulele,calm piano"] [--out DIR] [--lang ko|en]
"""
import argparse
import json
import os

import numpy as np

from common import load, ast_model, clap_model, db, pick

# AudioSet mood labels
MOOD = {'Happy music': ('행복한', 'happy'), 'Funny music': ('웃긴·장난스러운', 'funny'), 'Sad music': ('슬픈', 'sad'),
        'Tender music': ('다정한·잔잔한', 'tender'), 'Exciting music': ('신나는', 'exciting'), 'Angry music': ('화난', 'angry'),
        'Scary music': ('무서운', 'scary')}
# CLAP text prompts (English prompt, Korean label)
DESC = [('cute and playful', '귀엽고 장난스러운'), ('cheerful and bright', '밝고 경쾌한'), ('calm and relaxing', '잔잔하고 편안한'),
        ('warm and heartwarming', '따뜻하고 뭉클한'), ('sad and melancholic', '슬프고 쓸쓸한'), ('energetic and exciting', '힘차고 신나는'),
        ('tense and suspenseful', '긴장되는'), ('dreamy and soft', '몽글몽글 꿈같은'), ('funny and comical', '우스꽝스러운'),
        ('nostalgic', '그리운·추억 같은')]
KEYS_KO = ['도', '도#', '레', '레#', '미', '파', '파#', '솔', '솔#', '라', '라#', '시']
KEYS_EN = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])  # Krumhansl-Schmuckler
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def tags(model, fe, y):
    import torch
    x = fe(y, sampling_rate=16000, return_tensors='pt')
    with torch.no_grad():
        return torch.sigmoid(model(**x).logits[0]).numpy()


def main():
    ap = argparse.ArgumentParser(description='Listen to an audio/video file and describe it.')
    ap.add_argument('file')
    ap.add_argument('--seg', type=float, default=5.0, help='segment length in seconds (default 5; use 1-2 for short sound effects)')
    ap.add_argument('--describe', default='', help='extra CLAP mood prompts in English, comma-separated')
    ap.add_argument('--out', default='', help='folder to save a JSON report (for compare.py)')
    ap.add_argument('--lang', choices=['ko', 'en'], default='en')
    a = ap.parse_args()
    L = a.lang
    import torch
    import librosa

    au = load(a.file, no_audio_msg=pick(L, '소리 없음 — 소리 트랙이 없는 파일', 'No audio track in this file'))
    y16, y48 = au[16000], au[48000]
    dur = len(y16) / 16000
    print(pick(L, f'파일: {a.file}  길이 {dur:.1f}초', f'File: {a.file}  duration {dur:.1f}s'))

    fe, m = ast_model()
    lab = m.config.id2label
    idx = {v: k for k, v in lab.items()}

    # 1. what is in each segment
    seg = []
    for i in range(max(1, int(np.ceil(dur / a.seg)))):
        s, e = i * a.seg, min(dur, (i + 1) * a.seg)
        part = y16[int(s * 16000):int(e * 16000)]
        if len(part) < 1600:
            continue
        p = tags(m, fe, part)
        top = [(lab[j], float(p[j])) for j in np.argsort(-p)[:5]]
        rms = db(part)
        sp, mu = p[idx['Speech']], p[idx['Music']]
        kind = 'silence' if rms < -50 else ('speech' if sp > 0.5 and sp > mu else ('music' if mu > 0.2 else 'sfx/other'))
        seg.append({'start': s, 'end': e, 'kind': kind, 'speech': float(sp), 'music': float(mu), 'rms_db': rms, 'top': top})

    # 2. overall mood and tags (10 s chunks averaged)
    P = [tags(m, fe, y16[i:i + 160000]) for i in range(0, len(y16), 160000) if len(y16[i:i + 160000]) > 16000]
    Pm = np.mean(P, axis=0)
    mood = sorted([(pick(L, *MOOD[k]), float(Pm[idx[k]])) for k in MOOD], key=lambda t: -t[1])
    top_tags = [(lab[j], float(Pm[j])) for j in np.argsort(-Pm)[:12]]

    # 3. CLAP text-audio match (first 30 s)
    cp, cm = clap_model()
    desc = DESC + [(d.strip(), d.strip()) for d in a.describe.split(',') if d.strip()]
    inp = cp(text=[f'{d[0]} music' for d in desc], audio=[y48[:48000 * 30]], sampling_rate=48000, return_tensors='pt', padding=True)
    with torch.no_grad():
        lp = cm(**inp).logits_per_audio[0].softmax(-1).numpy()
    clap = sorted([(desc[i][1] if L == 'ko' else desc[i][0], float(lp[i])) for i in range(len(desc))], key=lambda t: -t[1])

    # 4. tempo, brightness, density, loudness, key
    tempo = float(np.atleast_1d(librosa.beat.beat_track(y=y48, sr=48000)[0])[0])
    half = tempo / 2 if tempo > 150 else None
    cent = float(np.mean(librosa.feature.spectral_centroid(y=y48, sr=48000)))
    loud = db(y48)
    onset = len(librosa.onset.onset_detect(y=y48, sr=48000)) / max(dur, 1e-6)
    chroma = librosa.feature.chroma_cqt(y=y48[:48000 * 60], sr=48000).mean(1)
    K = KEYS_KO if L == 'ko' else KEYS_EN
    cand = [(np.corrcoef(np.roll(MAJ, k), chroma)[0, 1], K[k] + pick(L, ' 장조(밝은 편)', ' major')) for k in range(12)] + \
           [(np.corrcoef(np.roll(MIN, k), chroma)[0, 1], K[k] + pick(L, ' 단조(어두운 편)', ' minor')) for k in range(12)]
    key = max(cand)[1]
    speed = pick(L, *(('느림', 'slow') if tempo < 80 else ('보통', 'moderate') if tempo < 115 else ('빠름', 'fast') if tempo < 145 else ('아주 빠름', 'very fast')))
    bright = pick(L, *(('어두움·묵직', 'dark') if cent < 1500 else ('보통', 'neutral') if cent < 2800 else ('밝음·쨍함', 'bright')))
    busy = pick(L, *(('듬성듬성', 'sparse') if onset < 1.5 else ('보통', 'moderate') if onset < 3.5 else ('촘촘·분주', 'busy')))

    KIND_KO = {'silence': '무음', 'speech': '말소리', 'music': '음악', 'sfx/other': '효과음·기타'}
    print(pick(L, '\n[구간별 소리]', '\n[Segments]'))
    for s in seg:
        k = KIND_KO[s['kind']] if L == 'ko' else s['kind']
        print(f"  {s['start']:6.1f}~{s['end']:6.1f}{pick(L, '초', 's')}  {k:<10} {s['rms_db']:6.1f} dB | " +
              ', '.join(f'{t}({v:.2f})' for t, v in s['top'][:3]))
    print(pick(L, '\n[전체 상위 태그]', '\n[Top tags overall]'))
    print('  ' + ', '.join(f'{t}({v:.2f})' for t, v in top_tags))
    print(pick(L, '\n[분위기 — AudioSet 분위기 항목 (값이 작아도 곡끼리 비교용)]', '\n[Mood — AudioSet mood labels (small values; compare across files)]'))
    print('  ' + ', '.join(f'{t} {v:.3f}' for t, v in mood))
    print(pick(L, '\n[분위기 — CLAP 글 맞춤, 합 1]', '\n[Mood — CLAP text match, sums to 1]'))
    print('  ' + ', '.join(f'{t} {v:.2f}' for t, v in clap[:6]))
    hs = pick(L, f', 반박자로 들으면 약 {half:.0f}', f', or ~{half:.0f} at half-time') if half else ''
    print(pick(L, f'\n[빠르기·결] 약 {tempo:.0f} BPM({speed}{hs}) · 밝기 {bright}({cent:.0f}Hz) · 소리 밀도 {busy}({onset:.1f}회/초) · 음량 {loud:.1f} dBFS · 조성 추정 {key}',
               f'\n[Tempo & texture] ~{tempo:.0f} BPM ({speed}{hs}) · brightness {bright} ({cent:.0f} Hz) · density {busy} ({onset:.1f} onsets/s) · loudness {loud:.1f} dBFS · key guess {key}'))
    print(pick(L, '\n※ 모델 추정값. 최종 판단은 귀로.', '\nNote: model estimates. Trust your ears for the final call.'))

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        rep = {'file': a.file, 'lang': L, 'duration': dur, 'segments': seg, 'mood_audioset': mood, 'top_tags': top_tags, 'clap': clap,
               'tempo_bpm': tempo, 'tempo_half': half, 'speed': speed, 'centroid_hz': cent, 'brightness': bright,
               'loudness_db': loud, 'onsets_per_s': onset, 'busy': busy, 'key_guess': key}
        name = os.path.splitext(os.path.basename(a.file))[0] + '.audio.json'
        with open(os.path.join(a.out, name), 'w', encoding='utf-8') as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
