"""Find short sound events (impacts, laughter, paper tearing, wings, ...) on a timeline.

Two detectors on sliding windows (default 1 s, hop 0.5 s):
  1. AST (AudioSet 527 classes), minus each class's median over the file -> only what *stands out* from the background.
     Classes active in more than --recur of the file are treated as background; instrument-like classes are ignored while music plays.
  2. CLAP similarity to named sounds ("laughing", "paper tearing", ...), robust z-score against the file's own baseline.
Quiet windows (< --quiet dB) are skipped: both models hallucinate on silence.

Usage:
  python scripts/events.py <audio-or-video> [--find "dog barking,keyboard typing"] [--start 0 --end 0] [--lang ko|en]
"""
import argparse
import hashlib
import os
import tempfile

import numpy as np

from common import load, ast_model, clap_model, feat, db, pick

GENERIC = {'Music', 'Speech', 'Silence', 'Musical instrument', 'Sound effect', 'Inside, small room', 'Inside, large room or hall',
           'Male speech, man speaking', 'Female speech, woman speaking', 'Conversation', 'Narration, monologue',
           'Background music', 'Soundtrack music', 'Video game music', 'Ambient music'}
# instrument-like classes that fire on melody notes; ignored in windows where Music > --music-on
MUSICLIKE = {'Tuning fork', 'Wind chime', 'Chime', 'Bell', 'Church bell', 'Jingle bell', 'Glockenspiel', 'Marimba, xylophone',
             'Vibraphone', 'Singing bowl', 'Sine wave', 'Sonar', 'Doorbell', 'Ding-dong', 'Ding', 'Synthesizer', 'Guitar',
             'Plucked string instrument', 'Piano', 'Keyboard (musical)', 'Music box', 'Mallet percussion', 'Tubular bells', 'Harp',
             'Zither', 'Chirp tone', 'Hum', 'Ping', 'Percussion', 'Timpani', 'Drum', 'Bass drum', 'Cymbal', 'Strum',
             'Bowed string instrument', 'Violin, fiddle', 'Flute', 'Organ'}
FIND = [('laughing', '웃음'), ('wings flapping', '날갯짓'), ('paper tearing', '종이 찢기'), ('door slam', '문 쾅'),
        ('footsteps', '발소리'), ('heartbeat', '심장 박동'), ('scratching', '긁는 소리'), ('object falling and thudding', '물건 떨어짐'),
        ('gasp', '숨 들이켬'), ('crying', '울음'), ('applause', '박수'), ('glass breaking', '유리 깨짐'),
        ('bird chirping', '새소리'), ('rain', '빗소리'), ('whoosh', '휙')]


def windows(y, sr, win, hop):
    n, h = int(win * sr), int(hop * sr)
    for i in range(0, max(1, len(y) - n + 1), h):
        yield i / sr, y[i:i + n]


def merge(hits, gap):
    """Join consecutive windows with the same label."""
    out = []
    for t, label, z in sorted(hits):
        if out and out[-1][2] == label and t - out[-1][1] <= gap:
            out[-1][1] = t
            out[-1][3] = max(out[-1][3], z)
        else:
            out.append([t, t, label, z])
    return out


def main():
    ap = argparse.ArgumentParser(description='Find short sound events on a timeline.')
    ap.add_argument('file')
    ap.add_argument('--find', default='', help='extra sound names to search for (English, comma-separated)')
    ap.add_argument('--win', type=float, default=1.0)
    ap.add_argument('--hop', type=float, default=0.5)
    ap.add_argument('--start', type=float, default=0)
    ap.add_argument('--end', type=float, default=0)
    ap.add_argument('--ast-th', type=float, default=0.10, help='AST score above the class median to count as an event')
    ap.add_argument('--clap-z', type=float, default=3.0, help='CLAP robust z-score to count as an event')
    ap.add_argument('--quiet', type=float, default=-50, help='skip windows quieter than this (dBFS)')
    ap.add_argument('--recur', type=float, default=0.06, help='classes active in more than this share of windows are background')
    ap.add_argument('--music-on', type=float, default=0.08, help='Music score above which instrument-like classes are ignored')
    ap.add_argument('--lang', choices=['ko', 'en'], default='en')
    a = ap.parse_args()
    L = a.lang
    import torch

    au = load(a.file, no_audio_msg=pick(L, '소리 없음 — 소리 트랙이 없는 파일', 'No audio track in this file'))
    y16, y48 = au[16000], au[48000]
    if a.start > 0 or a.end > 0:
        e = a.end if a.end > 0 else len(y16) / 16000
        y16, y48 = y16[int(a.start * 16000):int(e * 16000)], y48[int(a.start * 48000):int(e * 48000)]
    off = a.start
    print(pick(L, f'파일: {a.file}  ({off:.1f}초부터 {len(y16)/16000:.1f}초, 창 {a.win}초 / {a.hop}초 이동)',
               f'File: {a.file}  (from {off:.1f}s, {len(y16)/16000:.1f}s, window {a.win}s / hop {a.hop}s)'))

    # 1. AST over time (cached per file+params: the slow part)
    fe, m = ast_model()
    lab = m.config.id2label
    key = hashlib.md5(f'{os.path.abspath(a.file)}|{os.path.getmtime(a.file)}|{a.win}|{a.hop}|{a.start}|{a.end}'.encode()).hexdigest()[:12]
    cache = os.path.join(tempfile.gettempdir(), f'audio_understand_ast_{key}.npz')
    if os.path.exists(cache):
        d = np.load(cache)
        P, T = d['P'], d['T']
    else:
        T, P = [], []
        W = list(windows(y16, 16000, a.win, a.hop))
        for b in range(0, len(W), 16):
            chunk = W[b:b + 16]
            x = fe([c[1] for c in chunk], sampling_rate=16000, return_tensors='pt')
            with torch.no_grad():
                P.append(torch.sigmoid(m(**x).logits).numpy())
            T += [c[0] for c in chunk]
        P, T = np.concatenate(P), np.array(T)
        np.savez(cache, P=P, T=T)

    n16 = int(a.win * 16000)
    loud = np.array([db(y16[int(t * 16000):int(t * 16000) + n16]) for t in T]) > a.quiet
    Z = P - np.median(P[loud] if loud.any() else P, axis=0, keepdims=True)
    recur = ((Z > a.ast_th) & loud[:, None]).mean(0) > a.recur
    mi = {v: k for k, v in lab.items()}['Music']
    hits = []
    for i, t in enumerate(T):
        if not loud[i]:
            continue
        for j in np.argsort(-Z[i])[:3]:
            name = lab[j]
            if name in GENERIC or recur[j] or Z[i, j] < a.ast_th:
                continue
            if name in MUSICLIKE and P[i, mi] > a.music_on:
                continue
            hits.append((t + off, name, float(Z[i, j])))
    dropped = [lab[j] for j in np.where(recur)[0] if lab[j] not in GENERIC]

    # 2. CLAP named-sound search
    cp, cm = clap_model()
    find = FIND + [(f.strip(), f.strip()) for f in a.find.split(',') if f.strip()]
    with torch.no_grad():
        te = feat(cm.get_text_features(**cp(text=[f'the sound of {f[0]}' for f in find], return_tensors='pt', padding=True)))
        te = te / te.norm(dim=-1, keepdim=True)
        A = []
        W48 = [c[1] for c in windows(y48, 48000, a.win, a.hop)]
        for b in range(0, len(W48), 16):
            ae = feat(cm.get_audio_features(**cp(audio=W48[b:b + 16], sampling_rate=48000, return_tensors='pt')))
            A.append((ae / ae.norm(dim=-1, keepdim=True)).numpy())
    S = np.concatenate(A) @ te.numpy().T
    n = min(len(S), len(loud))
    S, lc = S[:n], loud[:n]
    base = S[lc] if lc.any() else S
    med = np.median(base, 0)
    mad = np.median(np.abs(base - med), 0) + 1e-6
    Zc = (S - med) / (1.4826 * mad)
    chits = []
    for i in range(n):
        if not lc[i]:
            continue
        for k in np.argsort(-Zc[i])[:2]:
            if Zc[i, k] >= a.clap_z:
                chits.append((T[i] + off, find[k][1] if L == 'ko' else find[k][0], float(Zc[i, k])))

    # 3. one line per event: overlapping results from both detectors are grouped
    ev = sorted([(s, e + a.win, lbl) for s, e, lbl, _ in merge(hits, a.hop * 2)] +
                [(s, e + a.win, lbl) for s, e, lbl, _ in merge(chits, a.hop * 2)])
    grp = []
    for s, e, lbl in ev:
        if grp and s < grp[-1][1]:
            grp[-1][1] = max(grp[-1][1], e)
            if lbl not in grp[-1][2]:
                grp[-1][2].append(lbl)
        else:
            grp.append([s, e, [lbl]])
    print(pick(L, f'\n[튀는 소리 — {len(grp)}개, 시각 순]', f'\n[Events — {len(grp)}, by time]'))
    for s, e, ls in grp:
        print(f"  {s:6.1f}~{e:6.1f}{pick(L, '초', 's')}  " + ' · '.join(ls[:5]))
    if dropped:
        print(pick(L, f'\n[배경으로 보고 뺀 소리 — 전체의 {a.recur*100:.0f}% 넘게 깔림] ',
                   f'\n[Treated as background — active in >{a.recur*100:.0f}% of the file] ') + ', '.join(dropped[:12]))
    print(pick(L, '\n※ 모델 추정. 후보 시각을 귀로 확인할 것.', '\nNote: model estimates. Listen at these timestamps to confirm.'))


if __name__ == '__main__':
    main()
