"""Rank several analyzed files by how well they match a wanted mood.

Usage:
  python scripts/analyze.py song1.mp3 --out reports
  python scripts/analyze.py song2.mp3 --out reports
  python scripts/compare.py reports --want "cute and playful"
"""
import argparse
import glob
import json
import os

from common import pick


def main():
    ap = argparse.ArgumentParser(description='Compare *.audio.json reports made by analyze.py --out.')
    ap.add_argument('folder')
    ap.add_argument('--want', default='cute and playful', help='a CLAP mood label as printed by analyze.py (same --lang)')
    ap.add_argument('--lang', choices=['ko', 'en'], default='en')
    a = ap.parse_args()
    L = a.lang
    rows = []
    for f in sorted(glob.glob(os.path.join(a.folder, '*.audio.json'))):
        with open(f, encoding='utf-8') as fh:
            d = json.load(fh)
        clap = dict(d['clap'])
        mood = dict(d['mood_audioset'])
        rows.append((clap.get(a.want, 0.0), os.path.basename(d['file']), clap, mood, d))
    if not rows:
        print(pick(L, '분석 파일이 없다. analyze.py --out 으로 먼저 만들 것.', 'No reports found. Run analyze.py --out first.'))
        return
    rows.sort(key=lambda r: -r[0])
    hk = ['행복한', '웃긴·장난스러운', '슬픈'] if L == 'ko' else ['happy', 'funny', 'sad']
    print(pick(L, f'원하는 느낌: {a.want} (점수 높은 순)\n', f'Wanted: {a.want} (best match first)\n'))
    print(pick(L, f'| 순위 | 파일 | {a.want} | CLAP 1위 | 행복한 | 웃긴 | 슬픈 | BPM | 밝기 |',
               f'| # | file | {a.want} | top CLAP mood | happy | funny | sad | BPM | brightness |'))
    print('|---|---|---|---|---|---|---|---|---|')
    for i, (w, name, clap, mood, d) in enumerate(rows, 1):
        top = max(clap.items(), key=lambda t: t[1])
        bpm = f"{d['tempo_bpm']:.0f}" + (f"(/{d['tempo_half']:.0f})" if d.get('tempo_half') else '')
        print(f"| {i} | {name} | {w:.2f} | {top[0]} {top[1]:.2f} | {mood.get(hk[0], 0):.3f} | {mood.get(hk[1], 0):.3f} | "
              f"{mood.get(hk[2], 0):.3f} | {bpm} | {d['brightness']} |")
    if len(rows) > 1 and rows[0][0] - rows[1][0] < 0.05:
        print(pick(L, '\n1·2위 차이 0.05 안쪽 — 비슷함. 귀로 고를 것.', '\nTop two within 0.05 — too close to call; listen.'))


if __name__ == '__main__':
    main()
