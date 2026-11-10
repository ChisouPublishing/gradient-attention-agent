#!/usr/bin/env python3
"""Colabで参照実装を動かすためのコード一式をZIPにする。ノートブックは結合しない。"""
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

CODE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, 'w', ZIP_DEFLATED) as z:
        for p in sorted(CODE.rglob('*')):
            rel = p.relative_to(CODE)
            if (set(rel.parts) & {'work', 'outputs', '__pycache__', '.venv', '.git', 'ann_index'}
                    or rel.parts[1:3] == ('blocks', 'data') or p.name.endswith('_raw.json')):
                continue
            if p.is_file() and p.suffix in {'.py', '.ipynb', '.sh', '.md', '.txt', '.yaml', '.json'} and not p.name.startswith('colab_'):
                z.write(p, Path('gradient-attention-agent') / rel)
    print(args.output)


if __name__ == '__main__':
    main()
