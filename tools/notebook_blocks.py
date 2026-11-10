"""本文ブロックを個別のJupyter Notebookへ変換する。"""
import json
from pathlib import Path
from textwrap import dedent

# 断片をつなぐ既存の検証ランナー。完結した本文ブロックは直接セルに入れる。
RECIPES = {
    '03': ['gd1d', 'linreg', 'optimizers', 'training_loop', 'scheduler', 'before_fix'],
    '06': ['attention', 'matrix_kvcache', 'matrix_kvcache', 'matrix_kvcache', 'position'],
    '07': ['grpo', 'sft_pipeline', 'sft_pipeline', 'sft_pipeline', 'sft_pipeline', 'prompt'],
    '08': ['sampling', 'sampler', 'sampler', 'sampler', 'outlines', 'vllm_0.5B', 'outlines', 'constrained', 'constrained', 'cot_0.5B', 'tot_got'],
    '09': ['online_softmax', 'exactness', 'exactness', 'online_softmax', 'exactness'],
    '12': ['rag'] * 11 + ['ann'],
    '13': ['agent', 'agent', 'agent', 'function_calling', 'mcp', 'mcp', 'agent', 'agent', 'memory'],
    '14': ['eval_statistics', 'otel', 'otel', 'otel', 'otel'],
}


def source_text(path):
    """ノートブックの本文対応セルだけを返す（補助セルを実装として読まない）。"""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    cells = [c for c in data['cells'] if c.get('metadata', {}).get('book_source')]
    if len(cells) != 1:
        raise ValueError(f'{path}: 本文対応セルは1個である必要があります')
    c = cells[0]
    source = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
    if c['cell_type'] == 'markdown':
        if not source.startswith('```python\n') or not source.endswith('```\n'):
            raise ValueError(f'{path}: 参照コードのフェンスが不正です')
        return source[len('```python\n'):-len('```\n')]
    return source


def make_notebook(rel, block, runs, code):
    ch = rel[2:4]
    n = int(Path(rel).name.split('_')[1])
    preferred = RECIPES.get(ch, [])[n - 1] if ch in RECIPES else None
    direct = next((r for r in runs if r[0] == ch and (preferred is None or r[1] == preferred)
                   and not r[2].startswith('run_') and r[2] in block['text']), None)
    name = direct[1] if direct else preferred
    if name is None:
        raise ValueError(f'{rel}: 実行方法が未登録です')
    run = next(r for r in runs if r[0] == ch and r[1] == name)
    docker = run[2] in {'run_vllm.sh', 'run_collector.py'}
    heavy = run[3]
    # 引数を読む測定コードはJupyter自身のargvを渡さず、既存ランナーで動かす。
    softmax_example = ch == '09' and n == 1
    standalone = softmax_example or (direct is not None and 'sys.argv' not in block['text'])
    cells = []

    def add(kind, source, book_source=False):
        cell = {'cell_type': kind, 'id': f'block-{len(cells):02d}',
                'metadata': {'book_source': True} if book_source else {},
                'source': source.splitlines(keepends=True)}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    note = '本文のコードをコードセルで直接実行します。' if standalone else (
        '本文の断片は参照表示し、末尾のセルで前提を補う既存ランナーを実行します。'
        '関数・変数・外部処理が省略されているため、断片だけを単独実行しないでください。')
    add('markdown', f'# {Path(rel).stem}\n\nコード：`{rel}`\n\n{note}\n\n'
        + ('モデル・データの取得、または長い計算を含みます。\n\n' if heavy else '')
        + ('この例はローカルDocker用です。Colab管理ランタイムでの実行対象にはしていません。\n\n' if docker else '')
        + '上からセルを実行してください。ローカルではコード一式と同じPython環境をJupyterのカーネルに選びます。'
          'Colabでコード一式が必要な場合は、準備セルで配布ZIPをアップロードします。'
          '実行確認の範囲は `VALIDATION_COLAB_20261110.md` を参照してください。\n')
    if not standalone:
        add('code', dedent('''\
            import os
            import pathlib
            import sys
            import zipfile

            CODE = next((p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents,
                                    pathlib.Path.cwd() / 'gradient-attention-agent']
                         if (p / 'run_all.py').is_file() and (p / 'common/blocks.py').is_file()), None)
            if CODE is None:
                from google.colab import files
                uploaded = files.upload()  # gradient-attention-agent-code.zip を選ぶ
                if len(uploaded) != 1:
                    raise ValueError('配布ZIPを1つ選んでください')
                dest = pathlib.Path.cwd().resolve()
                with zipfile.ZipFile(next(iter(uploaded))) as z:
                    for member in z.namelist():
                        if not (dest / member).resolve().is_relative_to(dest / 'gradient-attention-agent'):
                            raise ValueError('同梱のコード配布ZIPを選んでください')
                    z.extractall(dest)
                CODE = dest / 'gradient-attention-agent'
                del uploaded
            print('コード一式:', CODE)
        '''))
    packages = []
    if not docker:
        files = ['requirements-colab.txt']
        if heavy and name not in {'data', 'train', 'bandwidth'}:
            files.append('requirements-colab-models.txt')
        for file in files:
            packages += [l.strip() for l in (code / file).read_text().splitlines()
                         if l.strip() and not l.lstrip().startswith('#')]
        if name == 'baseline':
            packages = [p for p in packages if not p.startswith(('transformers', 'tiktoken', 'pydantic', 'outlines', 'lancedb', 'pyarrow', 'sentence-transformers'))]
        add('code', 'import subprocess\nimport sys\n\n'
            '# 既に導入済みなら不要な更新をせず、満たしていない依存だけを導入する。\n'
            f'subprocess.run([sys.executable, "-m", "pip", "install", *{packages!r}], check=True)\n')
    if standalone:
        if softmax_example:
            add('markdown', '本文で省略した入力を補います。fp16の大きな値を例にします。\n')
            add('code', 'import numpy as np\nfrom numpy import exp\nx = np.array([11.0, 11.1], dtype=np.float16)\n')
        add('code', block['text'], True)
        if softmax_example:
            add('code', 'print("safe softmax:", p)\n')
    else:
        add('markdown', '## 本文の掲載コード（参照）\n')
        add('markdown', '```python\n' + block['text'] + '```\n', True)
        add('markdown', '## 前提を補って実行する\n\n'
            '既存ランナーが関連する断片と必要なデータ・スタブを接続します。'
            '外部サービスを実行した証明や、掲載用AI応答の完全証跡にはなりません。\n')
        if docker:
            add('code', 'import shutil\nimport subprocess\n\n'
                'if shutil.which("docker") is None:\n'
                '    raise RuntimeError("この例はDockerの動くローカル環境で実行してください")\n'
                '(CODE / "ch08/outputs").mkdir(parents=True, exist_ok=True)\n'
                'subprocess.run(["sh", str(CODE / "ch08/run_vllm.sh"), '
                '"Qwen/Qwen2.5-0.5B-Instruct"], cwd=CODE, check=True)\n')
        else:
            add('code', 'import subprocess\nimport sys\n\n'
                f'subprocess.run([sys.executable, str(CODE / "run_all.py"), "--only", "{ch}", '
                f'"--name={name}", "--skip-docker"], cwd=CODE, check=True)\n')
    return {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
        'book': {'path': rel, 'runner': name, 'standalone': standalone,
                 'heavy': heavy, 'docker': docker}}, 'nbformat': 4, 'nbformat_minor': 5}
