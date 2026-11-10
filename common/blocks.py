"""検証スクリプトから、本文に対応する個別ノートブックのコードを読む。"""
import json
import pathlib

CODE = pathlib.Path(__file__).resolve().parent.parent


def notebook_source(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = [c for c in data["cells"] if c.get("metadata", {}).get("book_source")]
    if len(cells) != 1:
        raise ValueError(f"{path}: 本文対応セルは1個である必要があります")
    c = cells[0]
    source = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
    if c["cell_type"] == "markdown":
        if not source.startswith("```python\n") or not source.endswith("```\n"):
            raise ValueError(f"{path}: 参照コードのフェンスが不正です")
        return source[len("```python\n"):-len("```\n")]
    return source


def block(ch, key):
    """本文対応セルだけを検索し、本文と同じ文字のソースを返す。"""
    for p in sorted((CODE / f"ch{ch}" / "blocks").glob("*")):
        if p.suffix == ".ipynb":
            text = notebook_source(p)
        elif p.suffix == ".yaml":
            text = p.read_text(encoding="utf-8")
        else:
            continue
        if key in text:
            return text
    raise KeyError(f"{ch} 章に「{key}」を含むブロックが無い")
