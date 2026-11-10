#!/usr/bin/env python3
"""本文（../chapters/*.md）のコードブロックを、載っている順・載っている文字のまま chNN/blocks/ に書き出す。

  python3 tools/extract_blocks.py          # 書き出す（manifest.json も更新）
  python3 tools/extract_blocks.py --check  # 本文と blocks/ が一致しているかだけ確かめる（不一致なら終了コード 1）

--label は各コード欄の直前にリポジトリ相対パスを付ける。

ファイル名は「章_順番_見出し.拡張子」。見出しは、直前の「実装例」の見出し（無ければ節の見出し）から作る。
"""
import importlib.util, json, pathlib, re, sys
from notebook_blocks import make_notebook, source_text

CODE = pathlib.Path(__file__).resolve().parent.parent
BOOK = CODE.parent
EXT = {"python": "ipynb", "yaml": "yaml"}   # text（数式・実行結果の写し）はコードではないので入れない

def slug(heading):
    s = re.sub(r"^#+\s*(実装例：|参照実装：|\d+\s+)?", "", heading)
    s = re.sub(r"[（(].*?[）)]", "", s)
    s = re.sub(r"[\s/／・：:、，,。「」『』]+", "_", s).strip("_")
    return s[:24] or "block"

def extract():
    out = {}
    for md in sorted((BOOK / "chapters").glob("*.md")):
        ch = md.name[:2]
        lines = md.read_text(encoding="utf-8").split("\n")
        h2, impl, i, k = "", "", 0, 0
        while i < len(lines):
            l = lines[i]
            if l.startswith("## "):
                h2, impl = l, ""                     # 節が変わったら「実装例」の見出しを忘れる
            elif re.match(r"^#{3,4} (実装例|参照実装)", l):
                impl = l
            heading = impl or h2
            m = re.match(r"^(`{3,4})(python|yaml|text)$", l)
            if m:
                fence, lang = m.group(1), m.group(2)
                j = i + 1
                while lines[j] != fence:
                    j += 1
                if lang not in EXT:
                    i = j + 1
                    continue
                k += 1
                name = f"{ch}_{k:02d}_{slug(heading)}.{EXT[lang]}"
                out[f"ch{ch}/blocks/{name}"] = {"text": "\n".join(lines[i + 1:j]) + "\n",
                                                "chapter": md.name, "line": i + 1, "heading": heading}
                i = j
            i += 1
    return out

def main():
    blocks = extract()
    if "--label" in sys.argv:
        by_chapter = {}
        for rel, b in blocks.items():
            by_chapter.setdefault(b["chapter"], []).append((b["line"], rel))
        for chapter, entries in by_chapter.items():
            path = BOOK / "chapters" / chapter
            lines = path.read_text(encoding="utf-8").split("\n")
            for line, rel in sorted(entries, reverse=True):
                label = f"コード：`{rel}`"
                if line >= 3 and lines[line - 3].startswith("コード：`"):
                    lines[line - 3] = label
                else:
                    lines[line - 1:line - 1] = [label, ""]
            path.write_text("\n".join(lines), encoding="utf-8")
        blocks = extract()
    def read_source(rel):
        return source_text(CODE / rel) if rel.endswith(".ipynb") else (CODE / rel).read_text(encoding="utf-8")
    def label_matches(rel, b):
        lines = (BOOK / "chapters" / b["chapter"]).read_text(encoding="utf-8").split("\n")
        return b["line"] >= 3 and lines[b["line"] - 3] == f"コード：`{rel}`"

    if "--check" in sys.argv:
        bad = [p for p, b in blocks.items() if not (CODE / p).is_file() or read_source(p) != b["text"] or not label_matches(p, b)]
        extra = [str(p.relative_to(CODE)) for p in CODE.glob("ch*/blocks/*") if p.is_file() and p.suffix in {".py", ".ipynb", ".yaml"} and str(p.relative_to(CODE)) not in blocks]
        for p in bad: print("本文と違う／無い:", p)
        for p in extra: print("本文に無い:", p)
        readme = (CODE / "README.md").read_text(encoding="utf-8")
        section = readme.split("<!-- code-file-index:start -->", 1)[-1].split("<!-- code-file-index:end -->", 1)[0]
        listed = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", section)
        index_bad = (len(listed) != len(blocks) or {target for _, target in listed} != set(blocks)
                     or any(label != target for label, target in listed))
        if index_bad:
            print("READMEのコードファイル一覧に不足・不一致があります")
        print("一致" if not bad and not extra and not index_bad else f"不一致 {len(bad) + len(extra) + int(index_bad)} 件")
        return 1 if bad or extra or index_bad else 0
    spec = importlib.util.spec_from_file_location("book_runs", CODE / "run_all.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    expected = set(blocks)
    stale = [p for p in CODE.glob("ch*/blocks/*") if p.is_file() and p.suffix in {".py", ".ipynb", ".yaml"} and str(p.relative_to(CODE)) not in expected]
    if stale:
        raise SystemExit("旧ファイルが残っています。先にgit mvで改名してください: " + ", ".join(str(p) for p in stale))
    for p, b in blocks.items():
        (CODE / p).parent.mkdir(parents=True, exist_ok=True)
        text = (json.dumps(make_notebook(p, b, runner.RUNS, CODE), ensure_ascii=False, indent=1) + "\n"
                if p.endswith(".ipynb") else b["text"])
        (CODE / p).write_text(text, encoding="utf-8")
    manifest = {p: {k: v for k, v in b.items() if k != "text"} for p, b in blocks.items()}
    (CODE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    readme = CODE / "README.md"
    text = readme.read_text(encoding="utf-8")
    start, end = "<!-- code-file-index:start -->", "<!-- code-file-index:end -->"
    if start in text and end in text:
        rows = ["", "", "| ファイル | 形式 |", "| --- | --- |"]
        rows += [f"| [{rel}]({rel}) | {pathlib.Path(rel).suffix} |" for rel in sorted(blocks)]
        readme.write_text(text.split(start, 1)[0] + start + "\n".join(rows) + "\n\n" + end
                          + text.split(end, 1)[1], encoding="utf-8")
    print(f"{len(blocks)} ブロックを書き出した（README一覧も同期）")
    return 0

if __name__ == "__main__":
    sys.exit(main())
