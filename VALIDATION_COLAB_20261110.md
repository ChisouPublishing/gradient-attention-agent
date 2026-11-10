# 個別ノートブックの検証（2026-11-10）

## 結果

- **71個の.ipynbをローカルJupyterで全セル実行し、71/71件が例外なく完了。** 先行39件も追加32件と同じPython3.13環境で再実行した。
- 実行済みノートブックと配布版の全コードセルを照合した。配布版の出力は空に保ち、コードのSHA-256と各ファイルの結果を [VALIDATION_RESULTS_20261110.json](VALIDATION_RESULTS_20261110.json) に記録。
- 本文のPythonブロック71件は個別.ipynb、YAML1件は.yaml。本文72か所の相対パス、READMEのファイル一覧、本文対応セルの文字列、nbformatと実行セルの構文を検査。
- モデル・トークナイザ取得を伴うSFTデータ作成、プロンプト比較、サンプリング、Outlines、自作制約、CoT比較、RAG11件、近似検索、Function Calling、記憶を実行。帯域計測とFashionMNISTの取得・3エポック学習も実行。
- vLLM例はローカルDockerのCPU版 `vllm/vllm-openai-cpu:v0.33.0-arm64` で実処理を実行。既存ランナーに従い、本文の8Bモデルから `Qwen/Qwen2.5-0.5B-Instruct` へモデル名を差し替えて検査した。GPUの検証ではない。
- 元の保存済みoutputsを保持し、ノートブックごとの一時コピーで実行した。書籍編集環境の `verification/individual_notebooks_20261110/` に実行済みノートブック、ログ、環境記録を保存。新しい生成原文は配布コードへ含めない。

## 環境と修正

- Python：3.13.3 (main, Apr  8 2025, 13:54:08) [Clang 16.0.0 (clang-1600.0.26.6)]
- NumPy 2.5.4、torch 2.15.0、torchvision 0.29.2、transformers 5.22.1、sentence-transformers 6.2.0。
- LanceDB 0.41.0、Outlines 1.3.3、pydantic 2.14.2、mcp 2.3.0。
- transformers5.22.1がPython3.11以上を要求するため、READMEの前提をPython3.11以降へ訂正。検査には独立したPython3.13のvenvを使用。
- vLLM用ノートブックの表示だけだったセルを、ローカルDockerの既存ランナーを実行するセルへ修正。
- Notebookの実行で生成される重み・データ・生JSONを配布ZIPから除外し、原稿抽出ツールが実行生成物を掲載コードと誤認しないよう修正。

## 確認範囲

成功はプログラムが最後まで実行されたという意味であり、各モデルの全回答の正しさを保証するものではない。外部サービスや業務処理をスタブで補う範囲はREADMEと各ノートブックに記す。

Colab管理ランタイム、GPU、実際の業務API、掲載用AI応答の完全証跡は今回の検証対象外。過去の生成結果の証跡不足を解消したとは扱わず、新しい生成結果も本文へ追加掲載しない。

## 編集・再検査

書籍編集環境では `python tools/extract_blocks.py --label` で個別.ipynbと本文パス・README一覧を同期し、`python tools/extract_blocks.py --check` で照合する。コード専用cloneでは抽出ツールを実行せず、個別.ipynbをJupyterで開く。まとめノートブックは用意しない。
