# 『LLMを仕組みから理解する』のコード

本書に載せたコードと、それを確かめるためのスクリプトです。本文に載っているコードは、`chNN/blocks/` の個別ノートブックの本文対応セルに同じ文字のまま入っています。

## 各コードを.ipynbで開く

本文のPythonコードは **1ブロックにつき1つの `.ipynb`** として `chNN/blocks/` に置いている。たとえば `ch02/blocks/02_01_正解率_ベースライン_リーク.ipynb`。本文の各コード欄にも同じ相対パスを記載している。すべてをまとめるノートブックは用意しない。YAMLは `.yaml` のまま維持する。

### ローカルJupyter / VS Code

コード一式のフォルダで、Python 3.11以降の仮想環境を作成して開く。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab -r requirements-colab.txt
jupyter lab
```

実行したい `chNN/blocks/*.ipynb` を開き、同じ仮想環境をカーネルに選んで上から実行する。各ノートブックに依存関係の導入セルを含む。モデル取得などを伴う例は追加の依存を導入する。

### Google Colab

実行したい個別の `.ipynb` をColabの「ノートブックをアップロード」で開く。完結したプログラムは掲載コードを直接コードセルで実行する。本文が関数・変数・外部クライアントを省略した断片の場合は、参照コードを表示し、末尾のセルで既存ランナーに前提・データ・スタブを補わせる。断片には対応するランナー名と対象範囲を記している。

断片をつなぐランナーはコード一式を必要とする。ローカルでは親フォルダから自動検出し、Colabでは準備セルでコード配布ZIPをアップロードする。コード一式のZIPは `python tools/package_code.py gradient-attention-agent-code.zip` で作れる（個別ノートブックを結合する処理ではない）。

vLLMのDocker例はローカル環境向けとして区別し、Colabでは実行しない。MCPを含む断片の検証ランナーは別Pythonプロセスで起動する。必要なimportは本文コードまたは準備セルに含め、手作業で一括追加する必要はない。実モデル例はメモリ・利用時間の制限を受ける。GPUランタイムの選択だけで全処理がGPUへ移るわけではない。実行結果はセッション終了前に保存する（[Colab FAQ](https://research.google.com/colaboratory/faq.html)）。確認範囲は [VALIDATION_COLAB_20261110.md](VALIDATION_COLAB_20261110.md) を参照。

## 2026-11-10の実行確認

Python3.13の同一環境で個別ノートブック71件の全セルを実行し、71/71件が完了しました。モデル・データ取得とローカルCPU版vLLMも含みます。詳細と各ファイルの記録は [検証記録](VALIDATION_COLAB_20261110.md) と [機械可読の結果](VALIDATION_RESULTS_20261110.json) を参照してください。Colab管理ランタイム・GPU・実際の業務APIは未検証です。

## ローカルでの動かし方

Python 3.13 で確かめています（2026 年 11 月 5 日）。

```bash
cd gradient-attention-agent
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python run_all.py             # すべて実行し、chNN/outputs/ に出力を保存する
python run_all.py --only 08   # 章を選ぶ（例：8 章だけ）
python run_all.py --fast      # モデルの取得や長い計算が要るものを飛ばす
python run_all.py --name=vllm # 名前の先頭で選ぶ
python run_all.py --skip-docker # Dockerを使う例を明示的に省く
```

`--fast` を付けないと、Hugging Face から次のモデルを取得します（初回のみ。合計で数 GB）：`Qwen/Qwen2.5-0.5B-Instruct`、`Qwen/Qwen2.5-1.5B-Instruct`、`intfloat/multilingual-e5-large`、`BAAI/bge-reranker-v2-m3`。tiktoken の語彙と、付録 A の FashionMNIST（約 30 MB）の取得にもネットワークを使います。GPU は要りません（CPU で動きます）。

### 実行権限

上の手順は `python run_all.py` とPython経由で起動するため、スクリプトへの実行権限の付与は不要です。

`./run_all.py` と直接起動する場合には、仮想環境を有効にしてから `chmod u+x run_all.py` で自分の実行権限を付与してください。`chmod 777` は不要です。

### 実行結果の例

Python 3.13 の仮想環境で `python run_all.py` を実行した結果の一例です。実行時間はCPU・メモリ・モデルやデータの取得済み状況などによって変わります。`OK` は各処理が終了コード0で完了したことを表し、モデルの全回答が正しいことを示すものではありません。

```text
OK  02 baseline           25.3 秒
OK  03 gd1d                0.0 秒
OK  03 linreg              0.1 秒
OK  03 optimizers          0.1 秒
OK  03 training_loop       6.8 秒
OK  03 before_fix          1.4 秒
OK  03 scheduler           1.4 秒
OK  04 backprop            0.8 秒
OK  05 tokens              8.4 秒
OK  06 attention           0.1 秒
OK  06 position            0.1 秒
OK  06 matrix_kvcache      0.7 秒
OK  07 grpo                0.1 秒
OK  07 sft_pipeline        2.5 秒
OK  07 prompt             98.9 秒
OK  08 sampling            0.1 秒
OK  08 sampler             6.1 秒
OK  08 outlines           13.0 秒
OK  08 constrained         4.5 秒
OK  08 cot_0.5B          436.3 秒
OK  08 cot_1.5B          928.2 秒
OK  08 vllm_0.5B         112.9 秒
OK  08 vllm_1.5B         112.9 秒
OK  08 vllm_0.5B_allow_ws  296.1 秒
OK  08 vllm_1.5B_allow_ws  524.3 秒
OK  08 tot_got             0.7 秒
OK  09 online_softmax      0.1 秒
OK  09 exactness           0.9 秒
OK  11 bandwidth          10.2 秒
OK  12 rag                24.0 秒
OK  12 ann                21.4 秒
OK  13 agent               0.0 秒
OK  13 mcp                 1.2 秒
OK  13 memory              7.7 秒
OK  13 function_calling   84.1 秒
OK  14 eval_statistics     0.0 秒
OK  14 otel                7.4 秒
OK  14 collector          26.8 秒
OK  18 tensor_parallel     0.1 秒
OK  18 ssm                 0.1 秒
OK  20 tensor              0.7 秒
OK  20 data                1.3 秒
OK  20 model               0.6 秒
OK  20 autograd            0.6 秒
OK  20 train               7.4 秒
OK  20 save                0.6 秒
46/46 件が終了コード 0
```

## 個別コードのファイル一覧

本文のパスと同じ、コードリポジトリ直下からの相対パスです。Pythonは `.ipynb`、設定ファイルは `.yaml` です。一覧は `tools/extract_blocks.py` が同期し、`--check` で記載漏れも検査します。

<!-- code-file-index:start -->

| ファイル | 形式 |
| --- | --- |
| [ch02/blocks/02_01_正解率_ベースライン_リーク.ipynb](ch02/blocks/02_01_正解率_ベースライン_リーク.ipynb) | .ipynb |
| [ch03/blocks/03_01_1_次元の勾配降下.ipynb](ch03/blocks/03_01_1_次元の勾配降下.ipynb) | .ipynb |
| [ch03/blocks/03_02_線形回帰をバッチと_SGD_で学習する.ipynb](ch03/blocks/03_02_線形回帰をバッチと_SGD_で学習する.ipynb) | .ipynb |
| [ch03/blocks/03_03_SGD_Momentum_Adam_の比較.ipynb](ch03/blocks/03_03_SGD_Momentum_Adam_の比較.ipynb) | .ipynb |
| [ch03/blocks/03_04_AdamW_＋_Warmup_＋_Cosine_.ipynb](ch03/blocks/03_04_AdamW_＋_Warmup_＋_Cosine_.ipynb) | .ipynb |
| [ch03/blocks/03_05_AdamW_＋_Warmup_＋_Cosine_.ipynb](ch03/blocks/03_05_AdamW_＋_Warmup_＋_Cosine_.ipynb) | .ipynb |
| [ch03/blocks/03_06_AdamW_＋_Warmup_＋_Cosine_.ipynb](ch03/blocks/03_06_AdamW_＋_Warmup_＋_Cosine_.ipynb) | .ipynb |
| [ch04/blocks/04_01_手計算の逆伝播と自動微分の検算.ipynb](ch04/blocks/04_01_手計算の逆伝播と自動微分の検算.ipynb) | .ipynb |
| [ch05/blocks/05_01_BPE_の_5_手と_4_つのトークナイザでの比.ipynb](ch05/blocks/05_01_BPE_の_5_手と_4_つのトークナイザでの比.ipynb) | .ipynb |
| [ch06/blocks/06_01_自己注意の_5_工程.ipynb](ch06/blocks/06_01_自己注意の_5_工程.ipynb) | .ipynb |
| [ch06/blocks/06_02_自己注意の_5_工程.ipynb](ch06/blocks/06_02_自己注意の_5_工程.ipynb) | .ipynb |
| [ch06/blocks/06_03_自己注意の_5_工程.ipynb](ch06/blocks/06_03_自己注意の_5_工程.ipynb) | .ipynb |
| [ch06/blocks/06_04_自己注意の_5_工程.ipynb](ch06/blocks/06_04_自己注意の_5_工程.ipynb) | .ipynb |
| [ch06/blocks/06_05_正弦波_学習型_RoPE_の比較.ipynb](ch06/blocks/06_05_正弦波_学習型_RoPE_の比較.ipynb) | .ipynb |
| [ch07/blocks/07_01_グループ内の相対アドバンテージ.ipynb](ch07/blocks/07_01_グループ内の相対アドバンテージ.ipynb) | .ipynb |
| [ch07/blocks/07_02_棄却サンプリングで_SFT_データを作り_損失マ.ipynb](ch07/blocks/07_02_棄却サンプリングで_SFT_データを作り_損失マ.ipynb) | .ipynb |
| [ch07/blocks/07_03_棄却サンプリングで_SFT_データを作り_損失マ.ipynb](ch07/blocks/07_03_棄却サンプリングで_SFT_データを作り_損失マ.ipynb) | .ipynb |
| [ch07/blocks/07_04_棄却サンプリングで_SFT_データを作り_損失マ.ipynb](ch07/blocks/07_04_棄却サンプリングで_SFT_データを作り_損失マ.ipynb) | .ipynb |
| [ch07/blocks/07_05_棄却サンプリングで_SFT_データを作り_損失マ.ipynb](ch07/blocks/07_05_棄却サンプリングで_SFT_データを作り_損失マ.ipynb) | .ipynb |
| [ch07/blocks/07_06_指示の書き方を変えて正解率を測る.ipynb](ch07/blocks/07_06_指示の書き方を変えて正解率を測る.ipynb) | .ipynb |
| [ch08/blocks/08_01_温度_top-k_top-p_min-p.ipynb](ch08/blocks/08_01_温度_top-k_top-p_min-p.ipynb) | .ipynb |
| [ch08/blocks/08_02_サンプラーを_40_行で書く_温度_top-k_.ipynb](ch08/blocks/08_02_サンプラーを_40_行で書く_温度_top-k_.ipynb) | .ipynb |
| [ch08/blocks/08_03_サンプラーを_40_行で書く_温度_top-k_.ipynb](ch08/blocks/08_03_サンプラーを_40_行で書く_温度_top-k_.ipynb) | .ipynb |
| [ch08/blocks/08_04_サンプラーを_40_行で書く_温度_top-k_.ipynb](ch08/blocks/08_04_サンプラーを_40_行で書く_温度_top-k_.ipynb) | .ipynb |
| [ch08/blocks/08_05_制約デコーディングを使う_自分で書く.ipynb](ch08/blocks/08_05_制約デコーディングを使う_自分で書く.ipynb) | .ipynb |
| [ch08/blocks/08_06_制約デコーディングを使う_自分で書く.ipynb](ch08/blocks/08_06_制約デコーディングを使う_自分で書く.ipynb) | .ipynb |
| [ch08/blocks/08_07_制約デコーディングを使う_自分で書く.ipynb](ch08/blocks/08_07_制約デコーディングを使う_自分で書く.ipynb) | .ipynb |
| [ch08/blocks/08_08_制約デコーディングを使う_自分で書く.ipynb](ch08/blocks/08_08_制約デコーディングを使う_自分で書く.ipynb) | .ipynb |
| [ch08/blocks/08_09_制約デコーディングを使う_自分で書く.ipynb](ch08/blocks/08_09_制約デコーディングを使う_自分で書く.ipynb) | .ipynb |
| [ch08/blocks/08_10_4_つの解かせ方で_正解と出力トークンを測る.ipynb](ch08/blocks/08_10_4_つの解かせ方で_正解と出力トークンを測る.ipynb) | .ipynb |
| [ch08/blocks/08_11_ToTと_GoTの骨組み.ipynb](ch08/blocks/08_11_ToTと_GoTの骨組み.ipynb) | .ipynb |
| [ch09/blocks/09_01_数値オーバーフロー_なぜ_最大値を引く_のか.ipynb](ch09/blocks/09_01_数値オーバーフロー_なぜ_最大値を引く_のか.ipynb) | .ipynb |
| [ch09/blocks/09_02_オンライン_softmax_3_パスを_1_パス.ipynb](ch09/blocks/09_02_オンライン_softmax_3_パスを_1_パス.ipynb) | .ipynb |
| [ch09/blocks/09_03_オンライン_softmax_3_パスを_1_パス.ipynb](ch09/blocks/09_03_オンライン_softmax_3_パスを_1_パス.ipynb) | .ipynb |
| [ch09/blocks/09_04_オンライン_softmax_の内部状態.ipynb](ch09/blocks/09_04_オンライン_softmax_の内部状態.ipynb) | .ipynb |
| [ch09/blocks/09_05_タイル化_FlashAttention.ipynb](ch09/blocks/09_05_タイル化_FlashAttention.ipynb) | .ipynb |
| [ch11/blocks/11_01_上限の見積もりと実測.ipynb](ch11/blocks/11_01_上限の見積もりと実測.ipynb) | .ipynb |
| [ch12/blocks/12_01_①〜③_取り込み_チャンク分割_埋め込み_格納.ipynb](ch12/blocks/12_01_①〜③_取り込み_チャンク分割_埋め込み_格納.ipynb) | .ipynb |
| [ch12/blocks/12_02_①〜③_取り込み_チャンク分割_埋め込み_格納.ipynb](ch12/blocks/12_02_①〜③_取り込み_チャンク分割_埋め込み_格納.ipynb) | .ipynb |
| [ch12/blocks/12_03_①〜③_取り込み_チャンク分割_埋め込み_格納.ipynb](ch12/blocks/12_03_①〜③_取り込み_チャンク分割_埋め込み_格納.ipynb) | .ipynb |
| [ch12/blocks/12_04_④⑤_ハイブリッド検索_RRF_融合_リランキン.ipynb](ch12/blocks/12_04_④⑤_ハイブリッド検索_RRF_融合_リランキン.ipynb) | .ipynb |
| [ch12/blocks/12_05_④⑤_ハイブリッド検索_RRF_融合_リランキン.ipynb](ch12/blocks/12_05_④⑤_ハイブリッド検索_RRF_融合_リランキン.ipynb) | .ipynb |
| [ch12/blocks/12_06_④⑤_ハイブリッド検索_RRF_融合_リランキン.ipynb](ch12/blocks/12_06_④⑤_ハイブリッド検索_RRF_融合_リランキン.ipynb) | .ipynb |
| [ch12/blocks/12_07_⑥_生成_引用を強制し_根拠がなければ棄権させる.ipynb](ch12/blocks/12_07_⑥_生成_引用を強制し_根拠がなければ棄権させる.ipynb) | .ipynb |
| [ch12/blocks/12_08_⑥_生成_引用を強制し_根拠がなければ棄権させる.ipynb](ch12/blocks/12_08_⑥_生成_引用を強制し_根拠がなければ棄権させる.ipynb) | .ipynb |
| [ch12/blocks/12_09_検索側と生成側を分けて測る.ipynb](ch12/blocks/12_09_検索側と生成側を分けて測る.ipynb) | .ipynb |
| [ch12/blocks/12_10_検索側と生成側を分けて測る.ipynb](ch12/blocks/12_10_検索側と生成側を分けて測る.ipynb) | .ipynb |
| [ch12/blocks/12_11_検索側と生成側を分けて測る.ipynb](ch12/blocks/12_11_検索側と生成側を分けて測る.ipynb) | .ipynb |
| [ch12/blocks/12_12_索引の種類と_再現率_速度.ipynb](ch12/blocks/12_12_索引の種類と_再現率_速度.ipynb) | .ipynb |
| [ch13/blocks/13_01_ReAct_ループとツール呼び出しの最小構成.ipynb](ch13/blocks/13_01_ReAct_ループとツール呼び出しの最小構成.ipynb) | .ipynb |
| [ch13/blocks/13_02_ReAct_ループとツール呼び出しの最小構成.ipynb](ch13/blocks/13_02_ReAct_ループとツール呼び出しの最小構成.ipynb) | .ipynb |
| [ch13/blocks/13_03_ReAct_ループとツール呼び出しの最小構成.ipynb](ch13/blocks/13_03_ReAct_ループとツール呼び出しの最小構成.ipynb) | .ipynb |
| [ch13/blocks/13_04_ツールの定義を渡し_呼び出しを実行し_結果を返す.ipynb](ch13/blocks/13_04_ツールの定義を渡し_呼び出しを実行し_結果を返す.ipynb) | .ipynb |
| [ch13/blocks/13_05_MCP_サーバを書く_エージェントから使う.ipynb](ch13/blocks/13_05_MCP_サーバを書く_エージェントから使う.ipynb) | .ipynb |
| [ch13/blocks/13_06_MCP_サーバを書く_エージェントから使う.ipynb](ch13/blocks/13_06_MCP_サーバを書く_エージェントから使う.ipynb) | .ipynb |
| [ch13/blocks/13_07_致命的な組み合わせを構造で断つ.ipynb](ch13/blocks/13_07_致命的な組み合わせを構造で断つ.ipynb) | .ipynb |
| [ch13/blocks/13_08_致命的な組み合わせを構造で断つ.ipynb](ch13/blocks/13_08_致命的な組み合わせを構造で断つ.ipynb) | .ipynb |
| [ch13/blocks/13_09_3_層の記憶.ipynb](ch13/blocks/13_09_3_層の記憶.ipynb) | .ipynb |
| [ch14/blocks/14_01_成功条件と対応のある比較を計算する.ipynb](ch14/blocks/14_01_成功条件と対応のある比較を計算する.ipynb) | .ipynb |
| [ch14/blocks/14_02_OpenTelemetry_で_RAG＋エージェ.ipynb](ch14/blocks/14_02_OpenTelemetry_で_RAG＋エージェ.ipynb) | .ipynb |
| [ch14/blocks/14_03_OpenTelemetry_で_RAG＋エージェ.ipynb](ch14/blocks/14_03_OpenTelemetry_で_RAG＋エージェ.ipynb) | .ipynb |
| [ch14/blocks/14_04_OpenTelemetry_で_RAG＋エージェ.ipynb](ch14/blocks/14_04_OpenTelemetry_で_RAG＋エージェ.ipynb) | .ipynb |
| [ch14/blocks/14_05_OpenTelemetry_で_RAG＋エージェ.ipynb](ch14/blocks/14_05_OpenTelemetry_で_RAG＋エージェ.ipynb) | .ipynb |
| [ch14/blocks/14_06_OpenTelemetry_で_RAG＋エージェ.yaml](ch14/blocks/14_06_OpenTelemetry_で_RAG＋エージェ.yaml) | .yaml |
| [ch18/blocks/18_01_列で分けて_行で分けて_1_回だけ足す.ipynb](ch18/blocks/18_01_列で分けて_行で分けて_1_回だけ足す.ipynb) | .ipynb |
| [ch18/blocks/18_02_再帰と畳み込みの一致_選択的にすると崩れること_.ipynb](ch18/blocks/18_02_再帰と畳み込みの一致_選択的にすると崩れること_.ipynb) | .ipynb |
| [ch20/blocks/20_01_テンソルを作る_計算する_形を変える_NumPy.ipynb](ch20/blocks/20_01_テンソルを作る_計算する_形を変える_NumPy.ipynb) | .ipynb |
| [ch20/blocks/20_02_自作の_Dataset_と_公開データセット.ipynb](ch20/blocks/20_02_自作の_Dataset_と_公開データセット.ipynb) | .ipynb |
| [ch20/blocks/20_03_3_層のネットを作り_パラメータを数える.ipynb](ch20/blocks/20_03_3_層のネットを作り_パラメータを数える.ipynb) | .ipynb |
| [ch20/blocks/20_04_勾配の求め方と_つまずきやすい_3_点.ipynb](ch20/blocks/20_04_勾配の求め方と_つまずきやすい_3_点.ipynb) | .ipynb |
| [ch20/blocks/20_05_FashionMNIST_を_3_エポック学習す.ipynb](ch20/blocks/20_05_FashionMNIST_を_3_エポック学習す.ipynb) | .ipynb |
| [ch20/blocks/20_06_state_dict_の保存と読み込み.ipynb](ch20/blocks/20_06_state_dict_の保存と読み込み.ipynb) | .ipynb |

<!-- code-file-index:end -->

## 補助スクリプトと設定ファイル

以下はノートブックから呼ぶスクリプトや編集・配布用ツールです。実行方法に合わせて `.py`／`.sh`／`.txt`／`.json` を維持しています。

| ファイル | 拡張子 |
| --- | --- |
| [ch03/run_scheduler.py](ch03/run_scheduler.py) | .py |
| [ch06/run_matrix_and_kvcache.py](ch06/run_matrix_and_kvcache.py) | .py |
| [ch07/run_sft_pipeline.py](ch07/run_sft_pipeline.py) | .py |
| [ch08/run_constrained.py](ch08/run_constrained.py) | .py |
| [ch08/run_outlines.py](ch08/run_outlines.py) | .py |
| [ch08/run_sampler.py](ch08/run_sampler.py) | .py |
| [ch08/run_vllm.py](ch08/run_vllm.py) | .py |
| [ch08/run_vllm.sh](ch08/run_vllm.sh) | .sh |
| [ch09/run_exactness.py](ch09/run_exactness.py) | .py |
| [ch12/run_rag.py](ch12/run_rag.py) | .py |
| [ch13/run_agent.py](ch13/run_agent.py) | .py |
| [ch13/run_mcp.py](ch13/run_mcp.py) | .py |
| [ch14/run_collector.py](ch14/run_collector.py) | .py |
| [ch14/run_otel.py](ch14/run_otel.py) | .py |
| [common/blocks.py](common/blocks.py) | .py |
| [manifest.json](manifest.json) | .json |
| [VALIDATION_RESULTS_20261110.json](VALIDATION_RESULTS_20261110.json) | .json |
| [requirements-colab-models.txt](requirements-colab-models.txt) | .txt |
| [requirements-colab.txt](requirements-colab.txt) | .txt |
| [requirements.txt](requirements.txt) | .txt |
| [run_all.py](run_all.py) | .py |
| [tools/extract_blocks.py](tools/extract_blocks.py) | .py |
| [tools/notebook_blocks.py](tools/notebook_blocks.py) | .py |
| [tools/package_code.py](tools/package_code.py) | .py |

## 中身

| 場所                        | 中身                                                                                                                     |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `chNN/blocks/`            | Pythonは個別.ipynb、YAMLは.yaml。本文対応セルは掲載順・掲載文字を維持。ファイル名は「章_順番_見出し」               |
| `chNN/run_*.py`           | 本文の断片（関数だけのコードなど）をつなぎ、スタブ（決まった応答を返す代わりの部品）や小さなモデルで動かす検証スクリプト |
| `chNN/outputs/`           | `run_all.py` の実行結果（`.out`）。失敗したときは `.err`                                                           |
| `chNN/data/`              | 検証に使うデータ（12 章の架空の社内規程など）                                                                            |
| `common/blocks.py`        | 検証スクリプトが`blocks/` からコードを探して読むための関数                                                             |
| `tools/notebook_blocks.py` | 個別.ipynbのセル構成と実行前提の生成 |
| `tools/package_code.py` | コード一式の配布ZIPを作る（ノートブックは結合しない） |
| `tools/extract_blocks.py` | 本文から`blocks/` を作り直す。`--check` で本文と一致しているかを確かめる                                             |
| `manifest.json`           | 各ブロックが本文のどの章・行・見出しにあるか                                                                             |
| `requirements-colab*.txt` | Colab用の基本依存・追加依存 |
| `requirements.txt`        | 確かめたときのライブラリの版                                                                                             |

`run_all.py` が実行するもの：

| 章 | 名前                                     | 種類          | 内容                                                                              |
| -- | ---------------------------------------- | ------------- | --------------------------------------------------------------------------------- |
| 2  | baseline                                 | ブロック      | 正解率の罠・ベースライン・データのリーク                                          |
| 3  | gd1d・linreg・optimizers                 | ブロック      | 1 次元の勾配降下、線形回帰、SGD / Momentum / Adam                                 |
| 3  | training_loop・before_fix・scheduler     | ブロック・run | AdamW＋Warmup＋Cosine、直す前の書き方、SequentialLR                               |
| 4  | backprop                                 | ブロック      | 逆伝播の手計算と自動微分、勾配消失                                                |
| 5  | tokens                                   | ブロック      | BPE の 5 手、4 つのトークナイザでのトークン数                                     |
| 6  | attention・position・matrix_kvcache      | ブロック・run | 自己注意の 5 工程、位置符号化、行列版と KV キャッシュの検算                       |
| 7  | grpo・sft_pipeline・prompt               | ブロック・run | GRPO、SFT データ作り（実トークナイザ）、プロンプトの書き方の実測（実モデル）      |
| 8  | sampling・sampler・outlines・constrained | ブロック・run | 温度と切り詰め、サンプラーと生成、Outlines の構造化出力、自作の制約デコーディング |
| 8  | cot_0.5B・cot_1.5B・tot_got              | ブロック      | 直接・CoT・Zero-shot CoT・多数決の実測（実モデル 2 つ）、ToT と GoT の骨組み      |
| 8  | vllm_0.5B・vllm_1.5B                     | run（Docker） | vLLM の構造化出力（CPU 版）                                                       |
| 9  | online_softmax・exactness                | ブロック・run | オンライン softmax、4 実装の一致                                                  |
| 11 | bandwidth                                | ブロック      | 帯域からの上限と実測（機械によって値が変わる）                                    |
| 12 | rag・ann                                 | run・ブロック | RAG 一式（実モデル）、近似検索の再現率と速度                                      |
| 13 | agent・mcp・memory・function_calling     | run・ブロック | エージェントの骨組み、MCP サーバとクライアント、3 層の記憶、Function Calling の往復（実モデル） |
| 14 | otel                                     | run           | OpenTelemetry の計装（メモリ上に記録）                                            |
| 14 | collector                                | run（Docker） | 実際の Collector で tail sampling を確かめる                                      |
| 18 | tensor_parallel・ssm                     | ブロック      | テンソル並列、状態空間モデル                                                      |
| 付録 A（ch20） | tensor・data・model・autograd・train・save | ブロック | PyTorch の基本（テンソル・Dataset/DataLoader・nn.Module・自動微分・学習・保存）  |

## Docker で動かすもの

次の 2 つは Docker（Apple Silicon の Mac では OrbStack か Docker Desktop）が要ります。Docker が動いていなければ、`run_all.py` は飛ばします。

| 章 | 名前 | 使うイメージ | 内容 |
| -- | ---- | ------------ | ---- |
| 8  | vllm_0.5B・vllm_1.5B（と `_allow_ws`） | `vllm/vllm-openai-cpu:v0.33.0-arm64`（約 1.2 GB） | 本文の vLLM のコードを vLLM の CPU 版で実行し、構造化出力が型として通るかを 9 回ずつ数える（モデル名だけ小型のものに差し替え）。`_allow_ws` は JSON の空白を許した場合との比較。`sh ch08/run_vllm.sh [モデル名] [allow_ws]` でも動く。x86 の機械では `VLLM_IMAGE=vllm/vllm-openai-cpu:v0.33.0` を指定する |
| 14 | collector | `otel/opentelemetry-collector-contrib:0.164.0` | 本文の tail sampling の設定で Collector を起動し、計装のコードから送ったトレースのうち何が残るかを数える |

## 実行していないもの

- vLLM の GPU での動き（GPU のメモリの割り当て、テンソル並列、速度）：CPU 版で動作だけを確かめています
- 実際の LLM の API を呼ぶ部分：LLM 呼び出しやツールの実装はスタブに置き換えています

## 本文を直したとき

本文（`../chapters/*.md`）のコードを直したら、`python3 tools/extract_blocks.py --label` で個別ノートブックと本文のパス表記を作り直し、`python run_all.py` で確かめてください。`python3 tools/extract_blocks.py --check` は、本文と `blocks/` が食い違っていると終了コード 1 で終わります。
