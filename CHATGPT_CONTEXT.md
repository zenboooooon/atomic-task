# ChatGPT向けの実験資料

このファイルは、実験の設定・結果・実装をChatGPTに伝えるための詳細資料です。人向けの概要は[README.md](README.md)、最新の複合評価は[DOWNSTREAM_VALIDATION.md](DOWNSTREAM_VALIDATION.md)、初回の全体記録は[EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)にあります。

## 対象と記録時点

- 現行実験: 原子 `gaze-jump-v0.10-single-color`。複合環境は `gaze-jump-v0.10-wide24-start-back4`、R追加後の統合計画は `gaze-jump-v0.10-wide24-start-back4-psfr`。
- 初回の全体スナップショット: **2026-10-20 01:23:16 JST**。この時点で原子3seedの2000更新と最終validationは完了し、複合seed0のP/S/Fを学習中、正式validationは0/90回でした。履歴として保持しています。
- **2026-10-20 02:02 JSTの変更**: R条件3seedを追加し、既存9学習ジョブの後に待機登録。01:23の数値スナップショットを新条件の結果に読み替えないでください。
- 最新の複合決定的validationスナップショット: **2026-10-20 17:15:53 JST、117/120評価**。P/S/Fの9ジョブは2000更新の最終評価まで完了、Rは各seedの1800更新まで取得済み。Rの2000更新は未評価であり、0%ではありません。
- 最新の複合評価は[results/LATEST_VALIDATION.json](results/LATEST_VALIDATION.json)、初回の全体記録は[results/LATEST.json](results/LATEST.json)の参照先と取得時刻を確認します。今回の評価スナップショットには新しい訓練累計は含めていません。

## 現行プロトコル

| 項目 | 実際の設定 |
|---|---|
| 原子課題 | look、approach、jump |
| 配色 | 全課題で目標・台＝橙、自機＝シアンの1組 |
| 共有部 | 幅256の空間ViT＋時間Transformer。4画像と間の3行動の履歴 |
| 原子ヘッド | 各タスクに独立したactor/critic各1組、合計3組 |
| 観測 | RGBカメラ1個。固定Depth Anything V2で推定深度を生成し、RGBと合わせ96×96×4入力 |
| 操作 | 移動2、視点2、ジャンプ1の5連続入力。全課題でジャンプ入力可能 |
| 学習 | PPO。画像・特徴・平均行動の一致等の補助損失なし |
| look成功 | 目標を中央7度以内に0.5秒連続保持。自由移動 |
| approach成功 | 目標接触 |
| jump成功 | 床から跳躍し、高さ1mの台上面へ着地 |
| 複合成功 | 静的障害物3個を回避し、台にジャンプして着地し、台上の宝へ接触 |
| 原子学習量 | seed0・1・2、各2000更新。各更新3タスク×256＝768判断、各seed1,536,000判断 |
| 原子保存・評価 | 100更新ごと保存。最終モデルを各タスク36エピソードで決定的評価 |
| 複合学習量 | P/S/F/R×3seed、各2000更新×256＝512,000判断 |
| 複合保存・評価 | 200更新ごとの10モデルを各32エピソードで評価、計120回 |
| 評価方策 | 環境seed9000、tanh平均行動 |
| 並列数 | 学習は最大3ジョブ、複合評価は別の1枠 |
| 複合開始順 | P0、S0、F0、P1、S1、F1、P2、S2、F2、R0、R1、R2 |

原子は元の18m四方です。複合は動画確認後に承認された24m四方版で、開始位置を宝と反対方向へ4m後退させ、開始水平距離10.05–10.85m、場外判定11.65mとしています。

台・宝・3障害物の位置と寸法、物理、報酬、成功条件、TPSカメラ、PPO、seed、学習量は広域化の際に変更していません。原子3seedの全学習・最終評価・プロセス終了後に複合へ移行しています。


## 読む順番

1. `DOWNSTREAM_VALIDATION.md` と `results/LATEST_VALIDATION.json` で最新の複合評価の取得時刻・条件・seed・更新数を確認。
2. 最新評価の参照先にある `summary.json` と `validation_results.csv` を確認。未評価セルは0%に置き換えない。
3. 初回の全体記録は `EXPERIMENT_REPORT.md` と `results/LATEST.json` の参照先の `summary.json`・`training_summary.csv`。原子の詳細は `primitive_updates.csv` と `raw/primitive_seed*/{report,validation}.json`。
4. 初回時点の複合訓練の詳細は `downstream_updates.csv` と `raw/downstream/`。現在の最終結果と取り違えない。
5. 実装は `PROTOCOL.md`、`configs/gaze_v10_contract.json`、`configs/gaze_v10_wide24_change.json`、`python/tps_transfer/`、`environment_sources/wide24/`。

## この実験で比較していること

原子学習から得た共有表現が、複合課題の学習に役立つかをP/S/F/Rで比較します。

- **P**: 原子学習済み共有部を固定し、新規actor/criticを学習。
- **S**: ランダムな共有部と新規actor/criticを学習。
- **F**: 原子学習済み共有部を解凍し、新規actor/criticとともに学習。
- **R**: Sと同じランダム初期共有部を固定し、新規actor/criticだけを学習。

同じseedのP/S/F/Rは、新規ヘッド・環境・行動サンプリングの初期seedを揃えます。原子のactor/criticは転移しません。

P/Fには原子事前学習の経験が加わります。複合での判断数が同じでも、Sと事前学習を含む総計算量・総経験量が同じとは限りません。

## R追加の実行・検証・読み取り

Rは共有部全体（空間ViT、時間モデル、行動埋め込み）を凍結します。network seed=`20000+seed`、head=`30000+seed`、環境=`10000+seed`、行動サンプリング=`40000+seed`。Rの共有部初期値は対応seedのSと同一、新規actor/critic初期値はP/S/F/Rで同一です。

既存P/S/Fの監督・ワーカー・出力・開始順を保つため、R専用監督を追加しました。既存9学習ジョブの完了とプロセス終了後にR0〜2を最大3並列で学習します。Rの正式評価30回は既存90評価の終了後に同じ1評価枠で行います。環境・PPO・報酬・観測・物理・学習量は追加前と同じです。

- R出力: `runs/gaze_v10_2000/downstream_wide24/R_seed0..2`。
- R監督: `scripts/run_gaze_v10_r_extension.py`。状態・計画は同出力親の `r_extension/`。
- 全12ジョブの基準: `combined.run.plan.json`、`combined.progress.json`、`combined.pipeline.status`。元の無接頭辞ファイルは既存P/S/Fの9ジョブ・90評価の記録です。
- `scripts/read_gaze_v10_status.py` で学習指標と正式評価を読み取り集計。元の監督が正常終了してもR未完了なら全体完了ではありません。
- 定期動画は50バッチ（原子10、P/S/F/R各10）。全12学習・120評価・50動画バッチの終了と配信後に監視を終了します。
- 1更新の別試験で、R共有部がS初期値と完全一致し、更新後も不変、actor/criticは両方更新されることを確認。モデル・PPO・既存実行コード・配置の固定ハッシュを照合しました。別試験は正式結果に含めません。
- 変更表と検証証拠: `configs/gaze_v10_r_extension_change.json`、`results/amendments/2026-10-20-add-R/`。

## 指標の区別

| 種類 | 方策・配置 | 意味 |
|---|---|---|
| 訓練累計 | 訓練配置・TanhGaussianの確率的行動 | 学習に採用された終了エピソードの累計 |
| 正式な決定的validation | 固定評価配置・環境seed9000・tanh平均行動 | 保存モデルの評価。原子は各タスク36、複合は各モデル32エピソード |
| 定期動画 | 保存モデルからの新しい確率的試行・訓練配置 | 観察用に成功と失敗を探した例。無作為標本の評価ではない |
| スクリプト操作 | 環境内部の位置を使う確認用制御 | 物理・配置・成功条件の動作確認。学習結果ではない |
| smoke | 短時間の別実行 | 接続・保存・固定／更新などの確認。正式な学習量には含めない |

訓練成功率は `sum(successes) / sum(successes + failures + interruptions)`。
同じupdateはseed・ジョブ内で重複排除し、未完了エピソードと破棄された余剰遷移を分母に入れません。0/0は率未算出です。

原子CSVの `agent_decisions_trained` は **seed全体の3タスク合計**です。タスクごとの3行に同じ値があるので足し合わせないでください。タスク別判断数はupdate×256です。`training_summary.csv` の `decisions_scope` と `task_decisions` でも区別しています。

## 現行条件を取り違えないための情報

- v0.10は全原子タスクで単一配色。目標・台は橙、自機はシアン。
- 観測センサーはRGBカメラ1個。RGBから固定の事前学習済み深度推定器で深度チャネルを生成します。環境の真の深度センサーを追加したものではありません。
- 共有部には画像特徴に加え過去行動の時間モデルを含みます。画像だけの空間ViT単体の比較ではありません。
- 実験時の幅は256。モデルクラスにある既定値128を現行条件と取り違えないでください。
- ジャンプ入力はlook/approachにもあります。jumpのときだけ有効にするマスクはありません。
- 原子は元の18m四方、複合は承認済みの24m四方・開始位置4m後退版。
- 広域化で台・宝・3障害物の位置／寸法、物理、報酬、観測、PPOは変えていません。場外判定は床に合わせ11.65mへ広げています。
- PPO以外の特徴一致・行動一致・分散／共分散などの補助損失はありません。
- v0.8とv0.9は停止済みです。過去に作られた補助損失付きの別v0.9と、修正版の実験を混ぜません。
- 同梱する過去版のコード・設定が、現在の正式実行に使われたとは限りません。実行計画とハッシュを優先してください。

## 結果の解釈

17:15 JST時点ではP/S/Fの2000更新最終評価は揃っていますが、Rは学習途中で2000更新最終評価が未取得です。P/S/F/R全条件の最終比較はまだ揃っていません。途中モデルの最高値を最終モデルの結果に置き換えないでください。訓練累計と最終モデルの決定的評価は、方策・配置・集計範囲が異なるため一致しません。特にseed1のjumpは訓練累計が高くても決定的最終評価では6/36であり、この違い自体を結果として保持しています。

この公開スナップショットはv0.10の記録です。v0.5等の優位性を示す過去実験の最終データが全て含まれているわけではありません。

## コードとデータの対応

| 保存先 | 内容 |
|---|---|
| `python/tps_transfer/` | モデル、PPO、観測処理、ロールアウト |
| `scripts/train_pretraining_pilot.py` | 原子3タスクの学習 |
| `scripts/train_downstream_pilot.py` | P/S/F/Rの複合学習 |
| `scripts/evaluate_multi_pilot.py` | 決定的validation |
| `scripts/run_gaze_v10_*.{py,sh}` | 正式実験の起動・待機・移行 |
| `configs/gaze_v10_*.json` | 現行の実験条件と変更履歴 |
| `manifests/gaze_v10_layouts.json` | 原子学習・最終評価の元の配置 |
| `manifests/gaze_v10_wide24_layouts.json` | 広い複合課題の配置 |
| `unity/` | 原子と元の複合環境のUnityソース |
| `environment_sources/wide24/` | 広域版に使ったC#・設定・シーンと差分確認 |
| `results/snapshots/` | 集計JSON、更新ごとのCSV、元report・正式評価JSON、監査記録 |
| `results/validation_snapshots/` | 時刻別の条件・seed・更新数別の複合決定的validation集計JSONと全評価枠CSV |
| `media/` | 学習済みモデルの別試行動画と環境確認用動画 |

`downstream_wide24`と監視用リンク`downstream`は同じ実行を指します。両方を別ジョブとして重複集計しません。旧18m版の複合計画は未開始の履歴として保存されています。

## 実行環境・識別情報・再取得

- Unity **6000.0.84f1**、Linux/CUDAで実行。[実行環境](environment/runtime.json)と[実際の依存パッケージ](environment/runtime_packages.txt)を収録。
- Depth Anything V2 Smallのソースcommit・重みSHA-256は[depth.py](python/tps_transfer/depth.py)に固定。深度推定器は学習しません。
- 原子ビルドと広域版ビルドは別々に保持。原子の稼働環境へ広域版ソースを上書きしない構成です。
- チェックポイント、Unity実行バイナリ、Python環境、外部の学習済み深度重みはGitに含めません。結果にチェックポイント・ビルド・配置のSHA-256を記録しています。
- 実行用スクリプトの`.tools`・`.venv-gpu`と絶対パスは元の実験機に合わせたものです。リポジトリ単体に外部ランタイムや重みは同梱されません。
- ラボ側の実行時commitと未コミット変更を区別し、共有用コピーに実際のコードを収録しました。スナップショットの`live_source_sha256.json`で照合できます。
- 設定の基準は[PROTOCOL.md](PROTOCOL.md)、[実験契約](configs/gaze_v10_contract.json)、[広域版の変更表](configs/gaze_v10_wide24_change.json)と、取得済みの実行計画です。

再取得コマンドは `scripts/export_results_snapshot.py --experiment-root <実験ルート> --output <新規出力先>` です。実験ファイルは読み取りのみで、学習の起動・停止・再開・設定変更をしません。GitHubへの定期自動pushは設定していません。
