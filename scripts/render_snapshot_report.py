"""Render the human-readable current report from an exported JSON snapshot."""
import argparse
import json
from pathlib import Path


def rate(value):
    return "未算出" if value is None else f"{value * 100:.1f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = args.snapshot.resolve()
    s = json.loads((folder / "summary.json").read_text())
    rel = folder.relative_to(root).as_posix()
    lines = ["# v0.10 実験結果", "", f"**取得時刻: {s['captured_end_jst']}**（JST）", "",
             "原子look/approach/jumpは全3seedの2000更新と最終validationを完了。広域版の複合P/S/Fは学習中です。",
             "", "## 1. 実験条件", "",
             "- 原子: 単一配色（目標・台＝橙、自機＝シアン）、タスクごとにactor/critic各1組、幅256の共有部。",
             "- 共有部: 空間ViTと時間Transformer。4フレームのRGB＋推定深度と、間の3行動を使用。",
             "- 操作: 移動2・視点2・ジャンプ1の5連続入力。全原子課題でジャンプ入力可能。",
             "- PPO以外の補助損失、外見ランダム化、追加センサーはなし。固定の既存深度推定を使用。",
             "- look: 目標を中央7度以内に0.5秒保持。approach: 目標接触。jump: 床から跳躍し、高さ1mの台上面へ着地。",
             "- 原子: 各seed2000更新、各更新3タスク×256＝768判断、1,536,000判断/seed。100更新ごと保存。",
             "- 複合: 静的障害物3個を回避し、台へジャンプして着地し、台上の宝に接触。",
             "- 複合の広域化: 床18m→24m四方、プレイヤーを宝と反対方向へ4m後退、開始水平距離10.05–10.85m。場外判定8.65m→11.65m。",
             "- 台・宝・障害物の位置と寸法、物理・報酬・成功条件・観測・PPO・seed・学習量は維持。原子は元の18m版。",
             "- 複合: P/S/F×3seed、各2000更新×256＝512,000判断。200更新ごとの10モデルを各32エピソードで評価、合計90評価。",
             "- P: 学習済み共有部を固定、S: ランダム共有部を学習、F: 学習済み共有部を解凍。新規ヘッドの初期値と環境・行動seedを同じseed内で対応。",
             "- 原子のヘッドは転移せず、共有部だけを対応seedの最終モデルから移す。",
             "- 学習は最大3並列、複合評価は別の1枠。開始順P0,S0,F0,P1,S1,F1,P2,S2,F2。",
             "- 正式評価: 環境seed9000、tanh平均行動。原子は各タスク36エピソード。",
             "", "## 2. 原子の最終結果", "",
             "全seedが2000/2000更新、各1,536,000判断、最新保存はcheckpoint_2000.pt。訓練累計と最終モデルの決定的評価を別列に示します。",
             "", "| seed | タスク | 訓練成功 | 失敗 | 中断 | 終了合計 | 訓練成功率 | 最終決定的評価 | 評価成功率 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for seed, v in s["primitive"].items():
        for task, count in v["totals"].items():
            e = v["validation"][task]
            lines.append(f"| {seed} | {task} | {count['successes']:,} | {count['failures']:,} | {count['interruptions']:,} | {count['ended']:,} | {rate(count['success_rate'])} | {e['successes']}/{e['episodes']} | {rate(e['success_rate'])} |")
    lines += ["", "seed1のjumpは訓練累計81.9%に対して最終決定的評価6/36です。方策（確率的／平均行動）、配置（訓練／評価）、集計範囲（全学習中／最終モデル）が異なります。原因をこの集計だけから断定しません。",
              "", "## 3. 複合タスクの途中経過", "",
              "環境識別子は **gaze-jump-v0.10-wide24-start-back4**。実出力downstream_wide24と監視用リンクdownstreamは同じ実行であり、重複集計しません。旧18m版の複合学習は未開始のまま保存されています。",
              "", "| 条件・seed | 更新/2000 | 判断数 | 状態 | 最新保存 | 訓練成功/終了合計 | 成功率 |",
              "|---|---:|---:|---|---|---:|---:|"]
    for key, v in s["downstream"]["jobs"].items():
        count = v["totals"]
        status = "学習中" if v["update"] and v["update"] < 2000 else "学習完了" if v["update"] == 2000 else "待機"
        lines.append(f"| {key} | {v['update']}/2000 | {v['decisions']:,} | {status} | {v['latest_checkpoint'] or 'なし'} | {count['successes']}/{count['ended']} | {rate(count['success_rate'])} |")
    lines += ["", f"**正式な複合validation: {s['downstream']['completed_validation_files']}/90評価完了。** この時点では200更新の保存モデルにまだ到達しておらず、決定的評価結果はありません。",
              "", "P/S/Fの初期の訓練成功率は探索中の少数エピソードです。更新数の異なる途中値や原子の成績だけで、転移の優位性を結論づけません。P/Fには原子事前学習の経験・計算量が加わるため、複合の判断数だけを総コストとしません。",
              "", "## 4. 集計と評価の定義", "",
              "- 訓練成功率: 全unique updateのsuccesses合計 ÷（successes＋failures＋interruptions合計）。",
              "- 未完了エピソード、破棄された余剰遷移は分母に含めない。0/0は率未算出。",
              "- 原子はreport.json、複合の途中値はtrain.logの完結した1行JSONから集計。update番号で重複排除し、欠番がないことを確認。",
              "- 原子CSVのagent_decisions_trainedは3タスクを合わせたseed全体の累計。同じupdateの3タスク行に同じ値を記載しているので足さない。各タスクはupdate×256判断。",
              "- ファイルを順番に読み取ったスナップショットであり、進行中ジョブを全て同時刻に凍結した値ではない。開始・終了時刻と元ファイルの時刻を記録。",
              "- smoke、スクリプト操作、動画用別試行は正式訓練・validationへ混ぜない。",
              "", "## 5. 元データと再現情報", "",
              f"- [summary.json]({rel}/summary.json): スナップショットの全集計と進捗。",
              f"- [training_summary.csv]({rel}/training_summary.csv): seed・タスク・条件別の累計。",
              f"- [primitive_updates.csv]({rel}/primitive_updates.csv): 原子2000更新×3seed×3タスク、18,000行。",
              f"- [downstream_updates.csv]({rel}/downstream_updates.csv): 進行済み複合更新の結果とPPO指標。",
              f"- [raw/]({rel}/raw): 原子の元report・正式評価、複合のunique update JSONL、実行計画。",
              f"- [audit/]({rel}/audit): 広域化の差分検証・短時間動作確認・公開前の設定識別情報。",
              f"- [artifact_provenance.json]({rel}/artifact_provenance.json): 元ファイル名・SHA-256・mtime。",
              f"- [live_source_sha256.json]({rel}/live_source_sha256.json): 実行側ソースのファイル別SHA-256。",
              "", "### 原子最終チェックポイントの識別", "",
              "| seed | checkpoint_2000.pt SHA-256 |", "|---|---|"]
    for seed, v in s["primitive"].items():
        lines.append(f"| {seed} | `{v['final_checkpoint_sha256']}` |")
    lines += ["", "チェックポイント本体、Unity実行バイナリ、外部深度重みはこのGitに含めず、識別情報と結果を保存しています。",
              "", f"実行側commitは `{s['live_code_commit']}`、取得時の未コミット変更あり: `{s['live_code_dirty']}`。その変更を共有用コピーへ取り込んでいます。共有用コミットの作成によって稼働中の実験を変更していません。",
              "", "## 6. 観察動画", "",
              "- [原子2000更新の成功・失敗](media/primitive_update2000_success_failure.mp4): 学習済み方策の新しい確率的試行。lookは成功seed0/失敗seed1、approachは両方seed0、jumpは成功seed0/失敗seed1。左成功、右失敗。各側は別試行。",
              "- [広域複合のスクリプト操作プレビュー](media/composite_wide24_scripted_preview.mp4): 環境動作確認。学習結果ではない。",
              "", "動画は成功例と失敗例を選んだものなので、その割合から成功率を求めません。"]
    (root / "EXPERIMENT_REPORT.md").write_text("\n".join(lines) + "\n")
    (root / "results/LATEST.json").write_text(json.dumps({"snapshot": rel, "captured_jst": s["captured_end_jst"],
        "summary": f"{rel}/summary.json", "report": "EXPERIMENT_REPORT.md", "live_training_continues": True}, indent=2) + "\n")
    print(f"Rendered EXPERIMENT_REPORT.md from {rel}")


if __name__ == "__main__":
    main()
