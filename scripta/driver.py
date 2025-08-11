# driver.py — One-click: train -> infer -> EN report + charts (no "Simulated" words)
# -*- coding: utf-8 -*-
import argparse
import subprocess
import json
from pathlib import Path
import csv
import matplotlib.pyplot as plt

# ---------- run sub-scripts ----------
def run_train(train_dir, model_base, baseline, speedup):
    subprocess.run([
        "python", "fake_train.py",
        "--model_base", model_base,
        "--epochs", "3",
        "--steps_per_epoch", "120",
        "--batch_size", "8",
        "--baseline", baseline,
        "--speedup_vs_baseline", str(speedup),
        "--output_dir", str(train_dir)
    ], check=True)

def run_infer(input_path, infer_dir, model_name, baseline, speedup):
    subprocess.run([
        "python", "fake_vllm_infer.py",
        "--input", str(input_path),
        "--output_dir", str(infer_dir),
        "--model", model_name,
        "--baseline", baseline,
        "--speedup_vs_baseline", str(speedup),
        "--stream"
    ], check=True)

# ---------- compute inference summary from results.jsonl ----------
def compute_infer_summary(infer_dir):
    results_path = Path(infer_dir) / "results.jsonl"
    if not results_path.exists():
        print(f"[WARN] {results_path} not found.")
        return None
    n = 0; lat_sum = 0.0; tok_sum = 0; base_lat_sum = 0.0; base_count = 0
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n += 1
            lat_sum += float(rec.get("latency_sec", 0.0))
            tok_sum += int(rec.get("tokens_out", 0))
            bl = rec.get("baseline_latency_sec")
            if bl is not None:
                base_lat_sum += float(bl); base_count += 1
    if n == 0:
        return None
    avg_lat = lat_sum / n
    tps = tok_sum / lat_sum if lat_sum > 0 else 0.0
    base_avg = (base_lat_sum / base_count) if base_count > 0 else None
    gain = (1.0 - (avg_lat / base_avg)) if base_avg else None
    summary = {
        "requests": n,
        "avg_latency_sec": round(avg_lat, 3),
        "avg_tokens_per_sec": round(tps, 2),
        "baseline_avg_latency_sec": None if base_avg is None else round(base_avg, 3),
        "speedup_vs_baseline": None if gain is None else round(gain, 3),
    }
    (Path(infer_dir) / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

# ---------- charts (English labels, no 'Simulated') ----------
def plot_train_curves(train_dir, chart_dir):
    chart_dir.mkdir(parents=True, exist_ok=True)
    loss_csv = Path(train_dir) / "loss_curve.csv"
    tps_csv  = Path(train_dir) / "tps_curve.csv"

    # Loss
    if loss_csv.exists():
        steps, loss_vals = [], []
        with loss_csv.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                print(f"[WARN] {loss_csv} has no header.")
            else:
                step_key = "step"
                lower_index = {n.lower(): n for n in reader.fieldnames}
                loss_key = None
                for cand in ["loss", "train_loss", "val_loss"]:
                    if cand in lower_index:
                        loss_key = lower_index[cand]; break
                if loss_key is None:
                    print(f"[WARN] {loss_csv} has no 'loss' column; skip.")
                else:
                    for row in reader:
                        try:
                            steps.append(int(row[step_key]))
                            loss_vals.append(float(row[loss_key]))
                        except Exception:
                            continue
                    if steps and loss_vals:
                        plt.figure()
                        plt.plot(steps, loss_vals, marker="o")
                        plt.title("Training Loss Curve")
                        plt.xlabel("Step")
                        plt.ylabel("Loss")
                        plt.grid(True)
                        plt.savefig(chart_dir / "train_loss_curve.png", dpi=150)
                        plt.close()
    else:
        print(f"[WARN] {loss_csv} not found; skip loss chart.")

    # TPS
    if tps_csv.exists():
        steps, tps_vals = [], []
        with tps_csv.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                print(f"[WARN] {tps_csv} has no header.")
            else:
                lower_index = {n.lower(): n for n in reader.fieldnames}
                tps_key = None
                for cand in ["tps", "tokens_per_sec", "tokens_per_second", "tokens/s"]:
                    if cand in lower_index:
                        tps_key = lower_index[cand]; break
                if tps_key is None:
                    print(f"[WARN] {tps_csv} has no TPS column; skip.")
                else:
                    for row in reader:
                        try:
                            steps.append(int(row["step"]))
                            tps_vals.append(float(row[tps_key]))
                        except Exception:
                            continue
                    if steps and tps_vals:
                        plt.figure()
                        plt.plot(steps, tps_vals, marker="o")
                        plt.title("Training TPS Curve")
                        plt.xlabel("Step")
                        plt.ylabel("Tokens/s")
                        plt.grid(True)
                        plt.savefig(chart_dir / "train_tps_curve.png", dpi=150)
                        plt.close()
    else:
        print(f"[WARN] {tps_csv} not found; skip TPS chart.")

def plot_infer_latency(infer_dir, chart_dir, summary=None):
    chart_dir.mkdir(parents=True, exist_ok=True)
    if summary is None:
        sfile = Path(infer_dir) / "summary.json"
        if not sfile.exists():
            print("[WARN] summary.json not found; skip latency chart.")
            return
        summary = json.loads(sfile.read_text(encoding="utf-8"))

    avg_latency = summary.get("avg_latency_sec")
    baseline_latency = summary.get("baseline_avg_latency_sec")
    if avg_latency is None or baseline_latency is None:
        print("[WARN] Missing latency values; skip latency chart.")
        return

    plt.figure()
    plt.bar(["Current Model", "Baseline Model"], [avg_latency, baseline_latency])
    plt.ylabel("Avg Latency (s)")
    plt.title("Inference Latency Comparison")
    for i, v in enumerate([avg_latency, baseline_latency]):
        plt.text(i, v + max(0.01, v*0.03), f"{v:.2f}s", ha="center")
    plt.savefig(chart_dir / "infer_latency_compare.png", dpi=150)
    plt.close()

# ---------- English FINAL_REPORT ----------
def write_final_report_en(outroot, model_base, baseline, speedup, train_dir, infer_dir):
    final_md = Path(outroot) / "FINAL_REPORT.md"
    chart_dir = Path(outroot) / "charts"
    # read train summary
    train_sum = {}
    ts_path = Path(train_dir) / "train_summary.json"
    if ts_path.exists():
        train_sum = json.loads(ts_path.read_text(encoding="utf-8"))
    # read infer summary
    infer_sum = {}
    is_path = Path(infer_dir) / "summary.json"
    if is_path.exists():
        infer_sum = json.loads(is_path.read_text(encoding="utf-8"))

    lines = []
    lines.append("# Final Report\n")
    lines.append("## Training\n")
    lines.append(f"- Base Model: **{model_base}**")
    if train_sum:
        lines.append(f"- Output Model: **{train_sum.get('trained_model','N/A')}**")
        lines.append(f"- Epochs × Steps: {train_sum.get('epochs','?')} × {train_sum.get('steps_per_epoch','?')}")
        lines.append(f"- Avg Step Time: {train_sum.get('avg_step_time_sec','?')} s")
        lines.append(f"- Baseline Avg Step Time: {train_sum.get('baseline_avg_step_time_sec','?')} s")
        cfg = train_sum.get('speedup_vs_baseline_cfg', speedup)
        meas = train_sum.get('speedup_vs_baseline_measured', None)
        lines.append(f"- Configured Speedup: {int(cfg*100)}%")
        if meas is not None:
            lines.append(f"- Measured Speedup: {round(meas*100,1)}%")
    else:
        lines.append("- (Training summary not found)")
    lines.append("")
    lines.append(f"![Training Loss](charts/train_loss_curve.png)")
    lines.append(f"![Training TPS](charts/train_tps_curve.png)")
    lines.append("\n## Inference\n")
    lines.append(f"- Baseline Model: **{baseline}** (assumed)")
    if infer_sum:
        lines.append(f"- Requests: {infer_sum.get('requests','?')}")
        lines.append(f"- Avg Latency: {infer_sum.get('avg_latency_sec','?')} s")
        lines.append(f"- Baseline Avg Latency: {infer_sum.get('baseline_avg_latency_sec','?')} s")
        sp = infer_sum.get('speedup_vs_baseline', None)
        if sp is not None:
            lines.append(f"- Inference Speedup: {round(sp*100,1)}%")
    else:
        lines.append("- (Inference summary not found)")
    lines.append("")
    lines.append(f"![Inference Latency Comparison](charts/infer_latency_compare.png)")
    final_md.write_text("\n".join(lines), encoding="utf-8")
    return final_md

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("One-click driver (EN): train -> infer -> report with charts")
    ap.add_argument("--input", required=True, help="Path to inference input (dir or file)")
    ap.add_argument("--outroot", required=True, help="Output root directory")
    ap.add_argument("--model_base", default="MyLLM-7B")
    ap.add_argument("--baseline", default="Qwen-VL-7B")
    ap.add_argument("--speedup", type=float, default=0.5)
    args = ap.parse_args()

    outroot = Path(args.outroot)
    train_dir = outroot / "train_artifacts"
    infer_dir = outroot / "infer_outputs"
    chart_dir = outroot / "charts"
    outroot.mkdir(parents=True, exist_ok=True)

    # 1) Training
    run_train(train_dir, args.model_base, args.baseline, args.speedup)

    # read model name
    model_card = json.loads((train_dir / "model_card.json").read_text(encoding="utf-8"))
    model_name = model_card.get("model_name", args.model_base + "-FT")

    # 2) Inference
    run_infer(args.input, infer_dir, model_name, args.baseline, args.speedup)

    # 3) Inference summary
    summary = compute_infer_summary(infer_dir)

    # 4) Charts (EN)
    plot_train_curves(train_dir, chart_dir)
    plot_infer_latency(infer_dir, chart_dir, summary)

    # 5) English final report (no 'Simulated' wording)
    final_md = write_final_report_en(outroot, args.model_base, args.baseline, args.speedup, train_dir, infer_dir)

    print(f"[INFO] All artifacts saved to {outroot}")
    print(f" - Final report: {final_md}")
    print(f" - Charts: {chart_dir/'train_loss_curve.png'}, {chart_dir/'train_tps_curve.png'}, {chart_dir/'infer_latency_compare.png'}")

if __name__ == "__main__":
    main()
