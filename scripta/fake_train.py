# fake_train.py
# -*- coding: utf-8 -*-
import argparse, json, logging, os, random, time, uuid, csv
from datetime import datetime
from pathlib import Path

def now_iso(): return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def setup_logging(log_path: Path, level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s","%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(fh)

class JsonlWriter:
    def __init__(self, path: Path):
        self.f = path.open("w", encoding="utf-8")
    def write(self, obj):
        self.f.write(json.dumps(obj, ensure_ascii=False)+"\n"); self.f.flush()
    def close(self): self.f.close()

def main():
    ap = argparse.ArgumentParser("Fake training (mock) with baseline compare & artifacts")
    ap.add_argument("--model_base", default="MyLLM-7B")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps_per_epoch", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="./artifacts")
    ap.add_argument("--baseline", default="Qwen-VL-7B")
    ap.add_argument("--speedup_vs_baseline", type=float, default=0.5)
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    ensure_dir(out_dir)
    setup_logging(out_dir / "run.log", args.log_level)

    # 保存一次运行配置
    (out_dir/"run_config.yaml").write_text(
        "\n".join([
            f"model_base: {args.model_base}",
            f"epochs: {args.epochs}",
            f"steps_per_epoch: {args.steps_per_epoch}",
            f"batch_size: {args.batch_size}",
            f"lr: {args.lr}",
            f"baseline: {args.baseline}",
            f"speedup_vs_baseline: {args.speedup_vs_baseline}",
            f"seed: {args.seed}",
        ]), encoding="utf-8"
    )

    rng = random.Random(args.seed)
    exp_id = f"exp-{uuid.uuid4().hex[:8]}"
    logging.info(f"Experiment ID: {exp_id}")
    logging.info("Env: CUDA=12.1 | Driver=550.54 | cuDNN=9.1 | NCCL=2.20 (mock)")
    logging.info("Hardware: 4x A100-SXM4-80GB (mock), total_vram=320GB, reserve_kvcache=48GB")
    logging.info("AMP: bf16 | GradAccum: 4 | GradCheckpoint: ON | ZeroStage: 2 (mock)")

    jsonl = JsonlWriter(out_dir/"train_log.jsonl")
    loss_csv = (out_dir/"loss_curve.csv").open("w", newline="", encoding="utf-8")
    tps_csv  = (out_dir/"tps_curve.csv").open("w", newline="", encoding="utf-8")
    lcw, tcw = csv.writer(loss_csv), csv.writer(tps_csv)
    lcw.writerow(["step","loss"]); tcw.writerow(["step","tokens_per_sec"])

    stamp = datetime.now().strftime("%Y%m%d")
    model_name = f"{args.model_base}-FT-{stamp}"
    (out_dir / "model_card.json").write_text(json.dumps({
        "model_name": model_name,
        "base_model": args.model_base,
        "trained_at": now_iso(),
        "task": "Instruction Tuning (mock)",
        "data": "Power domain QA (mock)",
        "param_count": 7_532_451_000,  # ★ 新增：总参数量（整数）
        "param_count_billion": 7.5  # ★ 新增：以 B 为单位
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    init_loss = rng.uniform(2.5, 3.5)
    best_val, total_time, total_tokens, baseline_total_time = None, 0.0, 0, 0.0
    warn_used = False

    for ep in range(1, args.epochs+1):
        loss_trend = init_loss * (0.85 ** (ep-1))
        for step in range(1, args.steps_per_epoch+1):
            base_step_time = rng.uniform(0.045, 0.055)
            speedup = max(0.0, min(args.speedup_vs_baseline, 0.99))
            new_step_time = base_step_time * (1.0 - speedup)
            time.sleep(new_step_time * 0.20)  # 缩短真实等待，仅营造节奏
            tokens = args.batch_size * rng.randint(512, 768)

            total_time += new_step_time; total_tokens += tokens; baseline_total_time += base_step_time
            this_loss = max(0.05, loss_trend + rng.uniform(-0.05, 0.03) * (1.0/ep))
            tps = tokens / max(1e-6, new_step_time)
            gnorm = rng.uniform(0.6, 1.4)
            lcw.writerow([(ep-1)*args.steps_per_epoch+step, f"{this_loss:.4f}"])
            tcw.writerow([(ep-1)*args.steps_per_epoch+step, f"{tps:.1f}"])

            if (step % max(10, args.steps_per_epoch//10)) == 0:
                eta = (args.steps_per_epoch - step) * new_step_time
                logging.info(f"[Train] ep={ep}/{args.epochs} step={step}/{args.steps_per_epoch} "
                             f"loss={this_loss:.3f} lr={args.lr:.2e} grad_norm={gnorm:.2f} "
                             f"step_time={new_step_time:.3f}s tps={tps:.1f} ETA={eta:.1f}s")
                jsonl.write({"time": now_iso(),"epoch":ep,"step":step,"loss":round(this_loss,4),
                             "tokens_per_sec": round(tps,1),"step_time_sec": round(new_step_time,3)})

                # 注入一次轻微 WARNING
                if not warn_used and rng.random()<0.15:
                    warn_used=True
                    logging.warning("Throughput jitter detected, lowering page_size to stabilize KV (mock)")
                    logging.info("Recovered by adjusting page_size and prefill batching (mock)")

        # 验证 + ckpt
        val_loss = max(0.04, loss_trend*0.9 + rng.uniform(-0.02, 0.02))
        em = min(1.0, 0.30 + 0.10*ep + rng.uniform(-0.02, 0.02))
        f1 = min(1.0, 0.40 + 0.12*ep + rng.uniform(-0.02, 0.02))
        best_val = val_loss if best_val is None else min(best_val, val_loss)
        logging.info(f"[VAL] epoch={ep} val_loss={val_loss:.3f} EM={em:.3f} F1={f1:.3f}")

        ckpt = out_dir/f"ckpt_epoch_{ep}.pt"
        logging.info(f"[CKPT] saving checkpoint to {ckpt} (mock)")
        time.sleep(0.05); logging.info("[CKPT] saved.")

    jsonl.close(); loss_csv.close(); tps_csv.close()

    avg_step = total_time/(args.epochs*args.steps_per_epoch)
    base_avg = baseline_total_time/(args.epochs*args.steps_per_epoch)
    real_gain = 1.0 - (avg_step/base_avg) if base_avg>0 else None

    summary = {
        "trained_model": model_name,
        "base_model": args.model_base,
        "epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch,
        "batch_size": args.batch_size,
        "avg_step_time_sec": round(avg_step,4),
        "baseline_avg_step_time_sec": round(base_avg,4),
        "speedup_vs_baseline_cfg": args.speedup_vs_baseline,
        "speedup_vs_baseline_measured": None if real_gain is None else round(real_gain,3),
        "total_train_time_sec": round(total_time,2)
    }
    (out_dir/"train_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    # 计算测算提升的展示文案，避免嵌套 f-string
    measured = summary.get("speedup_vs_baseline_measured")
    measured_text = "None" if measured is None else f"{round(measured * 100, 1)}%"

    (out_dir / "train_summary.md").write_text(
        "\n".join([
            "# 训练总结（模拟）",
            f"- 基座模型：**{args.model_base}**",
            f"- 产出模型：**{model_name}**",
            f"- 训练轮次：{args.epochs} × {args.steps_per_epoch} steps",
            f"- 平均 step 时延：{summary['avg_step_time_sec']} s",
            f"- 基线平均 step 时延（推导）：{summary['baseline_avg_step_time_sec']} s",
            f"- 速度提升（配置）：{int(args.speedup_vs_baseline * 100)}%",
            f"- 速度提升（测算）：{measured_text}",  # ← 用上面已经算好的字符串
            f"- 总训练时间：{summary['total_train_time_sec']} s",
            "\n> 注：本报告为模拟。"
        ]),
        encoding="utf-8"
    )

    logging.info("Training finished (mock). Artifacts saved to: %s", out_dir)

if __name__ == "__main__":
    main()
