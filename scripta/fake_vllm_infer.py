# fake_vllm_infer.py
# -*- coding: utf-8 -*-
import argparse, json, logging, os, random, sys, time, uuid
from datetime import datetime
from pathlib import Path

def now_iso(): return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def setup_logging(log_path: Path, level="INFO"):
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s","%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(fh)
    v = logging.getLogger("vllm"); v.setLevel(getattr(logging, level.upper(), logging.INFO))
    return v

def iter_prompts_from_dir(input_path: Path):
    for f in sorted([p for p in input_path.rglob("*.txt")]):
        with f.open("r", encoding="utf-8", errors="ignore") as rf:
            for line in rf:
                line=line.strip()
                if line: yield f"{f.relative_to(input_path)}", line

def iter_prompts_from_file(input_file: Path):
    with input_file.open("r", encoding="utf-8", errors="ignore") as rf:
        for i, line in enumerate(rf, 1):
            line=line.strip()
            if line: yield f"{input_file.name}#L{i}", line

def simulate_load_model(vlog, model_name: str):
    vlog.info(f"Engine init: model={model_name}, dtype=auto, tp=1, pp=1")
    vlog.info("Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit) [mock]")
    for i in range(10):
        time.sleep(0.05); vlog.info(f"Loading shards [{i+1}/10] ... {int((i+1)*10)}%")
    vlog.info("KV Cache manager: reserved 6.0 GiB, paging=auto")
    vlog.info("Scheduler: Continuous batching ON, max_num_seqs=64")
    vlog.info("OpenAI /v1/chat/completions ready (mock)")

def domain_answer(prompt: str, rng: random.Random):
    p = prompt
    if any(k in p for k in ["绝缘","漏电"]):
        return "需检查绝缘子是否破损、污闪或老化；开展红外测温与爬电距离复核，必要时更换并做耐压试验。"
    if any(k in p for k in ["山火","火源"]):
        return "结合热成像与烟雾识别进行火情监测，按风速风向评估威胁等级，清理隔离带并与林火部门联动预警。"
    if any(k in p for k in ["带电","作业"]):
        return "严格执行《电力安全工作规程》，限定作业边界，使用合格绝缘工器具并建立现场监护与停送电许可。"
    repo_t = [
        "核对一次接线与通道净空，复测导线弧垂与风偏；对可疑点进行无人机近距拍照并建档复评。",
        "对通道异物实行先期隔离与停靠监护，满足气象窗口后组织清障，记录在通道台账。"
    ]
    repo_s = [
        "对异常间隔进行红外/超声复测；校核继保动作与告警门槛；必要时降负荷并安排停电检修。",
        "对 GIS 轻渗漏开展 SF6 补气与检漏，评估年漏率与环境排放合规，制定消缺计划。"
    ]
    repo_d = [
        "核查低压侧负荷结构与接地系统；对可疑支路做分段试送与绝缘测试，定位故障点。",
        "环网柜检修流程：停电-验电-接地-遮拦与标识-两票复核，确认隔离可靠后再作业。"
    ]
    repo_sa = [
        "强化两票三制执行，设置作业围栏与监护人；恶劣天气停止露天高处作业。",
        "工器具定检定校，带电作业执行三级确认与复唱制度，违章从严问责并纳入闭环整改。"
    ]
    pools = [repo_t, repo_s, repo_d, repo_sa]
    return rng.choice(rng.choice(pools))

def tokenize_fake(text: str):
    tokens, buf = [], ""
    for ch in text:
        if ch in " ，。；、：,.?!；：!?\n":
            if buf: tokens.append(buf); buf=""
            tokens.append(ch)
        else: buf += ch
    if buf: tokens.append(buf)
    return tokens

class JsonlWriter:
    def __init__(self, path: Path):
        self.f = path.open("w", encoding="utf-8")
    def write(self, obj):
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n"); self.f.flush()
    def close(self): self.f.close()

class SummaryMeter:
    def __init__(self): self.n=0; self.lat=0.0; self.toks=0; self.base=0.0
    def add(self, lat, toks, base=None):
        self.n+=1; self.lat+=lat; self.toks+=toks
        if base is not None: self.base+=base
    def report(self):
        if self.n==0: return {}
        avg = self.lat/self.n; tps = self.toks/max(1e-6,self.lat)
        base_avg = self.base/self.n if self.base>0 else None
        gain = None if base_avg is None else 1.0 - (avg/base_avg)
        return {"requests":self.n,"avg_latency_sec":round(avg,3),
                "avg_tokens_per_sec":round(tps,2),
                "baseline_avg_latency_sec":None if base_avg is None else round(base_avg,3),
                "speedup_vs_baseline":None if gain is None else round(gain,3)}

def run_once(vlog, rid, prompt, args, rng, out_dir: Path, writer: JsonlWriter, meter: SummaryMeter):
    client_ip = f"127.0.0.1:{random.randint(40000,65000)}"
    vlog.info(f'{client_ip} - "POST /v1/chat/completions HTTP/1.1" 200 OK (rid={rid})')

    params = {"temperature": args.temperature,"top_p":args.top_p,"max_tokens":args.max_tokens,
              "seed":args.seed,"stream":args.stream,"model":args.model}
    vlog.info(f"REQUEST ACCEPTED rid={rid} len={len(prompt)} params={json.dumps(params,ensure_ascii=False)}")

    t0=time.time()
    answer = domain_answer(prompt, rng)
    toks = tokenize_fake(answer)[:args.max_tokens]

    sys.stdout.write(f"\n> Prompt[{rid}]: {prompt}\n< Completion: "); sys.stdout.flush()
    emitted=[]
    # 模拟 TTFT
    ttft = round(0.12 + random.random()*0.08, 3); time.sleep(ttft*0.2)

    for i, tk in enumerate(toks,1):
        time.sleep(max(0.01, 0.035*(1.0+args.temperature)))
        emitted.append(tk)
        if args.stream:
            sys.stdout.write(tk); sys.stdout.flush()
            if i%4==0:
                sse={"id":rid,"object":"chat.completion.chunk","choices":[{"delta":{"content":tk},"index":0,"finish_reason":None}]}
                print("data: "+json.dumps(sse,ensure_ascii=False))
        # 伪 GPU 监控 & 调度迹象
        if i%12==0:
            util = 58 + (i%20)   # 58~78
            vram = 18.0 + i*0.02
            vlog.info(f"GPU0 util={util}% vram={vram:.1f}/80.0GB | KV-cache=ok (mock)")
        if i%max(6, int(24*(1-args.top_p)))==0:
            vlog.info(f"Scheduler step rid={rid} generated={i}/{len(toks)}")
    if args.stream: print("data: [DONE]")
    else: sys.stdout.write("".join(emitted)); sys.stdout.flush()
    sys.stdout.write("\n")
    latency = time.time()-t0
    tpot = round(len(emitted)/max(1e-6,latency),2)

    # 基线推导
    baseline_latency=None
    if args.baseline:
        gain = max(0.0, min(args.speedup_vs_baseline, 0.99))
        baseline_latency = latency/(1.0-gain)

    ensure_dir(out_dir)
    (out_dir/f"{rid}.txt").write_text(
        f"prompt: {prompt}\ncompletion: {''.join(emitted)}\n"
        f"latency_sec: {round(latency,3)}\nTTFT: {ttft}\nTPOT: {tpot}\n"
        + (f"baseline_model: {args.baseline}\nbaseline_latency_sec: {round(baseline_latency,3)}\nspeedup_vs_baseline: {args.speedup_vs_baseline}\n" if baseline_latency is not None else "")
        + f"timestamp: {now_iso()}\n", encoding="utf-8"
    )

    rec={"request_id":rid,"prompt":prompt,"completion":"".join(emitted),
         "tokens_out":len(emitted),"params":params,"created_at":now_iso(),
         "latency_sec":round(latency,3),"TTFT":ttft,"TPOT":tpot,
         "baseline_model":args.baseline,
         "baseline_latency_sec":None if baseline_latency is None else round(baseline_latency,3),
         "speedup_vs_baseline":None if baseline_latency is None else args.speedup_vs_baseline}
    writer.write(rec)
    meter.add(latency,len(emitted),baseline_latency)
    vlog.info(f"METRICS rid={rid} TTFT={ttft}s, RTT={round(latency,3)}s, TPOT={tpot} tok/s, out_tokens={len(emitted)}")

def write_report(path: Path, args, summary: dict):
    lines = [ "# 模拟推理报告（含基线对比）",
        f"- 当前模型：**{args.model}**",
        f"- 基线模型：**{args.baseline}**（假设当前模型较其提速 **{int(args.speedup_vs_baseline*100)}%**）" if args.baseline else "- 基线模型：无",
        f"- 请求数量：{summary.get('requests',0)}",
        f"- 平均时延：{summary.get('avg_latency_sec')} s",
        f"- 平均吞吐：{summary.get('avg_tokens_per_sec')} tokens/s"]
    if args.baseline:
        lines.append(f"- 基线平均时延（推导）：{summary.get('baseline_avg_latency_sec')} s")
        sp = summary.get("speedup_vs_baseline")
        lines.append(f"- 推理加速：{None if sp is None else f'{round(sp*100,1)}%'}")
    lines.append("\n> 注：模拟数据，仅用于演示。")
    path.write_text("\n".join(lines), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser("Fake vLLM inference (mock, server-like logs)")
    ap.add_argument("--model", default="Mock-LLM-7B")
    ap.add_argument("--input", help="目录或文件；缺省则交互模式")
    ap.add_argument("--output_dir", default="./mock_outputs")
    ap.add_argument("--max_tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--log_level", default=os.environ.get("VLLM_LOGGING_LEVEL", "INFO"))
    ap.add_argument("--baseline", default="Qwen-VL-7B")
    ap.add_argument("--speedup_vs_baseline", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    ensure_dir(out_dir)
    vlog = setup_logging(out_dir/"run.log", args.log_level)
    rng = random.Random(args.seed)

    simulate_load_model(vlog, args.model)
    writer = JsonlWriter(out_dir/"results.jsonl")
    meter = SummaryMeter()

    try:
        if not args.input:
            vlog.info("Interactive mode (mock). Ctrl+C to exit.")
            while True:
                try:
                    prompt = input("\n请输入问题（回车生成，Ctrl+C 退出）：").strip()
                except EOFError:
                    break
                if not prompt: continue
                rid = f"chatcmpl-{uuid.uuid4().hex[:10]}"
                run_once(vlog, rid, prompt, args, rng, out_dir, writer, meter)
        else:
            src = Path(args.input)
            if not src.exists():
                logging.error(f"输入路径不存在：{src}"); sys.exit(1)
            prompts=[]
            if src.is_dir():
                for origin, line in iter_prompts_from_dir(src): prompts.append((origin,line))
            else:
                for origin, line in iter_prompts_from_file(src): prompts.append((origin,line))
            if not prompts: logging.warning("未发现任何可用的文本输入。"); sys.exit(0)
            vlog.info(f"Batch mode: {len(prompts)} requests queued.")
            for origin, prompt in prompts:
                rid = f"chatcmpl-{uuid.uuid4().hex[:10]}"
                vlog.info(f"DISPATCH rid={rid} source={origin}")
                run_once(vlog, rid, prompt, args, rng, out_dir, writer, meter)
    finally:
        writer.close()
        summary = meter.report()
        write_report(out_dir/"summary_report.md", args, summary)
        vlog.info(f"All done. Results saved to: {out_dir}")
        if summary:
            logging.info("==== Summary ====")
            logging.info(json.dumps(summary, ensure_ascii=False, indent=2))
            logging.info(f"Report: {out_dir/'summary_report.md'}")

if __name__ == "__main__":
    main()
