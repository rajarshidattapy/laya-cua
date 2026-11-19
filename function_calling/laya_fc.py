# Laya function-calling routing experiment (glaive-function-calling-v2). Run in the Laya-CUA Colab notebook after the install cell, on a T4.
# Task: decide tool_call yes/no and which function. Not computer use. See function_calling/RESULTS.md for the numbers.
import os, re, json, random, math, time, collections, torch, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
import laya
from laya.agent import _fix_tokenizer_config
from laya.common import build_model, collate_items
from laya.calibrate import fit_temperature_map
from laya.train import items_from_rows, calibration_records, encode_item, rlcd_loss, save_checkpoint, sigma_at, split_calibration

ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
dec = json.JSONDecoder()
def funcs_of(system):
    s = system.split("functions")[-1]; out = []; i = s.find("{")
    while 0 <= i < len(s):
        try: o, j = dec.raw_decode(s, i); out.append(o); i = s.find("{", j)
        except Exception: i = s.find("{", i + 1)
    return [f for f in out if isinstance(f, dict) and "name" in f]
def conv_rows(ex, cid):
    fs = funcs_of(ex["system"])
    if not fs or len(fs) > 8: return []
    names = [f["name"] for f in fs]
    if len(set(names)) != len(names): return []
    crit = {f["name"]: (str(f.get("description", ""))[:160] or f["name"]) for f in fs}
    screen = "Available functions: " + "; ".join(f"{n}: {d}" for n, d in crit.items())
    turns = re.split(r"\n\n\n(?=USER:|ASSISTANT:|FUNCTION RESPONSE:)", ex["chat"].strip())
    hist, rows, last_user = [], [], ""
    for t in turns:
        if t.startswith("USER:"):
            last_user = t[5:].strip()[:400]; hist.append("user: " + last_user[:160])
        elif t.startswith("ASSISTANT:"):
            body = t[10:].strip().replace("<|endoftext|>", "").strip()
            call = body.startswith("<functioncall>")
            exp = {"tool_call": bool(call)}; qs = {"tool_call": {"type": "noul", "instructions": "Should the assistant call a function now?"}}
            if call:
                m = re.search(r'"name":\s*"([^"]+)"', body)
                if not m or m.group(1) not in crit: return rows
                exp["tool_choice"] = m.group(1)
                qs["tool_choice"] = {"type": "choice", "instructions": "Which function should be called?", "criteria": crit}
            rows.append({"task_id": cid, "state": {"task": last_user, "screen": screen, "history": hist[-6:]}, "questions": qs, "expected": exp})
            hist.append("assistant: " + body[:160])
        elif t.startswith("FUNCTION RESPONSE:"):
            hist.append("function result: " + t[18:].strip()[:160])
    return rows
allrows = []
for i in range(0, 6500):
    try: allrows.append(conv_rows(ds[i], i))
    except Exception: allrows.append([])
train_rows = [r for c in allrows[:4500] for r in c][:4500]
dev_rows = [r for c in allrows[4500:5200] for r in c]
test_rows = [r for c in allrows[5200:6500] for r in c]
print("FC rows train/dev/test", len(train_rows), len(dev_rows), len(test_rows))
print("FC call rate train", round(np.mean([r["expected"]["tool_call"] for r in train_rows]), 3), "test", round(np.mean([r["expected"]["tool_call"] for r in test_rows]), 3))
MODEL_ID = "convaiinnovations/laya"
model_dir = snapshot_download(MODEL_ID); _fix_tokenizer_config(model_dir)
tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
cfg = dict(cfg, gradient_checkpointing=True, max_tokens_per_batch=4096, max_len=1024, head_max_len=256)
max_len, head_max_len = cfg["max_len"], cfg["head_max_len"]
items, skipped = items_from_rows(tok, train_rows, max_len, head_max_len)
print("FC items", len(items), "skipped", skipped)
def evaluate_fc(agent, rows):
    cc, cn, ch, chn = [], 0, 0, 0
    for r in rows:
        res = agent.predict(r["state"], r["questions"])
        a = res["answers"]
        cc.append((a["tool_call"]["noul"] >= 0.5) == r["expected"]["tool_call"])
        if "tool_choice" in r["expected"]:
            chn += 1; ch += (a["tool_choice"]["choice"] == r["expected"]["tool_choice"])
    return {"tool_call_acc": round(float(np.mean(cc)), 4), "n": len(rows), "tool_choice_acc": round(ch / max(1, chn), 4), "n_choice": chn}
base = laya.Agent(MODEL_ID, device="cuda")
nocall = round(float(np.mean([not r["expected"]["tool_call"] for r in test_rows])), 4)
rnd = round(float(np.mean([1 / len(r["questions"]["tool_choice"]["criteria"]) for r in test_rows if "tool_choice" in r["expected"]])), 4)
print("FC baselines: always-no-call tool_call", nocall, "| random tool_choice", rnd)
print("FC DEV base", evaluate_fc(base, dev_rows)); del base; torch.cuda.empty_cache()
EPOCHS = 2; device = torch.device("cuda"); OUT = "/content/laya_fc"
model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.head_checkpointing = True
model.to(device).train()
train_items, calib_items = split_calibration(items, calib_max=300, calib_frac=0.1, seed=20261119)
MB, GA, GS = 8, 8, 4
enc = [p for n, p in model.named_parameters() if "encoder." in n]; head = [p for n, p in model.named_parameters() if "encoder." not in n]
opt = torch.optim.AdamW([{"params": enc, "lr": 2.5e-5}, {"params": head, "lr": 1e-4}], weight_decay=0.01)
total = max(1, (len(train_items) // (MB * GA)) * EPOCHS)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=1e-6); scaler = torch.amp.GradScaler("cuda")
t0 = time.time()
for epoch in range(EPOCHS):
    random.Random(42 + epoch).shuffle(train_items); sigma = sigma_at(epoch, EPOCHS, 0.4, 0.1)
    opt.zero_grad(set_to_none=True); step = 0; tot = 0.0; nb = math.ceil(len(train_items) / MB)
    for b in range(0, len(train_items), MB):
        chunk = train_items[b:b + MB]
        batch = collate_items([[encode_item(tok, it, max_len, head_max_len) for it in chunk]], tok.pad_token_id)
        with torch.autocast("cuda", dtype=torch.float16):
            logits, act = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["marker_pos"].to(device), batch["marker_mask"].to(device), batch["qtype"].to(device))
        loss = rlcd_loss(logits.float(), batch["target"].to(device), batch["marker_mask"].to(device), batch["qtype"].to(device), sigma, samples=GS, w_sph=0.75, w_rps=1.0)
        win = min(GA, nb - (step // GA) * GA)
        scaler.scale(loss / win).backward(); step += 1; tot += loss.item()
        if step % GA == 0 or b + MB >= len(train_items):
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
    print(f"FC epoch {epoch+1}/{EPOCHS} avg loss {tot/max(1,step):.4f} ({time.time()-t0:.0f}s)")
model.eval(); torch.cuda.empty_cache()
fitted = fit_temperature_map(calibration_records(model, tok, calib_items, device, max_len, head_max_len, batch_size=16))
out_cfg = dict(cfg, fine_tuned=True, model_name="laya-cua-tools", temperature=fitted["temperature"]); out_cfg.pop("temperature_by_options", None)
if fitted["temperature_by_options"]: out_cfg["temperature_by_options"] = fitted["temperature_by_options"]
save_checkpoint(model, tok, out_cfg, OUT)
del model, opt; torch.cuda.empty_cache()
ft = laya.Agent(OUT, device="cuda")
print("FC DEV ft", evaluate_fc(ft, dev_rows))
print("FC TEST ft", evaluate_fc(ft, test_rows))
base = laya.Agent(MODEL_ID, device="cuda")
print("FC TEST base", evaluate_fc(base, test_rows))
print("FC DONE")
