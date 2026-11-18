"""Convert OpenCUA AgentNet trajectories (MIT) into Laya typed-decision rows.

Each trajectory step becomes one row:
  state = {task, screen (text description of the current screen), history (previous action texts)}
  expected = {action_type: label from the step's pyautogui code, task_done: true on a successful terminate}
The step's own `thought` and `action` text are NOT in the state (they describe the answer).
`needs_human` has no label in AgentNet, so it is left unlabelled and skipped in training.
Screenshots are not used: Laya reads text only.
"""
import json, re, random, itertools

ACTION_MAP = {
    "click": "click", "doubleClick": "click", "tripleClick": "click", "rightClick": "click", "middleClick": "click",
    "moveTo": "move", "write": "type", "typewrite": "type", "press": "key", "hotkey": "key", "keyDown": "key", "keyUp": "key",
    "scroll": "scroll", "hscroll": "scroll", "dragTo": "drag", "drag": "drag",
    "terminate": "done", "wait": "wait", "sleep": "wait",
}

def parse_code(code):
    """Return (action_type, success_flag_or_None), or (None, None) if unknown."""
    code = (code or "").strip()
    m = re.match(r"(?:pyautogui|computer)\.(\w+)", code)
    if not m:
        return None, None
    name = m.group(1)
    kind = ACTION_MAP.get(name)
    if kind is None:
        return None, None
    ok = None
    if name == "terminate":
        ok = "success" in code
    return kind, ok

def trajectory_rows(task, history_len=4, max_obs_chars=1800, require_correct=True):
    traj = task.get("traj") or []
    instr = task.get("instruction") or task.get("natural_language_task") or ""
    rows, hist = [], []
    for step in traj:
        v = step.get("value") or {}
        kind, ok = parse_code(v.get("code"))
        text = (v.get("action") or "").strip()
        if kind is None or not v.get("observation"):
            if text: hist.append(text[:160])
            continue
        if not (require_correct and v.get("last_step_correct") is False):
            rows.append({
                "task_id": task.get("task_id"),
                "state": {"task": instr, "screen": v["observation"][:max_obs_chars], "history": hist[-history_len:]},
                "expected": {"action_type": kind, "task_done": bool(ok)},
            })
        if text: hist.append(text[:160])
    return rows

def stream_agentnet(n_tasks=600, file="agentnet_ubuntu_5k.jsonl", repo="xlangai/AgentNet", seed=0, skip=0):
    """Stream the first lines of an AgentNet jsonl over HTTP and stop after n_tasks (no 200 GB download)."""
    import requests
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{file}"
    out, kept = [], 0
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        for i, line in enumerate(r.iter_lines()):
            if i < skip or not line: continue
            try: task = json.loads(line)
            except Exception: continue
            rows = trajectory_rows(task)
            if rows: out.extend(rows); kept += 1
            if kept >= n_tasks: break
    return out

def split_by_task(rows, eval_frac=0.1, seed=7):
    """Hold out whole tasks, so no task leaks across train and eval."""
    ids = sorted({r["task_id"] for r in rows}); random.Random(seed).shuffle(ids)
    ev = set(ids[:max(1, int(len(ids) * eval_frac))])
    return [r for r in rows if r["task_id"] not in ev], [r for r in rows if r["task_id"] in ev]
