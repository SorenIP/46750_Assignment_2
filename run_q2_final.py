"""Final Question 2 runs, one after the other. Started by run_q2_final.bat (double-click).

  Step 1  IR against the better of "no contract" and "own individual contract" (C_ref), price per member.
          Uses the queue of run_dp_queue.py with the job option  "ir_ref": "best"   (about 1 minute).
  Step 2  Break-even points where the community contract stops paying (G = C^R - C^C = 0):
          cap scale phi in [0.8, 0.9] and discount beta in [0, 0.5], by bisection plus a small table (model C is an LP).
  Step 3  Redraw the sweep figure with the new model names.

Output: results/q2_final/  (breakeven.csv, step1_summary.txt, log.txt) and ../report/figs/q2_sweeps_v2.pdf/.png
Send the folder results/q2_final and results/base/DP_queue/queue_summary.csv back to Claude.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "q2_final"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "log.txt"


def log(msg: str) -> None:
    line = time.strftime("%H:%M:%S") + "  " + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def step1() -> None:
    log("STEP 1: IR against C_ref = min(C_R, C_I), price per member")
    jobs_path = ROOT / "queue" / "jobs.json"
    sys.path.insert(0, str(ROOT))
    import run_dp_queue as q
    jobs = q.load_jobs()                                         # creates the file if it does not exist
    name = "ind_ir_ref"
    if not any(j["name"] == name for j in jobs):
        jobs.append({"name": name, "kind": "ir", "groups": "individual", "ir_ref": "best", "limit_s": 900})
        jobs_path.write_text(json.dumps(jobs, indent=1), encoding="utf-8")
    done = ROOT / "queue" / "done" / f"{name}.ok"
    if done.exists():
        done.unlink()                                            # run it again
    (ROOT / "queue" / "done").mkdir(parents=True, exist_ok=True)
    job = next(j for j in jobs if j["name"] == name)
    info = q.run_job(job)
    done.write_text(f"{info['seconds']:.0f} s\n")
    summ = pd.read_csv(q.OUT / "queue_summary.csv")
    row = summ[summ["job"] == name]
    text = row.T.to_string()
    (OUT / "step1_summary.txt").write_text(text, encoding="utf-8")
    log("STEP 1 done:\n" + text)


def step2() -> None:
    log("STEP 2: break-even points (model C is an LP, fast)")
    import run_dp_queue as q

    def G(**kw) -> float:
        com, R, C, _ = q.prepare(kw)
        return float(R.objective - C.objective)

    def bisect(f, lo, hi, n=16):
        flo = f(lo)
        if (flo > 0) == (f(hi) > 0):
            log(f"  no sign change of G between {lo} and {hi}: break-even not inside this range")
            return float("nan")
        for _ in range(n):
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            if (fm > 0) == (flo > 0):
                lo, flo = mid, fm
            else:
                hi = mid
        return 0.5 * (lo + hi)

    rows = []
    for phi in (0.80, 0.84, 0.86, 0.88, 0.90, 0.92):
        rows.append({"parameter": "cap_scale", "value": phi, "G_DKK": G(cap_scale=phi)})
    for beta in (0.0, 0.40, 0.44, 0.48, 0.50, 0.55):
        rows.append({"parameter": "beta", "value": beta, "G_DKK": G(beta=beta)})
    phi_star = bisect(lambda x: G(cap_scale=x), 0.8, 0.9)
    beta_star = bisect(lambda x: G(beta=x), 0.0, 0.5)
    cap0 = float(q.load_community("base", day="d1").contract.capacity_limit_kW[0])
    rows.append({"parameter": "BREAK-EVEN cap_scale", "value": phi_star, "G_DKK": 0.0, "cap_kW": phi_star * cap0})
    rows.append({"parameter": "BREAK-EVEN beta", "value": beta_star, "G_DKK": 0.0})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "breakeven.csv", index=False)
    log("STEP 2 done:\n" + df.to_string(index=False))


def step3() -> None:
    log("STEP 3: sweep figure with new names")
    code = ROOT.parent / "code" / "plot_q2_sweeps_v2.py"
    src = ROOT / "results" / "base" / "DP_queue" / "queue_summary.csv"
    out = ROOT.parent / "report" / "figs" / "q2_sweeps_v2"
    r = subprocess.run([sys.executable, str(code), str(src), str(out)], capture_output=True, text=True)
    log("STEP 3 output: " + (r.stdout + r.stderr).strip()[-500:])


if __name__ == "__main__":
    for fn in (step2, step3, step1):                              # fast steps first, MILP last
        try:
            fn()
        except Exception:
            log(f"{fn.__name__} FAILED:\n" + traceback.format_exc())
    log("ALL DONE")
