"""Unattended job queue for the Question 2 pricing models (DP variants).

    python run_dp_queue.py                 # run the jobs in queue/jobs.json (created on first start), then wait for more
    python run_dp_queue.py --only proof    # only jobs whose name starts with "proof"
    python run_dp_queue.py --no-wait       # stop when no job is left (default: wait for new jobs, see below)

How it behaves
  * Jobs are read from ``queue/jobs.json`` before every job, so jobs can be added (or edited) while the queue runs.
  * A job that has a file ``queue/done/<name>.ok`` is skipped: after a crash or a restart, start the queue again and it
    continues where it stopped.
  * Every job has its own time limit. Create ``queue/STOP_<name>.txt`` to stop the running stage of a job early (the best
    solution found is kept and saved). Create ``queue/STOP_QUEUE.txt`` to end the whole queue after the current stage.
  * When no job is left, the queue waits (polling every 60 s, at most ``--idle-hours``) for new jobs in jobs.json.
  * Results of job <name> go to ``results/base/DP_queue/<name>/`` (members, hourly prices, per-member responses);
    one line per job in ``results/base/DP_queue/queue_summary.csv``; progress in ``results/base/DP_queue/progress_log.txt``.

Job fields (all optional except name, kind)
  kind:        cap_only | bb | ir | two_stage
                 cap_only  - cap penalty only
                 bb        - budget balance added
                 ir        - budget balance and individual rationality (cost_i <= R_i + eps)
                 two_stage - budget balance, IR relaxed with per-member shortfalls: stage 1 minimises the total shortfall,
                             stage 2 minimises the community cost with the shortfall kept at its stage-1 value
  groups:      uniform (default) | asset | individual   (how many price vectors the manager posts)
  eps:         IR tolerance in DKK per member (kind ir)
  beta, alpha, cap_scale:  contract terms (default: the contract file); cap_scale multiplies the cap
  limit_s:     time limit of the job in seconds (two_stage: limit1_s and limit2_s for the two stages)
  ir_ref:      R (default: IR against no contract) | best (IR against the better of no contract and the member's own
               individual contract; columns "R_cost_DKK"/"benefit_vs_R_DKK" of the member files then refer to that reference)
  gap:         relative MIP gap at which a stage counts as solved (default 1e-4)
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import time
import traceback
from pathlib import Path

import pandas as pd

from src.community_model import CommunityModel
from src.data_loader import load_community
from src.dp_variants import _fmt_time, _save_variant, _solve_variant

ROOT = Path(__file__).resolve().parent
QUEUE = ROOT / "queue"
OUT = ROOT / "results" / "base" / "DP_queue"
H = 3600


def default_jobs() -> list[dict]:
    """The planned jobs, in the order they are run (see RUN_QUEUE.md for the reasoning)."""
    jobs: list[dict] = []
    # --- Thursday: can the requirements be met, and what do the fixes cost? --------------------------------------------
    for g in ("asset", "individual"):                                         # price discrimination: the extension
        jobs.append({"name": f"disc_{g}_ir", "kind": "ir", "groups": g, "limit_s": 1 * H})
    for eps in (1.0, 2.0, 4.0, 8.0):                                          # relax IR by eps DKK per member
        jobs.append({"name": f"eps{eps:g}_ir_uniform", "kind": "ir", "eps": eps, "limit_s": 2400})
    jobs.append({"name": "proof_ir_uniform", "kind": "ir", "limit_s": int(1.5 * H)})        # strict IR, tight formulation
    jobs.append({"name": "proof_two_stage_uniform", "kind": "two_stage", "limit1_s": 3 * H, "limit2_s": 1 * H})
    # --- Friday: sensitivity to the contract terms, one term at a time ----------------------------------------------------
    sweep = [("cap_scale", v) for v in (0.8, 0.9, 1.1, 1.25, 1.5)] + [("beta", v) for v in (0.0, 0.5, 1.0)]
    for param, v in sweep:
        tag = f"{param}{v:g}"
        jobs.append({"name": f"sweep_{tag}_cap", "kind": "cap_only", param: v, "limit_s": 600})
        jobs.append({"name": f"sweep_{tag}_bb", "kind": "bb", param: v, "limit_s": 900})
        jobs.append({"name": f"sweep_{tag}_ir2", "kind": "two_stage", param: v, "limit1_s": 1500, "limit2_s": 900})
    return jobs


def load_jobs() -> list[dict]:
    path = QUEUE / "jobs.json"
    if not path.exists():
        QUEUE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(default_jobs(), indent=1), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = time.strftime("%H:%M:%S") + "  " + msg
    print(line, flush=True)
    with open(OUT / "progress_log.txt", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


_cache: dict = {}


def prepare(job: dict):
    """Community with the job's contract terms, plus model R (reference) and model C (central dispatch)."""
    com = load_community("base", day="d1")
    keep = os.environ.get("DP_TEST_MEMBERS")                  # for quick tests only: "C1_PV,C4"
    if keep:
        com.members = [m for m in com.members if m.consumer_id in keep.split(",")]
    ct = com.contract
    ct = dataclasses.replace(
        ct,
        tariff_discount_ratio=float(job.get("beta", ct.tariff_discount_ratio)),
        penalty_DKK_per_kWh=float(job.get("alpha", ct.penalty_DKK_per_kWh)),
        capacity_limit_kW=ct.capacity_limit_kW * float(job.get("cap_scale", 1.0)),
    )
    com = dataclasses.replace(com, contract=ct)
    key = (ct.tariff_discount_ratio, ct.penalty_DKK_per_kWh, float(job.get("cap_scale", 1.0)), keep)
    if key not in _cache:
        _cache[key] = (CommunityModel(com, "R").solve(), CommunityModel(com, "C").solve())
    R, C = _cache[key]
    return com, R, C, key


def run_job(job: dict) -> dict:
    name, kind = job["name"], job["kind"]
    com, R, C, key = prepare(job)
    R_orig = R
    if job.get("ir_ref", "R") == "best":
        # IR against the best alternative of a member: no contract (R) or its own individual contract (I)
        if ("I",) + key not in _cache:
            _cache[("I",) + key] = CommunityModel(com, "I").solve()
        I = _cache[("I",) + key]
        R = copy.copy(R)
        R.member_cost = R_orig.member_cost.copy()
        R.member_cost["cost_DKK"] = pd.concat([R_orig.member_cost["cost_DKK"], I.member_cost["cost_DKK"]], axis=1).min(axis=1)
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    groups = job.get("groups", "uniform")
    gap = float(job.get("gap", 1e-4))
    stop_file = QUEUE / f"STOP_{name}.txt"
    log_path = OUT / "progress_log.txt"
    if stop_file.exists():
        stop_file.unlink()                                       # stale stop file from an earlier run
    start = C.grid["pi"].to_numpy()
    common = dict(speedups=True, lower_bound=C.objective, log_path=log_path, stop_file=stop_file,
                  mip_gap_target=gap, price_groups=groups, start_prices=start)
    t0 = time.time()
    if kind in ("cap_only", "bb", "ir"):
        flags = {"cap_only": (False, False, False), "bb": (True, False, False), "ir": (True, True, False)}[kind]
        res = _solve_variant(com, R, name, *flags, label=name, ir_eps=float(job.get("eps", 0.0)),
                             time_limit=job.get("limit_s"), best_path=out / "best_so_far_prices.csv", **common)
        stages = {name: res}
    elif kind == "two_stage":
        r1 = _solve_variant(com, R, f"{name}_stage1", True, False, True, label=f"{name} stage 1",
                            shortfall_mode="shortfall_only", time_limit=job.get("limit1_s"),
                            best_path=out / "best_so_far_stage1.csv", **common)
        stages = {f"{name}_stage1": r1}
        if "metrics" in r1:
            cap_value = r1["metrics"]["total_IR_shortfall_DKK"] * 1.001 + 1e-4
            common2 = dict(common, start_prices=r1["hourly"]["posted_price"].to_numpy())
            log(f"{name}: stage 1 shortfall {r1['metrics']['total_IR_shortfall_DKK']:.4f} DKK; stage 2 starts")
            r2 = _solve_variant(com, R, f"{name}_stage2", True, False, True, label=f"{name} stage 2",
                                shortfall_mode="cost_given_shortfall", shortfall_cap=cap_value,
                                time_limit=job.get("limit2_s"), best_path=out / "best_so_far_stage2.csv", **common2)
            stages[f"{name}_stage2"] = r2
    else:
        raise ValueError(f"unknown kind {kind!r}")

    rows = []
    for stage_name, res in stages.items():
        row = _save_variant(out, stage_name, res)
        row = {"job": name, "stage": stage_name, "kind": kind, "groups": groups, "beta": com.contract.tariff_discount_ratio,
               "alpha": com.contract.penalty_DKK_per_kWh, "cap_scale": job.get("cap_scale", 1.0), "eps": job.get("eps", 0.0), "ir_ref": job.get("ir_ref", "R"),
               "C_cost_DKK": C.objective, "R_cost_DKK": R_orig.objective, **{k: v for k, v in row.items() if k != "variant"}}
        if "members" in res:
            m = res["members"]
            bad = m[(m["benefit_vs_R_DKK"] < -1e-6)]
            row["blocking_members"] = " ".join(f"{i}:{-b:.2f}" for i, b in bad["benefit_vs_R_DKK"].items())
        rows.append(row)
        log(f"{stage_name}: {res['status']} | " + (f"cost {res['metrics']['community_cost_DKK']:.3f} (C {C.objective:.3f}) "
            f"shortfall {res['metrics']['total_IR_shortfall_DKK']:.3f} gap {res['metrics']['mip_gap'] * 100:.2f}%"
            if "metrics" in res else "no result") + f" | {_fmt_time(res['runtime_s'])}")
    summary = OUT / "queue_summary.csv"
    new = pd.DataFrame(rows)
    if summary.exists():
        old = pd.read_csv(summary)
        new = pd.concat([old[~old["stage"].isin(new["stage"])], new], ignore_index=True)
    new.to_csv(summary, index=False)
    return {"seconds": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="run only jobs whose name starts with this text")
    ap.add_argument("--no-wait", action="store_true", help="stop when no job is left instead of waiting for new ones")
    ap.add_argument("--idle-hours", type=float, default=24.0, help="longest wait for new jobs (default 24 h)")
    args = ap.parse_args()
    (QUEUE / "done").mkdir(parents=True, exist_ok=True)
    stale = QUEUE / "STOP_QUEUE.txt"
    if stale.exists():                                       # left over from the previous session
        stale.unlink()
    log("QUEUE START")
    idle_since = None
    while True:
        if (QUEUE / "STOP_QUEUE.txt").exists():
            log("QUEUE STOP requested")
            break
        todo = [j for j in load_jobs()
                if not (QUEUE / "done" / f"{j['name']}.ok").exists() and (args.only is None or j["name"].startswith(args.only))]
        if not todo:
            if args.no_wait:
                break
            idle_since = idle_since or time.time()
            if time.time() - idle_since > args.idle_hours * H:
                break
            time.sleep(60)
            continue
        idle_since = None
        job = todo[0]
        log(f"JOB START {job['name']} {json.dumps({k: v for k, v in job.items() if k != 'name'})}")
        try:
            info = run_job(job)
            (QUEUE / "done" / f"{job['name']}.ok").write_text(f"{info['seconds']:.0f} s\n")
            log(f"JOB END {job['name']} ({_fmt_time(info['seconds'])})")
        except Exception:                                    # one failing job must not stop the queue
            log(f"JOB FAILED {job['name']}:\n" + traceback.format_exc())
            (QUEUE / "done" / f"{job['name']}.ok").write_text("failed\n")
    log("QUEUE END")


if __name__ == "__main__":
    main()
