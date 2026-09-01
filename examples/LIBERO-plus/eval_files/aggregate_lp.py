#!/usr/bin/env python
"""Aggregate the sharded LIBERO-plus results into per-dimension and overall scores.

Usage: aggregate_lp.py <LOGDIR>   (reads <LOGDIR>/*_to_*.json)

Two overall numbers are reported, because they answer different questions:

  * dimension mean — the unweighted mean of the seven perturbation-dimension
    success rates. This is what the LIBERO-plus leaderboard ranks on.
  * episode mean — successes over all episodes, so dimensions with more
    episodes count for more.
"""
import glob
import json
import os
import sys

logdir = sys.argv[1]
cats = {}
files = sorted(glob.glob(os.path.join(logdir, "*_to_*.json")))
for f in files:
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f"[warn] skip {f}: {e}")
        continue
    for cat, v in d.items():
        c = cats.setdefault(cat, {"total": 0, "success": 0})
        c["total"] += int(v.get("total_count", 0))
        c["success"] += int(v.get("success_count", 0))

order = ["Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout"]
ordered = [c for c in order if c in cats] + [c for c in sorted(cats) if c not in order]
sr = {c: 100.0 * cats[c]["success"] / cats[c]["total"] if cats[c]["total"] else 0.0
      for c in cats}
tot_t = sum(c["total"] for c in cats.values())
tot_s = sum(c["success"] for c in cats.values())
episode_mean = 100.0 * tot_s / tot_t if tot_t else 0.0
dim_mean = sum(sr[c] for c in ordered) / len(ordered) if ordered else 0.0

print(f"# shard files: {len(files)}")
print(f"{'Dimension':<14}{'SR%':>8}{'  succ/total':>14}")
for cat in ordered:
    print(f"{cat:<14}{sr[cat]:>7.1f}%{cats[cat]['success']:>8}/{cats[cat]['total']:<6}")
print(f"{'dimension mean':<14}{dim_mean:>7.1f}%   (leaderboard score)")
print(f"{'episode mean':<14}{episode_mean:>7.1f}%{tot_s:>8}/{tot_t:<6}")

out = {"dimensions": {k: {**cats[k], "sr": round(sr[k], 2)} for k in cats},
       "dimension_mean": round(dim_mean, 2),
       "episode_mean": {"success": tot_s, "total": tot_t, "sr": round(episode_mean, 2)},
       "n_shard_files": len(files)}
json.dump(out, open(os.path.join(logdir, "overall_results.json"), "w"), indent=2)
print(f"-> {os.path.join(logdir, 'overall_results.json')}")
