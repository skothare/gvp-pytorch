"""CPU-only exploratory paired-seed analysis of existing, audited CSV results.

Never changes training artifacts. Exact two-sided sign-flip tests of paired
mean differences; Holm correction across all 24 comparisons/metrics below.
These test training-seed variability on a fixed test set, not new-data error.
"""
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import permutation_test

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "logs/completed_lba_audit_20260925/paired_statistics"
SOURCES = {
    "audited": ROOT / "logs/completed_lba_audit_20260925/verified_results.csv",
    "historical_ep5": ROOT / "logs/v9_decoder_production_20260918T161209Z/results.csv",
}


def main():
    rows = list(csv.DictReader(SOURCES["audited"].open()))
    for row in csv.DictReader(SOURCES["historical_ep5"].open()):
        rows.append(dict(row, group="historical_ep5"))
    pairs = [
        ("controlled", "ep5", "controlled", "baseline"),
        ("controlled", "ep20", "controlled", "baseline"),
        ("controlled", "ep5", "controlled", "ep20"),
        ("parent", "ep20", "parent", "baseline"),
        ("parent", "ep20", "controls", "raw_qr"),
        ("parent", "ep20", "controls", "random_ep"),
        ("parent", "ep20", "controls", "shuffled_ep"),
        ("historical_ep5", "v9_real", "historical_ep5", "baseline"),
    ]
    result = []
    for ga, a, gb, b in pairs:
        left = {int(r["seed"]): r for r in rows if r["group"] == ga and r["arm"] == a}
        right = {int(r["seed"]): r for r in rows if r["group"] == gb and r["arm"] == b}
        assert left and left.keys() == right.keys()
        seeds = sorted(left)
        signs = np.array(list(itertools.product((-1, 1), repeat=len(seeds))))
        for metric in ("rmse", "pearson_r", "spearman_r"):
            d = np.array([float(left[s][metric])-float(right[s][metric]) for s in seeds])
            observed = float(d.mean())
            null = signs @ d / len(d)
            p = float(np.mean(np.abs(null) >= abs(observed)-1e-14))
            # Independent library cross-check: samples swaps, not pairings.
            check = permutation_test((d,), np.mean, permutation_type="samples",
                                     n_resamples=np.inf, alternative="two-sided")
            assert abs(p-float(check.pvalue)) < 1e-12
            result.append(dict(group_a=ga, arm_a=a, group_b=gb, arm_b=b,
                               metric=metric, n=len(d), seeds=";".join(map(str,seeds)),
                               mean_a_minus_b=observed, sd_difference_ddof0=float(d.std()),
                               a_better_count=int(np.sum(d < 0 if metric == "rmse" else d > 0)),
                               p_two_sided_exact=p))
    order = sorted(range(len(result)), key=lambda i: result[i]["p_two_sided_exact"])
    previous = 0.0
    for rank, i in enumerate(order):
        previous = max(previous, min(1., (len(result)-rank)*result[i]["p_two_sided_exact"]))
        result[i]["p_holm_all_24"] = previous
    OUT.mkdir(parents=True, exist_ok=False)
    with (OUT / "paired_tests.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    metadata = dict(exploratory=True, null="Paired differences symmetric about zero / labels exchangeable within seed under null",
                    scope="Training-seed uncertainty conditional on fixed ATOM3D test set; no equivalence or population generalization test",
                    method="Exact two-sided sign-flip enumeration of paired mean; verified using scipy permutation_test(samples)",
                    correction="Holm across all 24 tests (8 contrasts x RMSE/Pearson/Spearman)",
                    sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES.values()},
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / "method.json").write_text(json.dumps(metadata, indent=2)+"\n")
    for r in result:
        print(f"{r['group_a']} {r['arm_a']}-{r['arm_b']} {r['metric']}: n={r['n']} diff={r['mean_a_minus_b']:+.6f} p={r['p_two_sided_exact']:.6f} Holm={r['p_holm_all_24']:.6f}")


if __name__ == "__main__":
    main()
