"""ATOM3D ligand fingerprints and matched EP-20 ridge probes.

Reuse ProteinShake's SDF fingerprint helper and the existing ATOM3D chemistry
validator. Keep the current EP-20 source files and Slurm chain unchanged.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import lba_ep20 as ep

ESTATS = ep.WORKSPACE / "Estats"
sys.path.insert(0, str(ESTATS))
sys.path.insert(0, str(ep.EVAL))
MODES = ("morgan", "maccs", "maccs_morgan")
WIDTHS = dict(morgan=1024, maccs=167, maccs_morgan=1191)
SCHEMA = "atom3d-lba-ligands-v1"


def source_hashes():
    paths = [Path(__file__), ESTATS/"proteinshake_eval/pdbbind.py",
             ESTATS/"proteinshake_eval/hashing.py", ep.EVAL/"v9_lba_probes.py",
             ep.EVAL/"metrics.py", ep.GVP/"lba_ep20.py"]
    return {ep.relative(p): ep.sha(p) for p in paths}


def ligand_input_hash(elem):
    """Bind ordered ligand atoms/bonds/coordinates, source SMILES and target."""
    atoms = elem["atoms_ligand"].reset_index(drop=True)
    bonds = elem["bonds"].reset_index(drop=True)
    payload = dict(id=str(elem["id"]), smiles=str(elem["smiles"]),
                   atoms=atoms.to_json(orient="split", double_precision=15),
                   bonds=bonds.to_json(orient="split", double_precision=15),
                   target=float(elem["scores"]["neglog_aff"]))
    import hashlib
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def verified_fingerprints(elem, ligand_root=None):
    from rdkit import rdBase
    from v9_lba_probes import ligand_features
    source_path = None
    if ligand_root is not None:
        from v9_lba import safe_id
        pid = safe_id(elem["id"])
        candidates = [Path(ligand_root)/pid/(pid+"_ligand.sdf"),
                      Path(ligand_root)/(pid+"_ligand.sdf")]
        found = [p for p in candidates if p.is_file()]
        if len(found) > 1:
            raise ValueError(pid + ": ambiguous bound ligand SDF sources")
        if found:
            source_path = found[0]
            before = ep.sha(source_path)
    # Existing validator checks heavy-atom chemistry, and bound coordinates when
    # SDF is used. It never repairs invalid SMILES or guesses bonds from xyz.
    with rdBase.BlockLogs():
        legacy, smiles = ligand_features(elem, ligand_root if source_path is not None else None)
    origin = dict(kind="LMDB SMILES verified against ligand atom/bond graph")
    if source_path is not None:
        from proteinshake_eval.pdbbind import _ligand_attributes
        path = source_path
        # Direct reuse of the peer's generator; no ProteinShake dataset rebuild.
        attributes = _ligand_attributes(path)
        morgan = np.asarray(attributes["fp_morgan_r2"], dtype=np.uint8)
        maccs = np.asarray(attributes["fp_maccs"], dtype=np.uint8)
        if before != ep.sha(path):
            raise ValueError("Ligand source changed during extraction")
        if not np.array_equal(np.concatenate([maccs, morgan]), legacy):
            raise ValueError("ProteinShake/ATOM3D fingerprint parity mismatch")
        smiles = attributes["ligand_smiles"]
        origin = dict(kind="verified bound SDF; ProteinShake _ligand_attributes",
                      path=ep.relative(path), sha256=before)
    else:
        # Legacy ATOM3D generator has the same radius/bit/chirality settings.
        maccs, morgan = legacy[:167], legacy[167:]
    for name, vector in [("maccs", maccs), ("morgan", morgan)]:
        if vector.shape != (WIDTHS[name],) or not np.isin(vector, [0, 1]).all():
            raise ValueError("Invalid " + name + " fingerprint")
    return dict(morgan=morgan, maccs=maccs), dict(
        canonical_smiles=smiles, source=origin, ligand_input_sha256=ligand_input_hash(elem))


def prepare(run_root, out, ligand_root=None, limit=None):
    import rdkit
    from atom3d.datasets import LMDBDataset
    from v9_lba import dataset_ids
    if limit is not None and limit <= 0:
        raise ValueError("Limit must be positive")
    run_root, out = Path(run_root).resolve(), Path(out).resolve()
    if ligand_root is not None and not Path(ligand_root).is_dir():
        raise ValueError("Ligand source directory does not exist")
    info = ep.config(run_root)  # Does not wait for the EP cache.
    for name, digest in info["dataset_sha256"].items():
        if ep.sha(ep.resolve(name)) != digest:
            raise ValueError("LMDB changed since parent experiment initialization")
    out.mkdir(parents=True, exist_ok=False)
    initial_sources = source_hashes()
    arrays, records, errors, expected = {}, {}, [], {}
    for split, count in info["counts"].items():
        ds = LMDBDataset(str(ep.resolve(info["lmdb_root"])/split))
        ids = dataset_ids(ds)
        if ids != info["split_ids"][split] or len(ids) != count:
            raise ValueError("Parent/ligand cohort mismatch")
        indices = range(min(limit, len(ds)) if limit else len(ds))
        expected[split] = [ids[i] for i in indices]
        accepted, targets, morgans, maccs = [], [], [], []
        records[split] = []
        for i in indices:
            elem = ds[i]
            pid = str(elem["id"])
            if pid != ids[i]:
                raise ValueError("Dataset index/ID mismatch")
            try:
                fp, provenance = verified_fingerprints(elem, ligand_root)
                y = float(elem["scores"]["neglog_aff"])
                if not np.isfinite(y):
                    raise ValueError("Nonfinite affinity")
            except Exception as error:
                errors.append(dict(split=split, id=pid, error=str(error)))
                continue
            accepted.append(pid)
            targets.append(y)
            morgans.append(fp["morgan"])
            maccs.append(fp["maccs"])
            records[split].append(dict(id=pid, target=y, **provenance))
        arrays.update({split+"_ids": np.asarray(accepted, dtype=str),
                       split+"_y": np.asarray(targets, dtype=float),
                       split+"_morgan": np.asarray(morgans, dtype=np.uint8),
                       split+"_maccs": np.asarray(maccs, dtype=np.uint8)})
        print(split, "accepted", len(accepted), "failed",
              sum(e["split"] == split for e in errors), flush=True)
    validation = dict(passed=not errors, expected_ids=expected, records=records,
                      errors=errors, parent_run=ep.relative(run_root),
                      parent_run_sha256=ep.sha(run_root/"run.json"),
                      source_sha256=initial_sources, rdkit=rdkit.__version__,
                      ligand_root=ep.relative(ligand_root) if ligand_root else None)
    ep.write_new(out/"ligand_validation.json", validation)
    with (out/"ligand_failures.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "id", "error"])
        writer.writeheader()
        writer.writerows(errors)
    if errors:
        raise ValueError(f"{len(errors)} ligand records failed; no feature cache published. See {out/'ligand_validation.json'}")
    if source_hashes() != initial_sources:
        raise ValueError("Fingerprint source changed during preparation")
    np.savez_compressed(out/"fingerprints.npz", **arrays)
    ep.write_new(out/"manifest.json", dict(
        schema=SCHEMA, complete=True, smoke=limit is not None,
        parent_run=ep.relative(run_root), parent_run_sha256=ep.sha(run_root/"run.json"),
        counts={s: len(v) for s, v in expected.items()}, ids=expected,
        dataset_sha256=info["dataset_sha256"], source_sha256=initial_sources,
        npz_sha256=ep.sha(out/"fingerprints.npz"),
        validation_sha256=ep.sha(out/"ligand_validation.json"),
        failures_sha256=ep.sha(out/"ligand_failures.csv"), rdkit=rdkit.__version__,
        protocol=dict(morgan=dict(radius=2, bits=1024, useChirality=False,
                                  useFeatures=False, useBondTypes=True),
                      maccs=dict(bits=167), concatenation_order=["maccs", "morgan"]),
        created_at=time.time()))
    print("LIGAND_CACHE_COMPLETE", ep.relative(out), flush=True)


def load_ligands(root, allow_smoke=False):
    root = Path(root).resolve()
    m = ep.read(root/"manifest.json")
    if m["schema"] != SCHEMA or not m["complete"] or (m["smoke"] and not allow_smoke):
        raise ValueError("Incomplete or smoke-only ligand dataset")
    if m["source_sha256"] != source_hashes():
        raise ValueError("Fingerprint/probe source changed")
    run_root = ep.resolve(m["parent_run"])
    info = ep.config(run_root)
    if ep.sha(run_root/"run.json") != m["parent_run_sha256"]:
        raise ValueError("Parent run changed")
    for path, key in [("fingerprints.npz", "npz_sha256"),
                      ("ligand_validation.json", "validation_sha256"),
                      ("ligand_failures.csv", "failures_sha256")]:
        if ep.sha(root/path) != m[key]:
            raise ValueError("Prepared ligand artifact changed: " + path)
    validation = ep.read(root/"ligand_validation.json")
    if validation["passed"] is not True or validation["errors"]:
        raise ValueError("Ligand chemistry audit failed")
    with np.load(root/"fingerprints.npz", allow_pickle=False) as d:
        arrays = {k: d[k] for k in d.files}
    for split in info["counts"]:
        ids = arrays[split+"_ids"].tolist()
        expected = m["ids"][split] if m["smoke"] else info["split_ids"][split]
        if ids != expected or len(ids) != len(set(ids)) or len(ids) != m["counts"][split]:
            raise ValueError("Ligand ID coverage/order changed")
        if arrays[split+"_y"].shape != (len(ids),) or not np.isfinite(arrays[split+"_y"]).all():
            raise ValueError("Invalid ligand labels")
        targets = {r["id"]: r["target"] for r in validation["records"][split]}
        if set(ids) != set(targets) or not np.array_equal(
                arrays[split+"_y"], np.asarray([targets[i] for i in ids])):
            raise ValueError("Ligand labels disagree with chemistry audit")
        for mode in ("morgan", "maccs"):
            x = arrays[split+"_"+mode]
            if x.shape != (len(ids), WIDTHS[mode]) or not np.isin(x, [0, 1]).all():
                raise ValueError("Wrong fingerprint dimension/values")
    return m, arrays


def ep_features(run_root):
    """Use the existing queued ridge's audited features; never rerun EP inference."""
    run_root = Path(run_root)
    ep.ready(run_root)
    receipt = ep.read(run_root/"completed/ep20_only.json")
    if (receipt["arm"] != "ep20_only" or receipt["seed"] is not None or
            receipt["run_sha256"] != ep.sha(run_root/"run.json")):
        raise ValueError("Wrong EP-only completion")
    for name, digest in receipt["files"].items():
        if ep.sha(ep.resolve(name, run_root)) != digest:
            raise ValueError("EP-only artifact changed: " + name)
    path = next(p for p in receipt["files"] if p.endswith("/features.npz"))
    with np.load(ep.resolve(path, run_root), allow_pickle=False) as d:
        arrays = {k: d[k] for k in d.files}
    return arrays, receipt


def ligand_matrix(data, split, mode):
    if mode not in MODES:
        raise ValueError("Unknown fingerprint mode")
    if mode == "maccs_morgan":
        x = np.concatenate([data[split+"_maccs"], data[split+"_morgan"]], axis=1)
    else:
        x = data[split+"_"+mode]
    return x.astype(float)


def design(data, split, mode, protein=None):
    ligand = ligand_matrix(data, split, mode)
    if protein is None:
        return ligand
    ids = data[split+"_ids"].tolist()
    ep_ids = protein[split+"_ids"].tolist()
    if len(ep_ids) != len(set(ep_ids)) or set(ep_ids) != set(ids):
        raise ValueError("EP and ligand cohorts differ")
    lookup = {pid: i for i, pid in enumerate(ep_ids)}
    indices = [lookup[pid] for pid in ids]
    y = protein[split+"_y"][indices]
    x = protein[split+"_x"][indices]
    if not np.array_equal(y, data[split+"_y"]):
        raise ValueError("EP and ligand targets differ")
    if x.shape != (len(ids), 256) or not np.isfinite(x).all():
        raise ValueError("Invalid pooled EP features")
    return np.concatenate([x.astype(float), ligand], axis=1)


def fit(data_root, results_root, mode, combined=False, allow_smoke=False):
    from v9_lba_probes import fit_probe
    import sklearn, scipy
    data_root, results_root = Path(data_root).resolve(), Path(results_root).resolve()
    manifest, arrays = load_ligands(data_root, allow_smoke)
    run_root = ep.resolve(manifest["parent_run"])
    protein = ep_features(run_root)[0] if combined else None
    # Current production EP-only features cover the entire cohort. Smoke combined
    # fits need an explicitly matched fixture, never implicit subset selection.
    arm = ("ep20_" if combined else "") + mode
    out = results_root/arm
    with ep.lock(results_root, arm):
        if out.exists():
            raise ValueError("Result directory exists; use a fresh results root")
        x = {s: design(arrays, s, mode, protein) for s in ("train", "val", "test")}
        scaler, model, search = fit_probe(x["train"], arrays["train_y"], x["val"], arrays["val_y"])
        predictions = model.predict(scaler.transform(x["test"]))
        if not np.isfinite(predictions).all():
            raise ValueError("Nonfinite test predictions")
        out.mkdir(parents=True)
        np.savez(out/"model.npz", coef=model.coef_, intercept=model.intercept_,
                 mean=scaler.mean_, scale=scaler.scale_)
        with (out/"predictions.csv").open("x", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "target", "prediction"])
            writer.writerows(zip(arrays["test_ids"], arrays["test_y"], predictions))
        metrics = ep.validate_predictions(out/"predictions.csv",
                                         dict(zip(arrays["test_ids"], arrays["test_y"])))
        # Catch source/input changes during fitting before publishing completion.
        load_ligands(data_root, allow_smoke)
        if combined:
            ep_features(run_root)
        ep.write_new(out/"result.json", dict(
            arm=arm, mode=mode, includes_ep=combined, smoke=manifest["smoke"],
            readout="linear ridge", input_width=int(x["train"].shape[1]), alpha=float(model.alpha),
            validation_search=search, metrics=metrics,
            protocol="train-only StandardScaler and ridge; validation alpha selection; no train+val refit",
            data_root=ep.relative(data_root), data_manifest_sha256=ep.sha(data_root/"manifest.json"),
            parent_run_sha256=manifest["parent_run_sha256"],
            ep_completion_sha256=ep.sha(run_root/"completed/ep20_only.json") if combined else None,
            checkpoint_sha256=ep.CHECKPOINT_SHA if combined else None,
            prediction_sha256=ep.sha(out/"predictions.csv"), model_sha256=ep.sha(out/"model.npz"),
            versions=dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__)))
    print("LIGAND_PROBE_COMPLETE", arm, json.dumps(metrics), flush=True)


def verify_result(path, data_root, arrays, mode, combined):
    path, data_root = Path(path), Path(data_root)
    result = ep.read(path/"result.json")
    expected_arm = ("ep20_" if combined else "") + mode
    manifest = ep.read(data_root/"manifest.json")
    if (result["smoke"] or result["arm"] != expected_arm or result["mode"] != mode or
            result["includes_ep"] != combined or
            result["parent_run_sha256"] != manifest["parent_run_sha256"] or
            result["checkpoint_sha256"] != (ep.CHECKPOINT_SHA if combined else None) or
            result["data_manifest_sha256"] != ep.sha(data_root/"manifest.json")):
        raise ValueError("Wrong result identity/cohort")
    for name, key in [("predictions.csv", "prediction_sha256"), ("model.npz", "model_sha256")]:
        if ep.sha(path/name) != result[key]:
            raise ValueError("Result artifact changed")
    values = ep.validate_predictions(path/"predictions.csv",
                                    dict(zip(arrays["test_ids"], arrays["test_y"])))
    if values != result["metrics"]:
        raise ValueError("Reported metrics disagree with predictions")
    return result


def summarize(data_root, results_root, mode):
    data_root, results_root = Path(data_root).resolve(), Path(results_root).resolve()
    manifest, arrays = load_ligands(data_root)
    run_root = ep.resolve(manifest["parent_run"])
    protein, receipt = ep_features(run_root)
    for split in ("train", "val", "test"):
        design(arrays, split, mode, protein)  # Exact common cohort and labels.
    ligand = verify_result(results_root/mode, data_root, arrays, mode, False)
    combined = verify_result(results_root/("ep20_"+mode), data_root, arrays, mode, True)
    if combined["ep_completion_sha256"] != ep.sha(run_root/"completed/ep20_only.json"):
        raise ValueError("Combined fit used different EP features")
    result_path = next(p for p in receipt["files"] if p.endswith("/result.json"))
    prediction_path = next(p for p in receipt["files"] if p.endswith("/predictions.csv"))
    old = ep.read(ep.resolve(result_path, run_root))
    metrics = ep.validate_predictions(ep.resolve(prediction_path, run_root),
                                     dict(zip(arrays["test_ids"], arrays["test_y"])))
    if metrics != old["metrics"] or metrics != receipt["metrics"]:
        raise ValueError("EP-only result metrics changed")
    rows = [dict(arm=mode, alpha=ligand["alpha"], **ligand["metrics"]),
            dict(arm="ep20_only", alpha=old["alpha"], **metrics),
            dict(arm="ep20_"+mode, alpha=combined["alpha"], **combined["metrics"])]
    folder = results_root/("comparison_"+mode)
    folder.mkdir(exist_ok=False)
    with (folder/"results.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# ATOM3D LBA-30: matched ligand / EP-20 ridge probes", "",
             "Identical complexes and train-only preprocessing; validation selects alpha.",
             "One deterministic fit per arm. No seed SD or significance claim.", "",
             "| Arm | Alpha | RMSE | Pearson r | Spearman rho | R² |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| "+row["arm"]+" | "+str(row["alpha"])+" | "+
                     " | ".join(f"{row[k]:.5f}" for k in ep.METRICS)+" |")
    (folder/"COMPARISON.md").write_text("\n".join(lines)+"\n")
    ep.write_new(folder/"audit.json", dict(passed=True, counts=manifest["counts"],
                 ligand_manifest_sha256=ep.sha(data_root/"manifest.json"),
                 ep_completion_sha256=ep.sha(run_root/"completed/ep20_only.json"),
                 ligand_result_sha256=ep.sha(results_root/mode/"result.json"),
                 combined_result_sha256=ep.sha(results_root/("ep20_"+mode)/"result.json"),
                 ep_result_sha256=ep.sha(ep.resolve(result_path, run_root))))
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--run-root", required=True)
    prep.add_argument("--out", required=True)
    prep.add_argument("--ligand-root")
    prep.add_argument("--limit", type=int)
    for name in ("run", "summarize"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--data-root", required=True)
        cmd.add_argument("--results-root", required=True)
        cmd.add_argument("--mode", choices=MODES, default="morgan")
        if name == "run":
            cmd.add_argument("--with-ep", action="store_true")
            cmd.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.run_root, args.out, args.ligand_root, args.limit)
    elif args.action == "run":
        fit(args.data_root, args.results_root, args.mode, args.with_ep, args.allow_smoke)
    else:
        summarize(args.data_root, args.results_root, args.mode)


if __name__ == "__main__":
    main()
