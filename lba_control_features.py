"""Feature operations for isolated ATOM3D LBA controls; no training-side EP model."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import lba_ep20 as ep

ARMS = ("raw_qr", "random_ep", "shuffled_ep")
WIDTHS = {"raw_qr": 2, "random_ep": 256, "shuffled_ep": 256}
LABELS = {"raw_qr": "GVP + raw PQR charges/radii",
          "random_ep": "GVP + frozen random V9 decoder",
          "shuffled_ep": "GVP + within-pocket shuffled EP-20 decoder"}


def tensor_hash(value):
    from v8_lba import tensor_hash as digest
    return digest(value)


def validate_features(value, n, width):
    if (not isinstance(value, torch.Tensor) or value.shape != (n, width)
            or value.dtype != torch.float32 or value.device.type != "cpu"
            or not torch.isfinite(value).all() or value.requires_grad):
        raise ValueError("Expected finite, frozen CPU float32 features with exact atom count/width")


def raw_features(elem, lut, stems):
    from pqr_charges import lookup_charges_radii
    q, r, _ = lookup_charges_radii(elem["atoms_pocket"], lut, stems)
    value = torch.tensor(np.column_stack([q, r]), dtype=torch.float32)
    validate_features(value, len(elem["atoms_pocket"]), 2)
    if (value[:, 1] < 0).any():
        raise ValueError("Negative radius")
    return value


def permutation(n, split, identifier, seed):
    if n < 1 or split not in ("train", "val", "test"):
        raise ValueError("Invalid permutation domain")
    payload = json.dumps(["within-pocket-row-permutation-v1", int(seed), split, str(identifier)])
    key = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:16], "little")
    return torch.from_numpy(np.random.default_rng(key).permutation(n).astype(np.int64))


def validate_permutation(value, n):
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.int64
            or value.device.type != "cpu" or value.shape != (n,)
            or not torch.equal(torch.sort(value).values, torch.arange(n))):
        raise ValueError("Invalid atom-row permutation")


def initialize_random_model(architecture, seed, device="cpu"):
    """Initialize the whole model without ever reading/loading pretrained weights."""
    sys.path.insert(0, str(ep.WORKSPACE / "Estats"))
    from proteinshake_eval._electroprot_loader import load_model_module
    kwargs = dict(architecture)
    has_norm = kwargs.pop("encoder_final_norm")
    module = load_model_module("V9", "lba_controls")
    # Initialization must not consume the caller's downstream training RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = module.create_model_with_distance_features(**kwargs)
    if not has_norm:
        model.encoder.final_norm = torch.nn.Identity()
    return model.to(device).eval().requires_grad_(False)


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fit_scaler(values):
    """Stable atom-weighted population statistics; caller supplies TRAIN only."""
    count, mean, m2 = 0, np.zeros(2), np.zeros(2)
    for value in values:
        validate_features(value, len(value), 2)
        x = value.numpy().astype(np.float64)
        if not len(x):
            raise ValueError("Empty raw feature block")
        n, mu = len(x), x.mean(axis=0)
        delta = mu - mean
        m2 += np.square(x - mu).sum(axis=0) + delta**2 * count*n/(count+n)
        mean += delta*n/(count+n)
        count += n
    if not count:
        raise ValueError("Empty training scaler input")
    variance = np.maximum(m2/count, 0)
    scale = np.sqrt(variance)
    scale[scale == 0] = 1
    return dict(fit_split="train", weighting="pocket atoms", count=count,
                mean=mean.tolist(), variance=variance.tolist(), scale=scale.tolist())


def standardize(value, scaler):
    if scaler.get("fit_split") != "train":
        raise ValueError("Raw features require a train-only scaler")
    mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
    if (mean.shape != (2,) or scale.shape != (2,) or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError("Invalid raw feature scaler")
    result = ((value.double() - torch.tensor(mean))/torch.tensor(scale)).float()
    validate_features(result, len(value), 2)
    return result


def load_features(directory, identifier, record, width):
    from v9_lba import safe_id
    payload = torch.load(Path(directory)/(safe_id(identifier)+"_pocket.pt"),
                         map_location="cpu", weights_only=True)
    value = payload["features"]
    validate_features(value, record["n_atoms"], width)
    if tensor_hash(value) != record["tensor_sha256"]:
        raise ValueError(identifier + ": changed control tensor")
    if "permutation_sha256" in record:
        perm = payload["permutation"]
        validate_permutation(perm, record["n_atoms"])
        if tensor_hash(perm) != record["permutation_sha256"]:
            raise ValueError(identifier + ": changed permutation")
    elif "permutation" in payload:
        raise ValueError("Unexpected permutation")
    return payload


def attach_features(graph, value):
    n = int((~graph.lig_flag).sum())
    validate_features(value, n, value.shape[1])
    block = torch.zeros(graph.num_nodes, value.shape[1], dtype=torch.float32)
    block[~graph.lig_flag] = value
    graph.v3_emb = block
    return graph


class ControlTransform:
    """Preserve the baseline graph; add an audited, fixed per-pocket feature block."""
    def __init__(self, directory, manifest, scaler=None):
        from gvp.atom3d import LBATransform
        self.base = LBATransform()
        self.directory = str(directory)
        self.manifest = manifest
        self.scaler = scaler
        self.arm = manifest["arm"]

    def __call__(self, elem):
        from v8_lba import input_fingerprint
        record = self.manifest["entries"][str(elem["id"])]
        if input_fingerprint(elem) != record["input_sha256"]:
            raise ValueError("Ordered pocket/protein inputs changed")
        value = load_features(self.directory, str(elem["id"]), record, WIDTHS[self.arm])["features"]
        if self.arm == "raw_qr":
            value = standardize(value, self.scaler)
        return attach_features(self.base(elem), value)
