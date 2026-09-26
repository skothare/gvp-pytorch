"""Reported direct-input and frozen-random controls (no shuffle arm)."""
import hashlib, json
from pathlib import Path
import numpy as np
import torch
from cache_utils import tensor_hash
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
