import copy
import numpy as np
import pandas as pd
import pytest
import torch
import control_features as f
from pqr_charges import atom_key
def element():
    df = pd.DataFrame(dict(element=["N", "C", "H"], resname=["ALA"]*3,
        name=["N", "CA", "H"], chain=["A"]*3, residue=[1]*3,
        x=[0., 1., 2.], y=[0., 0., 0.], z=[0., 0., 0.]))
    return dict(id="toy", atoms_pocket=df, atoms_protein=df.copy(),
                atoms_ligand=df.iloc[:1].copy(), scores={"neglog_aff": 5.})

def lookup():
    return {atom_key("A", 1, "ALA", name, ""): pair for name, pair in
            [("N", (-.5, 1.5)), ("CA", (.25, 2.)), ("H", (.25, 0.))]}

def test_qr_uses_existing_lookup_order_zero_radii_and_fallback():
    elem = element()
    actual = f.raw_features(elem, lookup(), {})
    torch.testing.assert_close(actual, torch.tensor([[-.5, 1.5], [.25, 2.], [.25, 0.]]))
    elem["atoms_pocket"] = elem["atoms_pocket"].iloc[[2, 0, 1]]
    assert torch.equal(f.raw_features(elem, lookup(), {}), actual[[2, 0, 1]])
    elem["atoms_pocket"] = elem["atoms_pocket"].copy()
    elem["atoms_pocket"].loc[0, ["name", "element", "resname"]] = ["ZN", "ZN", "ZN"]
    from pqr_charges import FALLBACK_RADIUS_DEFAULT
    assert torch.equal(f.raw_features(elem, lookup(), {})[1], torch.tensor([0., FALLBACK_RADIUS_DEFAULT]))

def test_scaler_is_atom_weighted_train_only_and_constant_safe():
    train = [torch.tensor([[-1., 2.], [1., 2.]]), torch.tensor([[6., 2.]])]
    s = f.fit_scaler(iter(train))
    x = torch.cat(train).numpy().astype(float)
    assert s["count"] == 3
    np.testing.assert_allclose(s["mean"], x.mean(0))
    np.testing.assert_allclose(s["variance"], x.var(0))
    np.testing.assert_allclose(f.standardize(torch.cat(train), s).mean(0), 0, atol=1e-7)
    original = copy.deepcopy(s)
    f.standardize(torch.tensor([[100000., 99.]]), s)
    assert s == original
    with pytest.raises(ValueError, match="train-only"):
        f.standardize(train[0], dict(s, fit_split="val"))
