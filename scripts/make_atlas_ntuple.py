"""Generate a small ATLAS-ntuple-style ROOT sample for awkward-zipper tests."""

import awkward as ak
import numpy as np
import uproot

rng = np.random.default_rng(42)
N = 20


def jagged(maxn, lo, hi, dtype="float32"):
    counts = rng.integers(0, maxn + 1, size=N)
    flat = rng.uniform(lo, hi, size=int(counts.sum())).astype(dtype)
    return ak.unflatten(ak.Array(flat), counts)


def like(ref, lo, hi, dtype="float32"):
    counts = ak.num(ref)
    flat = rng.uniform(lo, hi, size=int(ak.sum(counts))).astype(dtype)
    return ak.unflatten(ak.Array(flat), counts)


branches = {}
# event-level singletons (event_ids)
branches["eventNumber"] = np.arange(N, dtype=np.uint64) + 1000
branches["runNumber"] = np.full(N, 654321, dtype=np.uint32)
branches["lumiBlock"] = np.full(N, 12390123, dtype=np.uint32)
branches["mcChannelNumber"] = np.full(N, 410470, dtype=np.uint32)
branches["actualInteractionsPerCrossing"] = rng.uniform(20, 40, N).astype("float32")
branches["averageInteractionsPerCrossing"] = rng.uniform(20, 40, N).astype("float32")
branches["dataTakingYear"] = np.full(N, 2018, dtype=np.uint32)
branches["mcEventWeights"] = jagged(3, 0.9, 1.1)

# electrons: nominal + a systematic variation
el_pt = jagged(4, 10e3, 200e3)
branches["el_pt_NOSYS"] = el_pt
branches["el_pt_EG_SCALE_ALL__1up"] = like(el_pt, 10e3, 200e3)
branches["el_eta"] = like(el_pt, -2.5, 2.5)
branches["el_phi"] = like(el_pt, -np.pi, np.pi)
branches["el_charge"] = like(el_pt, -1, 1, "int32")
branches["el_select_baseline_NOSYS"] = like(el_pt, 0, 2, "int32")

# jets: nominal + systematic; 'm' gets renamed to 'mass' by the schema
jet_pt = jagged(6, 20e3, 500e3)
branches["jet_pt_NOSYS"] = jet_pt
branches["jet_pt_JET_Res__1up"] = like(jet_pt, 20e3, 500e3)
branches["jet_eta"] = like(jet_pt, -4.5, 4.5)
branches["jet_phi"] = like(jet_pt, -np.pi, np.pi)
branches["jet_m"] = like(jet_pt, 1e3, 50e3)

# muons and photons (photons get full_like mass+charge)
mu_pt = jagged(3, 10e3, 150e3)
branches["mu_pt_NOSYS"] = mu_pt
branches["mu_eta"] = like(mu_pt, -2.7, 2.7)
branches["mu_phi"] = like(mu_pt, -np.pi, np.pi)
ph_pt = jagged(2, 20e3, 300e3)
branches["ph_pt_NOSYS"] = ph_pt
branches["ph_eta"] = like(ph_pt, -2.5, 2.5)
branches["ph_phi"] = like(ph_pt, -np.pi, np.pi)

# MET: 'met' aliased to 'rho' by the schema
branches["met_met_NOSYS"] = rng.uniform(0, 300e3, N).astype("float32")
branches["met_phi_NOSYS"] = rng.uniform(-np.pi, np.pi, N).astype("float32")

# weight / pass / trigPassed collections
branches["weight_mc_NOSYS"] = rng.uniform(0.5, 1.5, N).astype("float32")
branches["weight_pileup_NOSYS"] = rng.uniform(0.8, 1.2, N).astype("float32")
branches["pass_SR_NOSYS"] = rng.integers(0, 2, N).astype(bool)
branches["trigPassed_HLT_e26_lhtight"] = rng.integers(0, 2, N).astype(bool)

with uproot.recreate("tests/samples/atlas_ntuple.root") as f:
    f["analysis"] = branches
print("wrote tests/samples/atlas_ntuple.root")
with uproot.open("tests/samples/atlas_ntuple.root")["analysis"] as t:
    print("branches:", len(t.keys()))
