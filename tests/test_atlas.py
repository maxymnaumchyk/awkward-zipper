import atexit
import pathlib
import tempfile

import awkward
import numpy as np
import uproot
from atlas_schema.schema import NtupleSchema
from coffea.nanoevents import NanoEventsFactory

from awkward_zipper import Ntuple

N_EVENTS = 20


def write_sample(path, tree_name):
    """Write a small ATLAS-ntuple-style sample.

    atlas-schema ships no ROOT file, so one is generated here rather than committed:
    when the ntuple layout changes only this function has to be updated, never a
    binary fixture.
    """
    rng = np.random.default_rng(42)

    def jagged(maxn, lo, hi, dtype="float32"):
        counts = rng.integers(0, maxn + 1, size=N_EVENTS)
        flat = rng.uniform(lo, hi, size=int(counts.sum())).astype(dtype)
        return awkward.unflatten(awkward.Array(flat), counts)

    def like(ref, lo, hi, dtype="float32"):
        counts = awkward.num(ref)
        flat = rng.uniform(lo, hi, size=int(awkward.sum(counts))).astype(dtype)
        return awkward.unflatten(awkward.Array(flat), counts)

    branches = {}
    # event-level singletons (the schema's event_ids)
    branches["eventNumber"] = np.arange(N_EVENTS, dtype=np.uint64) + 1000
    branches["runNumber"] = np.full(N_EVENTS, 654321, dtype=np.uint32)
    branches["lumiBlock"] = np.full(N_EVENTS, 12390123, dtype=np.uint32)
    branches["mcChannelNumber"] = np.full(N_EVENTS, 410470, dtype=np.uint32)
    branches["actualInteractionsPerCrossing"] = rng.uniform(20, 40, N_EVENTS).astype(
        "float32"
    )
    branches["averageInteractionsPerCrossing"] = rng.uniform(20, 40, N_EVENTS).astype(
        "float32"
    )
    branches["dataTakingYear"] = np.full(N_EVENTS, 2018, dtype=np.uint32)
    branches["mcEventWeights"] = jagged(3, 0.9, 1.1)

    # electrons: nominal plus a systematic variation
    el_pt = jagged(4, 10e3, 200e3)
    branches["el_pt_NOSYS"] = el_pt
    branches["el_pt_EG_SCALE_ALL__1up"] = like(el_pt, 10e3, 200e3)
    branches["el_eta"] = like(el_pt, -2.5, 2.5)
    branches["el_phi"] = like(el_pt, -np.pi, np.pi)
    branches["el_charge"] = like(el_pt, -1, 1, "int32")
    branches["el_select_baseline_NOSYS"] = like(el_pt, 0, 2, "int32")

    # jets: nominal plus a systematic; 'm' gets renamed to 'mass'
    jet_pt = jagged(6, 20e3, 500e3)
    branches["jet_pt_NOSYS"] = jet_pt
    branches["jet_pt_JET_Res__1up"] = like(jet_pt, 20e3, 500e3)
    branches["jet_eta"] = like(jet_pt, -4.5, 4.5)
    branches["jet_phi"] = like(jet_pt, -np.pi, np.pi)
    branches["jet_m"] = like(jet_pt, 1e3, 50e3)

    # muons and photons (photons get a synthesized mass and charge)
    mu_pt = jagged(3, 10e3, 150e3)
    branches["mu_pt_NOSYS"] = mu_pt
    branches["mu_eta"] = like(mu_pt, -2.7, 2.7)
    branches["mu_phi"] = like(mu_pt, -np.pi, np.pi)
    ph_pt = jagged(2, 20e3, 300e3)
    branches["ph_pt_NOSYS"] = ph_pt
    branches["ph_eta"] = like(ph_pt, -2.5, 2.5)
    branches["ph_phi"] = like(ph_pt, -np.pi, np.pi)

    # MET: 'met' is aliased to 'rho' by the schema
    branches["met_met_NOSYS"] = rng.uniform(0, 300e3, N_EVENTS).astype("float32")
    branches["met_phi_NOSYS"] = rng.uniform(-np.pi, np.pi, N_EVENTS).astype("float32")

    # event-level weight / pass / trigPassed collections
    branches["weight_mc_NOSYS"] = rng.uniform(0.5, 1.5, N_EVENTS).astype("float32")
    branches["weight_pileup_NOSYS"] = rng.uniform(0.8, 1.2, N_EVENTS).astype("float32")
    branches["pass_SR_NOSYS"] = rng.integers(0, 2, N_EVENTS).astype(bool)
    branches["trigPassed_HLT_e26_lhtight"] = rng.integers(0, 2, N_EVENTS).astype(bool)

    with uproot.recreate(path) as file:
        file[tree_name] = branches


# generate the test file once, when this module is imported, into a temporary
# directory that is removed again when the interpreter exits, so nothing is written
# into the repository
_tmp_dir = tempfile.TemporaryDirectory()
atexit.register(_tmp_dir.cleanup)
file_name = str(pathlib.Path(_tmp_dir.name) / "atlas_ntuple.root")
tree_name = "analysis"
write_sample(file_name, tree_name)

# --- eager ---
tree = uproot.open(file_name)[tree_name]
array = tree.arrays(ak_add_doc=True)
zipper_array = Ntuple()(array)

reference_events = NanoEventsFactory.from_root(
    {file_name: tree_name},
    schemaclass=NtupleSchema,
    mode="eager",
)
reference_array = reference_events.events()

# --- virtual ---
access_log_zipper = []
array_virtual = uproot.open(file_name)[tree_name].arrays(
    virtual=True,
    ak_add_doc=True,
    access_log=access_log_zipper,
)
zipper_array_virtual = Ntuple()(array_virtual)
# snapshot the access log right after construction (later comparisons materialize data)
construction_access_log = list(access_log_zipper)

reference_events_virtual = NanoEventsFactory.from_root(
    {file_name: tree_name},
    schemaclass=NtupleSchema,
    mode="virtual",
)
reference_array_virtual = reference_events_virtual.events()


def test_atlas_whole_eager():
    # awkward-zipper adds some additional parameters, so check_parameters=False
    assert awkward.array_equal(
        zipper_array, reference_array, check_parameters=False, equal_nan=True
    )


def test_atlas_whole_virtual():
    assert awkward.array_equal(
        zipper_array_virtual,
        reference_array_virtual,
        check_parameters=False,
        equal_nan=True,
    )


def test_no_materialization():
    # construction is fully lazy: no buffers (neither offsets/Index nor data)
    # are materialized while building the layout
    assert len(construction_access_log) == 0


def test_collections_and_systematics():
    # nominal collections carry their mixin behaviors
    assert zipper_array.el.layout.content.parameters["__record__"] == "Electron"
    assert zipper_array.jet.layout.content.parameters["__record__"] == "Jet"
    assert zipper_array.met.layout.parameters["__record__"] == "MissingET"

    # 'jet_m' is renamed to 'mass', 'met' is aliased to 'rho', and the leptons and
    # photons get a synthesized 'mass'
    assert "mass" in zipper_array.jet.fields
    assert "m" not in zipper_array.jet.fields
    assert "rho" in zipper_array.met.fields
    assert "mass" in zipper_array.el.fields
    assert {"mass", "charge"}.issubset(zipper_array.ph.fields)

    # systematic variations sit alongside the nominal collections
    systematics = zipper_array.layout.parameters["metadata"]["systematics"]
    assert systematics == ["EG_SCALE_ALL__1up", "JET_Res__1up"]
    for systematic in systematics:
        assert zipper_array[systematic].layout.parameters["__record__"] == "Systematic"

    # the varied branch really is different from the nominal one
    assert not awkward.all(zipper_array.EG_SCALE_ALL__1up.el.pt == zipper_array.el.pt)
    # a collection with no variation for this systematic falls back to nominal
    assert awkward.array_equal(zipper_array.EG_SCALE_ALL__1up.mu.pt, zipper_array.mu.pt)


def test_record_projections():
    # atlas-schema registers projections only on the *Array classes, so per-element
    # access raises there; awkward-zipper registers the *Record variants too
    jet = zipper_array.jet[0][0]
    assert jet.to_Vector2D().fields
    assert jet.to_Vector3D().fields


def test_suggested_behavior():
    assert Ntuple.suggested_behavior("truthjet") == "Jet"
    assert Ntuple.suggested_behavior("SignalElectron") == "Electron"
    assert Ntuple.suggested_behavior("generatorWeight") == "Weight"
    assert Ntuple.suggested_behavior("aVeryStrangelyNamedBranchWithNoMatch") == (
        "NanoCollection"
    )


def test_behaviors():
    # behavior of the reference implementation and zipper should be the same
    # except for Systematics, since they are not included in zipper
    diff = set(reference_array.behavior) - set(zipper_array.behavior)
    for behavior in diff.copy():
        if any(
            string in str(behavior)
            for string in ("Systematic", "UpDownSystematic", "UpDownMultiSystematic")
        ):
            diff.remove(behavior)
    assert len(diff) == 0


if __name__ == "__main__":
    test_atlas_whole_eager()
    test_atlas_whole_virtual()
    test_no_materialization()
    test_collections_and_systematics()
    test_record_projections()
    test_suggested_behavior()
    test_behaviors()
