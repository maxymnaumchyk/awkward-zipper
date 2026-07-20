from uuid import uuid4

import awkward
import numpy as np
from atlas_schema.schema import NtupleSchema
from coffea.nanoevents import NanoEventsFactory
from coffea.nanoevents.mapping import SimplePreloadedColumnSource

from awkward_zipper import Ntuple

# atlas-schema ships no ROOT sample and builds its test inputs in memory, so the
# columns below are the fixture for both implementations
N_EVENTS = 20


def _build_columns():
    rng = np.random.default_rng(42)

    def jagged(maxn, lo, hi, dtype="float32"):
        counts = rng.integers(0, maxn + 1, size=N_EVENTS)
        flat = rng.uniform(lo, hi, size=int(counts.sum())).astype(dtype)
        return awkward.unflatten(awkward.Array(flat), counts)

    def like(ref, lo, hi, dtype="float32"):
        counts = awkward.num(ref)
        flat = rng.uniform(lo, hi, size=int(awkward.sum(counts))).astype(dtype)
        return awkward.unflatten(awkward.Array(flat), counts)

    def flat(lo, hi, dtype="float32"):
        return awkward.Array(rng.uniform(lo, hi, size=N_EVENTS).astype(dtype))

    columns = {}
    # event-level singletons (the schema's event_ids)
    columns["eventNumber"] = awkward.Array(np.arange(N_EVENTS, dtype=np.uint64) + 1000)
    columns["runNumber"] = awkward.Array(np.full(N_EVENTS, 654321, dtype=np.uint32))
    columns["lumiBlock"] = awkward.Array(np.full(N_EVENTS, 12390123, dtype=np.uint32))
    columns["mcChannelNumber"] = awkward.Array(
        np.full(N_EVENTS, 410470, dtype=np.uint32)
    )
    columns["actualInteractionsPerCrossing"] = flat(20, 40)
    columns["averageInteractionsPerCrossing"] = flat(20, 40)
    columns["dataTakingYear"] = awkward.Array(np.full(N_EVENTS, 2018, dtype=np.uint32))
    columns["mcEventWeights"] = jagged(3, 0.9, 1.1)

    # electrons: nominal plus a systematic variation
    el_pt = jagged(4, 10e3, 200e3)
    columns["el_pt_NOSYS"] = el_pt
    columns["el_pt_EG_SCALE_ALL__1up"] = like(el_pt, 10e3, 200e3)
    columns["el_eta"] = like(el_pt, -2.5, 2.5)
    columns["el_phi"] = like(el_pt, -np.pi, np.pi)
    columns["el_charge"] = like(el_pt, -1, 1, "int32")
    columns["el_select_baseline_NOSYS"] = like(el_pt, 0, 2, "int32")

    # jets: nominal plus a systematic; 'm' gets renamed to 'mass'
    jet_pt = jagged(6, 20e3, 500e3)
    columns["jet_pt_NOSYS"] = jet_pt
    columns["jet_pt_JET_Res__1up"] = like(jet_pt, 20e3, 500e3)
    columns["jet_eta"] = like(jet_pt, -4.5, 4.5)
    columns["jet_phi"] = like(jet_pt, -np.pi, np.pi)
    columns["jet_m"] = like(jet_pt, 1e3, 50e3)

    # muons and photons (photons get a synthesized mass and charge)
    mu_pt = jagged(3, 10e3, 150e3)
    columns["mu_pt_NOSYS"] = mu_pt
    columns["mu_eta"] = like(mu_pt, -2.7, 2.7)
    columns["mu_phi"] = like(mu_pt, -np.pi, np.pi)
    ph_pt = jagged(2, 20e3, 300e3)
    columns["ph_pt_NOSYS"] = ph_pt
    columns["ph_eta"] = like(ph_pt, -2.5, 2.5)
    columns["ph_phi"] = like(ph_pt, -np.pi, np.pi)

    # MET: 'met' is aliased to 'rho' by the schema
    columns["met_met_NOSYS"] = flat(0, 300e3)
    columns["met_phi_NOSYS"] = flat(-np.pi, np.pi)

    # event-level weight / pass / trigPassed collections
    columns["weight_mc_NOSYS"] = flat(0.5, 1.5)
    columns["weight_pileup_NOSYS"] = flat(0.8, 1.2)
    columns["pass_SR_NOSYS"] = awkward.Array(rng.integers(0, 2, N_EVENTS).astype(bool))
    columns["trigPassed_HLT_e26_lhtight"] = awkward.Array(
        rng.integers(0, 2, N_EVENTS).astype(bool)
    )
    return columns


columns = _build_columns()


def _as_record_array(columns):
    """Zip the columns into the one record array a layout builder is handed."""
    contents = [awkward.to_layout(column) for column in columns.values()]
    return awkward.Array(
        awkward.contents.RecordArray(contents, list(columns), length=N_EVENTS)
    )


# --- eager ---
array = _as_record_array(columns)
zipper_array = Ntuple()(array)

reference_array = NanoEventsFactory.from_preloaded(
    SimplePreloadedColumnSource(columns, uuid4(), N_EVENTS, object_path="/Events"),
    metadata={"dataset": "test"},
    schemaclass=NtupleSchema,
).events()

# --- virtual ---
# coffea's preloaded source is eager, so there is no virtual reference to compare
# against; instead feed awkward-zipper the same buffers behind generators that record
# every materialization, so construction can be shown to touch none of them
materialized = []


def _as_virtual(array):
    form, length, buffers = awkward.to_buffers(array)
    nplike = awkward._nplikes.numpy.Numpy.instance()

    def generator(key, buffer):
        def generate():
            materialized.append(key)
            return buffer

        return generate

    # buffer shapes are declared up front, the way uproot declares them when reading
    # with virtual=True; without that awkward has to read offsets[-1] to learn how
    # long each content is, which would materialize buffers before the builder runs
    virtual_buffers = {
        key: awkward._nplikes.virtual.VirtualNDArray(
            nplike=nplike,
            shape=buffer.shape,
            dtype=buffer.dtype,
            generator=generator(key, buffer),
            shape_generator=None,
        )
        for key, buffer in buffers.items()
    }
    return awkward.from_buffers(form, length, virtual_buffers)


array_virtual = _as_virtual(array)
zipper_array_virtual = Ntuple()(array_virtual)
# snapshot right after construction (the comparisons below materialize data)
construction_materialized = list(materialized)


def test_atlas_whole_eager():
    # awkward-zipper adds some additional parameters, so check_parameters=False
    assert awkward.array_equal(
        zipper_array, reference_array, check_parameters=False, equal_nan=True
    )


def test_atlas_whole_virtual():
    assert awkward.array_equal(
        zipper_array_virtual,
        reference_array,
        check_parameters=False,
        equal_nan=True,
    )


def test_no_materialization():
    # construction is fully lazy: no buffers (neither offsets/Index nor data)
    # are materialized while building the layout
    assert construction_materialized == []


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
