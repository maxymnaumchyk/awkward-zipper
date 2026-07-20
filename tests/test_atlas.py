import awkward
import uproot
from atlas_schema.schema import NtupleSchema
from coffea.nanoevents import NanoEventsFactory

from awkward_zipper import Ntuple

# load a test root file
file_name = "tests/samples/atlas_ntuple.root"
tree_name = "analysis"

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
    test_suggested_behavior()
    test_behaviors()
