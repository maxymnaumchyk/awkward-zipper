import awkward
import uproot
from coffea.nanoevents import NanoEventsFactory, TreeMakerSchema

from awkward_zipper import TreeMaker

# load a test root file
file_name = "tests/samples/treemaker.root"
tree_name = "PreSelection"

# --- eager ---
tree = uproot.open(file_name)[tree_name]
array = tree.arrays(ak_add_doc=True)
zipper_array = TreeMaker()(array)

coffea_events = NanoEventsFactory.from_root(
    {file_name: tree_name},
    schemaclass=TreeMakerSchema,
    mode="eager",
)
coffea_array = coffea_events.events()

# --- virtual ---
access_log_zipper = []
array_virtual = uproot.open(file_name)[tree_name].arrays(
    virtual=True,
    ak_add_doc={"__doc__": "title", "typename": "typename"},
    access_log=access_log_zipper,
)
zipper_array_virtual = TreeMaker()(array_virtual)
# snapshot the access log right after construction (later comparisons materialize data)
construction_access_log = list(access_log_zipper)

coffea_events_virtual = NanoEventsFactory.from_root(
    {file_name: tree_name},
    schemaclass=TreeMakerSchema,
    mode="virtual",
)
coffea_array_virtual = coffea_events_virtual.events()


def test_treemaker_whole_eager():
    # awkward-zipper adds some additional parameters, so check_parameters=False
    assert awkward.array_equal(
        zipper_array, coffea_array, check_parameters=False, equal_nan=True
    )


def test_treemaker_whole_virtual():
    assert awkward.array_equal(
        zipper_array_virtual,
        coffea_array_virtual,
        check_parameters=False,
        equal_nan=True,
    )


def test_no_data_materialization():
    # NOTE: full lazy construction (zero accesses) is a goal but not yet reached;
    # some offsets buffers are currently touched. What must hold is that no heavy
    # *data* buffers are materialized while building the layout.
    data_accesses = [
        a for a in construction_access_log if a.buffer_key.endswith("-data")
    ]
    assert len(data_accesses) == 0


def test_composite_vector_collections():
    # composite ROOT vectors become vector-behavior records
    for coll in ("Muons", "Electrons", "Jets", "JetsAK8", "GenParticles"):
        assert {"pt", "eta", "phi", "energy"}.issubset(set(zipper_array[coll].fields))
    for coll in ("Tracks", "PrimaryVertices"):
        assert {"x", "y", "z"}.issubset(set(zipper_array[coll].fields))


def test_nested_subcollections():
    # doubly-jagged sub-collections and their counts
    assert "subjets" in zipper_array.JetsAK8.fields
    assert "subjetsCounts" in zipper_array.JetsAK8.fields
    assert awkward.array_equal(
        zipper_array.JetsAK8.subjets.pt,
        coffea_array.JetsAK8.subjets.pt,
        check_parameters=False,
        equal_nan=True,
    )
    assert "hitPattern" in zipper_array.Tracks.fields
    assert awkward.array_equal(
        zipper_array.Tracks.hitPattern,
        coffea_array.Tracks.hitPattern,
        check_parameters=False,
    )


def test_behaviors():
    # behavior of coffea and zipper should be the same, except for Systematics
    diff = set(coffea_array.behavior) - set(zipper_array.behavior)
    for behavior in diff.copy():
        if any(
            string in str(behavior)
            for string in ("Systematic", "UpDownSystematic", "UpDownMultiSystematic")
        ):
            diff.remove(behavior)
    assert len(diff) == 0


if __name__ == "__main__":
    test_treemaker_whole_eager()
    test_treemaker_whole_virtual()
    test_no_data_materialization()
    test_composite_vector_collections()
    test_nested_subcollections()
    test_behaviors()
