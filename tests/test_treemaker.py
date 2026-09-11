import awkward
import coffea
import pytest
import uproot
from coffea.nanoevents import NanoEventsFactory, TreeMakerSchema
from packaging.version import parse as parse_version

from awkward_zipper import TreeMaker

# coffea up to 2026.7.0 truncates TreeMaker subbranch field names to one character
# (scikit-hep/coffea#1582); awkward-zipper follows the fixed behaviour
_COFFEA_TRUNCATES_FIELD_NAMES = parse_version(coffea.__version__) <= parse_version(
    "2026.7.0"
)

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


@pytest.mark.skipif(
    _COFFEA_TRUNCATES_FIELD_NAMES,
    reason="installed coffea truncates TreeMaker subbranch field names (pre-#1582)",
)
def test_treemaker_whole_eager():
    # awkward-zipper adds some additional parameters, so check_parameters=False
    assert awkward.array_equal(
        zipper_array, coffea_array, check_parameters=False, equal_nan=True
    )


@pytest.mark.skipif(
    _COFFEA_TRUNCATES_FIELD_NAMES,
    reason="installed coffea truncates TreeMaker subbranch field names (pre-#1582)",
)
def test_treemaker_whole_virtual():
    assert awkward.array_equal(
        zipper_array_virtual,
        coffea_array_virtual,
        check_parameters=False,
        equal_nan=True,
    )


def test_no_materialization():
    # construction is fully lazy: no buffers (neither offsets/Index nor data)
    # are materialized while building the layout
    assert len(construction_access_log) == 0


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


def _record_fields(form):
    """Nested record field names, in order (array_equal ignores field order)."""
    if isinstance(form, awkward.forms.RecordForm):
        return [
            (f, _record_fields(c))
            for f, c in zip(form.fields, form.contents, strict=True)
        ]
    if isinstance(form, awkward.forms.UnionForm):
        return [_record_fields(c) for c in form.contents]
    if hasattr(form, "content"):
        return _record_fields(form.content)
    return None


def _collection_fields(form):
    """Per-collection field order; coffea builds the top-level record from a set,
    so its own order varies from run to run and is compared as a mapping."""
    return {
        f: _record_fields(c) for f, c in zip(form.fields, form.contents, strict=True)
    }


def test_field_order():
    # array_equal matches record fields by name; the layouts must also list them
    # in the same order as coffea
    assert _collection_fields(zipper_array.layout.form) == _collection_fields(
        coffea_array.layout.form
    )
    assert _collection_fields(zipper_array_virtual.layout.form) == _collection_fields(
        coffea_array_virtual.layout.form
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
    test_no_materialization()
    test_composite_vector_collections()
    test_nested_subcollections()
    test_field_order()
    test_behaviors()
