import awkward
import uproot
from coffea.nanoevents import NanoEventsFactory, PDUNESchema

from awkward_zipper import PDUNE

file_name = "tests/samples/pduneana.root"
tree_name = "pduneana/beamana"

# --- eager ---
array = uproot.open(file_name)[tree_name].arrays(ak_add_doc=True)
zipper_array = PDUNE()(array)
coffea_array = NanoEventsFactory.from_root(
    {file_name: tree_name}, schemaclass=PDUNESchema, mode="eager"
).events()

# --- virtual ---
access_log_zipper = []
array_virtual = uproot.open(file_name)[tree_name].arrays(
    virtual=True,
    ak_add_doc={"__doc__": "title", "typename": "typename"},
    access_log=access_log_zipper,
)
zipper_array_virtual = PDUNE()(array_virtual)
construction_access_log = list(access_log_zipper)

coffea_array_virtual = NanoEventsFactory.from_root(
    {file_name: tree_name}, schemaclass=PDUNESchema, mode="virtual"
).events()


def test_pdune_whole_eager():
    assert awkward.array_equal(
        zipper_array, coffea_array, check_parameters=False, equal_nan=True
    )


def test_pdune_whole_virtual():
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


def test_nested_vector_records():
    # deeply-nested records and 3D-vector sub-records are built
    assert {"RecoBeam", "Tracks", "Showers", "TrueBeam"} == set(zipper_array.fields)
    assert "start3D" in zipper_array.RecoBeam.fields
    assert {"x", "y", "z"}.issubset(set(zipper_array.RecoBeam.start3D.fields))
    assert awkward.array_equal(
        zipper_array.Showers.start3D.x,
        coffea_array.Showers.start3D.x,
        check_parameters=False,
        equal_nan=True,
    )


def test_behaviors():
    diff = set(coffea_array.behavior) - set(zipper_array.behavior)
    for behavior in diff.copy():
        if any(
            string in str(behavior)
            for string in ("Systematic", "UpDownSystematic", "UpDownMultiSystematic")
        ):
            diff.remove(behavior)
    assert len(diff) == 0


if __name__ == "__main__":
    test_pdune_whole_eager()
    test_pdune_whole_virtual()
    test_no_materialization()
    test_nested_vector_records()
    test_behaviors()
