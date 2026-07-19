import awkward
import uproot
from coffea.nanoevents import NanoEventsFactory, PHYSLITESchema

from awkward_zipper import PHYSLITE

file_name = "tests/samples/PHYSLITE_example.root"
tree_name = "CollectionTree"
branch_filter = ["*AuxDyn*"]

# --- eager ---
array = uproot.open(file_name)[tree_name].arrays(
    filter_name=branch_filter, ak_add_doc=True
)
zipper_array = PHYSLITE()(array)
coffea_array = NanoEventsFactory.from_root(
    {file_name: tree_name},
    schemaclass=PHYSLITESchema,
    mode="eager",
    iteritems_options={"filter_name": branch_filter},
).events()

# --- virtual ---
access_log_zipper = []
array_virtual = uproot.open(file_name)[tree_name].arrays(
    filter_name=branch_filter,
    virtual=True,
    ak_add_doc={"__doc__": "title", "typename": "typename"},
    access_log=access_log_zipper,
)
zipper_array_virtual = PHYSLITE()(array_virtual)
construction_access_log = list(access_log_zipper)

coffea_array_virtual = NanoEventsFactory.from_root(
    {file_name: tree_name},
    schemaclass=PHYSLITESchema,
    mode="virtual",
    iteritems_options={"filter_name": branch_filter},
).events()


def test_physlite_whole_eager():
    assert awkward.array_equal(
        zipper_array, coffea_array, check_parameters=False, equal_nan=True
    )


def test_physlite_whole_virtual():
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


def test_eventindex_and_derived_fields():
    # synthetic _eventindex column
    assert "_eventindex" in zipper_array.Electrons.fields
    assert awkward.array_equal(
        zipper_array.Electrons._eventindex,
        coffea_array.Electrons._eventindex,
        check_parameters=False,
    )
    # TrackParticle derived fields from qOverP/theta
    tp = zipper_array.InDetTrackParticles
    for fld in ("p", "pt", "tau"):
        assert fld in tp.fields
        assert awkward.array_equal(
            tp[fld],
            coffea_array.InDetTrackParticles[fld],
            check_parameters=False,
            equal_nan=True,
        )
    # Muon mass added
    assert awkward.array_equal(
        zipper_array.Muons.m,
        coffea_array.Muons.m,
        check_parameters=False,
        equal_nan=True,
    )


def test_elementlink_reconstitution():
    # split ElementLink members are reconstituted into a sub-record
    assert set(zipper_array.Electrons.ambiguityLink.fields) == {
        "m_persKey",
        "m_persIndex",
    }


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
    test_physlite_whole_eager()
    test_physlite_whole_virtual()
    test_no_materialization()
    test_eventindex_and_derived_fields()
    test_elementlink_reconstitution()
    test_behaviors()
