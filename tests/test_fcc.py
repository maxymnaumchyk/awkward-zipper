import awkward
import uproot
from coffea.nanoevents import FCC as CoffeaFCC
from coffea.nanoevents import NanoEventsFactory

from awkward_zipper import EDM4HEP, FCC, FCCSchema, FCCSchema_edm4hep1

# the Winter2023 sample reads cleanly once the unreadable PARAMETERS/*Map branches
# are filtered out (the same filter coffea's own FCC tests use)
branch_filter = "/^(?!.*(PARAMETERS|_.*Map))/"
file_name = "tests/samples/test_FCC_Winter2023.root"
tree_name = "events"

tree = uproot.open(file_name)[tree_name]
typenames = tree.typenames()

# --- eager ---
array = tree.arrays(filter_name=branch_filter, ak_add_doc=True)
zipper_array = FCCSchema()(array, typenames=typenames)
coffea_array = NanoEventsFactory.from_root(
    {file_name: tree_name},
    schemaclass=CoffeaFCC.get_schema(version="pre-edm4hep1"),
    mode="eager",
    iteritems_options={"filter_name": branch_filter},
).events()

# --- virtual ---
access_log_zipper = []
array_virtual = tree.arrays(
    filter_name=branch_filter,
    virtual=True,
    ak_add_doc={"__doc__": "title", "typename": "typename"},
    access_log=access_log_zipper,
)
zipper_array_virtual = FCCSchema()(array_virtual, typenames=typenames)
construction_access_log = list(access_log_zipper)

coffea_array_virtual = NanoEventsFactory.from_root(
    {file_name: tree_name},
    schemaclass=CoffeaFCC.get_schema(version="pre-edm4hep1"),
    mode="virtual",
    iteritems_options={"filter_name": branch_filter},
).events()


def test_fcc_whole_eager():
    assert awkward.array_equal(
        zipper_array, coffea_array, check_parameters=False, equal_nan=True
    )


def test_fcc_whole_virtual():
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


def test_idx_and_subcollections():
    # ObjectID '#N' branches become 'idxN' collections
    assert any(name.endswith("idx0") for name in zipper_array.fields)
    # three-vector subcollections are zipped
    assert {"x", "y", "z"}.issubset(
        set(zipper_array.ReconstructedParticles.referencePoint.fields)
    )
    # MC parent/daughter global range indexers
    assert awkward.array_equal(
        zipper_array.Particle.parents.Particleidx0_rangesG,
        coffea_array.Particle.parents.Particleidx0_rangesG,
        check_parameters=False,
    )


def test_schema_dispatch_and_inheritance():
    # mirrors coffea's FCC.get_schema / class hierarchy
    assert FCC.get_schema("pre-edm4hep1") is FCCSchema
    assert FCC.get_schema("latest") is FCCSchema_edm4hep1
    assert FCC.get_schema("edm4hep1") is FCCSchema_edm4hep1
    assert issubclass(FCCSchema_edm4hep1, EDM4HEP)


def test_edm4hep1_variant_builds():
    # the edm4hep1 variant is EDM4HEP-based and works on edm4hep>=1 samples
    edm4hep_file = "tests/samples/p8_ee_WW_ecm240_edm4hep.root"
    t = uproot.open(edm4hep_file)[tree_name]
    events = FCCSchema_edm4hep1()(
        t.arrays(filter_name=branch_filter, ak_add_doc=True),
        typenames=t.typenames(),
    )
    assert len(events) > 0
    assert "ReconstructedParticles" in events.fields


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
    test_fcc_whole_eager()
    test_fcc_whole_virtual()
    test_no_materialization()
    test_idx_and_subcollections()
    test_schema_dispatch_and_inheritance()
    test_edm4hep1_variant_builds()
    test_behaviors()
