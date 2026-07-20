import awkward
import uproot
from coffea.nanoevents import EDM4HEPSchema, NanoEventsFactory

from awkward_zipper import EDM4HEP, EDM4HEP_v00_99_00, edm4hep_version

file_name = "tests/samples/p8_ee_WW_ecm240_edm4hep.root"
tree_name = "events"

tree = uproot.open(file_name)[tree_name]
typenames = tree.typenames()

# --- eager ---
array = tree.arrays(ak_add_doc=True)
zipper_array = EDM4HEP()(array, typenames=typenames)
coffea_array = NanoEventsFactory.from_root(
    {file_name: tree_name}, schemaclass=EDM4HEPSchema, mode="eager"
).events()

# --- virtual ---
access_log_zipper = []
array_virtual = tree.arrays(
    virtual=True,
    ak_add_doc={"__doc__": "title", "typename": "typename"},
    access_log=access_log_zipper,
)
zipper_array_virtual = EDM4HEP()(array_virtual, typenames=typenames)
construction_access_log = list(access_log_zipper)

coffea_array_virtual = NanoEventsFactory.from_root(
    {file_name: tree_name}, schemaclass=EDM4HEPSchema, mode="virtual"
).events()


def test_edm4hep_whole_eager():
    assert awkward.array_equal(
        zipper_array, coffea_array, check_parameters=False, equal_nan=True
    )


def test_edm4hep_whole_virtual():
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


def test_components_and_relations():
    # component members are zipped into sub-records
    assert {"x", "y", "z"}.issubset(set(zipper_array.CalorimeterHits.position.fields))
    # OneToMany relations get a global index into the target collection
    assert "daughters_idx_Particle_index_Global" in zipper_array.Particle.fields
    assert awkward.array_equal(
        zipper_array.Particle.daughters_idx_Particle_index_Global,
        coffea_array.Particle.daughters_idx_Particle_index_Global,
        check_parameters=False,
    )
    # VectorMembers (begin/end ranges) become doubly-jagged
    assert awkward.array_equal(
        zipper_array.EFlowTrack.trackStates.covMatrix,
        coffea_array.EFlowTrack.trackStates.covMatrix,
        check_parameters=False,
        equal_nan=True,
    )


def test_builds_without_typenames():
    # typenames are optional; without them the datatypes are inferred from the yaml
    # model on a best-effort basis (pass typenames for exact coffea parity)
    inferred = EDM4HEP()(array)
    assert len(inferred) == len(zipper_array)
    assert set(zipper_array.fields).issubset(set(inferred.fields))


def test_version_dispatch_and_inheritance():
    assert edm4hep_version("latest") is EDM4HEP
    assert edm4hep_version("00.99.00") is EDM4HEP_v00_99_00
    assert issubclass(EDM4HEP_v00_99_00, EDM4HEP)
    assert EDM4HEP_v00_99_00.edm4hep_version == "00-99-00"


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
    test_edm4hep_whole_eager()
    test_edm4hep_whole_virtual()
    test_no_materialization()
    test_components_and_relations()
    test_builds_without_typenames()
    test_version_dispatch_and_inheritance()
    test_behaviors()
