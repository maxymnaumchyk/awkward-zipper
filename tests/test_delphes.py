import awkward
import uproot
from coffea.nanoevents import DelphesSchema, NanoEventsFactory

from awkward_zipper import Delphes

file_name = "tests/samples/delphes.root"
tree_name = "Delphes"

# --- eager ---
array = uproot.open(file_name)[tree_name].arrays(ak_add_doc=True)
zipper_array = Delphes()(array)
coffea_array = NanoEventsFactory.from_root(
    {file_name: tree_name}, schemaclass=DelphesSchema, mode="eager"
).events()

# --- virtual ---
access_log_zipper = []
array_virtual = uproot.open(file_name)[tree_name].arrays(
    virtual=True,
    ak_add_doc={"__doc__": "title", "typename": "typename"},
    access_log=access_log_zipper,
)
zipper_array_virtual = Delphes()(array_virtual)
construction_access_log = list(access_log_zipper)

coffea_array_virtual = NanoEventsFactory.from_root(
    {file_name: tree_name}, schemaclass=DelphesSchema, mode="virtual"
).events()


def test_delphes_whole_eager():
    assert awkward.array_equal(
        zipper_array, coffea_array, check_parameters=False, equal_nan=True
    )


def test_delphes_whole_virtual():
    assert awkward.array_equal(
        zipper_array_virtual,
        coffea_array_virtual,
        check_parameters=False,
        equal_nan=True,
    )


def test_no_data_materialization():
    # offsets buffers may be touched during construction (to be tightened later),
    # but no heavy *data* buffers should be materialized while building the layout
    data_accesses = [
        a for a in construction_access_log if a.buffer_key.endswith("-data")
    ]
    assert len(data_accesses) == 0


def test_lorentz_and_tref_conversions():
    # ROOT TLorentzVector leaves become LorentzVector records
    assert set(zipper_array.Jet.Area.fields) == {"x", "y", "z", "t"}
    # ROOT TRef leaves are reduced to their `ref` member (matches coffea)
    assert zipper_array.Electron.Particle.fields == ["ref"]


def test_singletons_flattened():
    # length-1 vector collections are flattened (25 * event, not 25 * var * event)
    assert str(awkward.type(zipper_array.MissingET)).endswith("missingET")
    assert awkward.array_equal(
        zipper_array.MissingET.rho,
        coffea_array.MissingET.rho,
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
    test_delphes_whole_eager()
    test_delphes_whole_virtual()
    test_no_data_materialization()
    test_lorentz_and_tref_conversions()
    test_singletons_flattened()
    test_behaviors()
