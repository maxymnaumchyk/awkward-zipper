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


def test_no_materialization():
    # construction is fully lazy: no buffers (neither offsets/Index nor data)
    # are materialized while building the layout
    assert len(construction_access_log) == 0


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
    test_no_materialization()
    test_lorentz_and_tref_conversions()
    test_singletons_flattened()
    test_field_order()
    test_behaviors()
