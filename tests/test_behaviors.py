import awkward
import pytest

from awkward_zipper.behaviors import candidate, nanoaod, vector

# (name, behavior module, fields) for classes that carry a charge
CHARGED = [
    (
        "Candidate",
        candidate,
        {"x": [[1.0]], "y": [[2.0]], "z": [[3.0]], "t": [[10.0]], "charge": [[1]]},
    ),
    (
        "PtEtaPhiMCandidate",
        candidate,
        {
            "pt": [[10.0]],
            "eta": [[0.5]],
            "phi": [[0.1]],
            "mass": [[0.105]],
            "charge": [[1]],
        },
    ),
    (
        "PtEtaPhiECandidate",
        candidate,
        {
            "pt": [[10.0]],
            "eta": [[0.5]],
            "phi": [[0.1]],
            "energy": [[11.0]],
            "charge": [[1]],
        },
    ),
    (
        "GenVisTau",
        nanoaod,
        {
            "pt": [[10.0]],
            "eta": [[0.5]],
            "phi": [[0.1]],
            "mass": [[0.105]],
            "charge": [[1]],
        },
    ),
]


@pytest.mark.parametrize(("name", "module", "fields"), CHARGED)
def test_same_class_sum_propagates_charge(name, module, fields):
    # a module-level copy_behaviors() call running *before* @mixin_class would
    # pre-seed (add, X, X) -> LorentzVector.add and silently drop charge
    # (see scikit-hep/coffea#1578)
    array = awkward.zip(fields, with_name=name, behavior=module.behavior)
    summed = array + array
    assert "charge" in summed.fields
    assert awkward.all(summed.charge == 2)


@pytest.mark.parametrize(("name", "module", "fields"), CHARGED)
def test_same_class_sum_matches_coffea(name, module, fields):
    coffea_methods = pytest.importorskip(
        "coffea.nanoevents.methods.candidate"
        if module is candidate
        else "coffea.nanoevents.methods.nanoaod"
    )
    zipper = awkward.zip(fields, with_name=name, behavior=module.behavior)
    coffea = awkward.zip(fields, with_name=name, behavior=coffea_methods.behavior)
    coffea_summed = coffea + coffea
    if "charge" not in coffea_summed.fields:
        # coffea older than scikit-hep/coffea#1585 still drops charge here, so it
        # is not a valid reference for this class until that release lands
        pytest.skip(f"installed coffea drops charge on {name} + {name} (pre-#1585)")
    assert awkward.array_equal(
        zipper + zipper, coffea_summed, check_parameters=False, equal_nan=True
    )


PROJECTION_ATTRS = [
    "ProjectionClass2D",
    "ProjectionClass3D",
    "ProjectionClass4D",
    "MomentumClass",
]


@pytest.mark.parametrize("attr", PROJECTION_ATTRS)
@pytest.mark.parametrize("module_name", ["candidate", "vector", "nanoaod"])
def test_record_projections_match_coffea(module_name, attr):
    # projections must be registered on the Record classes too, otherwise
    # per-element access (e.g. events.Muon[0][0].to_Vector2D()) raises
    zipper_module = {"candidate": candidate, "vector": vector, "nanoaod": nanoaod}[
        module_name
    ]
    coffea_module = pytest.importorskip(f"coffea.nanoevents.methods.{module_name}")

    def registered(module):
        return {
            name
            for name, obj in vars(module).items()
            if name.endswith("Record") and isinstance(obj, type) and hasattr(obj, attr)
        }

    zipper_records = registered(zipper_module)
    assert zipper_records
    assert zipper_records == registered(coffea_module)


def test_record_projection_roundtrip():
    muon = awkward.zip(
        {
            "pt": [[10.0]],
            "eta": [[0.5]],
            "phi": [[0.1]],
            "mass": [[0.105]],
            "charge": [[1]],
        },
        with_name="Muon",
        behavior=nanoaod.behavior,
    )[0][0]
    assert muon.to_Vector2D().fields
    assert muon.to_Vector3D().fields
    assert muon.to_Vector4D().fields
