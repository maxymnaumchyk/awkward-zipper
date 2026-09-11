import pickle
import sys
import types

import awkward
import pytest
import uproot
from coffea.nanoevents import FCC as CoffeaFCC
from coffea.nanoevents import NanoEventsFactory

from awkward_zipper import (
    EDM4HEP,
    FCC,
    FCCSchema,
    FCCSchema_edm4hep1,
    podio_collection_types,
)
from awkward_zipper.behaviors import fcc as fcc_behaviors

TREE_NAME = "events"
# PARAMETERS and *Map branches are unreadable by uproot (the filter coffea's tests use)
PARAMETERS_FILTER = "/^(?!.*(PARAMETERS|_.*Map))/"
# unsplit vector<edm4hep::RecoParticleRefData> branches of the Spring2021 sample that
# uproot cannot interpret; coffea leaves them out of its base form
SPRING2021_UNREADABLE = {"Electron", "Muon", "AllMuon", "Photon"}

# sample -> (file, FCC version, uproot read options, coffea options): the same
# files and filters coffea tests with
SAMPLES = {
    "Spring2021": (
        "tests/samples/test_FCC_Spring2021.root",
        "pre-edm4hep1",
        {"filter_branch": lambda branch: branch.name not in SPRING2021_UNREADABLE},
        {},
    ),
    "Winter2023": (
        "tests/samples/test_FCC_Winter2023.root",
        "pre-edm4hep1",
        {"filter_name": PARAMETERS_FILTER},
        {"iteritems_options": {"filter_name": PARAMETERS_FILTER}},
    ),
    "edm4hep1": (
        "tests/samples/p8_ee_WW_ecm240_edm4hep.root",
        "latest",
        {"filter_name": PARAMETERS_FILTER},
        {"iteritems_options": {"filter_name": PARAMETERS_FILTER}},
    ),
}


def _build(file_name, version, uproot_kwargs, coffea_kwargs):
    """Build the zipper and coffea arrays (eager and virtual) for one sample."""
    zipper_cls = FCC.get_schema(version)
    coffea_cls = CoffeaFCC.get_schema(version)
    tree = uproot.open(file_name)[TREE_NAME]
    zipper_kwargs = {"typenames": tree.typenames()}
    if issubclass(zipper_cls, EDM4HEP):
        zipper_kwargs["podio_collection_types"] = podio_collection_types(tree)

    # --- eager ---
    zipper_eager = zipper_cls()(
        tree.arrays(ak_add_doc=True, **uproot_kwargs), **zipper_kwargs
    )
    coffea_eager = NanoEventsFactory.from_root(
        {file_name: TREE_NAME}, schemaclass=coffea_cls, mode="eager", **coffea_kwargs
    ).events()

    # --- virtual ---
    access_log = []
    array_virtual = tree.arrays(
        virtual=True,
        ak_add_doc={"__doc__": "title", "typename": "typename"},
        access_log=access_log,
        **uproot_kwargs,
    )
    zipper_virtual = zipper_cls()(array_virtual, **zipper_kwargs)
    # snapshot the access log right after construction (later comparisons materialize data)
    construction_access_log = list(access_log)
    coffea_virtual = NanoEventsFactory.from_root(
        {file_name: TREE_NAME}, schemaclass=coffea_cls, mode="virtual", **coffea_kwargs
    ).events()

    return types.SimpleNamespace(
        zipper=zipper_eager,
        coffea=coffea_eager,
        zipper_virtual=zipper_virtual,
        coffea_virtual=coffea_virtual,
        construction_access_log=construction_access_log,
    )


_cache = {}


def _built(sample):
    if sample not in _cache:
        _cache[sample] = _build(*SAMPLES[sample])
    return _cache[sample]


@pytest.fixture(params=list(SAMPLES))
def built(request):
    return _built(request.param)


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


def _assert_equal(zipper_array, coffea_array):
    assert awkward.array_equal(
        zipper_array, coffea_array, check_parameters=False, equal_nan=True
    )
    assert _record_fields(zipper_array.layout.form) == _record_fields(
        coffea_array.layout.form
    )


def test_fcc_whole_eager(built):
    assert awkward.array_equal(
        built.zipper, built.coffea, check_parameters=False, equal_nan=True
    )


def test_fcc_whole_virtual(built):
    assert awkward.array_equal(
        built.zipper_virtual,
        built.coffea_virtual,
        check_parameters=False,
        equal_nan=True,
    )


def test_no_materialization(built):
    # construction is fully lazy: no buffers (neither offsets/Index nor data)
    # are materialized while building the layout
    assert len(built.construction_access_log) == 0


def test_field_order(built):
    # array_equal matches record fields by name; the layouts must also list them
    # in the same order as coffea
    assert _record_fields(built.zipper.layout.form) == _record_fields(
        built.coffea.layout.form
    )
    assert _record_fields(built.zipper_virtual.layout.form) == _record_fields(
        built.coffea_virtual.layout.form
    )


def test_idx_and_subcollections():
    built = _built("Winter2023")
    # ObjectID '#N' branches become 'idxN' collections
    assert any(name.endswith("idx0") for name in built.zipper.fields)
    # three-vector subcollections are zipped
    assert {"x", "y", "z"}.issubset(
        set(built.zipper.ReconstructedParticles.referencePoint.fields)
    )
    # MC parent/daughter global range indexers
    _assert_equal(
        built.zipper.Particle.parents.Particleidx0_rangesG,
        built.coffea.Particle.parents.Particleidx0_rangesG,
    )


@pytest.mark.parametrize("sample", ["Spring2021", "Winter2023"])
def test_pre_edm4hep1_relations(sample):
    # the relation and link behaviors give the same arrays as coffea's
    built = _built(sample)
    daughters = built.zipper.Particle.get_daughters
    assert daughters.layout.branch_depth[1] == 3
    assert daughters.fields == built.zipper.Particle.fields
    _assert_equal(daughters, built.coffea.Particle.get_daughters)

    parents = built.zipper.Particle.get_parents
    assert parents.layout.branch_depth[1] == 3
    _assert_equal(parents, built.coffea.Particle.get_parents)

    reco_mc = built.zipper.MCRecoAssociations.reco_mc
    assert reco_mc.layout.branch_depth[1] == 3
    _assert_equal(reco_mc, built.coffea.MCRecoAssociations.reco_mc)
    # for these samples the mc and reco masses differ by less than 1 GeV
    reco, mc = reco_mc[:, :, 0], reco_mc[:, :, 1]
    assert awkward.all(awkward.flatten(mc.mass - reco.mass) < 1.0)

    _assert_equal(
        built.zipper.ReconstructedParticles.match_gen,
        built.coffea.ReconstructedParticles.match_gen,
    )


# collection -> methods of the edm4hep1 variant, as exercised by coffea's tests
EDM4HEP1_METHODS = {
    "Particle": ["get_daughters", "get_parents"],
    "ReconstructedParticles": [
        "match_gen",
        "get_cluster_photons",
        "get_reconstructedparticles",
        "get_tracks",
    ],
    "ParticleIDs": ["get_reconstructedparticles"],
    "EFlowNeutralHadron": ["get_cluster_photons", "get_hits"],
    "EFlowPhoton": ["get_cluster_photons", "get_hits"],
    "EFlowTrack": ["get_tracks"],
}


@pytest.mark.parametrize("collection", list(EDM4HEP1_METHODS))
def test_edm4hep1_methods(collection):
    built = _built("edm4hep1")
    for method in EDM4HEP1_METHODS[collection]:
        _assert_equal(
            getattr(built.zipper[collection], method),
            getattr(built.coffea[collection], method),
        )


def test_edm4hep1_copies_links_to_targets():
    # like coffea's FCCSchema_edm4hep1, the MC-reco links are copied onto the
    # ReconstructedParticles collection (and not onto Jet, per _datatype_priority)
    built = _built("edm4hep1")
    assert {"Link_from_ReconstructedParticles", "Link_to_Particle"}.issubset(
        set(built.zipper.ReconstructedParticles.fields)
    )
    assert not any(name.startswith("Link_") for name in built.zipper.Jet.fields)
    # for this sample the mc and reco masses differ by less than 1 GeV
    reco = built.zipper.ReconstructedParticles
    assert awkward.all(awkward.flatten(reco.match_gen.mass - reco.mass) < 1.0)


def test_schema_dispatch_and_inheritance():
    # mirrors coffea's FCC.get_schema / class hierarchy
    assert FCC.get_schema("pre-edm4hep1") is FCCSchema
    assert FCC.get_schema("latest") is FCCSchema_edm4hep1
    assert FCC.get_schema("edm4hep1") is FCCSchema_edm4hep1
    assert issubclass(FCCSchema_edm4hep1, EDM4HEP)
    assert FCCSchema_edm4hep1.edm4hep_version == "00-99-01"
    with pytest.raises(ValueError, match="Invalid FCC schema version"):
        FCC.get_schema("bad-version")


@pytest.mark.parametrize("sample", ["Winter2023", "edm4hep1"])
def test_behaviors(sample):
    built = _built(sample)
    diff = set(built.coffea.behavior) - set(built.zipper.behavior)
    for behavior in diff.copy():
        if any(
            string in str(behavior)
            for string in ("Systematic", "UpDownSystematic", "UpDownMultiSystematic")
        ):
            diff.remove(behavior)
    assert len(diff) == 0


@pytest.mark.parametrize(
    "behavior", [fcc_behaviors.behavior, fcc_behaviors.behavior_edm4hep1]
)
def test_behavior_classes_pickle_by_reference(behavior):
    # the edm4hep1 overloads must not shadow the pre-edm4hep1 classes: a shadowed
    # class can no longer be pickled by reference (coffea #1603)
    for key, value in behavior.items():
        if isinstance(value, type):
            module = sys.modules[value.__module__]
            assert getattr(module, value.__qualname__, None) is value, key
            assert pickle.loads(pickle.dumps(value)) is value, key
