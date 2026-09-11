import copy
import pickle
import types
import typing as tp

import awkward
import numpy as np
import pytest
import uproot
from coffea.nanoevents import EDM4HEPSchema, NanoEventsFactory
from coffea.nanoevents.assets import edm4hep_ver as coffea_edm4hep_ver
from coffea.nanoevents.assets import versions as coffea_versions
from coffea.nanoevents.schemas import edm4hep as coffea_edm4hep

from awkward_zipper import EDM4HEP, edm4hep_version, podio_collection_types
from awkward_zipper.assets import edm4hep_ver, versions
from awkward_zipper.layouts import edm4hep as zipper_edm4hep

TREE_NAME = "events"
# PARAMETERS and *Map branches are unreadable by uproot (the filter coffea's tests use)
PARAMETERS_FILTER = "/^(?!.*(PARAMETERS|_.*Map))/"

# sample -> (file, edm4hep.yaml version, branch filter): the files coffea tests with
SAMPLES = {
    "p8_ee_WW": ("tests/samples/p8_ee_WW_ecm240_edm4hep.root", "latest", None),
    "key4hep_00-99-01": ("tests/samples/edm4hep.root", "00.99.01", PARAMETERS_FILTER),
    # EDM4hep's own example files (backwards-compat inputs and a CI artifact)
    "example_00-99-02": (
        "tests/samples/edm4hep_example_v00-99-02_podio_v01-03.root",
        "00.99.02",
        PARAMETERS_FILTER,
    ),
    "example_00-99-03": (
        "tests/samples/edm4hep_example_v00-99-03_podio_v01-06.root",
        "00.99.03",
        PARAMETERS_FILTER,
    ),
    "example_00-99-04": (
        "tests/samples/edm4hep_example_v00-99-04_podio_v01-06.root",
        "00.99.04",
        PARAMETERS_FILTER,
    ),
    "example_01-01": (
        "tests/samples/edm4hep_example_v01-01_podio_v01-07.root",
        "01.01",
        PARAMETERS_FILTER,
    ),
    # ILD reconstruction, EDM4hep 1.0.0 written by podio 1.7: links are generic
    # podio::LinkCollection<From,To> and several collection names contain underscores
    "ILD_latest": (
        "tests/samples/edm4hep_ILD_mumuH_v01-00_podio_v01-07_10ev.root",
        "latest",
        None,
    ),
    "ILD_01-00": (
        "tests/samples/edm4hep_ILD_mumuH_v01-00_podio_v01-07_10ev.root",
        "01.00",
        None,
    ),
}


def _build(file_name, ver, filter_name):
    """Build the zipper and coffea arrays (eager and virtual) for one sample."""
    zipper_cls = edm4hep_version(ver)
    coffea_cls = EDM4HEPSchema.version(ver)
    tree = uproot.open(file_name)[TREE_NAME]
    zipper_kwargs = {
        "typenames": tree.typenames(),
        "podio_collection_types": podio_collection_types(tree),
    }
    uproot_kwargs = {} if filter_name is None else {"filter_name": filter_name}
    coffea_kwargs = (
        {}
        if filter_name is None
        else {"iteritems_options": {"filter_name": filter_name}}
    )

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


def _assert_global_index(index, index_global, target):
    """index_Global is the local index shifted by the per-event offsets of the target."""
    offsets = np.concatenate([[0], np.cumsum(awkward.num(target))[:-1]])
    valid = index >= 0
    assert awkward.sum(valid) > 0
    assert awkward.all((index_global == index + offsets)[valid])


def test_edm4hep_whole_eager(built):
    assert awkward.array_equal(
        built.zipper, built.coffea, check_parameters=False, equal_nan=True
    )


def test_edm4hep_whole_virtual(built):
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


def test_components_and_relations():
    built = _built("p8_ee_WW")
    # component members are zipped into sub-records
    assert {"x", "y", "z"}.issubset(set(built.zipper.CalorimeterHits.position.fields))
    # OneToMany relations get a global index into the target collection
    assert "daughters_idx_Particle_index_Global" in built.zipper.Particle.fields
    assert awkward.array_equal(
        built.zipper.Particle.daughters_idx_Particle_index_Global,
        built.coffea.Particle.daughters_idx_Particle_index_Global,
        check_parameters=False,
    )
    # VectorMembers (begin/end ranges) become doubly-jagged
    assert awkward.array_equal(
        built.zipper.EFlowTrack.trackStates.covMatrix,
        built.coffea.EFlowTrack.trackStates.covMatrix,
        check_parameters=False,
        equal_nan=True,
    )


def test_interface_links():
    # a Link whose target is an interface (edm4hep::TrackerHit) resolves to the
    # collections of the interfaced datatypes
    events = _built("key4hep_00-99-01").zipper
    assert events.TrackerHitSimTrackerHitLinkCollection.fields == [
        "Link_from_TrackerHit3DCollection",
        "Link_from_TrackerHitPlaneCollection",
        "Link_to_SimTrackerHitCollection",
        "weight",
    ]


@pytest.mark.parametrize("sample", ["example_00-99-02", "example_01-01"])
def test_upstream_example_file(sample):
    events = _built(sample).zipper
    link = events.RecoMCParticleLinkCollection
    assert link.fields == [
        "Link_from_ReconstructedParticleCollection",
        "Link_to_MCParticleCollection",
        "weight",
    ]
    to = link.Link_to_MCParticleCollection
    assert to.fields == ["index", "collectionID", "index_Global"]
    assert awkward.all(to.index < awkward.num(events.MCParticleCollection))


@pytest.mark.parametrize("sample", ["ILD_latest", "ILD_01-00"])
def test_generic_links(sample):
    events = _built(sample).zipper
    link = events.RecoMCTruthLink
    assert {"weight", "Link_from_PandoraPFOs", "Link_to_MCParticles"} <= set(
        link.fields
    )
    to = link.Link_to_MCParticles
    _assert_global_index(to.index, to.index_Global, events.MCParticles)

    # link whose target collection is underscore-named
    src = events.SiTracksMCTruthLink.Link_from_SiTracks_Refitted
    _assert_global_index(src.index, src.index_Global, events.SiTracks_Refitted)

    # underscore-named link collection (its "to" side is unset in this sample)
    vertex_link = events.BuildUpVertices_associatedParticles
    assert {"Link_from_BuildUpVertices", "Link_to_PandoraPFOs"} <= set(
        vertex_link.fields
    )
    src = vertex_link.Link_from_BuildUpVertices
    _assert_global_index(src.index, src.index_Global, events.BuildUpVertices)


def test_underscore_named_collections():
    file_name = SAMPLES["ILD_latest"][0]
    events = _built("ILD_latest").zipper

    # vector member of an underscore-named collection keeps every leaf
    track_states = events.SiTracks_Refitted.trackStates
    assert track_states.fields == events.SiTracks.trackStates.fields
    raw = uproot.open(file_name)[TREE_NAME][
        "_SiTracks_Refitted_trackStates/_SiTracks_Refitted_trackStates.D0"
    ].array()
    assert awkward.all(
        awkward.flatten(track_states.D0, axis=None) == awkward.flatten(raw, axis=None)
    )

    # one-to-one relation whose branch name is prefixed by two other collection names
    dqdx = events.SiTracks_Refitted_dQdx
    _assert_global_index(
        dqdx.track_idx_SiTracks_Refitted_index,
        dqdx.track_idx_SiTracks_Refitted_index_Global,
        events.SiTracks_Refitted,
    )


def test_relation_branches_exact_match():
    forms = {
        "_X_Y_m/_X_Y_m.index": "xy_m",
        "_X_Y_members/_X_Y_members.index": "xy_members",
        "_X_Y_m": "xy_m_flat",
        "_X_m/_X_m.index": "x_m",
    }
    assert zipper_edm4hep._relation_branches(forms, "X_Y", "m") == {"m.index": "xy_m"}
    assert zipper_edm4hep._relation_branches(forms, "X", "m") == {"m.index": "x_m"}
    assert set(forms) == {"_X_Y_members/_X_Y_members.index", "_X_Y_m"}


def test_unresolved_links_error_and_override():
    file_name = SAMPLES["ILD_latest"][0]
    tree = uproot.open(file_name)[TREE_NAME]
    array = tree.arrays(ak_add_doc=True)
    typenames = tree.typenames()

    # generic podio links cannot be typed without the podio_metadata information
    with pytest.raises(
        RuntimeError, match=r"BuildUpVertices_associatedParticles.*extra_mixins"
    ):
        EDM4HEP()(array, typenames=typenames)

    link_types = {
        (link["From"], link["To"]): name.split("::")[-1]
        for name, link in zipper_edm4hep.load_edm4hep(EDM4HEP.edm4hep_version)[0][
            "links"
        ].items()
    }
    overrides = {}
    for name, datatype in podio_collection_types(tree).items():
        endpoints = zipper_edm4hep._link_collection.match(datatype)
        if endpoints and name in tree:
            overrides[name] = link_types[endpoints.groups()]

    class TypedLinks(EDM4HEP):
        extra_mixins: tp.ClassVar = {**EDM4HEP.extra_mixins, **overrides}

    events = TypedLinks()(array, typenames=typenames)
    assert "Link_to_MCParticles" in events.RecoMCTruthLink.fields


def test_builds_without_typenames():
    # typenames are optional; without them the datatypes are inferred from the yaml
    # model on a best-effort basis (pass typenames for exact coffea parity)
    file_name = SAMPLES["p8_ee_WW"][0]
    array = uproot.open(file_name)[TREE_NAME].arrays(ak_add_doc=True)
    inferred = EDM4HEP()(array)
    reference = _built("p8_ee_WW").zipper
    assert len(inferred) == len(reference)
    assert set(reference.fields).issubset(set(inferred.fields))


def test_version_selection():
    assert edm4hep_version("latest") is EDM4HEP
    assert EDM4HEP.version("latest") is EDM4HEP
    assert EDM4HEP.edm4hep_version == versions[-1]
    assert EDM4HEP.edm4hep_version == EDM4HEPSchema.edm4hep_version
    assert edm4hep_version("00.99.01") is edm4hep_version("00-99-01")
    assert edm4hep_version("00.99.01") is EDM4HEP.version("00.99.01")
    assert issubclass(edm4hep_version("00.99.00"), EDM4HEP)
    assert edm4hep_version("00.99.00").edm4hep_version == "00-99-00"
    with pytest.raises(ValueError, match="not found"):
        edm4hep_version("99.99.99")


def test_bundled_versions_match_coffea():
    assert versions == coffea_versions
    for ver in versions:
        assert edm4hep_ver[ver]() == coffea_edm4hep_ver[ver]()


@pytest.mark.parametrize("ver", versions)
def test_bundled_version_parses(ver):
    builder = edm4hep_version(ver)
    assert builder.edm4hep_version == ver
    assert pickle.loads(pickle.dumps(builder)) is builder

    loaded = edm4hep_ver[ver]()
    parsed = zipper_edm4hep.parse_yaml(loaded, copy.deepcopy(loaded))
    assert "edm4hep::MCParticle" in parsed["datatypes"]
    if ver >= "00-99":
        link = parsed["datatypes"]["edm4hep::RecoMCParticleLink"]
        assert "weight" in link["Members"]
        assert link["OneToOneRelations"]["from"]["target"] == "ReconstructedParticle"
        assert link["OneToOneRelations"]["to"]["target"] == "MCParticle"

    # the parsed data model is coffea's (up to the wording of the ObjectID stub)
    coffea_parsed = coffea_edm4hep.parse_yaml(loaded, copy.deepcopy(loaded))
    parsed["datatypes"].pop("edm4hep::ObjectID")
    coffea_parsed["datatypes"].pop("edm4hep::ObjectID")
    assert parsed == coffea_parsed


def test_yaml_cache_is_readonly():
    # the parsed yaml is loaded once and shared across all builds, so a build must
    # treat it as read-only
    raw, parsed = zipper_edm4hep.load_edm4hep(EDM4HEP.edm4hep_version)
    raw_snapshot = copy.deepcopy(raw)
    parsed_snapshot = copy.deepcopy(parsed)

    # a full build exercises every path that reads the cached dicts
    file_name = SAMPLES["ILD_latest"][0]
    tree = uproot.open(file_name)[TREE_NAME]
    EDM4HEP()(
        tree.arrays(ak_add_doc=True),
        typenames=tree.typenames(),
        podio_collection_types=podio_collection_types(tree),
    )

    raw_after, parsed_after = zipper_edm4hep.load_edm4hep(EDM4HEP.edm4hep_version)
    assert raw_after is raw
    assert parsed_after is parsed
    assert raw_after == raw_snapshot
    assert parsed_after == parsed_snapshot


def test_behaviors():
    built = _built("p8_ee_WW")
    diff = set(built.coffea.behavior) - set(built.zipper.behavior)
    for behavior in diff.copy():
        if any(
            string in str(behavior)
            for string in ("Systematic", "UpDownSystematic", "UpDownMultiSystematic")
        ):
            diff.remove(behavior)
    assert len(diff) == 0
