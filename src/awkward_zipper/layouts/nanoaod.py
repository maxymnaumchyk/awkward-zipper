import typing as tp
import warnings

import awkward

from awkward_zipper.awkward_util import (
    _check_equal_lengths,
    _non_materializing_get_field,
    _rewrap,
)
from awkward_zipper.kernels import (
    children,
    counts2nestedindex,
    counts2offsets,
    distinct_children_deep,
    distinct_parent,
    local2globalindex,
    nestedindex,
)
from awkward_zipper.layouts.base import BaseLayoutBuilder


class NanoAOD(BaseLayoutBuilder):
    """NanoAOD layout builder

    The NanoAOD layout is built from all branches found in the supplied file, based on
    the naming pattern of the branches. The following additional arrays are constructed:

    - Any branches named ``n{name}`` are assumed to be counts branches and converted to offsets ``o{name}``
    - Any local index branches with names matching ``{source}_{target}Idx*`` are converted to global indexes for the event chunk (postfix ``G``)
    - Any `nested_items` are constructed, if the necessary branches are available
    - Any `special_items` are constructed, if the necessary branches are available

    From those arrays, NanoAOD collections are formed as collections of branches grouped by name, where:

    - one branch exists named ``name`` and no branches start with ``name_``, interpreted as a single flat array;
    - one branch exists named ``name``, one named ``n{name}``, and no branches start with ``name_``, interpreted as a single jagged array;
    - no branch exists named ``{name}`` and many branches start with ``name_*``, interpreted as a flat table; or
    - one branch exists named ``n{name}`` and many branches start with ``name_*``, interpreted as a jagged table.

    Collections are assigned mixin types according to the `mixins` mapping.
    All collections are then zipped into one `base.NanoEvents` record and returned.

    There is a class-level variable ``warn_missing_crossrefs`` which will alter the behavior of
    NanoAOD. If warn_missing_crossrefs is true then when a missing global index cross-ref
    target is encountered a warning will be issued. Regardless, the cross-reference is dropped.

    The same holds for ``error_missing_events_id``. If error_missing_events_id is true, then when the 'run', 'event',
    or 'luminosityBlock' fields are missing, an exception will be thrown; if it is false, just a warning will be issued.
    """

    warn_missing_crossrefs = True  # If True, issues a warning when a missing global index cross-ref target is encountered
    error_missing_event_ids = True  # If True, raises an exception when 'run', 'event', or 'luminosityBlock' fields are missing

    event_ids: tp.ClassVar = ["run", "luminosityBlock", "event"]
    """List of NanoAOD event IDs
    """

    mixins: tp.ClassVar = {
        "CaloMET": "MissingET",
        "ChsMET": "MissingET",
        "GenMET": "MissingET",
        "MET": "MissingET",
        "METFixEE2017": "MissingET",
        "PuppiMET": "MissingET",
        "RawMET": "MissingET",
        "RawPuppiMET": "MissingET",
        "TkMET": "MissingET",
        # pseudo-lorentz: pt, eta, phi, mass=0
        "IsoTrack": "PtEtaPhiMCollection",
        "SoftActivityJet": "PtEtaPhiMCollection",
        "TrigObj": "PtEtaPhiMCollection",
        # True lorentz: pt, eta, phi, mass
        "FatJet": "FatJet",
        "GenDressedLepton": "PtEtaPhiMCollection",
        "GenIsolatedPhoton": "PtEtaPhiMCollection",
        "GenJet": "PtEtaPhiMCollection",
        "GenJetAK8": "PtEtaPhiMCollection",
        "Jet": "Jet",
        "LHEPart": "PtEtaPhiMCollection",
        "SubGenJetAK8": "PtEtaPhiMCollection",
        "SubJet": "PtEtaPhiMCollection",
        # Candidate: lorentz + charge
        "Electron": "Electron",
        "LowPtElectron": "LowPtElectron",
        "Muon": "Muon",
        "Photon": "Photon",
        "FsrPhoton": "FsrPhoton",
        "Tau": "Tau",
        "GenVisTau": "GenVisTau",
        # special
        "GenPart": "GenParticle",
        "PV": "Vertex",
        "SV": "SecondaryVertex",
    }
    """Default configuration for mixin types, based on the collection name.

    The types are implemented in the `coffea.nanoevents.methods.nanoaod` module.
    """
    all_cross_references: tp.ClassVar = {
        "Electron_genPartIdx": "GenPart",
        "Electron_jetIdx": "Jet",
        "Electron_photonIdx": "Photon",
        "LowPtElectron_electronIdx": "Electron",
        "LowPtElectron_genPartIdx": "GenPart",
        "LowPtElectron_photonIdx": "Photon",
        "FatJet_genJetAK8Idx": "GenJetAK8",
        "FatJet_subJetIdx1": "SubJet",
        "FatJet_subJetIdx2": "SubJet",
        "FsrPhoton_muonIdx": "Muon",
        "GenPart_genPartIdxMother": "GenPart",
        "GenVisTau_genPartIdxMother": "GenPart",
        "Jet_electronIdx1": "Electron",
        "Jet_electronIdx2": "Electron",
        "Jet_genJetIdx": "GenJet",
        "Jet_muonIdx1": "Muon",
        "Jet_muonIdx2": "Muon",
        "Muon_fsrPhotonIdx": "FsrPhoton",
        "Muon_genPartIdx": "GenPart",
        "Muon_jetIdx": "Jet",
        "Photon_electronIdx": "Electron",
        "Photon_genPartIdx": "GenPart",
        "Photon_jetIdx": "Jet",
        "Tau_genPartIdx": "GenPart",
        "Tau_jetIdx": "Jet",
    }
    """Cross-references, where an index is to be interpreted with respect to another collection

    Each such cross-reference will be converted to a global indexer, so that arbitrarily sliced events
    can still resolve the indirection back the parent events
    """
    nested_items: tp.ClassVar = {
        "FatJet_subJetIdxG": ["FatJet_subJetIdx1G", "FatJet_subJetIdx2G"],
        "Jet_muonIdxG": ["Jet_muonIdx1G", "Jet_muonIdx2G"],
        "Jet_electronIdxG": ["Jet_electronIdx1G", "Jet_electronIdx2G"],
    }
    """Nested collections, where nesting is accomplished by a fixed-length set of indexers"""
    nested_index_items: tp.ClassVar = {
        "Jet_pFCandsIdxG": ("Jet_nConstituents", "JetPFCands"),
        "FatJet_pFCandsIdxG": ("FatJet_nConstituents", "FatJetPFCands"),
        "GenJet_pFCandsIdxG": ("GenJet_nConstituents", "GenJetCands"),
        "GenFatJet_pFCandsIdxG": ("GenJetAK8_nConstituents", "GenFatJetCands"),
    }
    """Nested collections, where nesting is accomplished by assuming the target can be unflattened according to a source counts"""
    special_items: tp.ClassVar = {
        "GenPart_distinctParentIdxG": (
            distinct_parent,
            ("GenPart_genPartIdxMotherG", "GenPart_pdgId"),
        ),
        "GenPart_childrenIdxG": (
            children,
            (
                "nGenPart",
                "GenPart_genPartIdxMotherG",
            ),
        ),
        "GenPart_distinctChildrenIdxG": (
            children,
            (
                "nGenPart",
                "GenPart_distinctParentIdxG",
            ),
        ),
        "GenPart_distinctChildrenDeepIdxG": (
            distinct_children_deep,
            (
                "nGenPart",
                "GenPart_genPartIdxMotherG",
                "GenPart_pdgId",
            ),
        ),
    }
    """Special arrays, where the callable and input arrays are specified in the value"""

    def __init__(self, version="latest"):
        self._version = version
        self.cross_references = dict(self.all_cross_references)
        if version == "latest":
            pass
        else:
            if int(version) < 7:
                del self.cross_references["FatJet_genJetAK8Idx"]
            if int(version) < 6:
                del self.cross_references["FsrPhoton_muonIdx"]
                del self.cross_references["Muon_fsrPhotonIdx"]

    @classmethod
    def v7(cls):
        """Build the NanoEvents assuming NanoAODv7

        For example, one can use ``NanoEventsFactory.from_root("file.root", schemaclass=NanoAOD.v7)``
        to ensure NanoAODv7 compatibility.

        Returns
        -------
            out: NanoAOD
                Schema assuming NanoAODv7
        """
        return cls(version="7")

    @classmethod
    def v6(cls):
        """Build the NanoEvents assuming NanoAODv6

        Returns
        -------
            out: NanoAOD
                Schema assuming NanoAODv6
        """
        return cls(version="6")

    @classmethod
    def v5(cls):
        """Build the NanoEvents assuming NanoAODv5

        Returns
        -------
            out: NanoAOD
                Schema assuming NanoAODv5
        """
        return cls(version="5")

    def __call__(self, array: awkward.Array) -> awkward.Array:
        fields = set(array.fields)

        def _get_collection_fields(name, collection):
            return set(filter(lambda f: f.startswith(name), collection))

        # branches that start with "n"
        counter_fields = _get_collection_fields("n", fields)

        # parse into high-level records (collections, list collections, and singletons)
        collections = {k.split("_", maxsplit=1)[0] for k in fields - counter_fields}

        # check if data or simulation
        is_data = "GenPart" not in collections

        new_fields = {}
        # # Create offsets virtual arrays
        # for name in counter_fields:
        #     arr = _non_materializing_get_field(array, name)
        #     new_fields[name.replace("n", "o", 1)] = counts2offsets(arr)

        # Check the presence of the event_ids
        missing_event_ids = [
            event_id for event_id in self.event_ids if event_id not in fields
        ]
        if len(missing_event_ids) > 0:
            if self.error_missing_event_ids:
                msg = f"There are missing event ID fields: {missing_event_ids} \n\n\
                    The event ID fields {self.event_ids} are necessary to perform sub-run identification \
                    (e.g. for corrections and sub-dividing data during different detector conditions),\
                    to cross-validate MC and Data (i.e. matching events for comparison), and to generate event displays. \
                    It's advised to never drop these branches from the dataformat.\n\n\
                    This error can be demoted to a warning by setting the class level variable error_missing_event_ids to False."
                raise RuntimeError(msg)
            warnings.warn(
                f"Missing event_ids : {missing_event_ids}",
                RuntimeWarning,
                stacklevel=2,
            )

        # Create global index virtual arrays for indirection
        for indexer, target in self.all_cross_references.items():
            if target.startswith("Gen") and is_data:
                continue
            if indexer not in fields:
                if self.warn_missing_crossrefs:
                    warnings.warn(
                        f"Missing cross-reference index for {indexer} => {target}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                continue
            if "n" + target not in fields:
                if self.warn_missing_crossrefs:
                    warnings.warn(
                        f"Missing cross-reference target for {indexer} => {target}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                continue
            # convert nWhatever to a global index
            # this used to be transforms.counts2offsets_form + transforms.local2global_form in coffea
            arr_indexer = _non_materializing_get_field(array, indexer)
            arr_target = _non_materializing_get_field(array, "n" + target)
            new_fields[indexer + "G"] = local2globalindex(arr_indexer, arr_target)

        # Create nested indexer from Idx1, Idx2, ... arrays
        for name, indexers in self.nested_items.items():
            if all(idx in new_fields for idx in indexers):
                new_fields[name] = nestedindex(
                    [_non_materializing_get_field(new_fields, idx) for idx in indexers]
                )

        # Create nested indexer from n* counts arrays
        for name, (local_counts, target) in self.nested_index_items.items():
            if local_counts in fields and "n" + target in fields:
                arr_local_counts = _non_materializing_get_field(array, local_counts)
                arr_target = _non_materializing_get_field(array, "n" + target)
                # this used to be transforms.counts2nestedindex_form + transforms.local2global_form in coffea
                new_fields[name] = counts2nestedindex(arr_local_counts, arr_target)

        # TODO: make those kernels work with virtual arrays
        # Create any special arrays
        for name, (fcn, args) in self.special_items.items():
            breakpoint()
            if all(k in fields for k in args):
                new_fields[name] = fcn(*(_non_materializing_get_field(array, k) for k in args))

        output = {}
        for name in collections:
            name_with_underscore = name + "_"
            mixin = self.mixins.get(name, "NanoCollection")
            if "n" + name in fields and name not in fields:
                content = {}
                # buffers in `array`
                for field in _get_collection_fields(name_with_underscore, fields):
                    arr = _non_materializing_get_field(array, field)

                    *_, buffers = awkward.to_buffers(arr)
                    assert {"node0-offsets", "node1-data"} == set(buffers)
                    # take flat data
                    content[field.removeprefix(name_with_underscore)] = (
                        awkward.contents.NumpyArray(
                            buffers["node1-data"],
                            parameters=arr.layout.parameters,
                        )
                    )

                # new buffers in `new_fields`
                for field in _get_collection_fields(name_with_underscore, new_fields):
                    arr = _non_materializing_get_field(new_fields, field)
                    parameters = arr.layout.parameters
                    *_, buffers = awkward.to_buffers(arr)
                    if field in self.nested_items | self.nested_index_items:
                        # doubly-jagged case
                        assert {"node0-offsets", "node1-offsets", "node2-data"} == set(
                            buffers
                        )
                        # take singly jagged array
                        content[field.removeprefix(name_with_underscore)] = (
                            awkward.contents.ListOffsetArray(
                                offsets=awkward.index.Index(buffers["node1-offsets"]),
                                content=awkward.contents.NumpyArray(
                                    buffers["node2-data"]
                                ),
                                parameters=parameters,
                            )
                        )
                    else:
                        assert {"node0-offsets", "node1-data"} == set(buffers)
                        # take flat data
                        content[field.removeprefix(name_with_underscore)] = (
                            awkward.contents.NumpyArray(
                                buffers["node1-data"], parameters=parameters
                            )
                        )

                _content = (*content.values(),)
                _fields = (*content.keys(),)
                _length = _check_equal_lengths(_content)

                # combine contents in a RecordArray
                record = awkward.contents.RecordArray(
                    _content, _fields, length=_length, parameters={}
                )
                # update parameters
                counts = _non_materializing_get_field(array, "n" + name)
                record.parameters.update(
                    {
                        "collection_name": name,
                        "__record__": mixin,
                        "__doc__": counts.layout.parameters.get("__doc__"),
                    }
                )

                # wrap as jagged array
                offsets = counts2offsets(counts)
                offsets = awkward.index.Index(offsets)
                output[name] = awkward.contents.ListOffsetArray(
                    offsets=offsets, content=record
                )
            elif ("n" + name) in fields or name in fields:
                # list singleton (can use branch's own offsets) or singleton
                arr = _non_materializing_get_field(array, name)
                output[name] = awkward.to_layout(arr)
            else:
                # simple collection
                content = {}
                for field in _get_collection_fields(name_with_underscore, fields):
                    arr = _non_materializing_get_field(array, field)

                    *_, buffers = awkward.to_buffers(arr)
                    assert {"node0-data"} == set(buffers)
                    # take flat data
                    content[field.removeprefix(name_with_underscore)] = (
                        awkward.contents.NumpyArray(
                            buffers["node0-data"],
                            # forward parameters
                            parameters=arr.layout.parameters,
                        )
                    )

                _content = (*content.values(),)
                _fields = (*content.keys(),)
                _length = _check_equal_lengths(_content)

                output[name] = awkward.contents.RecordArray(
                    _content, _fields, length=_length, parameters={}
                )
                # update parameters
                output[name].parameters.update(
                    {
                        "collection_name": name,
                        "__record__": mixin,
                    }
                )

        # final nanoevents (most outer) zip
        _content = (*output.values(),)
        _fields = (*output.keys(),)
        _length = awkward.num(array, axis=0)

        nanoevents = awkward.Array(
            awkward.contents.RecordArray(_content, _fields, length=_length),
            behavior=self.behavior(),
        )

        # fix virtual array shape generators by re-running from buffers:
        nanoevents = awkward.with_name(
            _rewrap(nanoevents),
            name="NanoEvents",
        )

        # add ref to itself in attrs
        nanoevents.attrs["@original_array"] = nanoevents

        return nanoevents

    @classmethod
    def behavior(cls):
        """Behaviors necessary to implement this schema (dict)"""
        from awkward_zipper.behaviors import nanoaod

        return nanoaod.behavior


class PFNanoAOD(NanoAOD):
    """PFNano schema builder

    PFNano is an extended NanoAOD format that includes PF candidates and secondary vertices
    More info at https://github.com/cms-jet/PFNano
    """

    mixins: tp.ClassVar = {
        **NanoAOD.mixins,
        "JetSVs": "AssociatedSV",
        "FatJetSVs": "AssociatedSV",
        "GenJetSVs": "AssociatedSV",
        "GenFatJetSVs": "AssociatedSV",
        "JetPFCands": "AssociatedPFCand",
        "FatJetPFCands": "AssociatedPFCand",
        "GenJetCands": "AssociatedPFCand",
        "GenFatJetCands": "AssociatedPFCand",
        "PFCands": "PFCand",
        "GenCands": "PFCand",
    }
    all_cross_references: tp.ClassVar = {
        **NanoAOD.all_cross_references,
        "FatJetPFCands_jetIdx": "FatJet",  # breaks pattern
        "FatJetPFCands_pFCandsIdx": "PFCands",
        "FatJetSVs_jetIdx": "FatJet",  # breaks pattern
        "FatJetSVs_sVIdx": "SV",
        "FatJet_electronIdx3SJ": "Electron",
        "FatJet_muonIdx3SJ": "Muon",
        "GenFatJetCands_jetIdx": "GenJetAK8",  # breaks pattern
        "GenFatJetCands_pFCandsIdx": "GenCands",  # breaks pattern
        "GenFatJetSVs_jetIdx": "GenJetAK8",  # breaks pattern
        "GenFatJetSVs_sVIdx": "SV",
        "GenJetCands_jetIdx": "GenJet",  # breaks pattern
        "GenJetCands_pFCandsIdx": "GenCands",  # breaks pattern
        "GenJetSVs_jetIdx": "GenJet",  # breaks pattern
        "GenJetSVs_sVIdx": "SV",
        "JetPFCands_jetIdx": "Jet",
        "JetPFCands_pFCandsIdx": "PFCands",
        "JetSVs_jetIdx": "Jet",
        "JetSVs_sVIdx": "SV",
        "SubJet_subGenJetAK8Idx": "SubGenJetAK8",
    }


class ScoutingNanoAOD(NanoAOD):
    """ScoutingNano schema builder

    ScoutingNano is a NanoAOD format that includes Scouting objects
    """

    mixins: tp.ClassVar = {
        **NanoAOD.mixins,
        "ScoutingJet": "Jet",
        "ScoutingFatJet": "FatJet",
        "ScoutingMET": "MissingET",
        "ScoutingMuonNoVtxDisplacedVertex": "Vertex",
        "ScoutingMuonVtxDisplacedVertex": "Vertex",
        "ScoutingPrimaryVertex": "Vertex",
        "ScoutingElectron": "Electron",
        "ScoutingPhoton": "Photon",
        "ScoutingMuonNoVtx": "Muon",
        "ScoutingMuonVtx": "Muon",
    }

    all_cross_references: tp.ClassVar = {**NanoAOD.all_cross_references}
