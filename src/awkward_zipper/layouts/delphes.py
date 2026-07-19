import typing as tp

import awkward
import numpy as np

from awkward_zipper.awkward_util import (
    _jagged_content,
    _non_materializing_get_field,
    _rewrap,
)
from awkward_zipper.kernels import counts2offsets, full_like_from_counts, met_to_rho
from awkward_zipper.layouts.base import BaseLayoutBuilder


def _total_items(offsets):
    """Number of flat items implied by an offsets array (``offsets[-1]``).

    Returns an unknown length when the offsets are virtual, so that reading it
    does not materialize the counts branch.
    """
    if isinstance(offsets, np.ndarray):
        return int(offsets[-1])
    return awkward._nplikes.shape.unknown_length


_LIST_LIKE = (
    awkward.contents.ListOffsetArray,
    awkward.contents.ListArray,
    awkward.contents.RegularArray,
)


def _lorentz_from_tlv(content):
    """Convert a ROOT ``TLorentzVector`` record content to a ``LorentzVector``.

    Matches a ``RecordArray`` with ``fE`` and ``fP`` (a ``TVector3`` with
    ``fX``/``fY``/``fZ``) and returns ``{x, y, z, t}``. Recurses through
    list-like wrappers. Returns ``None`` when nothing is converted, so callers
    can leave unrelated leaves untouched.
    """
    if isinstance(content, awkward.contents.RecordArray) and {"fE", "fP"} <= set(
        content.fields
    ):
        fP = content.contents[content.fields.index("fP")]
        fE = content.contents[content.fields.index("fE")]
        fX = fP.contents[fP.fields.index("fX")]
        fY = fP.contents[fP.fields.index("fY")]
        fZ = fP.contents[fP.fields.index("fZ")]
        return awkward.contents.RecordArray(
            [fX, fY, fZ, fE],
            ["x", "y", "z", "t"],
            length=content.length,
            parameters={"__record__": "LorentzVector"},
        )
    if isinstance(content, _LIST_LIKE):
        inner = _lorentz_from_tlv(content.content)
        if inner is not None:
            return content.copy(content=inner)
    return None


def _strip_tref(content):
    """Drop the ``@other*`` members of a ROOT ``TRef`` record, keeping only ``ref``.

    uproot's eager reader already reduces a ``TRef`` to ``{ref}`` (as does
    coffea), but the ``virtual=True`` reader keeps the full
    ``{ref, @other1, @other2}`` whose ``@other`` buffers cannot be materialized.
    Returns ``None`` when there is nothing to strip.
    """
    if isinstance(content, awkward.contents.RecordArray):
        keep = [f for f in content.fields if not f.startswith("@")]
        if len(keep) < len(content.fields):
            indices = [content.fields.index(f) for f in keep]
            return awkward.contents.RecordArray(
                [content.contents[i] for i in indices],
                keep,
                length=content.length,
                parameters=content.parameters,
            )
    if isinstance(content, _LIST_LIKE):
        inner = _strip_tref(content.content)
        if inner is not None:
            return content.copy(content=inner)
    return None


class Delphes(BaseLayoutBuilder):
    """Delphes layout builder

    The Delphes layout is built from all branches found in the supplied file,
    based on the naming pattern of the branches. Any branches named
    ``{name}_size`` are counts branches, converted to offsets. Every Delphes
    object collection is a jagged list of records; the split leaf branches
    (``{name}.{var}``) are grouped under the collection ``{name}`` and given the
    aliases required by the vector/candidate behaviors (``pt``, ``eta``, ``phi``,
    ``mass``, ``rho``).

    A companion ``{name}.offsets`` field is emitted for each collection (matching
    coffea). Collections stored as length-1 vectors (see ``singletons``) are
    flattened, removing an unnecessary level of nesting.
    """

    mixins: tp.ClassVar = {
        "CaloJet02": "Jet",
        "CaloJet04": "Jet",
        "CaloJet08": "Jet",
        "CaloJet15": "Jet",
        "EFlowNeutralHadron": "Tower",
        "EFlowPhoton": "Photon",
        "EFlowTrack": "Track",
        "Electron": "Electron",
        "ElectronCHS": "Electron",
        "GenJet": "Jet",
        "GenJet02": "Jet",
        "GenJet04": "Jet",
        "GenJet08": "Jet",
        "GenJetAK8": "Jet",
        "GenJet15": "Jet",
        "GenMissingET": "MissingET",
        "GenPileUpMissingET": "MissingET",
        "Jet": "Jet",
        "JetAK8": "Jet",
        "JetPUPPI": "Jet",
        "FatJet": "Jet",
        "JetPUPPIAK8": "Jet",
        "MissingET": "MissingET",
        "PuppiMissingET": "MissingET",
        "Muon": "Muon",
        "MuonTight": "Muon",
        "MuonLoose": "Muon",
        "MuonTightCHS": "Muon",
        "MuonLooseCHS": "Muon",
        "Particle": "Particle",
        "ParticleFlowJet02": "Jet",
        "ParticleFlowJet04": "Jet",
        "ParticleFlowJet08": "Jet",
        "ParticleFlowJet15": "Jet",
        "Photon": "Photon",
        "PhotonCHS": "Photon",
        "Tower": "Tower",
        "Track": "Track",
        "TrackJet02": "Jet",
        "TrackJet04": "Jet",
        "TrackJet08": "Jet",
        "TrackJet15": "Jet",
        "Weight": "Weight",
        "WeightLHEF": "WeightLHEF",
        # the following are also singletons
        "Event": "Event",
        "EventLHEF": "EventLHEF",
        "HepMCEvent": "HepMCEvent",
        "LHCOEvent": "LHCOEvent",
        "Rho": "Rho",
        "ScalarHT": "ScalarHT",
    }
    """Default configuration for mixin types, based on the collection name."""

    # These are stored as length-1 vectors unnecessarily
    singletons: tp.ClassVar = [
        "Event",
        "EventLHEF",
        "HepMCEvent",
        "LHCOEvent",
        "Rho",
        "ScalarHT",
        "MissingET",
    ]
    """Fields stored as length-1 vectors in Delphes, flattened out in nanoevents."""

    def __init__(self, version="latest"):
        self._version = version

    def __call__(self, array: awkward.Array) -> awkward.Array:
        fields = list(array.fields)
        n_events = int(awkward.num(array, axis=0))
        forms = {f: _non_materializing_get_field(array, f) for f in fields}

        # collections are the branch prefixes, excluding the counts branches
        collections = {k.split(".")[0] for k in fields}
        collections -= {k for k in collections if k.endswith("_size")}
        # every real collection has a matching counts branch
        collections = {c for c in collections if (c + "_size") in forms}

        output = {}
        for name in sorted(collections):
            size = forms[name + "_size"]
            offsets = counts2offsets(size)
            mixin = self.mixins.get(name, "NanoCollection")

            content = {}
            for k in fields:
                if k.startswith(name + "."):
                    c = _jagged_content(forms[k])
                    # convert ROOT TLorentzVector leaves to LorentzVector records
                    converted = _lorentz_from_tlv(c)
                    if converted is not None:
                        c = converted
                    # reduce ROOT TRef leaves to just their ``ref`` member
                    stripped = _strip_tref(c)
                    if stripped is not None:
                        c = stripped
                    content[k[len(name) + 1 :]] = c

            # add aliases expected by the vector/candidate behaviors
            if mixin == "MissingET":
                content["rho"] = met_to_rho(content["MET"], content["Eta"])
                content["eta"] = content["Eta"]
                content["phi"] = content["Phi"]
            elif mixin == "Vertex":
                content["t"] = content["T"]
                content["x"] = content["X"]
                content["y"] = content["Y"]
                content["z"] = content["Z"]
            elif mixin in ("Particle", "Jet", "Track"):
                content.pop("E", None)
                content["pt"] = content["PT"]
                content["eta"] = content["Eta"]
                content["phi"] = content["Phi"]
                content["mass"] = content["Mass"]
            elif mixin in ("MasslessParticle", "Photon", "Electron", "Muon", "Tower"):
                content.pop("E", None)
                if "PT" not in content and "ET" in content:
                    content["PT"] = content["ET"]
                content["pt"] = content["PT"]
                content["eta"] = content["Eta"]
                content["phi"] = content["Phi"]
                content["mass"] = full_like_from_counts(size, 0.0).layout.content

            # handle branch names like Edges[4] and Tau[5]
            content = {
                k.replace("[", "_").replace("]", ""): v for k, v in content.items()
            }

            _content = tuple(content.values())
            _fields = tuple(content.keys())
            # some ROOT leaves (e.g. fBits) carry an over-sized data buffer; the
            # collection offsets define the real item count, so set the record
            # length from offsets[-1] and let longer contents be truncated
            record = awkward.contents.RecordArray(
                _content,
                _fields,
                length=_total_items(offsets),
                parameters={"__record__": mixin, "collection_name": name},
            )

            # companion offsets field (matches coffea; truncated to n_events)
            output[name + ".offsets"] = awkward.contents.NumpyArray(offsets)

            if name in self.singletons:
                # flatten: promote the (length-1) inner record up one level
                output[name] = record
            else:
                output[name] = awkward.contents.ListOffsetArray(
                    offsets=awkward.index.Index(offsets), content=record
                )

        contents = tuple(output.values())
        names = tuple(output.keys())
        nanoevents = awkward.Array(
            awkward.contents.RecordArray(contents, names, length=n_events),
            behavior=self.behavior(),
        )
        nanoevents = awkward.with_name(_rewrap(nanoevents), name="NanoEvents")
        nanoevents.layout.parameters.update({"metadata": {"version": self._version}})
        nanoevents.attrs["@original_array"] = nanoevents
        return nanoevents

    @classmethod
    def behavior(cls):
        """Behaviors necessary to implement this schema (dict)"""
        from awkward_zipper.behaviors import delphes

        return delphes.behavior
