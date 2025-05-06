"""Mixins for the CMS NanoAOD schema"""

import typing as tp
import warnings

import awkward

from awkward_zipper.behaviors import base, candidate, vector

behavior = {}
behavior.update(base.behavior)
# vector behavior is included in candidate behavior
behavior.update(candidate.behavior)


class _NanoAODEvents(behavior["NanoEvents"]):
    def __repr__(self):
        return "<NanoAOD event>"


behavior["NanoEvents"] = _NanoAODEvents


def _set_repr_name(classname):
    def namefcn(self):
        return classname

    behavior[classname].__repr__ = namefcn


behavior.update(
    awkward._util.copy_behaviors(
        "PtEtaPhiMLorentzVector", "PtEtaPhiMCollection", behavior
    )
)


@awkward.mixin_class(behavior)
class PtEtaPhiMCollection(vector.PtEtaPhiMLorentzVector, base.NanoCollection):
    """Generic collection that has Lorentz vector properties"""


PtEtaPhiMCollectionArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
PtEtaPhiMCollectionArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
PtEtaPhiMCollectionArray.ProjectionClass4D = PtEtaPhiMCollectionArray  # noqa: F821
PtEtaPhiMCollectionArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821


behavior.update(
    awkward._util.copy_behaviors("PtEtaPhiMLorentzVector", "GenParticle", behavior)
)


@awkward.mixin_class(behavior)
class GenParticle(vector.PtEtaPhiMLorentzVector, base.NanoCollection):
    """NanoAOD generator-level particle object, including parent and child self-references

    Parent and child self-references are constructed from the ``genPartIdxMother`` column, where
    for each entry, the mother entry index is recorded, or -1 if no mother exists.
    """

    FLAGS: tp.ClassVar = [
        "isPrompt",
        "isDecayedLeptonHadron",
        "isTauDecayProduct",
        "isPromptTauDecayProduct",
        "isDirectTauDecayProduct",
        "isDirectPromptTauDecayProduct",
        "isDirectHadronDecayProduct",
        "isHardProcess",
        "fromHardProcess",
        "isHardProcessTauDecayProduct",
        "isDirectHardProcessTauDecayProduct",
        "fromHardProcessBeforeFSR",
        "isFirstCopy",
        "isLastCopy",
        "isLastCopyBeforeFSR",
    ]
    """bit-packed statusFlags interpretations.  Use `GenParticle.hasFlags` to query"""

    def hasFlags(self, *flags):
        """Check if one or more status flags are set

        Parameters
        ----------
            flags : str or list
                A list of flags that are required to be set true. If the first argument
                is a list, it is expanded and subsequent arguments ignored.
                Possible flags are enumerated in the `FLAGS` attribute

        Returns a boolean array
        """
        if len(flags) == 0:
            msg = "No flags specified"
            raise ValueError(msg)
        if isinstance(flags[0], list):
            flags = flags[0]
        mask = 0
        for flag in flags:
            mask |= 1 << self.FLAGS.index(flag)
        return (self.statusFlags & mask) == mask

    def parent(self):
        """
        Accessor to the direct parent of this particle.
        """
        return self._events().GenPart._apply_global_index(self.genPartIdxMotherG)

    def distinctParent(self):
        """
        Accessor to distinct (different PDG id) parent particle.
        """
        return self._events().GenPart._apply_global_index(self.distinctParentIdxG)

    def children(self):
        """
        Accessor to direct children of this particle (not grandchildren). Includes particles
        with the same PDG ID as this particle.
        """
        return self._events().GenPart._apply_global_index(self.childrenIdxG)

    def distinctChildren(self):
        """
        Accessor to direct children of this particle which do not have the same PDG ID as
        this particle. Note that this implies the summed four-momentum of the distinctChildren
        may not sum to the four-momentum of this particle (for example, if this particle
        radiates another particle type). If that behavior is desired, see `distinctChildrenDeep`.
        """
        return self._events().GenPart._apply_global_index(self.distinctChildrenIdxG)

    def distinctChildrenDeep(self):
        """
        Accessor to distinct child particles with different PDG id, or last ones in the chain.
        Note that this does not always find the correct children, since this sometimes depends
        on the MC generator! See `here <https://github.com/scikit-hep/coffea/pull/698>` for more
        information.
        """
        warnings.warn(
            "distinctChildrenDeep may not give correct answers for all generators!",
            stacklevel=2,
        )
        return self._events().GenPart._apply_global_index(self.distinctChildrenDeepIdxG)


_set_repr_name("GenParticle")

GenParticleArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
GenParticleArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
GenParticleArray.ProjectionClass4D = GenParticleArray  # noqa: F821
GenParticleArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821

behavior.update(
    awkward._util.copy_behaviors("PtEtaPhiMLorentzVector", "GenVisTau", behavior)
)


@awkward.mixin_class(behavior)
class GenVisTau(candidate.PtEtaPhiMCandidate, base.NanoCollection):
    """NanoAOD visible tau object"""

    def parent(self):
        """Accessor to the parent particle"""
        return self._events().GenPart._apply_global_index(self.genPartIdxMotherG)


_set_repr_name("GenVisTau")

GenVisTauArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
GenVisTauArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
GenVisTauArray.ProjectionClass4D = GenVisTauArray  # noqa: F821
GenVisTauArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821

behavior.update(
    awkward._util.copy_behaviors("PtEtaPhiMCandidate", "Electron", behavior)
)


@awkward.mixin_class(behavior)
class Electron(candidate.PtEtaPhiMCandidate, base.NanoCollection):
    """NanoAOD electron object"""

    FAIL = 0
    "cutBased selection minimum value"
    VETO = 1
    "cutBased selection minimum value"
    LOOSE = 2
    "cutBased selection minimum value"
    MEDIUM = 3
    "cutBased selection minimum value"
    TIGHT = 4
    "cutBased selection minimum value"

    @property
    def isVeto(self):
        """Returns a boolean array marking veto cut-based electrons"""
        return self.cutBased >= self.VETO

    @property
    def isLoose(self):
        """Returns a boolean array marking loose cut-based electrons"""
        return self.cutBased >= self.LOOSE

    @property
    def isMedium(self):
        """Returns a boolean array marking medium cut-based electrons"""
        return self.cutBased >= self.MEDIUM

    @property
    def isTight(self):
        """Returns a boolean array marking tight cut-based electrons"""
        return self.cutBased >= self.TIGHT

    def matched_gen(self):
        """The matched gen-level particle as determined by the NanoAOD branch genPartIdx"""
        return self._events().GenPart._apply_global_index(self.genPartIdxG)

    def matched_jet(self):
        """The matched jet as determined by the NanoAOD branch jetIdx"""
        return self._events().Jet._apply_global_index(self.jetIdxG)

    def matched_photon(self):
        """The associated photon as determined by the NanoAOD branch photonIdx"""
        return self._events().Photon._apply_global_index(self.photonIdxG)


_set_repr_name("Electron")

ElectronArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
ElectronArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
ElectronArray.ProjectionClass4D = ElectronArray  # noqa: F821
ElectronArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821

behavior.update(
    awkward._util.copy_behaviors("PtEtaPhiMCandidate", "LowPtElectron", behavior)
)


@awkward.mixin_class(behavior)
class LowPtElectron(candidate.PtEtaPhiMCandidate, base.NanoCollection):
    """NanoAOD low-pt electron object"""

    def matched_gen(self):
        """The matched gen-level particle as determined by the NanoAOD branch genPartIdx"""
        return self._events().GenPart._apply_global_index(self.genPartIdxG)

    def matched_electron(self):
        """The matched gen-level electron as determined by the NanoAOD branch electronIdx"""
        return self._events().Electron._apply_global_index(self.electronIdxG)

    def matched_photon(self):
        """The associated photon as determined by the NanoAOD branch photonIdx"""
        return self._events().Photon._apply_global_index(self.photonIdxG)


_set_repr_name("LowPtElectron")

LowPtElectronArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
LowPtElectronArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
LowPtElectronArray.ProjectionClass4D = LowPtElectronArray  # noqa: F821
LowPtElectronArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821

behavior.update(awkward._util.copy_behaviors("PtEtaPhiMCandidate", "Muon", behavior))


@awkward.mixin_class(behavior)
class Muon(candidate.PtEtaPhiMCandidate, base.NanoCollection):
    """NanoAOD muon object"""

    def matched_fsrPhoton(self):
        """The matched FSR photon with the lowest dR/ET2. Accessed via the NanoAOD branch fsrPhotonIdx"""
        return self._events().FsrPhoton._apply_global_index(self.fsrPhotonIdxG)

    def matched_gen(self):
        """The matched gen-level particle as determined by the NanoAOD branch genPartIdx"""
        return self._events().GenPart._apply_global_index(self.genPartIdxG)

    def matched_jet(self):
        """The matched jet as determined by the NanoAOD branch jetIdx"""
        return self._events().Jet._apply_global_index(self.jetIdxG)


_set_repr_name("Muon")

MuonArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
MuonArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
MuonArray.ProjectionClass4D = MuonArray  # noqa: F821
MuonArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821

behavior.update(awkward._util.copy_behaviors("PtEtaPhiMCandidate", "Tau", behavior))


@awkward.mixin_class(behavior)
class Tau(candidate.PtEtaPhiMCandidate, base.NanoCollection):
    """NanoAOD tau object"""

    def matched_gen(self):
        """The matched gen-level particle as determined by the NanoAOD branch genPartIdx"""
        return self._events().GenPart._apply_global_index(self.genPartIdxG)

    def matched_jet(self):
        """The matched jet as determined by the NanoAOD branch jetIdx"""
        return self._events().Jet._apply_global_index(self.jetIdxG)


_set_repr_name("Tau")

TauArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
TauArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
TauArray.ProjectionClass4D = TauArray  # noqa: F821
TauArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821

behavior.update(awkward._util.copy_behaviors("PtEtaPhiMCandidate", "Photon", behavior))


@awkward.mixin_class(behavior)
class Photon(candidate.PtEtaPhiMCandidate, base.NanoCollection):
    """NanoAOD photon object"""

    FAIL = 0
    "cutBased selection minimum value"
    LOOSE = 1
    "cutBased selection minimum value"
    MEDIUM = 2
    "cutBased selection minimum value"
    TIGHT = 3
    "cutBased selection minimum value"

    @property
    def mass(self):
        return awkward.zeros_like(self.pt)

    @property
    def charge(self):
        return awkward.zeros_like(self.pt)

    @property
    def isLoose(self):
        """Returns a boolean array marking loose cut-based photons"""
        # For NanoAOD v9+ the cutBasedBitmap was changed to a cutBased integer
        if "cutBased" in self.fields:
            return self.cutBased >= self.LOOSE
        return (self.cutBasedBitmap & (1 << (self.LOOSE - 1))) != 0

    @property
    def isMedium(self):
        """Returns a boolean array marking medium cut-based photons"""
        # For NanoAOD v9+ the cutBasedBitmap was changed to a cutBased integer
        if "cutBased" in self.fields:
            return self.cutBased >= self.MEDIUM
        return (self.cutBasedBitmap & (1 << (self.MEDIUM - 1))) != 0

    @property
    def isTight(self):
        """Returns a boolean array marking tight cut-based photons"""
        # For NanoAOD v9+ the cutBasedBitmap was changed to a cutBased integer
        if "cutBased" in self.fields:
            return self.cutBased >= self.TIGHT
        return (self.cutBasedBitmap & (1 << (self.TIGHT - 1))) != 0

    def matched_electron(self):
        """The matched electron as determined by the NanoAOD branch electronIdx"""
        return self._events().Electron._apply_global_index(self.electronIdxG)

    def matched_gen(self):
        """The matched gen-level particle as determined by the NanoAOD branch genPartIdx"""
        return self._events().GenPart._apply_global_index(self.genPartIdxG)

    def matched_jet(self):
        """The matched jet as determined by the NanoAOD branch jetIdx"""
        return self._events().Jet._apply_global_index(self.jetIdxG)


_set_repr_name("Photon")

PhotonArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
PhotonArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
PhotonArray.ProjectionClass4D = PhotonArray  # noqa: F821
PhotonArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821

behavior.update(
    awkward._util.copy_behaviors("PtEtaPhiMCandidate", "FsrPhoton", behavior)
)


@awkward.mixin_class(behavior)
class FsrPhoton(candidate.PtEtaPhiMCandidate, base.NanoCollection):
    """NanoAOD fsr photon object"""

    def matched_muon(self):
        """The matched muon as determined by the NanoAOD branch muonIdx"""
        return self._events().Muon._apply_global_index(self.muonIdxG)


_set_repr_name("FsrPhoton")

FsrPhotonArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
FsrPhotonArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
FsrPhotonArray.ProjectionClass4D = FsrPhotonArray  # noqa: F821
FsrPhotonArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821

behavior.update(awkward._util.copy_behaviors("PtEtaPhiMCandidate", "Jet", behavior))


@awkward.mixin_class(behavior)
class Jet(candidate.PtEtaPhiMCandidate, base.NanoCollection):
    """NanoAOD narrow radius jet object"""

    LOOSE = 0
    "jetId bit position"
    TIGHT = 1
    "jetId bit position"
    TIGHTLEPVETO = 2
    "jetId bit position"

    @property
    def charge(self):
        return awkward.zeros_like(self.pt)

    @property
    def isLoose(self):
        """Returns a boolean array marking loose jets according to jetId index"""
        return (self.jetId & (1 << self.LOOSE)) != 0

    @property
    def isTight(self):
        """Returns a boolean array marking tight jets according to jetId index"""
        return (self.jetId & (1 << self.TIGHT)) != 0

    @property
    def isTightLeptonVeto(self):
        """Returns a boolean array marking tight jets with explicit lepton veto according to jetId index"""
        return (self.jetId & (1 << self.TIGHTLEPVETO)) != 0

    def matched_electrons(self):
        """
        The matched electrons as determined by the NanoAOD branch electronIdx. The resulting awkward
        array has two entries per jet, where if there are fewer than 2 electrons matched to a jet, the
        innermost dimensions are padded with None to be of size 2.
        """
        return self._events().Electron._apply_global_index(self.electronIdxG)

    def matched_muons(self):
        """
        The matched muons as determined by the NanoAOD branch muonIdx. The resulting awkward
        array has two entries per jet, where if there are fewer than 2 muons matched to a jet, the
        innermost dimensions are padded with None to be of size 2.
        """
        return self._events().Muon._apply_global_index(self.muonIdxG)

    def matched_gen(self):
        """
        AK4 jets made with visible genparticles, matched to this jet via the NanoAOD branch genJetIdx
        """
        return self._events().GenJet._apply_global_index(self.genJetIdxG)

    def constituents(self):
        if "pFCandsIdxG" not in self.fields:
            msg = "PF candidates are only available for PFNano"
            raise RuntimeError(msg)
        return self._events().JetPFCands._apply_global_index(self.pFCandsIdxG)


_set_repr_name("Jet")

JetArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
JetArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
JetArray.ProjectionClass4D = JetArray  # noqa: F821
JetArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821

behavior.update(awkward._util.copy_behaviors("PtEtaPhiMCandidate", "FatJet", behavior))


@awkward.mixin_class(behavior)
class FatJet(candidate.PtEtaPhiMCandidate, base.NanoCollection):
    """NanoAOD large radius jet object"""

    LOOSE = 0
    "jetId bit position"
    TIGHT = 1
    "jetId bit position"
    TIGHTLEPVETO = 2
    "jetId bit position"

    @property
    def charge(self):
        return awkward.zeros_like(self.pt)

    @property
    def isLoose(self):
        """Returns a boolean array marking loose jets according to jetId index"""
        return (self.jetId & (1 << self.LOOSE)) != 0

    @property
    def isTight(self):
        """Returns a boolean array marking tight jets according to jetId index"""
        return (self.jetId & (1 << self.TIGHT)) != 0

    @property
    def isTightLeptonVeto(self):
        """Returns a boolean array marking tight jets with explicit lepton veto according to jetId index"""
        return (self.jetId & (1 << self.TIGHTLEPVETO)) != 0

    def subjets(self):
        return self._events().SubJet._apply_global_index(self.subJetIdxG)

    def matched_gen(self):
        """AK8 jets made of visible genparticles, matched via the NanoAOD branch genJetAK8Idx"""
        return self._events().GenJetAK8._apply_global_index(self.genJetAK8IdxG)

    def constituents(self):
        if "pFCandsIdxG" not in self.fields:
            msg = "PF candidates are only available for PFNano"
            raise RuntimeError(msg)
        return self._events().FatJetPFCands._apply_global_index(self.pFCandsIdxG)


_set_repr_name("FatJet")

FatJetArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
FatJetArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
FatJetArray.ProjectionClass4D = FatJetArray  # noqa: F821
FatJetArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821

behavior.update(awkward._util.copy_behaviors("PolarTwoVector", "MissingET", behavior))


@awkward.mixin_class(behavior)
class MissingET(vector.PolarTwoVector, base.NanoCollection):
    """NanoAOD Missing transverse energy object"""

    @property
    def r(self):
        """Distance from origin in XY plane"""
        return self["pt"]


_set_repr_name("MissingET")

MissingETArray.ProjectionClass2D = MissingETArray  # noqa: F821
MissingETArray.ProjectionClass3D = vector.SphericalThreeVectorArray  # noqa: F821
MissingETArray.ProjectionClass4D = vector.LorentzVectorArray  # noqa: F821
MissingETArray.MomentumClass = MissingETArray  # noqa: F821


@awkward.mixin_class(behavior)
class Vertex(base.NanoCollection):
    """NanoAOD vertex object"""

    @property
    def pos(self):
        """Vertex position as a three vector"""
        return awkward.zip(
            {
                "x": self["x"],
                "y": self["y"],
                "z": self["z"],
            },
            with_name="ThreeVector",
            behavior=self.behavior,
        )


_set_repr_name("Vertex")


@awkward.mixin_class(behavior)
class SecondaryVertex(Vertex):
    """NanoAOD secondary vertex object"""

    @property
    def p4(self):
        """4-momentum vector of tracks associated to this SV"""
        return awkward.zip(
            {
                "pt": self["pt"],
                "eta": self["eta"],
                "phi": self["phi"],
                "mass": self["mass"],
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=self.behavior,
        )


_set_repr_name("SecondaryVertex")


@awkward.mixin_class(behavior)
class AssociatedPFCand(base.NanoCollection):
    """PFNano PF candidate to jet association object"""

    collection_map: tp.ClassVar = {
        "JetPFCands": ("Jet", "PFCands"),
        "FatJetPFCands": ("FatJet", "PFCands"),
        "GenJetCands": ("GenJet", "GenCands"),
        "GenFatJetCands": ("GenJetAK8", "GenCands"),
    }

    def jet(self):
        collection = self.collection_map[self._collection_name()][0]
        return self._events()[collection]._apply_global_index(self.jetIdxG)

    def pf(self):
        collection = self.collection_map[self._collection_name()][1]
        return self._events()[collection]._apply_global_index(self.pFCandsIdxG)


_set_repr_name("AssociatedPFCand")


@awkward.mixin_class(behavior)
class AssociatedSV(base.NanoCollection):
    """PFNano secondary vertex to jet association object"""

    collection_map: tp.ClassVar = {
        "JetSVs": ("Jet", "SV"),
        "FatJetSVs": ("FatJet", "SV"),
        # these two are unclear
        "GenJetSVs": ("GenJet", "SV"),
        "GenFatJetSVs": ("GenJetAK8", "SV"),
    }

    def jet(self):
        collection = self._events()[self.collection_map[self._collection_name()][0]]
        return self._events()[collection]._apply_global_index(self.jetIdxG)

    def sv(self):
        collection = self.collection_map[self._collection_name()][1]
        return self._events()[collection]._apply_global_index(self.sVIdxG)


_set_repr_name("AssociatedSV")


@awkward.mixin_class(behavior)
class PFCand(candidate.PtEtaPhiMCandidate, base.NanoCollection):
    """PFNano particle flow candidate object"""


_set_repr_name("PFCand")

PFCandArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
PFCandArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
PFCandArray.ProjectionClass4D = PFCandArray  # noqa: F821
PFCandArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821

__all__ = [
    "AssociatedPFCand",
    "AssociatedSV",
    "Electron",
    "FatJet",
    "FsrPhoton",
    "GenParticle",
    "GenVisTau",
    "Jet",
    "LowPtElectron",
    "MissingET",
    "Muon",
    "PFCand",
    "Photon",
    "PtEtaPhiMCollection",
    "SecondaryVertex",
    "Tau",
    "Vertex",
]
