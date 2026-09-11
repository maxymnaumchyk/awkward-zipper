"""Mixins for the ATLAS Ntuple schema"""

from functools import reduce
from operator import ior

import awkward

from awkward_zipper.behaviors import base, candidate, vector
from awkward_zipper.behaviors.atlas_enums import PhotonID

behavior = {}
behavior.update(base.behavior)
# vector behavior is included in candidate behavior
behavior.update(candidate.behavior)


def _set_repr_name(classname):
    def namefcn(_self):
        return classname

    behavior["__typestr__", classname] = classname[0].lower() + classname[1:]
    behavior[classname].__repr__ = namefcn


class NtupleEvents(behavior["NanoEvents"]):
    """Individual systematic variation of events."""

    def __repr__(self):
        return f"<event {getattr(self, 'runNumber', '??')}:{getattr(self, 'eventNumber', '??')}:{getattr(self, 'mcChannelNumber', '??')}>"

    def __getitem__(self, key):
        """Support accessing systematic variations via bracket notation.

        Args:
            key: The systematic variation name. "NOSYS" returns the nominal events.

        Returns:
            The requested systematic variation or nominal events for "NOSYS".
        """
        if isinstance(key, str) and key == "NOSYS":
            return self
        return super().__getitem__(key)

    @property
    def systematic(self):
        """Get the systematic variation name for this event collection."""
        return "nominal"

    @property
    def systematic_names(self):
        """Get all systematic variations available in this event collection.

        Returns a list of systematic variation names, including 'NOSYS' for nominal.
        """
        # Get systematics from metadata stored during schema building
        systematics = self.metadata.get("systematics", [])
        return ["NOSYS", *systematics]

    @property
    def systematics(self):
        """Get all systematic variations available in this event collection.

        Returns a list of systematic variation names, excluding 'nominal'.
        """
        # Get systematics from metadata stored during schema building
        return [
            getattr(self, systematic)
            for systematic in self.systematic_names
            if systematic != "NOSYS"
        ]


behavior["NtupleEvents"] = NtupleEvents


class NtupleEventsArray(behavior["*", "NanoEvents"]):
    """Collection of NtupleEvents objects, one for each systematic variation."""

    def __getitem__(self, key):
        """Support accessing systematic variations via bracket notation.

        Args:
            key: The systematic variation name. "NOSYS" returns the nominal events.

        Returns:
            The requested systematic variation or nominal events for "NOSYS".
        """
        if isinstance(key, str) and key == "NOSYS":
            return self
        return super().__getitem__(key)

    @property
    def systematic_names(self):
        """Get all systematic variations available in this event collection.

        Returns a list of systematic variation names, including 'NOSYS' for nominal.
        """
        # Get systematics from metadata stored during schema building
        systematics = self.metadata.get("systematics", [])
        return ["NOSYS", *systematics]

    @property
    def systematics(self):
        """Get all systematic variations available in this event collection.

        Returns a list of systematic variation names, excluding 'nominal'.
        """
        # Get systematics from metadata stored during schema building
        return [
            getattr(self, systematic)
            for systematic in self.systematic_names
            if systematic != "NOSYS"
        ]


behavior["*", "NtupleEvents"] = NtupleEventsArray


@awkward.mixin_class(behavior)
class Systematic(base.NanoCollection):
    """Base class for systematic variations."""

    @property
    def metadata(self):
        """Arbitrary metadata"""
        return self.layout.purelist_parameter("metadata")

    @property
    def systematic(self):
        """Get the systematic variation name for this event collection."""
        return self.metadata["systematic"]

    def __repr__(self):
        return f"<event {self.systematic}>"


_set_repr_name("Systematic")


@awkward.mixin_class(behavior)
class Weight(base.NanoCollection): ...


_set_repr_name("Weight")


@awkward.mixin_class(behavior)
class Pass(base.NanoCollection): ...


_set_repr_name("Pass")

behavior.update(
    awkward._util.copy_behaviors("PtEtaPhiMLorentzVector", "Particle", behavior)
)


@awkward.mixin_class(behavior)
class Particle(vector.PtEtaPhiMLorentzVector):
    """Generic particle collection that has Lorentz vector properties

    Also handles the following additional branches:
    - '{obj}_select'
    """

    def passes(self, name):
        return self[f"select_{name}"] == 1


_set_repr_name("Particle")

ParticleArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
ParticleArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
ParticleArray.ProjectionClass4D = ParticleArray  # noqa: F821
ParticleArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821
ParticleRecord.ProjectionClass2D = vector.TwoVectorRecord  # noqa: F821
ParticleRecord.ProjectionClass3D = vector.ThreeVectorRecord  # noqa: F821
ParticleRecord.ProjectionClass4D = ParticleRecord  # noqa: F821
ParticleRecord.MomentumClass = vector.LorentzVectorRecord  # noqa: F821


behavior.update(awkward._util.copy_behaviors("PolarTwoVector", "MissingET", behavior))


@awkward.mixin_class(behavior)
class MissingET(vector.PolarTwoVector, base.NanoCollection):
    """Missing transverse energy collection."""


_set_repr_name("MissingET")

MissingETArray.ProjectionClass2D = MissingETArray  # noqa: F821
MissingETArray.ProjectionClass3D = vector.SphericalThreeVectorArray  # noqa: F821
MissingETArray.ProjectionClass4D = vector.LorentzVectorArray  # noqa: F821
MissingETArray.MomentumClass = MissingETArray  # noqa: F821
MissingETRecord.ProjectionClass2D = MissingETRecord  # noqa: F821
MissingETRecord.ProjectionClass3D = vector.SphericalThreeVectorRecord  # noqa: F821
MissingETRecord.ProjectionClass4D = vector.LorentzVectorRecord  # noqa: F821
MissingETRecord.MomentumClass = MissingETRecord  # noqa: F821

behavior.update(awkward._util.copy_behaviors("Particle", "Photon", behavior))


@awkward.mixin_class(behavior)
class Photon(Particle, base.NanoCollection):
    """Photon particle collection."""

    @property
    def isEM(self):
        return self.isEM_syst.NOSYS == 0

    def pass_isEM(self, words: list[PhotonID]):
        # 0 is pass, 1 is fail
        return (
            self.isEM_syst.NOSYS & reduce(ior, (1 << word.value for word in words))
        ) == 0


_set_repr_name("Photon")

PhotonArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
PhotonArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
PhotonArray.ProjectionClass4D = PhotonArray  # noqa: F821
PhotonArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821
PhotonRecord.ProjectionClass2D = vector.TwoVectorRecord  # noqa: F821
PhotonRecord.ProjectionClass3D = vector.ThreeVectorRecord  # noqa: F821
PhotonRecord.ProjectionClass4D = PhotonRecord  # noqa: F821
PhotonRecord.MomentumClass = vector.LorentzVectorRecord  # noqa: F821

behavior.update(awkward._util.copy_behaviors("Particle", "Electron", behavior))


@awkward.mixin_class(behavior)
class Electron(Particle, base.NanoCollection):
    """Electron particle collection."""


_set_repr_name("Electron")

ElectronArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
ElectronArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
ElectronArray.ProjectionClass4D = ElectronArray  # noqa: F821
ElectronArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821
ElectronRecord.ProjectionClass2D = vector.TwoVectorRecord  # noqa: F821
ElectronRecord.ProjectionClass3D = vector.ThreeVectorRecord  # noqa: F821
ElectronRecord.ProjectionClass4D = ElectronRecord  # noqa: F821
ElectronRecord.MomentumClass = vector.LorentzVectorRecord  # noqa: F821

behavior.update(awkward._util.copy_behaviors("Particle", "Muon", behavior))


@awkward.mixin_class(behavior)
class Muon(Particle, base.NanoCollection):
    """Muon particle collection."""


_set_repr_name("Muon")

MuonArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
MuonArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
MuonArray.ProjectionClass4D = MuonArray  # noqa: F821
MuonArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821
MuonRecord.ProjectionClass2D = vector.TwoVectorRecord  # noqa: F821
MuonRecord.ProjectionClass3D = vector.ThreeVectorRecord  # noqa: F821
MuonRecord.ProjectionClass4D = MuonRecord  # noqa: F821
MuonRecord.MomentumClass = vector.LorentzVectorRecord  # noqa: F821

behavior.update(awkward._util.copy_behaviors("Particle", "Tau", behavior))


@awkward.mixin_class(behavior)
class Tau(Particle, base.NanoCollection):
    """Tau particle collection."""


_set_repr_name("Tau")

TauArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
TauArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
TauArray.ProjectionClass4D = TauArray  # noqa: F821
TauArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821
TauRecord.ProjectionClass2D = vector.TwoVectorRecord  # noqa: F821
TauRecord.ProjectionClass3D = vector.ThreeVectorRecord  # noqa: F821
TauRecord.ProjectionClass4D = TauRecord  # noqa: F821
TauRecord.MomentumClass = vector.LorentzVectorRecord  # noqa: F821


behavior.update(awkward._util.copy_behaviors("Particle", "Jet", behavior))


@awkward.mixin_class(behavior)
class Jet(Particle, base.NanoCollection):
    """Jet particle collection."""


_set_repr_name("Jet")

JetArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
JetArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
JetArray.ProjectionClass4D = JetArray  # noqa: F821
JetArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821
JetRecord.ProjectionClass2D = vector.TwoVectorRecord  # noqa: F821
JetRecord.ProjectionClass3D = vector.ThreeVectorRecord  # noqa: F821
JetRecord.ProjectionClass4D = JetRecord  # noqa: F821
JetRecord.MomentumClass = vector.LorentzVectorRecord  # noqa: F821

__all__ = [
    "Electron",
    "ElectronArray",  # noqa: F822
    "ElectronRecord",  # noqa: F822
    "Jet",
    "JetArray",  # noqa: F822
    "JetRecord",  # noqa: F822
    "MissingET",
    "MissingETArray",  # noqa: F822
    "MissingETRecord",  # noqa: F822
    "Muon",
    "MuonArray",  # noqa: F822
    "MuonRecord",  # noqa: F822
    "NtupleEvents",
    "Particle",
    "ParticleArray",  # noqa: F822
    "ParticleRecord",  # noqa: F822
    "Pass",
    "Photon",
    "PhotonArray",  # noqa: F822
    "PhotonRecord",  # noqa: F822
    "Weight",
]
