from __future__ import annotations

import sys
from enum import Enum, IntEnum

if sys.version_info >= (3, 11):
    from enum import EnumType
else:
    from enum import EnumMeta as EnumType

from collections.abc import Callable
from typing import TypeVar, cast

_E = TypeVar("_E", bound=Enum)


class MultipleEnumAccessMeta(EnumType):
    """
    Enum Metaclass to provide a way to access multiple values all at once.
    """

    def __getitem__(cls: type[_E], key: str | tuple[str]) -> _E | list[_E]:  # type:ignore[misc,override]
        getitem = cast(Callable[[str], _E], super().__getitem__)  # type:ignore[misc]
        if isinstance(key, tuple):
            return [getitem(name) for name in key]
        return getitem(key)


class ParticleType(IntEnum, metaclass=MultipleEnumAccessMeta):
    """
    Taken from `ATLAS Truth Utilities for ParticleType <https://gitlab.cern.ch/atlas/athena/-/blob/74f43ff0910edb2a2bd3778880ccbdad648dc037/Generators/TruthUtils/TruthUtils/TruthClasses.h#L8-49>`_.
    """

    Unknown = 0
    UnknownElectron = 1
    IsoElectron = 2
    NonIsoElectron = 3
    BkgElectron = 4
    UnknownMuon = 5
    IsoMuon = 6
    NonIsoMuon = 7
    BkgMuon = 8
    UnknownTau = 9
    IsoTau = 10
    NonIsoTau = 11
    BkgTau = 12
    UnknownPhoton = 13
    IsoPhoton = 14
    NonIsoPhoton = 15
    BkgPhoton = 16
    Hadron = 17
    Neutrino = 18
    NuclFrag = 19
    NonPrimary = 20
    GenParticle = 21
    SUSYParticle = 22
    OtherBSMParticle = 39
    BBbarMesonPart = 23
    BottomMesonPart = 24
    CCbarMesonPart = 25
    CharmedMesonPart = 26
    BottomBaryonPart = 27
    CharmedBaryonPart = 28
    StrangeBaryonPart = 29
    LightBaryonPart = 30
    StrangeMesonPart = 31
    LightMesonPart = 32
    BJet = 33
    CJet = 34
    LJet = 35
    GJet = 36
    TauJet = 37
    UnknownJet = 38


class ParticleOrigin(IntEnum, metaclass=MultipleEnumAccessMeta):
    """
    Taken from `ATLAS Truth Utilities for ParticleOrigin <https://gitlab.cern.ch/atlas/athena/-/blob/74f43ff0910edb2a2bd3778880ccbdad648dc037/Generators/TruthUtils/TruthUtils/TruthClasses.h#L51-103>`_.
    """

    NonDefined = 0
    SingleElec = 1
    SingleMuon = 2
    SinglePhot = 3
    SingleTau = 4
    PhotonConv = 5
    DalitzDec = 6
    ElMagProc = 7
    Mu = 8
    TauLep = 9
    top = 10
    QuarkWeakDec = 11
    WBoson = 12
    ZBoson = 13
    Higgs = 14
    HiggsMSSM = 15
    HeavyBoson = 16
    WBosonLRSM = 17
    NuREle = 18
    NuRMu = 19
    NuRTau = 20
    LQ = 21
    SUSY = 22
    OtherBSM = 46
    LightMeson = 23
    StrangeMeson = 24
    CharmedMeson = 25
    BottomMeson = 26
    CCbarMeson = 27
    JPsi = 28
    BBbarMeson = 29
    LightBaryon = 30
    StrangeBaryon = 31
    CharmedBaryon = 32
    BottomBaryon = 33
    PionDecay = 34
    KaonDecay = 35
    BremPhot = 36
    PromptPhot = 37
    UndrPhot = 38
    ISRPhot = 39
    FSRPhot = 40
    NucReact = 41
    PiZero = 42
    DiBoson = 43
    ZorHeavyBoson = 44
    MultiBoson = 47
    QCD = 45


class PhotonID(IntEnum, metaclass=MultipleEnumAccessMeta):
    """
    Taken from the `EGamma Identification CP group's twiki <https://twiki.cern.ch/twiki/bin/viewauth/AtlasProtected/EGammaIdentificationRun2#Photon_isEM_word>`_.
    """

    Rhad = 10  # ClusterHadronicLeakage_Photon
    E277 = 11  # ClusterMiddleEnergy_Photon
    Reta = 12  # ClusterMiddleEratio37_Photon
    Rphi = 13  # ClusterMiddleEratio33_Photon
    Weta2 = 14  # ClusterMiddleWidth_Photon
    f1 = 15  # ClusterStripsEratio_Photon
    DeltaE = 17  # ClusterStripsDeltaE_Photon
    Wstot = 18  # ClusterStripsWtot_Photon
    fside = 19  # ClusterStripsFracm_Photon
    Ws3 = 20  # ClusterStripsWeta1c_Photon
    ERatio = 21  # ClusterStripsDEmaxs1_Photon
