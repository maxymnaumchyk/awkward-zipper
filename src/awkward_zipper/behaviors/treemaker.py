"""Behaviors for the TreeMaker schema

TreeMaker collections use only the generic vector behaviors (plus the base
``NanoEvents``/``NanoCollection`` mixins), so the behavior dict is simply the
union of the base and vector behaviors.
"""

from awkward_zipper.behaviors import base, vector

behavior = {}
behavior.update(base.behavior)
behavior.update(vector.behavior)


class _TreeMakerEvents(behavior["NanoEvents"]):
    def __repr__(self):
        return "<TreeMaker event>"


behavior["NanoEvents"] = _TreeMakerEvents

__all__ = ["behavior"]
