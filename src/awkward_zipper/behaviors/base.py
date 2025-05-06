"""Basic NanoEvents and NanoCollection mixins"""

from typing import Any

import awkward

behavior = {}


class _ClassMethodFn:
    def __init__(self, attr: str, **kwargs: Any) -> None:
        self.attr = attr

    def __call__(self, coll: awkward.Array, *args: Any, **kwargs: Any) -> awkward.Array:
        return getattr(coll, self.attr)(*args, **kwargs)


behavior[("__typestr__", "NanoEvents")] = "event"


@awkward.mixin_class(behavior)
class NanoEvents:
    """NanoEvents mixin class

    This mixin class is used as the top-level type for NanoEvents objects.
    """

    def metadata(self):
        """Arbitrary metadata"""
        return self.layout.purelist_parameter("metadata")


@awkward.mixin_class(behavior)
class NanoCollection:
    """A NanoEvents collection

    This mixin provides some helper methods useful for creating cross-references
    and other advanced mixin types.
    """

    def _collection_name(self):
        """The name of the collection (i.e. the field under events where it is found)"""
        return self.layout.purelist_parameter("collection_name")

    def _getlistarray(self):
        """Do some digging to find the initial listarray"""

        def descend(layout, depth, **kwargs):
            islistarray = isinstance(
                layout,
                awkward.contents.ListOffsetArray,
            )
            if islistarray and layout.content.parameter("collection_name") is not None:
                return layout
            return None

        return awkward.transform(descend, self.layout, highlevel=False)

    def _content(self):
        """Internal method to get jagged collection content

        This should only be called on the original unsliced collection array.
        Used with global indexes to resolve cross-references"""
        return self._getlistarray().content

    def _apply_global_index(self, index):
        """Internal method to take from a collection using a flat index

        This is often necessary to be able to still resolve cross-references on
        reduced arrays or single records.
        """
        if isinstance(index, int):
            out = self._content()[index]
            return awkward.Record(out, behavior=self.behavior)

        def flat_take(layout):
            idx = awkward.Array(layout)
            return self._content()[idx.mask[idx >= 0]]

        def descend(layout, depth, **kwargs):
            if layout.purelist_depth == 1:
                return flat_take(layout)
            return None

        (index_out,) = awkward.broadcast_arrays(index)
        layout_out = awkward.transform(descend, index_out.layout, highlevel=False)
        return awkward.Array(layout_out, behavior=self.behavior, attrs=self.attrs)

    def _events(self):
        """Internal method to get the originally-constructed NanoEvents

        This can be called at any time from any collection, as long as
        the attr exists.

        This will not work automatically if you read serialized nanoevents."""
        return self.attrs["@original_array"]


__all__ = ["NanoCollection", "NanoEvents"]
