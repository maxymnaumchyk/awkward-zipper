import typing as tp
import warnings
from collections import defaultdict

import awkward

from awkward_zipper.awkward_util import (
    _check_equal_lengths,
    _jagged_content,
    _non_materializing_get_field,
    _rewrap,
    _total_items,
)
from awkward_zipper.kernels import (
    eventindex_content,
    full_like_from_content,
    qoverp_theta_to_pt,
    qoverp_to_p,
)
from awkward_zipper.layouts.base import BaseLayoutBuilder

# coffea's "List" form-class check: a list of fixed-size arrays (RegularArray)
# still counts as single-jagged, a list of lists does not
_LIST_CLASSES = (awkward.contents.ListOffsetArray, awkward.contents.ListArray)


def _zip_leaves(members):
    """Zip leaf layouts into one record (coffea's ``zip_forms``).

    Jagged leaves share the offsets of the first one; anything else is zipped
    into a flat per-event record.
    """
    layouts = list(members.values())
    if all(isinstance(layout, awkward.contents.ListOffsetArray) for layout in layouts):
        offsets = layouts[0].offsets
        return awkward.contents.ListOffsetArray(
            offsets,
            awkward.contents.RecordArray(
                [_jagged_content(layout) for layout in layouts],
                list(members.keys()),
                length=_total_items(offsets),
            ),
        )
    return awkward.contents.RecordArray(
        layouts, list(members.keys()), length=_check_equal_lengths(layouts)
    )


class PHYSLITE(BaseLayoutBuilder):
    """ATLAS DAOD_PHYSLITE layout builder

    Array-based re-implementation of coffea's ``PHYSLITESchema``. Branches named
    ``<Collection>Aux[Dyn].<field>`` are grouped by collection (with the
    ``Analysis``/``Aux``/``AuxDyn`` affixes stripped) and zipped into jagged
    records. Split ``ElementLink`` members (``link.m_persKey``/``link.m_persIndex``)
    are reconstituted into sub-records, a synthetic ``_eventindex`` column is added
    per collection, and ``TrackParticle``/``Muon`` collections get their derived
    ``p``/``pt``/``tau``/``m`` fields.
    """

    truth_collections: tp.ClassVar = [
        "TruthPhotons",
        "TruthMuons",
        "TruthNeutrinos",
        "TruthTaus",
        "TruthElectrons",
        "TruthBoson",
        "TruthBottom",
        "TruthTop",
    ]

    mixins: tp.ClassVar = {
        "Photons": "Particle",
        "Electrons": "Electron",
        "Muons": "Muon",
        "Jets": "Particle",
        "TauJets": "Particle",
        "CombinedMuonTrackParticles": "TrackParticle",
        "ExtrapolatedMuonTrackParticles": "TrackParticle",
        "GSFTrackParticles": "TrackParticle",
        "InDetTrackParticles": "TrackParticle",
        "MuonSpectrometerTrackParticles": "TrackParticle",
        "egammaClusters": "NanoCollection",
        **dict.fromkeys(truth_collections, "TruthParticle"),
    }

    def __call__(self, array: awkward.Array) -> awkward.Array:
        n_events = int(awkward.num(array, axis=0))

        # group (sub_key, layout) by collection name
        zip_groups = defaultdict(list)
        has_eventindex = defaultdict(bool)
        for key in array.fields:
            layout = _non_materializing_get_field(array, key).layout
            if isinstance(layout, awkward.contents.RecordArray) and not layout.fields:
                # skip empty records (branches containing only the base class)
                continue
            key_fields = key.split("/")[-1].split(".")
            top_key = key_fields[0]
            sub_key = ".".join(key_fields[1:])
            objname = (
                top_key.replace("Analysis", "").replace("AuxDyn", "").replace("Aux", "")
            )
            zip_groups[objname].append((sub_key, layout))

            # add a single _eventindex column per collection, from the first
            # single-jagged column
            if (
                not has_eventindex[objname]
                and isinstance(layout, awkward.contents.ListOffsetArray)
                and not isinstance(layout.content, _LIST_CLASSES)
            ):
                ev = awkward.contents.ListOffsetArray(
                    layout.offsets, eventindex_content(layout.offsets)
                )
                zip_groups[objname].append(("_eventindex", ev))
                has_eventindex[objname] = True

        contents = {}
        for objname, items in zip_groups.items():
            if len(items) == 1:
                contents[objname] = items[0][1]
                continue

            # reconstitute split ElementLink parents (e.g. ambiguityLink from
            # ambiguityLink.m_persKey / ambiguityLink.m_persIndex); like coffea this
            # is done unconditionally and the result replaces a same-named field
            to_collect = defaultdict(list)
            for sk, layout in items:
                if "." in sk:
                    skleft, skright = sk.split(".", 1)
                    to_collect[skleft].append((skright, layout))
            reconstituted = {
                skleft: _zip_leaves(dict(leaves))
                for skleft, leaves in to_collect.items()
            }

            # build the mapping of fields to zip (all share the collection offsets)
            to_zip = {}
            for sk, layout in items:
                if "." in sk:
                    # the reconstituted parent takes the place of its first leaf
                    # (coffea sees the parent branch at that position)
                    parent = sk.split(".", 1)[0]
                    to_zip.setdefault(parent, reconstituted[parent])
                    continue
                field_layout = layout
                if isinstance(layout, awkward.contents.RecordArray) and layout.fields:
                    # single-jagged ElementLink stored as RecordArray(ListOffsetArray):
                    # convert to ListOffsetArray(RecordArray)
                    fields = [f.split(".")[-1] for f in layout.fields]
                    field_layout = _zip_leaves(
                        dict(zip(fields, layout.contents, strict=True))
                    )
                to_zip[sk] = field_layout

            mixin = self.mixins.get(objname)
            try:
                if mixin == "TrackParticle":
                    offsets = self._collection_offsets(to_zip)
                    to_zip["p"] = self._wrap(
                        offsets, qoverp_to_p(to_zip["qOverP"].content)
                    )
                    to_zip["pt"] = self._wrap(
                        offsets,
                        qoverp_theta_to_pt(
                            to_zip["qOverP"].content, to_zip["theta"].content
                        ),
                    )
                    to_zip["tau"] = self._wrap(
                        offsets,
                        full_like_from_content(to_zip["theta"].content, 139.570),
                    )
                if mixin == "Muon" and "m" not in to_zip:
                    offsets = self._collection_offsets(to_zip)
                    to_zip["m"] = self._wrap(
                        offsets, full_like_from_content(to_zip["pt"].content, 105.658)
                    )
                contents[objname] = self._zip_collection(to_zip, objname, mixin)
            except (KeyError, NotImplementedError):
                warnings.warn(f"Can't zip collection {objname}", stacklevel=2)

        _contents = tuple(contents.values())
        _fields = tuple(contents.keys())
        nanoevents = awkward.Array(
            awkward.contents.RecordArray(_contents, _fields, length=n_events),
            behavior=self.behavior(),
        )
        nanoevents = awkward.with_name(_rewrap(nanoevents), name="NanoEvents")
        nanoevents.attrs["@original_array"] = nanoevents
        return nanoevents

    @staticmethod
    def _collection_offsets(to_zip):
        for layout in to_zip.values():
            if isinstance(layout, awkward.contents.ListOffsetArray):
                return layout.offsets
        raise NotImplementedError

    @staticmethod
    def _wrap(offsets, content):
        return awkward.contents.ListOffsetArray(offsets, content)

    @staticmethod
    def _zip_collection(to_zip, objname, mixin):
        names = list(to_zip.keys())
        layouts = list(to_zip.values())
        params = {"collection_name": objname}
        if mixin is not None:
            params["__record__"] = mixin
        if all(
            isinstance(layout, awkward.contents.ListOffsetArray) for layout in layouts
        ):
            # jagged collection: share offsets, use each field's flat content
            offsets = layouts[0].offsets
            record = awkward.contents.RecordArray(
                [layout.content for layout in layouts],
                names,
                length=_total_items(offsets),
                parameters=params,
            )
            return awkward.contents.ListOffsetArray(offsets=offsets, content=record)
        # mixed/flat collection: keep fields as-is in a per-event record
        length = awkward._util.maybe_length_of(layouts[0])
        return awkward.contents.RecordArray(
            layouts, names, length=length, parameters=params
        )

    @classmethod
    def behavior(cls):
        """Behaviors necessary to implement this schema (dict)"""
        from awkward_zipper.behaviors import physlite

        return physlite.behavior
