import typing as tp

import awkward

from awkward_zipper.awkward_util import (
    _append_record_fields,
    _check_equal_lengths,
    _jagged_content,
    _jagged_offsets,
    _non_materializing_get_field,
    _rewrap,
    _zip_jagged,
)
from awkward_zipper.kernels import counts2offsets
from awkward_zipper.layouts.base import BaseLayoutBuilder


class TreeMaker(BaseLayoutBuilder):
    """TreeMaker layout builder

    The TreeMaker layout is built from all branches found in the supplied file,
    based on the naming pattern of the branches. There are two steps to the
    generation of the array collections:

    - Objects with vector-like quantities (momentum, coordinate points) in the
      TreeMaker n-tuples are stored using ROOT ``PtEtaPhiEVector`` and
      ``XYZPoint`` classes with maximum TTree splitting. These split branches
      (``<Object>.fCoordinates.f{Pt,Eta,Phi,E}`` and
      ``<Object>.fCoordinates.f{X,Y,Z}``) are grouped into a single collection
      with the original object name, mapping the coordinate variables to the
      standard names used by the vector behaviors (``pt``, ``eta``, ``phi``,
      ``energy`` and ``x``, ``y``, ``z``).

    - Extended quantities of physics objects are stored as ``<Object>_<variable>``
      (e.g. ``Jets_jecFactor``) and are merged into the collection ``<Object>``.

    Sub-collections, signalled by a ``<Object>_<subcol>Counts`` branch, are
    nested as doubly-jagged arrays inside their parent collection.

    All collections are then zipped into one ``base.NanoEvents`` record.
    """

    # coordinate members of the ROOT composite vector classes and the field
    # name they are mapped to for the vector behaviors
    _lorentz_map: tp.ClassVar = {
        "pt": "fPt",
        "eta": "fEta",
        "phi": "fPhi",
        "energy": "fE",
    }
    _threevec_map: tp.ClassVar = {"x": "fX", "y": "fY", "z": "fZ"}

    def __call__(self, array: awkward.Array) -> awkward.Array:
        fields = list(array.fields)
        n_events = int(awkward.num(array, axis=0))

        # working dict of name -> low-level layout (Content); raw branches and
        # any collections built along the way live here together
        forms = {f: _non_materializing_get_field(array, f).layout for f in fields}

        self._build_composite_objects(forms)
        subcollections = self._build_collections(forms)
        self._nest_subcollections(forms, subcollections)

        # final (outermost) zip into NanoEvents
        contents = tuple(forms.values())
        names = tuple(forms.keys())
        nanoevents = awkward.Array(
            awkward.contents.RecordArray(contents, names, length=n_events),
            behavior=self.behavior(),
        )
        nanoevents = awkward.with_name(_rewrap(nanoevents), name="NanoEvents")
        nanoevents.attrs["@original_array"] = nanoevents
        return nanoevents

    def _build_composite_objects(self, forms):
        """Zip the split ROOT vector branches into vector-like collections."""
        # any split (dotted) branch is a composite object; a skimmed n-tuple stores
        # the object under a trailing underscore (``Jets_.fCoordinates.fPt``)
        composite_objects = sorted(
            {k.split(".")[0].rstrip("_") for k in forms if "." in k}
        )
        for objname in composite_objects:
            components = {
                k.split(".")[-1]: k
                for k in list(forms)
                if k.startswith((objname + ".", objname + "_."))
            }
            present = set(components)
            if present == set(self._lorentz_map.values()):
                mapping, record_name = self._lorentz_map, "PtEtaPhiELorentzVector"
            elif present == set(self._threevec_map.values()):
                mapping, record_name = self._threevec_map, "ThreeVector"
            else:
                msg = (
                    f"Unrecognized class with split branches of object "
                    f"{objname}: {list(components.values())}"
                )
                raise ValueError(msg)

            first_key = components[next(iter(mapping.values()))]
            offsets = _jagged_offsets(forms[first_key])
            members = {
                out: _jagged_content(forms.pop(components[src]))
                for out, src in mapping.items()
            }
            forms[objname] = _zip_jagged(members, offsets, record_name=record_name)

    def _build_collections(self, forms):
        """Merge/zip ``<Object>_<var>`` branches into their collections.

        Returns the list of discovered sub-collections to nest afterwards.
        """
        collection_names = [k for k in forms if "_" in k and not k.startswith("n")]
        collection_names = sorted(
            {
                "_".join(k.split("_")[:-1])
                for k in collection_names
                # exclude per-event variables with AK8 variants (Mjj, MT, ...)
                if k.split("_")[-1] != "AK8"
            },
            key=lambda name: name.count("_"),
            reverse=True,
        )

        subcollections = []
        for cname in collection_names:
            items = sorted(k for k in forms if k.startswith(cname + "_"))
            if len(items) == 0:
                continue

            # split off <collection>_<subcol>Counts sub-collections
            countitems = [x for x in items if x.endswith("Counts")]
            subcols = {x[:-6] for x in countitems}
            for subcol in subcols:
                items = [
                    k for k in items if not k.startswith(subcol) or k.endswith("Counts")
                ]
                subname = subcol[len(cname) + 1 :]
                subcollections.append(
                    {
                        "colname": cname,
                        "subcol": subcol,
                        "countname": subname + "Counts",
                        "subname": subname,
                    }
                )

            if cname in forms:
                if not isinstance(forms[cname], awkward.contents.ListOffsetArray):
                    msg = f"{cname} isn't a jagged array, not sure what to do"
                    raise NotImplementedError(msg)
                new_members = {
                    k[len(cname) + 1 :]: _jagged_content(forms.pop(k)) for k in items
                }
                forms[cname] = _append_record_fields(forms[cname], new_members)
            else:
                # pure "_"-grouped collection with no composite base
                members = {k[len(cname) + 1 :]: forms.pop(k) for k in items}
                forms[cname] = self._zip_members(members)

        return subcollections

    @staticmethod
    def _zip_members(members):
        """Zip member layouts into one record (coffea's ``zip_forms``).

        Jagged members share the offsets of the first one (all share the same
        per-event counts); anything else is zipped into a flat per-event record.
        """
        layouts = list(members.values())
        if all(
            isinstance(layout, awkward.contents.ListOffsetArray) for layout in layouts
        ):
            offsets = _jagged_offsets(layouts[0])
            contents = {k: _jagged_content(v) for k, v in members.items()}
            return _zip_jagged(contents, offsets)
        return awkward.contents.RecordArray(
            layouts, list(members.keys()), length=_check_equal_lengths(layouts)
        )

    def _nest_subcollections(self, forms, subcollections):
        for sub in subcollections:
            parent = forms[sub["colname"]]
            child = forms.pop(sub["subcol"])
            record = parent.content
            counts_content = record.contents[record.fields.index(sub["countname"])]
            inner_offsets = counts2offsets(awkward.Array(counts_content))
            inner = awkward.contents.ListOffsetArray(
                offsets=awkward.index.Index(inner_offsets),
                content=child.content,
            )
            forms[sub["colname"]] = _append_record_fields(
                parent, {sub["subname"]: inner}
            )

    @classmethod
    def behavior(cls):
        """Behaviors necessary to implement this schema (dict)"""
        from awkward_zipper.behaviors import treemaker

        return treemaker.behavior
