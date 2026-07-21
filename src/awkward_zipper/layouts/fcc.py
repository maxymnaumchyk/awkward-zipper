import re
import typing as tp

import awkward

from awkward_zipper.awkward_util import (
    _non_materializing_get_field,
    _record_length,
    _rewrap,
)
from awkward_zipper.kernels import (
    begin_end_mapping,
    grow_local_index_to_target_shape,
    local2global,
    nested_local2global,
)
from awkward_zipper.layouts.base import BaseLayoutBuilder
from awkward_zipper.layouts.edm4hep import EDM4HEP, sort_dict

_idxs = re.compile(r".*[#]+[0-9]+")
_trailing_under = re.compile(r".*_[0-9]")
_square_braces = re.compile(r".*\[.*\]")


def _zip_shared_offsets(members, record_name=None, parameters=None, offsets=None):
    """Zip layouts that share per-event offsets into one jagged record."""
    names = list(members.keys())
    layouts = list(members.values())
    if offsets is None:
        offsets = layouts[0].offsets
    contents = [layout.content for layout in layouts]
    params = {}
    if record_name is not None:
        params["__record__"] = record_name
    if parameters:
        params.update(parameters)
    record = awkward.contents.RecordArray(
        contents, names, length=_record_length(contents), parameters=params
    )
    return awkward.contents.ListOffsetArray(offsets=offsets, content=record)


class FCCSchema(BaseLayoutBuilder):
    """FCC (pre-edm4hep1) layout builder.

    Array-based re-implementation of coffea's ``FCCSchema``, tested against the
    Spring2021/Winter2023 pregenerated FCC samples (https://fcc-physics-events.web.cern.ch/).
    These follow the edm4hep structure:

    - vector components (``X.referencePoint.{x,y,z}``) are zipped into
      ``X.referencePoint`` sub-collections,
    - ``*_begin``/``*_end`` range pairs are zipped, and MC parent/daughter relations
      get ``*_ranges`` plus global (``G``) indexers,
    - ObjectID branches (``X#0.index``/``X#0.collectionID``) become ``Xidx0``
      collections,
    - ``*_[0-9]`` trailing-underscore branches form their own collections,
    - the remaining ``X.*`` branches are zipped into the ``X`` collection, using
      ``momentum.{x,y,z}`` (+ ``energy``) for the LorentzVector behavior.
    """

    extra_mixins: tp.ClassVar = {"*idx": "ObjectID"}

    _momentum_fields_e: tp.ClassVar = {
        "energy": "E",
        "momentum.x": "px",
        "momentum.y": "py",
        "momentum.z": "pz",
    }
    _replacement: tp.ClassVar = {**_momentum_fields_e}

    _threevec_fields: tp.ClassVar = {
        "position": ["position.x", "position.y", "position.z"],
        "directionError": ["directionError.x", "directionError.y", "directionError.z"],
        "vertex": ["vertex.x", "vertex.y", "vertex.z"],
        "endpoint": ["endpoint.x", "endpoint.y", "endpoint.z"],
        "referencePoint": [
            "referencePoint.x",
            "referencePoint.y",
            "referencePoint.z",
        ],
        "momentumAtEndpoint": [
            "momentumAtEndpoint.x",
            "momentumAtEndpoint.y",
            "momentumAtEndpoint.z",
        ],
        "spin": ["spin.x", "spin.y", "spin.z"],
    }

    all_cross_references: tp.ClassVar = {
        "MCRecoAssociations#1.index": "Particle",
        "MCRecoAssociations#0.index": "ReconstructedParticles",
        "Muon#0.index": "ReconstructedParticles",
        "Electron#0.index": "ReconstructedParticles",
    }

    mc_relations: tp.ClassVar = {
        "parents": "Particle#0.index",
        "daughters": "Particle#1.index",
    }

    def __call__(self, array: awkward.Array, typenames=None) -> awkward.Array:
        n_events = int(awkward.num(array, axis=0))

        forms = {}
        for field in array.fields:
            layout = _non_materializing_get_field(array, field).layout
            key = f"{field.split('.')[0]}/{field}" if "." in field else field
            forms[key] = layout

        self._create_mixin(forms, typenames)
        output = self._build_collections(forms)

        contents = tuple(output.values())
        names = tuple(output.keys())
        nanoevents = awkward.Array(
            awkward.contents.RecordArray(contents, names, length=n_events),
            behavior=self.behavior(),
        )
        nanoevents = awkward.with_name(_rewrap(nanoevents), name="NanoEvents")
        nanoevents.attrs["@original_array"] = nanoevents
        return nanoevents

    def _create_mixin(self, forms, typenames):
        all_collections = {key.split("/")[0] for key in forms if "/" in key}
        collections = {
            name
            for name in all_collections
            if not _idxs.match(name) and not _trailing_under.match(name)
        }
        mixins = {}
        for name in collections:
            datatype = (typenames or {}).get(name)
            if datatype is None:
                mixins[name] = "NanoCollection"
            elif datatype.startswith(r"vector<edm4hep::"):
                mixins[name] = (
                    datatype.split("::")[-1][:-5]
                    if datatype.endswith("Data>")
                    else "NanoCollection"
                )
            elif datatype.startswith(r"vector<podio::"):
                mixins[name] = datatype.split("::")[-1][:-1]
            else:
                mixins[name] = datatype
        self.mixins_dictionary = {**mixins, **self.extra_mixins}

    # ---------------- processors ----------------

    def _create_subcollections(self, forms, all_collections):
        field_names = list(forms)

        # python-friendly names for square-braced branches
        for name in field_names:
            if _square_braces.match(name):
                forms[name.replace("[", "_").replace("]", "_")] = forms.pop(name)

        field_names = list(forms)

        # zip *_begin / *_end pairs, adding MC parent/daughter range indexers
        begin_end = set()
        for fullname in field_names:
            if fullname.endswith("_begin"):
                begin_end.add(fullname.split("_begin")[0])
            elif fullname.endswith("_end"):
                begin_end.add(fullname.split("_end")[0])

        for name in begin_end:
            content = {
                k[len(name) + 1 :]: forms.pop(k)
                for k in field_names
                if k.startswith(name) and k in forms
            }
            if not content:
                continue
            offsets = next(iter(content.values())).offsets
            begin = next((v for k, v in content.items() if k.endswith("begin")), None)
            end = next((v for k, v in content.items() if k.endswith("end")), None)

            ranges = {}
            if begin is not None and end is not None:
                for key, target in self.mc_relations.items():
                    col_name = target.split(".")[0]
                    target_key = f"{col_name}/{target}"
                    if name.endswith(key) and target_key in forms:
                        range_name = f"{col_name.replace('#', 'idx')}_ranges"
                        mapped = begin_end_mapping(
                            begin, end, forms[target_key].content
                        )
                        ranges[range_name] = mapped
                        ranges[range_name + "G"] = awkward.contents.ListOffsetArray(
                            mapped.offsets,
                            awkward.contents.ListOffsetArray(
                                mapped.content.offsets,
                                nested_local2global(mapped, offsets),
                            ),
                        )
            forms[name] = _zip_shared_offsets(
                sort_dict({**content, **ranges}), offsets=offsets
            )

        # zip colorFlow.a / colorFlow.b
        field_names = list(forms)
        color = set()
        for name in field_names:
            if name.endswith("colorFlow.a"):
                color.add(name.split(".a")[0])
            elif name.endswith("colorFlow.b"):
                color.add(name.split(".b")[0])
        for name in color:
            content = {
                k[len(name) + 1 :]: forms.pop(k)
                for k in field_names
                if k.startswith(name) and k in forms
            }
            if content:
                forms[name] = _zip_shared_offsets(sort_dict(content))

        # three-vectors
        field_names = list(forms)
        for name in all_collections:
            for threevec_name, subfields in self._threevec_fields.items():
                if all(f"{name}/{name}.{sub}" in field_names for sub in subfields):
                    content = {
                        axis: forms.pop(f"{name}/{name}.{threevec_name}.{axis}")
                        for axis in ("x", "y", "z")
                    }
                    forms[f"{name}/{name}.{threevec_name}"] = _zip_shared_offsets(
                        sort_dict(content), record_name="ThreeVector"
                    )
        return forms

    def _global_indexers(self, forms, all_collections):
        for cross_ref, target in self.all_cross_references.items():
            collection_name, index_name = cross_ref.split(".")
            source_key = f"{collection_name}/{collection_name}.{index_name}"
            if source_key not in forms:
                continue
            available = [k for k in forms if k.startswith(f"{target}/{target}.")]
            if not available:
                continue
            target_offsets = forms[available[0]].offsets
            index = forms[source_key]
            grown = grow_local_index_to_target_shape(index, target_offsets)
            grown_jagged = awkward.contents.ListOffsetArray(target_offsets, grown)
            replaced = collection_name.replace("#", "idx")
            forms[f"{target}/{target}.{replaced}_{index_name}Global"] = (
                awkward.contents.ListOffsetArray(
                    target_offsets, local2global(grown_jagged, target_offsets)
                )
            )
        return forms

    def _idx_collections(self, output, forms, all_collections):
        field_names = list(forms)
        idxs = {k.split("/")[0] for k in all_collections if _idxs.match(k)}
        for k in field_names:
            if _idxs.match(k) and "/" not in k:
                forms.pop(k, None)

        for idx in idxs:
            repl = idx.replace("#", "idx")
            content = {
                k[2 * len(idx) + 2 :]: forms.pop(k)
                for k in field_names
                if k.startswith(f"{idx}/{idx}.") and k in forms
            }
            if not content:
                continue
            output[repl] = _zip_shared_offsets(
                sort_dict(content),
                record_name=self.mixins_dictionary.get("*idx", "NanoCollection"),
                parameters={"collection_name": repl},
            )

        # MCRecoAssociations idx0/idx1 join later as reco/mc
        if "MCRecoAssociationsidx0" in output and "MCRecoAssociationsidx1" in output:
            forms["MCRecoAssociations/MCRecoAssociations.reco"] = output.pop(
                "MCRecoAssociationsidx0"
            )
            forms["MCRecoAssociations/MCRecoAssociations.mc"] = output.pop(
                "MCRecoAssociationsidx1"
            )
        return output, forms

    def _trailing_underscore_collections(self, output, forms, all_collections):
        collections = {n for n in all_collections if _trailing_under.match(n)}
        for name in collections:
            forms.pop(name, None)
            field_names = list(forms)
            content = {
                k[2 * len(name) + 2 :]: forms.pop(k)
                for k in field_names
                if k.startswith(f"{name}/{name}.") and k in forms
            }
            if not content:
                continue
            output[name] = _zip_shared_offsets(
                sort_dict(content),
                record_name=self.mixins_dictionary.get(name, "NanoCollection"),
                parameters={"collection_name": name},
            )
        return output, forms

    def _main_collections(self, output, forms, all_collections):
        field_names = list(forms)
        collections = {
            n
            for n in all_collections
            if not _idxs.match(n) and not _trailing_under.match(n)
        }
        for name in collections:
            mixin = self.mixins_dictionary.get(name, "NanoCollection")
            content = {
                k[2 * len(name) + 2 :]: forms.pop(k)
                for k in field_names
                if k.startswith(f"{name}/{name}.") and k in forms
            }
            if not content:
                continue
            content = {self._replacement.get(k, k): v for k, v in content.items()}
            if mixin == "ReconstructedParticle":
                content.pop("E", None)
            output[name] = _zip_shared_offsets(
                sort_dict(content),
                record_name=mixin,
                parameters={"collection_name": name},
            )
            forms.pop(name, None)
        return output, forms

    def _unknown_collections(self, output, forms, all_collections):
        for name, layout in list(forms.items()):
            record = getattr(layout, "content", None)
            if isinstance(record, awkward.contents.RecordArray) and not record.fields:
                forms.pop(name)
                continue
            if isinstance(layout, awkward.contents.RecordArray) and not layout.fields:
                forms.pop(name)
                continue
            output[name] = forms.pop(name)
        return output, forms

    def _build_collections(self, forms):
        all_collections = {name.split("/")[0] for name in forms if "/" in name}

        forms = self._create_subcollections(forms, all_collections)
        forms = self._global_indexers(forms, all_collections)

        output = {}
        output, forms = self._idx_collections(output, forms, all_collections)
        output, forms = self._trailing_underscore_collections(
            output, forms, all_collections
        )
        output, forms = self._main_collections(output, forms, all_collections)
        output, forms = self._unknown_collections(output, forms, all_collections)
        return sort_dict(output)

    @classmethod
    def behavior(cls):
        """Behaviors necessary to implement this schema (dict)"""
        from awkward_zipper.behaviors import base, fcc, vector

        behavior = {}
        behavior.update(base.behavior)
        behavior.update(vector.behavior)
        behavior.update(fcc.behavior)
        return behavior


class FCCSchema_edm4hep1(EDM4HEP):
    """FCC layout builder for samples produced with edm4hep >= 1.

    Inherits from :class:`~awkward_zipper.EDM4HEP` (matching coffea's
    ``FCCSchema_edm4hep1(EDM4HEPSchema)``).

    .. note::
       ``copy_links_to_target_datatype`` is supported by the builder but left
       disabled here. Coffea copies a Link branch onto its target collection at the
       *form* level without checking shapes, which yields a record whose declared
       length exceeds the copied buffer (e.g. 625 vs 620 items). That is fine for
       coffea's lazy per-branch mapping but is not a valid awkward layout -- it
       fails to round-trip through ``ak.from_buffers``. Enable it once the shapes
       are reconciled upstream.
    """

    copy_links_to_target_datatype = False
    _datatype_priority: tp.ClassVar = {
        "ReconstructedParticle": "ReconstructedParticles"
    }

    @classmethod
    def behavior(cls):
        """Behaviors necessary to implement this schema (dict)"""
        from awkward_zipper.behaviors import base, fcc, vector

        behavior = {}
        behavior.update(base.behavior)
        behavior.update(vector.behavior)
        behavior.update(fcc.behavior_edm4hep1)
        return behavior


class FCC:
    """Choose the required variant of the FCC layout builder.

    Example usage::

        from awkward_zipper import FCC
        FCC.get_schema(version="latest")

    Available versions:

    - ``"latest"`` -> :class:`FCCSchema_edm4hep1`
    - ``"pre-edm4hep1"`` -> :class:`FCCSchema`
    - ``"edm4hep1"`` -> :class:`FCCSchema_edm4hep1`
    """

    def __init__(self, version="latest"):
        self._version = version

    @classmethod
    def get_schema(cls, version="latest"):
        if version in ("latest", "edm4hep1"):
            return FCCSchema_edm4hep1
        if version == "pre-edm4hep1":
            return FCCSchema
        msg = (
            f"Unknown FCC version {version}. "
            "Available: 'latest', 'pre-edm4hep1', 'edm4hep1'."
        )
        raise ValueError(msg)
