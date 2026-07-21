import copy
import typing as tp

import awkward

from awkward_zipper.assets import edm4hep_ver
from awkward_zipper.awkward_util import (
    _non_materializing_get_field,
    _rewrap,
)
from awkward_zipper.kernels import (
    begin_end_mapping,
    local2global,
    nested_local2global,
    regular_to_jagged,
)
from awkward_zipper.layouts.base import BaseLayoutBuilder


def parse_members_and_relations(members_and_relation_list, target_text=False):
    """Convert the raw yaml Members/Relations lists into a workable mapping."""
    parsed = {}
    for i in members_and_relation_list:
        separated = i.split("//", 1)
        declaration = separated[0].strip()
        doc_str = separated[1].strip() if len(separated) > 1 else ""

        type_str = declaration.split()[0]
        name_str = declaration.split()[1]
        if ("::" in declaration) and ("<" in declaration) and (">" in declaration):
            type_str = declaration.split(">", 1)[0] + ">"
            name_str = declaration.split(">", 1)[1]

        parsed[name_str.strip()] = {"type": type_str.strip(), "doc": doc_str}
        if target_text:
            parsed[name_str.strip()]["target"] = type_str.strip().split("::")[1]
    return parsed


def parse_yaml(loaded_dict, parsed_dict):
    """Post-process the loaded edm4hep yaml into the structure the builder needs."""
    for key in loaded_dict:
        if not isinstance(loaded_dict[key], dict):
            continue
        for subkey in loaded_dict[key]:
            if not isinstance(loaded_dict[key][subkey], dict):
                continue
            for subsubkey in loaded_dict[key][subkey]:
                if subsubkey in ["Members", "VectorMembers"]:
                    parsed_dict[key][subkey][subsubkey] = parse_members_and_relations(
                        loaded_dict[key][subkey][subsubkey]
                    )
                elif subsubkey in ["OneToOneRelations", "OneToManyRelations"]:
                    parsed_dict[key][subkey][subsubkey] = parse_members_and_relations(
                        loaded_dict[key][subkey][subsubkey], target_text=True
                    )
    # podio::ObjectID, kept under the edm4hep namespace for parsing compatibility
    parsed_dict["datatypes"]["edm4hep::ObjectID"] = {
        "Description": "podio ObjectID",
        "Members": {
            "index": {"type": "int64", "doc": "indices to the target collection"},
            "collectionID": {"type": "int64", "doc": "collection ID"},
        },
    }
    return parsed_dict


def sort_dict(d):
    return {k: d[k] for k in sorted(d)}


def _offsets_length(offsets, contents):
    """Item count implied by ``offsets`` (``offsets[-1]``), lazily.

    Contents may legitimately differ in length -- coffea copies Link branches onto
    target collections whose item counts differ slightly -- so the collection's own
    offsets define the record length rather than requiring equal lengths.
    """
    if not all(c.is_all_materialized for c in contents):
        return awkward._nplikes.shape.unknown_length
    data = offsets.data if isinstance(offsets, awkward.index.Index) else offsets
    # a copied Link can be shorter than the collection it is attached to; keep the
    # record valid by never exceeding the shortest content
    return min(int(data[-1]), *(c.length for c in contents))


def _zip_shared_offsets(members, record_name=None, parameters=None):
    """Zip layouts that share per-event offsets into one jagged record."""
    names = list(members.keys())
    layouts = list(members.values())
    offsets = layouts[0].offsets
    contents = [layout.content for layout in layouts]
    params = {}
    if record_name is not None:
        params["__record__"] = record_name
    if parameters:
        params.update(parameters)
    record = awkward.contents.RecordArray(
        contents, names, length=_offsets_length(offsets, contents), parameters=params
    )
    return awkward.contents.ListOffsetArray(offsets=offsets, content=record)


class EDM4HEP(BaseLayoutBuilder):
    """EDM4HEP layout builder (edm4hep 00.99.01).

    Array-based re-implementation of coffea's ``EDM4HEPSchema``. The layout is
    driven by the vendored EDM4HEP yaml data model, which describes each
    datatype's Members, VectorMembers, OneToOneRelations and OneToManyRelations:

    - component members (``Vector3f``/``Vector4f``/...) such as
      ``X.position.{x,y,z}`` are zipped into an ``X.position`` sub-record,
    - ``VectorMembers`` and ``OneToManyRelations`` stored as flat per-event arrays
      plus per-item ``{member}_begin``/``{member}_end`` ranges are regrouped into
      doubly-jagged arrays,
    - ``OneToOneRelations`` and Links get a global index (``index_Global``) into
      their target collection,
    - the remaining ``X.*`` branches are zipped into the ``X`` collection.

    The collection datatypes come from the TTree's branch typenames, which are not
    present in ``tree.arrays()`` output, so pass them explicitly::

        EDM4HEP()(tree.arrays(...), typenames=tree.typenames())

    If omitted, the datatypes are inferred by matching each collection's member
    branch names against the yaml data model.
    """

    edm4hep_version = "00-99-01"

    _components_mixins: tp.ClassVar = {
        "Vector4f": "LorentzVector",
        "Vector3f": "ThreeVector",
        "Vector3d": "ThreeVector",
        "Vector2i": "TwoVector",
        "Vector2f": "TwoVector",
        "TrackState": "TrackState",
        "Quantity": "Quantity",
        "covMatrix2f": "covMatrix",
        "covMatrix3f": "covMatrix",
        "covMatrix4f": "covMatrix",
        "covMatrix6f": "covMatrix",
    }

    extra_mixins: tp.ClassVar = {"*idx": "ObjectID"}

    _momentum_fields_e: tp.ClassVar = {
        "energy": "E",
        "momentum.x": "px",
        "momentum.y": "py",
        "momentum.z": "pz",
    }
    _two_vec_replacement: tp.ClassVar = {"a": "x", "b": "y"}
    _replacement: tp.ClassVar = {**_momentum_fields_e, **_two_vec_replacement}

    copy_links_to_target_datatype = False
    _datatype_priority: tp.ClassVar = {}

    def __call__(self, array: awkward.Array, typenames=None) -> awkward.Array:
        self.edm4hep = edm4hep_ver[self.edm4hep_version]()
        self.parsed_edm4hep = parse_yaml(self.edm4hep, copy.deepcopy(self.edm4hep))

        n_events = int(awkward.num(array, axis=0))

        # Work with coffea-style keys ("X/X.y") so the branch-name logic matches
        # coffea's; uproot's `tree.arrays()` gives them as "X.y".
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

    # ---------------- datatype mixins ----------------

    def _infer_datatype(self, name, forms):
        """Infer a collection's edm4hep datatype from its member branch names."""
        members = {
            key.split("/")[1][len(name) + 1 :].split(".")[0]
            for key in forms
            if key.startswith(f"{name}/{name}.")
        }
        best, best_score = None, 0.0
        for dt_name, dt in self.parsed_edm4hep["datatypes"].items():
            declared = set(dt.get("Members", {}))
            for rel in ("VectorMembers", "OneToOneRelations", "OneToManyRelations"):
                declared |= set(dt.get(rel, {}))
                declared |= {f"{k}_begin" for k in dt.get(rel, {})}
                declared |= {f"{k}_end" for k in dt.get(rel, {})}
            if not declared:
                continue
            overlap = len(declared & members)
            if overlap == 0:
                continue
            # prefer the datatype that explains the most of what is present while
            # declaring the fewest members that are absent
            score = overlap - 0.5 * len(declared - members)
            if score > best_score:
                best, best_score = dt_name, score
        return best

    def _create_mixin(self, forms, typenames):
        all_collections = {key.split("/")[0] for key in forms if "/" in key}
        collections = {c for c in all_collections if not c.startswith("_")}

        mixins = {}
        for name in collections:
            datatype = (typenames or {}).get(name)
            if datatype is None:
                inferred = self._infer_datatype(name, forms)
                mixins[name] = (
                    inferred.split("::")[-1] if inferred else "edm4hep_nanocollection"
                )
                continue
            if datatype.startswith(r"vector<edm4hep::"):
                if not datatype.endswith("Data>"):
                    msg = f"Unknown datatype: {datatype}"
                    raise RuntimeError(msg)
                mixins[name] = datatype.split("::")[-1][:-5]
            elif datatype.startswith(r"vector<podio::"):
                mixins[name] = datatype.split("::")[-1][:-1]
            else:
                mixins[name] = datatype

        self._datatype_mixins = {**mixins, **self.extra_mixins}

    def _datatype_spec(self, datatype):
        """yaml spec for a datatype, or None when it is not a real edm4hep type."""
        if datatype is None:
            return None
        return self.parsed_edm4hep["datatypes"].get("edm4hep::" + datatype)

    def _lookup_branch(self, collection_name, branch_name, key=None):
        datatype = self._datatype_mixins.get(collection_name)
        if collection_name.startswith("_"):
            col_name = collection_name[1:].split("_")[0]
            subcol_name = collection_name[1:].split("_")[-1]
            datatype = self._datatype_mixins.get(col_name)
        if datatype is None:
            return {"type": "unknown", "doc": "unknown"} if key is None else "unknown"
        collection_edm4hep = self.parsed_edm4hep["datatypes"].get(
            "edm4hep::" + datatype, {}
        )
        composite = {
            **collection_edm4hep.get("Members", {}),
            **collection_edm4hep.get("VectorMembers", {}),
            **collection_edm4hep.get("OneToOneRelations", {}),
        }
        if collection_name.startswith("_"):
            matched = composite.get(subcol_name, {"type": "unknown"})
            composite = {
                **composite,
                **self.parsed_edm4hep["components"].get(
                    matched["type"], {"Members": {}}
                )["Members"],
            }
        found = composite.get(branch_name, {"type": "unknown", "doc": "unknown"})
        return found[key] if key is not None else found

    # ---------------- processors ----------------

    def _zip_components(self, collection_name, component_branches, forms):
        inverted = {}
        for name, info in component_branches.items():
            var, subvar = info["branch_var"], info["branch_subvar"]
            inverted.setdefault(f"{var}@{info['type']}", []).append(
                {"name": name, "branch_subvar": subvar}
            )

        for var, branch_list in inverted.items():
            assign_name, type_str = var.split("@")
            if assign_name == "momentum" or type_str == "unknown":
                continue
            type_name = type_str.split("::")[-1]
            mixin = self._components_mixins.get(type_name)

            to_zip_raw = {
                item["branch_subvar"]: forms.pop(item["name"]) for item in branch_list
            }
            to_zip = {self._replacement.get(n, n): f for n, f in to_zip_raw.items()}
            key = f"{collection_name}/{collection_name}.{assign_name}"
            forms[key] = _zip_shared_offsets(
                sort_dict(to_zip),
                record_name=mixin,
                parameters={"collection_name": assign_name},
            )
        return forms

    def _process_components(self, forms, all_collections):
        def _process(forms):
            for collection in all_collections:
                component_branches = {}
                for name in list(forms):
                    slash = name.split("/")
                    if slash[0] != collection or len(slash) <= 1:
                        continue
                    parts = slash[1].split(".")
                    if len(parts) > 2:
                        branch_var, branch_subvar = parts[-2], parts[-1]
                        if branch_var == "momentum":
                            continue
                        component = self._lookup_branch(collection, branch_var)
                        component_branches[name] = {
                            "type": component["type"],
                            "branch_var": branch_var,
                            "branch_subvar": branch_subvar,
                        }
                forms = self._zip_components(collection, component_branches, forms)
            return forms

        # twice, to resolve nested components
        return _process(_process(forms))

    def _target_offsets(self, matched_collection, forms):
        """Per-event offsets of a target collection (from its first member branch)."""
        datatype = self._datatype_mixins.get(matched_collection)
        spec = self._datatype_spec(datatype)
        if spec is None:
            return None
        first_var = next(iter(spec["Members"]))
        key = f"{matched_collection}/{matched_collection}.{first_var}"
        return forms[key].offsets

    def _matched_collections(self, target_datatype):
        matched = [
            name
            for name, datatype in self._datatype_mixins.items()
            if "edm4hep::" + datatype == target_datatype
        ]
        if matched:
            return matched
        # Interfaces are only consulted for schema_version "2". Note coffea compares
        # against the *string* "2" while the yaml stores an int, so in practice the
        # interface fallback never fires; match that behavior exactly.
        if self.edm4hep["schema_version"] == "2":
            interfaces = self.parsed_edm4hep.get("interfaces", {})
            if target_datatype in interfaces:
                for i in interfaces[target_datatype]["Types"]:
                    matched += [
                        name
                        for name, datatype in self._datatype_mixins.items()
                        if "edm4hep::" + datatype == i
                    ]
        return matched

    def _process_vector_members(self, forms, all_collections):
        fieldnames = list(forms)
        for collection in all_collections:
            if collection.startswith("_"):
                continue
            datatype = self._datatype_mixins.get(collection)
            if datatype is None:
                continue
            spec = self._datatype_spec(datatype)
            vec_members = spec.get("VectorMembers") if spec else None
            if not vec_members:
                continue
            branch_var = {
                name.split("/")[1].split(".")[1]: forms[name]
                for name in fieldnames
                if name.split("/")[0] == collection and len(name.split("/")) > 1
            }
            for member in vec_members:
                if f"{member}_begin" not in branch_var:
                    continue
                begin = branch_var[member + "_begin"]
                end = branch_var[member + "_end"]
                forms.pop(f"{collection}/{collection}.{member}_begin", None)
                forms.pop(f"{collection}/{collection}.{member}_end", None)

                targets = {
                    name.split(".")[-1]: forms.pop(name)
                    for name in fieldnames
                    if name.startswith(f"_{collection}_{member}")
                    and len(name.split("/")) > 1
                    and name in forms
                }
                if len(targets) == 0:
                    bare = forms.pop(f"_{collection}_{member}", None)
                    if bare is None:
                        continue
                    target_form = begin_end_mapping(begin, end, bare.content)
                elif len(targets) == 1:
                    only = next(iter(targets.values()))
                    target_form = begin_end_mapping(begin, end, only.content)
                else:
                    vec_contents = {
                        name: begin_end_mapping(
                            begin, end, self._vector_member_target(name, layout)
                        )
                        for name, layout in targets.items()
                    }
                    target_form = _zip_shared_offsets(sort_dict(vec_contents))
                forms[f"{collection}/{collection}.{member}"] = target_form
        return forms

    @staticmethod
    def _vector_member_target(name, layout):
        """Flat content of a VectorMember target, ready for ``begin_end_mapping``.

        Fixed-size members such as ``covMatrix.values[21]`` were zipped into a
        single-field record by the component pass; coffea flattens those into
        variable-length float64 lists, so unwrap and convert them here.
        """
        content = layout.content
        if (
            name.endswith("covMatrix")
            and isinstance(content, awkward.contents.RecordArray)
            and len(content.fields) == 1
        ):
            inner = content.contents[0]
            if isinstance(inner, awkward.contents.RegularArray):
                return regular_to_jagged(inner)
        return content

    def _process_one_to_one_relations(self, forms, all_collections, links=False):
        fieldnames = list(forms)
        for collection in all_collections:
            if collection.startswith("_"):
                continue
            datatype = self._datatype_mixins.get(collection)
            if datatype is None:
                continue
            spec = self._datatype_spec(datatype)
            relations = spec.get("OneToOneRelations") if spec else None
            if not relations:
                continue
            is_link = all(k in relations for k in ("from", "to"))
            if links != is_link:
                continue
            # when copying links onto their targets, collect them per collection
            copy_targets = set()
            branches_to_copy = {}
            for member in relations:
                if (member in ("from", "to")) != links:
                    continue
                targets = {
                    name.split(".")[-1]: forms.pop(name)
                    for name in fieldnames
                    if name.startswith(f"_{collection}_{member}")
                    and len(name.split("/")) > 1
                    and name in forms
                }
                if not targets:
                    continue
                for matched in self._matched_collections(relations[member]["type"]):
                    target_offsets = self._target_offsets(matched, forms)
                    if target_offsets is None:
                        continue
                    index = targets["index"]
                    content = dict(targets)
                    content["index_Global"] = awkward.contents.ListOffsetArray(
                        index.offsets, local2global(index, target_offsets)
                    )
                    if links:
                        link_form = _zip_shared_offsets(content)
                        forms[f"{collection}/{collection}.Link_{member}_{matched}"] = (
                            link_form
                        )
                        if self._should_copy_link(member, matched, relations):
                            if member == "from":
                                copy_targets.add(matched)
                            branches_to_copy[f"Link_{member}_{matched}"] = link_form
                    else:
                        for name, layout in content.items():
                            forms[
                                f"{collection}/{collection}."
                                f"{member}_idx_{matched}_{name}"
                            ] = layout

            # copy the collected links onto their target collections
            if links and self.copy_links_to_target_datatype:
                for matched in copy_targets:
                    for name, layout in branches_to_copy.items():
                        forms[f"{matched}/{matched}.{name}"] = layout
        return forms

    def _should_copy_link(self, member, matched, relations):
        """Whether a Link branch should also be copied onto its target collection."""
        if not self.copy_links_to_target_datatype:
            return False
        if not self._datatype_priority:
            msg = "Cannot copy links if no _datatype_priority is given!"
            raise RuntimeError(msg)
        candidates = self._matched_collections(relations[member]["type"])
        if len(candidates) > 1:
            datatype = self._datatype_mixins.get(matched)
            return self._datatype_priority.get(datatype) == matched
        return True

    def _process_one_to_many_relations(self, forms, all_collections):
        fieldnames = list(forms)
        for collection in all_collections:
            if collection.startswith("_"):
                continue
            datatype = self._datatype_mixins.get(collection)
            if datatype is None:
                continue
            spec = self._datatype_spec(datatype)
            relations = spec.get("OneToManyRelations") if spec else None
            if not relations:
                continue
            branch_var = {
                name.split("/")[1].split(".")[1]: forms[name]
                for name in fieldnames
                if name.split("/")[0] == collection and len(name.split("/")) > 1
            }
            for member in relations:
                if member in ("from", "to") or f"{member}_begin" not in branch_var:
                    continue
                begin = branch_var[member + "_begin"]
                end = branch_var[member + "_end"]
                forms.pop(f"{collection}/{collection}.{member}_begin", None)
                forms.pop(f"{collection}/{collection}.{member}_end", None)

                targets = {
                    name.split(".")[-1]: forms.pop(name)
                    for name in fieldnames
                    if name.startswith(f"_{collection}_{member}")
                    and len(name.split("/")) > 1
                    and name in forms
                }
                if not targets:
                    continue
                for matched in self._matched_collections(relations[member]["type"]):
                    target_offsets = self._target_offsets(matched, forms)
                    if target_offsets is None:
                        continue
                    nested = {
                        name: begin_end_mapping(begin, end, layout.content)
                        for name, layout in targets.items()
                    }
                    to_zip = dict(nested)
                    if "index" in nested:
                        idx = nested["index"]
                        to_zip["index_Global"] = awkward.contents.ListOffsetArray(
                            idx.offsets,
                            awkward.contents.ListOffsetArray(
                                idx.content.offsets,
                                nested_local2global(idx, target_offsets),
                            ),
                        )
                    for name, layout in to_zip.items():
                        forms[
                            f"{collection}/{collection}.{member}_idx_{matched}_{name}"
                        ] = layout
        return forms

    def _make_collections(self, output, forms):
        field_names = list(forms)
        collections = {name.split("/")[0] for name in field_names if "/" in name}

        for name in collections:
            mixin = self._datatype_mixins.get(name, "edm4hep_nanocollection")
            content = {
                k[(2 * len(name) + 2) :]: forms.pop(k)
                for k in field_names
                if k.startswith(f"{name}/{name}.") and k in forms
            }
            if not content:
                continue
            content = {self._replacement.get(k, k): v for k, v in content.items()}
            if mixin == "ReconstructedParticle":
                content.pop("E", None)

            params = {}
            if mixin != "edm4hep_nanocollection":
                params = {
                    "collection_name": name,
                    "__doc__": self.parsed_edm4hep["datatypes"]
                    .get("edm4hep::" + mixin, {})
                    .get("Description", mixin),
                }
            output[name] = _zip_shared_offsets(
                sort_dict(content), record_name=mixin, parameters=params
            )
            forms.pop(name, None)
        return output, forms

    def _unknown_collections(self, output, forms):
        for name, layout in list(forms.items()):
            record = layout.content if hasattr(layout, "content") else None
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

        forms = self._process_components(forms, all_collections)
        forms = self._process_vector_members(forms, all_collections)
        forms = self._process_one_to_one_relations(forms, all_collections, links=False)
        forms = self._process_one_to_many_relations(forms, all_collections)
        forms = self._process_one_to_one_relations(forms, all_collections, links=True)

        output = {}
        output, forms = self._make_collections(output, forms)
        output, forms = self._unknown_collections(output, forms)
        return sort_dict(output)

    @classmethod
    def behavior(cls):
        """Behaviors necessary to implement this schema (dict)"""
        from awkward_zipper.behaviors import base, edm4hep, vector

        behavior = {}
        behavior.update(base.behavior)
        behavior.update(vector.behavior)
        behavior.update(edm4hep.behavior)
        return behavior


class EDM4HEP_v00_99_00(EDM4HEP):
    """EDM4HEP layout builder for edm4hep version 00.99.00"""

    edm4hep_version = "00-99-00"


class EDM4HEP_v00_10_05(EDM4HEP):
    """EDM4HEP layout builder for edm4hep version 00.10.05"""

    edm4hep_version = "00-10-05"


class EDM4HEP_v00_10_04(EDM4HEP):
    """EDM4HEP layout builder for edm4hep version 00.10.04"""

    edm4hep_version = "00-10-04"


class EDM4HEP_v00_10_03(EDM4HEP):
    """EDM4HEP layout builder for edm4hep version 00.10.03"""

    edm4hep_version = "00-10-03"


class EDM4HEP_v00_10_02(EDM4HEP):
    """EDM4HEP layout builder for edm4hep version 00.10.02"""

    edm4hep_version = "00-10-02"


class EDM4HEP_v00_10_01(EDM4HEP):
    """EDM4HEP layout builder for edm4hep version 00.10.01"""

    edm4hep_version = "00-10-01"


_VERSION_MATCH = {
    "latest": EDM4HEP,
    "00.99.01": EDM4HEP,
    "00.99.00": EDM4HEP_v00_99_00,
    "00.10.05": EDM4HEP_v00_10_05,
    "00.10.04": EDM4HEP_v00_10_04,
    "00.10.03": EDM4HEP_v00_10_03,
    "00.10.02": EDM4HEP_v00_10_02,
    "00.10.01": EDM4HEP_v00_10_01,
}


def edm4hep_version(ver="latest"):
    """Return the EDM4HEP layout builder class for a given edm4hep.yaml version."""
    schema = _VERSION_MATCH.get(ver)
    if schema is None:
        msg = (
            f"The given version {ver} is not found. "
            f"Available versions are: {', '.join(_VERSION_MATCH)}."
        )
        raise ValueError(msg)
    return schema
