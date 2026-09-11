import copy
import re
import typing as tp
import warnings
from functools import cache

import awkward

from awkward_zipper.assets import edm4hep_ver, versions
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

_link_collection = re.compile(r"podio::LinkCollection<(.+),(.+)>")


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


def _synthesize_link_datatypes(loaded_dict):
    """edm4hep >= 00-99-02 defines links in a ``links`` section; rebuild them in the
    datatype shape (float weight + from/to relations) the rest of the parser expects."""
    synthesized = {}
    for link_name, link_def in loaded_dict.get("links", {}).items():
        from_type = link_def["From"]
        to_type = link_def["To"]
        synthesized[link_name] = {
            "Description": link_def.get("Description", ""),
            "Members": ["float weight  // weight of this link"],
            "OneToOneRelations": [
                f"{from_type}  from  // reference to the source object of this link",
                f"{to_type}  to  // reference to the target object of this link",
            ],
        }
    return synthesized


def podio_collection_types(tree):
    """Map collection name to podio dataType from the file's ``podio_metadata`` tree.

    Returns None when the file has no metadata or predates the named leaf layout
    (podio < 1.3 writes positional ``_0.._3`` leaves).
    """
    directory = tree.file.root_directory
    if "podio_metadata" not in directory:
        return None
    metadata = directory["podio_metadata"]
    branch = f"{tree.name}___CollectionTypeInfo"
    if branch not in metadata or f"{branch}.name" not in metadata[branch]:
        return None
    names, datatypes = f"{branch}.name", f"{branch}.dataType"
    info = metadata[branch].arrays([names, datatypes], library="ak")[0]
    return dict(zip(info[names].tolist(), info[datatypes].tolist(), strict=True))


def _relation_branches(forms, collection, member):
    """Pop the leaves of the ``_{collection}_{member}`` relation or vector-member
    branch, keyed ``member.leaf``. The top-level name is matched exactly because
    collection names may themselves contain underscores (``SiTracks_Refitted``)."""
    top = f"_{collection}_{member}"
    return {
        name.split("/")[1][len(collection) + 2 :]: forms.pop(name)
        for name in list(forms)
        if "/" in name and name.split("/")[0] == top
    }


def parse_yaml(loaded_dict, parsed_dict):
    """Post-process the loaded edm4hep yaml into the structure the builder needs."""
    links = _synthesize_link_datatypes(loaded_dict)
    loaded_dict = {**loaded_dict, "datatypes": {**loaded_dict["datatypes"], **links}}
    parsed_dict["datatypes"].update(copy.deepcopy(links))

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


@cache
def load_edm4hep(version):
    """Load and parse the edm4hep yaml for a version, caching the result.

    The returned ``(raw, parsed)`` dicts are treated as read-only by the builder,
    so a single parse is shared across all builds for a given version.
    """
    raw = edm4hep_ver[version]()
    return raw, parse_yaml(raw, copy.deepcopy(raw))


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


def _zip_shared_offsets(members, record_name=None, parameters=None, offsets=None):
    """Zip layouts that share per-event offsets into one jagged record.

    ``offsets`` defaults to those of the first member (coffea's ``zip_forms``).
    """
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
        contents, names, length=_offsets_length(offsets, contents), parameters=params
    )
    return awkward.contents.ListOffsetArray(offsets=offsets, content=record)


class EDM4HEP(BaseLayoutBuilder):
    """EDM4HEP layout builder.

    Array-based re-implementation of coffea's ``EDM4HEPSchema`` for the newest
    bundled ``edm4hep.yaml`` version; use :func:`edm4hep_version` (or
    ``EDM4HEP.version(...)``) to pick an older one. The layout is driven by the
    vendored EDM4HEP yaml data model, which describes each datatype's Members,
    VectorMembers, OneToOneRelations and OneToManyRelations:

    - component members (``Vector3f``/``Vector4f``/...) such as
      ``X.position.{x,y,z}`` are zipped into an ``X.position`` sub-record,
    - ``VectorMembers`` and ``OneToManyRelations`` stored as flat per-event arrays
      plus per-item ``{member}_begin``/``{member}_end`` ranges are regrouped into
      doubly-jagged arrays,
    - ``OneToOneRelations`` and Links get a global index (``index_Global``) into
      their target collection,
    - the remaining ``X.*`` branches are zipped into the ``X`` collection.

    The collection datatypes come from the TTree's branch typenames, which are not
    present in ``tree.arrays()`` output, so pass them explicitly. Generic podio
    links (``vector<podio::LinkData>``, podio >= 1.3) are typed from the file's
    ``podio_metadata`` tree, which :func:`podio_collection_types` reads::

        EDM4HEP()(
            tree.arrays(...),
            typenames=tree.typenames(),
            podio_collection_types=podio_collection_types(tree),
        )

    ``extra_mixins`` overrides the link type of a collection. If ``typenames`` is
    omitted, the datatypes are inferred by matching each collection's member branch
    names against the yaml data model.
    """

    edm4hep_version = versions[-1]

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

    # By default, Links are not copied onto their target datatype collections: many
    # collections may share a datatype and not all of them are compatible targets.
    # Subclasses that know which link belongs to which collection can enable it and
    # provide ``_datatype_priority`` (datatype -> collection to copy to).
    copy_links_to_target_datatype = False
    _datatype_priority: tp.ClassVar = {}

    def __call__(
        self, array: awkward.Array, typenames=None, podio_collection_types=None
    ) -> awkward.Array:
        self.edm4hep, self.parsed_edm4hep = load_edm4hep(self.edm4hep_version)

        n_events = int(awkward.num(array, axis=0))

        # Work with coffea-style keys ("X/X.y") so the branch-name logic matches
        # coffea's; uproot's `tree.arrays()` gives them as "X.y".
        forms = {}
        for field in array.fields:
            layout = _non_materializing_get_field(array, field).layout
            key = f"{field.split('.')[0]}/{field}" if "." in field else field
            forms[key] = layout

        self._create_mixin(forms, typenames, podio_collection_types)
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

    @classmethod
    def version(cls, ver="latest"):
        """Return the layout builder class for a given edm4hep.yaml version.

        Parameters
        ----------
            ver : str, optional
                Version of edm4hep.yaml, written either as "00.99.04" or "00-99-04".
                "latest" (default) selects the newest bundled version. The available
                versions are listed in ``awkward_zipper.assets.versions``.
        """
        return edm4hep_version(ver)

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

    def _create_mixin(self, forms, typenames, podio_types):
        all_collections = {key.split("/")[0] for key in forms if "/" in key}
        self._all_collections = all_collections
        collections = {c for c in all_collections if not c.startswith("_")}

        # podio >= 1.3 stores generic links as vector<podio::LinkData>; the (From, To)
        # pair naming the link datatype lives only in podio_metadata
        link_types = {
            (link["From"], link["To"]): name.split("::")[-1]
            for name, link in self.edm4hep.get("links", {}).items()
        }
        podio_types = podio_types or {}
        mixins = {}
        for name in collections:
            if typenames is None:
                inferred = self._infer_datatype(name, forms)
                mixins[name] = (
                    inferred.split("::")[-1] if inferred else "edm4hep_nanocollection"
                )
            else:
                datatype = typenames.get(name, "edm4hep_nanocollection")
                if datatype.startswith(r"vector<edm4hep::"):
                    if not datatype.endswith("Data>"):
                        msg = f"Unknown datatype: {datatype}"
                        raise RuntimeError(msg)
                    mixins[name] = datatype.split("::")[-1][:-5]
                elif datatype.startswith(r"vector<podio::"):
                    mixins[name] = datatype.split("::")[-1][:-1]
                else:
                    mixins[name] = datatype

            if mixins[name] == "LinkData":
                endpoints = _link_collection.match(podio_types.get(name, ""))
                stem = name[: -len("Collection")] if name.endswith("Collection") else ""
                if endpoints and endpoints.groups() in link_types:
                    mixins[name] = link_types[endpoints.groups()]
                elif "edm4hep::" + stem in self.parsed_edm4hep["datatypes"]:
                    mixins[name] = stem

        mixins_dictionary = {**mixins, **self.extra_mixins}
        unresolved = sorted(n for n, m in mixins_dictionary.items() if m == "LinkData")
        if unresolved:
            msg = (
                f"Cannot determine the link type of {unresolved}: they are stored as "
                "podio::LinkData and no podio_collection_types named their From/To "
                "types (podio >= 1.3 writes them to the file's podio_metadata tree). "
                "Pass podio_collection_types=podio_collection_types(tree), or subclass "
                "EDM4HEP with extra_mixins = {<collection>: <link datatype>} using the "
                "names in load_edm4hep(version)[0]['links']."
            )
            raise RuntimeError(msg)
        self._datatype_mixins = mixins_dictionary

    def _datatype_spec(self, datatype):
        """yaml spec for a datatype, or None when it is not a real edm4hep type."""
        if datatype is None:
            return None
        return self.parsed_edm4hep["datatypes"].get("edm4hep::" + datatype)

    def _lookup_branch(self, collection_name, branch_name, key=None):
        """'type'/'doc' (or both) of a branch of a collection, from the yaml model."""
        unknown = {"type": "unknown", "doc": "unknown"}
        datatype = self._datatype_mixins.get(collection_name)
        if collection_name.startswith("_"):
            # _{collection}_{member}: the collection may contain underscores, so take
            # the longest known collection; no yaml member name contains one
            stem = collection_name[1:]
            col_name = max(
                (c for c in self._all_collections if stem.startswith(c + "_")),
                key=len,
                default=stem.split("_")[0],
            )
            subcol_name = stem[len(col_name) + 1 :]
            datatype = self._datatype_mixins.get(col_name)
        if datatype is None:
            return unknown if key is None else unknown[key]
        collection_edm4hep = self.parsed_edm4hep["datatypes"].get(
            "edm4hep::" + datatype, {}
        )
        composite = {
            **collection_edm4hep.get("Members", {}),
            **collection_edm4hep.get("VectorMembers", {}),
            **collection_edm4hep.get("OneToOneRelations", {}),
            **collection_edm4hep.get("OneToManyRelations", {}),
        }
        if collection_name.startswith("_"):
            matched = composite.get(subcol_name, unknown)
            composite = {
                **composite,
                **self.parsed_edm4hep["components"].get(
                    matched["type"], {"Members": {}}
                )["Members"],
            }
        found = composite.get(branch_name, unknown)
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
            if assign_name == "momentum":
                continue  # Used to create 4 vector for the whole collection, later.
            if type_str == "unknown":
                continue  # not in this edm4hep version
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
                        # skip momentum because it will be used later to create
                        # the 4 vector with E or mass
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

    def _matched_collections(self, target_datatype, interfaces=False):
        """Collections whose datatype is ``target_datatype``.

        With ``interfaces=True`` (Links) an interface target such as
        ``edm4hep::TrackerHit`` resolves to the collections of its interfaced
        types. OneToMany relations never consult the interfaces: coffea gates that
        on the *string* ``"2"`` while the yaml stores an int, so the fallback never
        fires there; match that behavior exactly.
        """
        matched = [
            name
            for name, datatype in self._datatype_mixins.items()
            if "edm4hep::" + datatype == target_datatype
        ]
        if matched or not interfaces:
            return matched
        interfaced = self.parsed_edm4hep.get("interfaces", {})
        if target_datatype not in interfaced:
            msg = f"No matched collection for {target_datatype} found!"
            raise RuntimeError(msg)
        return [
            name
            for i in interfaced[target_datatype]["Types"]
            for name, datatype in self._datatype_mixins.items()
            if "edm4hep::" + datatype == i
        ]

    def _process_vector_members(self, forms, all_collections):
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
                for name in list(forms)
                if name.split("/")[0] == collection and "/" in name
            }
            for member in vec_members:
                if f"{member}_begin" not in branch_var:
                    continue
                target_contents = _relation_branches(forms, collection, member)
                begin = branch_var[member + "_begin"]
                end = branch_var[member + "_end"]
                forms.pop(f"{collection}/{collection}.{member}_begin")
                forms.pop(f"{collection}/{collection}.{member}_end")

                leaves = list(target_contents)
                if len(leaves) == 0:
                    if vec_members[member]["type"].startswith("edm4hep::"):
                        msg = f"_{collection}_{member} not found!"
                        raise RuntimeError(msg)
                    # Example: _EventHeader_weights, a plain vector member stored
                    # flat ('weights' not to be confused with 'weight')
                    bare = forms.pop(f"_{collection}_{member}", None)
                    if bare is None:
                        continue
                    target_form = begin_end_mapping(begin, end, bare.content)
                elif len(leaves) == 1:
                    target_form = begin_end_mapping(
                        begin, end, target_contents[leaves[0]].content
                    )
                else:
                    # Example: _TrackCollection_trackStates.D0, ...phi, etc. where
                    # 'trackStates' is a VectorMember of 'TrackState' components
                    vec_contents = {
                        name.split(".")[1]: begin_end_mapping(
                            begin, end, self._vector_member_target(name, layout)
                        )
                        for name, layout in target_contents.items()
                    }
                    target_form = _zip_shared_offsets(vec_contents)
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
        """OneToOneRelations (``links=False``) or the from/to Links (``links=True``).

        With ``copy_links_to_target_datatype`` the Links are also copied onto the
        collection their ``from`` side points to, using ``_datatype_priority`` to
        pick one when several collections share the target datatype.
        """
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
            if links and not all(k in relations for k in ("from", "to")):
                continue
            copy_targets = set()
            branches_to_copy = {}
            for member in relations:
                if (member in ("from", "to")) != links:
                    continue
                target_contents = _relation_branches(forms, collection, member)
                target_datatype = relations[member]["type"]
                if not target_datatype.startswith("edm4hep::"):
                    msg = f"{member} does not point to a valid datatype({target_datatype})!"
                    raise RuntimeError(msg)
                if not target_contents:
                    continue
                matched_collections = self._matched_collections(
                    target_datatype, interfaces=links
                )
                if not matched_collections:
                    warnings.warn(
                        f"No matched collection for {target_datatype} found!\n skipping ...",
                        stacklevel=2,
                    )
                    continue
                for matched in matched_collections:
                    target_offsets = self._target_offsets(matched, forms)
                    if target_offsets is None:
                        continue
                    content = {
                        name.split(".")[1]: layout
                        for name, layout in target_contents.items()
                    }
                    index = content["index"]
                    content["index_Global"] = awkward.contents.ListOffsetArray(
                        index.offsets, local2global(index, target_offsets)
                    )
                    if links:
                        link_form = _zip_shared_offsets(content)
                        forms[f"{collection}/{collection}.Link_{member}_{matched}"] = (
                            link_form
                        )
                        if self._should_copy_link(matched, matched_collections):
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

    def _should_copy_link(self, matched, matched_collections):
        """Whether a Link branch should also be copied onto ``matched``."""
        if not self.copy_links_to_target_datatype:
            return False
        if not self._datatype_priority:
            msg = "Cannot copy links if no priority is given!"
            raise RuntimeError(msg)
        if len(matched_collections) > 1:
            # choose which one to copy
            datatype = self._datatype_mixins.get(matched)
            return self._datatype_priority.get(datatype) == matched
        return True

    def _process_one_to_many_relations(self, forms, all_collections):
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
                for name in list(forms)
                if name.split("/")[0] == collection and "/" in name
            }
            for member in relations:
                if member in ("from", "to") or f"{member}_begin" not in branch_var:
                    continue
                target_contents = _relation_branches(forms, collection, member)
                begin = branch_var[member + "_begin"]
                end = branch_var[member + "_end"]
                forms.pop(f"{collection}/{collection}.{member}_begin")
                forms.pop(f"{collection}/{collection}.{member}_end")

                target_datatype = relations[member]["type"]
                if not target_datatype.startswith("edm4hep::"):
                    msg = f"{member} does not point to a valid datatype({target_datatype})!"
                    raise RuntimeError(msg)
                if not target_contents:
                    continue
                for matched in self._matched_collections(target_datatype):
                    target_offsets = self._target_offsets(matched, forms)
                    if target_offsets is None:
                        continue
                    nested = {
                        name.split(".")[1]: begin_end_mapping(
                            begin, end, layout.content
                        )
                        for name, layout in target_contents.items()
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
            # the collection's offsets come from its first branch (before sorting):
            # a copied Link branch carries the offsets of the link collection
            offsets = next(iter(content.values())).offsets
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
                sort_dict(content),
                record_name=mixin,
                parameters=params,
                offsets=offsets,
            )
            # the empty grouping branch "X" that accompanies "X/X.y" carries no info
            forms.pop(name, None)
        return output, forms

    def _unknown_collections(self, output, forms):
        """Handle the unknown, empty or singleton branches that remain.

        Mirrors coffea: empty grouping records are dropped, a remaining jagged
        record is left out of the output, jagged/flat singletons pass through.
        """
        for name, layout in list(forms.items()):
            if isinstance(layout, awkward.contents.ListOffsetArray):
                if isinstance(layout.content, awkward.contents.RecordArray):
                    if not layout.content.fields:
                        forms.pop(name)
                    continue
                output[name] = forms.pop(name)
            elif isinstance(layout, awkward.contents.RecordArray):
                if not layout.fields:
                    continue
                record_name = name.split("/")[0]
                contents = {
                    k[2 * len(record_name) + 2 :]: forms.pop(k)
                    for k in list(forms)
                    if k.startswith(record_name + "/")
                }
                if not contents:
                    continue
                output[record_name] = _zip_shared_offsets(
                    sort_dict(contents),
                    record_name=self._datatype_mixins.get(
                        record_name, "edm4hep_nanocollection"
                    ),
                )
            else:
                output[name] = forms.pop(name)
        return output, forms

    def _build_collections(self, forms):
        all_collections = self._all_collections

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


def _versioned_builder(ver):
    name = "EDM4HEP_v" + ver.replace("-", "_")
    doc = f"EDM4HEP layout builder for edm4hep version {ver.replace('-', '.')}"
    return type(
        name,
        (EDM4HEP,),
        {"edm4hep_version": ver, "__doc__": doc, "__module__": __name__},
    )


# Module globals keep EDM4HEP_v* importable by name and picklable by reference.
_versioned_builders = {ver: _versioned_builder(ver) for ver in versions}
globals().update({b.__name__: b for b in _versioned_builders.values()})


def edm4hep_version(ver="latest"):
    """Return the EDM4HEP layout builder class for a given edm4hep.yaml version.

    ``ver`` is written either as ``"00.99.04"`` or ``"00-99-04"``; ``"latest"``
    (default) selects the newest bundled version. The available versions are
    listed in ``awkward_zipper.assets.versions``.
    """
    if ver == "latest":
        return EDM4HEP
    builder = _versioned_builders.get(ver.replace(".", "-"))
    if builder is None:
        msg = (
            f"The given version {ver} is not found. "
            f"Available versions are : {', '.join(versions)} ."
        )
        raise ValueError(msg)
    return builder
