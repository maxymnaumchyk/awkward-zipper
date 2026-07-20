"""Layout builder for ATLAS ntuples (the ``atlas-schema`` NtupleSchema)"""

import difflib
import typing as tp
import warnings

import awkward
import particle

from awkward_zipper.awkward_util import (
    _check_equal_lengths,
    _non_materializing_get_field,
    _rewrap,
)
from awkward_zipper.kernels import full_like_from_content
from awkward_zipper.layouts.base import BaseLayoutBuilder


class Ntuple(BaseLayoutBuilder):
    """Builder for ATLAS ntuples following the typical centralized formats.

    Collections are built from all branches found in the tree, based on the naming
    pattern of the branches, which is assumed to be

    .. code-block:: bash

       {collection:str}_{subcollection:str}_{systematic:str}

    So ``el_pt_NOSYS`` is the ``pt`` subcollection of the ``el`` collection, for the
    nominal (``NOSYS``) systematic. Branches carrying a systematic variation are
    additionally gathered into one record per systematic, sitting alongside the
    nominal collections, so that ``events.EG_SCALE_ALL__1up.el.pt`` reads the varied
    electron transverse momentum while ``events.el.pt`` reads the nominal one.

    Collections are assigned mixin types according to the `mixins` mapping; anything
    unrecognized falls back to `suggested_behavior`. All collections are then zipped
    into one `base.NanoEvents` record and returned.
    """

    #: warn when a cross-reference index or target is missing
    warn_missing_crossrefs: tp.ClassVar[bool] = True
    #: treat missing event-level branches as an error instead of a warning
    error_missing_event_ids: tp.ClassVar[bool] = False
    #: determine closest behavior for a given branch, or use `default_behavior`
    identify_closest_behavior: tp.ClassVar[bool] = True

    #: event IDs to expect in data datasets
    event_ids_data: tp.ClassVar = {
        "actualInteractionsPerCrossing",
        "averageInteractionsPerCrossing",
        "dataTakingYear",
        "lumiBlock",
    }
    #: event IDs to expect in MC datasets
    event_ids_mc: tp.ClassVar = {
        "eventNumber",
        "mcChannelNumber",
        "mcEventWeights",
        "runNumber",
    }
    #: all event IDs to expect in the dataset
    event_ids: tp.ClassVar = {*event_ids_data, *event_ids_mc}

    #: mapping from collection name to the behavior to use for that collection
    mixins: tp.ClassVar = {
        "el": "Electron",
        "jet": "Jet",
        "met": "MissingET",
        "mu": "Muon",
        "pass": "Pass",
        "ph": "Photon",
        "trigPassed": "Trigger",
        "weight": "Weight",
    }

    #: additional branches to pass through with no zipping or interpretation
    singletons: tp.ClassVar = set()

    #: docstrings to assign for specific subcollections across the collections
    docstrings: tp.ClassVar = {
        "charge": "charge",
        "eta": "pseudorapidity",
        "mass": "invariant mass [MeV]",
        "met": "missing transverse energy [MeV]",
        "phi": "azimuthal angle",
        "pt": "transverse momentum [MeV]",
    }

    #: default behavior to use for any collection
    default_behavior: tp.ClassVar[str] = "NanoCollection"

    #: fields to materialize as full_like arrays, keyed by behavior name.
    #: Each entry maps new_field -> (source_field, fill_value).
    full_like_items: tp.ClassVar = {
        "Electron": {"mass": ("pt", float(particle.literals.e_minus.mass))},
        "Muon": {"mass": ("pt", float(particle.literals.mu_minus.mass))},
        "Photon": {"mass": ("pt", 0.0), "charge": ("pt", 0.0)},
        "Tau": {"mass": ("pt", float(particle.literals.tau_minus.mass))},
    }
    #: fields to rename, keyed by behavior name. Maps new_field -> old_field.
    rename_items: tp.ClassVar = {
        "Jet": {"mass": "m"},
    }
    #: fields to alias, keyed by behavior name. Maps new_field -> source_field.
    alias_items: tp.ClassVar = {
        "MissingET": {"rho": "met"},  # vector reads 'rho', not 'r' or 'met'
    }

    def __init__(self, version="latest"):
        self._version = version

    @classmethod
    def v1(cls):
        """Build the NtupleEvents for version 1"""
        return cls(version="1")

    def _apply_vector_fields(self, behavior_name, collection_content):
        """Materialize vector coordinate fields into ``collection_content`` in place.

        Keyed by behavior name so custom collections mapped via `mixins` get the same
        treatment. Silently skips if the source field is absent (optional branches),
        warns and skips if the target field already exists.
        """
        for new_field, (source_field, fill_value) in self.full_like_items.get(
            behavior_name, {}
        ).items():
            if source_field not in collection_content:
                continue
            if new_field in collection_content:
                warnings.warn(
                    f"Field '{new_field}' already present in collection with behavior "
                    f"'{behavior_name}'; skipping materialization. [vector-field-exists]",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            collection_content[new_field] = full_like_from_content(
                collection_content[source_field], fill_value
            )
        for new_field, old_field in self.rename_items.get(behavior_name, {}).items():
            if old_field not in collection_content:
                continue
            if new_field in collection_content:
                warnings.warn(
                    f"Field '{new_field}' already present in collection with behavior "
                    f"'{behavior_name}'; skipping rename of '{old_field}'. [vector-field-exists]",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            collection_content[new_field] = collection_content.pop(old_field)
        for new_field, source_field in self.alias_items.get(behavior_name, {}).items():
            if source_field not in collection_content:
                continue
            if new_field in collection_content:
                warnings.warn(
                    f"Field '{new_field}' already present in collection with behavior "
                    f"'{behavior_name}'; skipping alias. [vector-field-exists]",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            collection_content[new_field] = collection_content[source_field]

    @staticmethod
    def _zip_collection(collection_content, collection_name, behavior_name):
        """Zip a mapping of field name -> content into one (jagged) RecordArray.

        The branches of an ATLAS ntuple carry their own offsets, so a jagged
        collection is rebuilt by sharing the offsets of its first field and zipping
        the flat contents underneath; event-level collections stay flat records.
        """
        contents = [
            awkward.to_layout(content) for content in collection_content.values()
        ]
        fields = list(collection_content)

        parameters = {
            "__record__": behavior_name,
            "collection_name": collection_name,
        }

        if all(content.is_list for content in contents):
            inner = tuple(content.content for content in contents)
            record = awkward.contents.RecordArray(
                inner,
                fields,
                length=_check_equal_lengths(inner),
                parameters=parameters,
            )
            return awkward.contents.ListOffsetArray(contents[0].offsets, record)

        return awkward.contents.RecordArray(
            tuple(contents),
            fields,
            length=_check_equal_lengths(tuple(contents)),
            parameters=parameters,
        )

    def _collection_behavior(self, collection_name, warn):
        behavior_name = self.mixins.get(collection_name, "")
        if behavior_name:
            return behavior_name
        behavior_name = self.suggested_behavior(collection_name)
        if warn:
            warnings.warn(
                f"I found a collection with no defined mixin: '{collection_name}'. "
                f"I will assume behavior: '{behavior_name}'. To suppress this warning "
                "next time, please define mixins for your custom collections. [mixin-undefined]",
                RuntimeWarning,
                stacklevel=2,
            )
        return behavior_name

    def __call__(self, array: awkward.Array) -> awkward.Array:
        branch_arrays = {
            field: _non_materializing_get_field(array, field) for field in array.fields
        }

        # parse into high-level records (collections, list collections, and singletons)
        collections = {
            k.split("_")[0] for k in branch_arrays if k not in self.singletons
        }
        collections -= self.event_ids
        collections -= set(self.singletons)

        # handle collections that are substrings of the items in the mixins
        bf_str = ",".join(branch_arrays.keys())
        for mixin in self.mixins:
            if mixin in collections:
                continue
            if f",{mixin}_" not in bf_str and not bf_str.startswith(f"{mixin}_"):
                continue
            if "_" in mixin:
                warnings.warn(
                    f"I identified a mixin that I did not automatically identify as a "
                    f"collection because it contained an underscore: '{mixin}'. I will add "
                    "this to the known collections. To suppress this warning next time, "
                    "please create your ntuples with collections without underscores. "
                    "[mixin-underscore]",
                    RuntimeWarning,
                    stacklevel=2,
                )
            collections.add(mixin)
            for collection in list(collections):
                if mixin.startswith(f"{collection}_"):
                    warnings.warn(
                        f"I found a misidentified collection: '{collection}'. I will remove "
                        "this from the known collections. To suppress this warning next time, "
                        "please create your ntuples with collections that are not similarly "
                        "named with underscores. [collection-subset]",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    collections.remove(collection)
                    break

        # rename needed because easyjet breaks the AMG assumptions
        # https://gitlab.cern.ch/easyjet/easyjet/-/issues/246
        for k in list(branch_arrays):
            if "NOSYS" not in k:
                continue
            branch_arrays[k.replace("_NOSYS", "") + "_NOSYS"] = branch_arrays.pop(k)

        # these are collections with systematic variations
        try:
            subcollections = {
                k.split("__")[0].split("_", 1)[1].replace("_NOSYS", "")
                for k in branch_arrays
                if "NOSYS" in k and k not in self.singletons
            }
        except IndexError as exc:
            msg = (
                "One of the branches does not follow the assumed pattern for this "
                "schema. [invalid-branch-name]"
            )
            raise RuntimeError(msg) from exc

        all_systematics = self._discover_systematics(
            branch_arrays, collections, subcollections
        )

        # pre-compute systematic branch patterns for O(1) lookups
        systematic_branch_patterns = set()
        for collection in collections:
            for subcoll in subcollections:
                for systematic in all_systematics:
                    if systematic != "NOSYS":
                        systematic_branch_patterns.add(
                            f"{collection}_{subcoll}_{systematic}"
                        )

        missing_event_ids = [
            event_id for event_id in self.event_ids if event_id not in branch_arrays
        ]
        missing_singletons = [
            singleton for singleton in self.singletons if singleton not in branch_arrays
        ]

        if len(missing_event_ids) > 0:
            if self.error_missing_event_ids:
                msg = f"There are missing event ID fields: {missing_event_ids} \n\n\
                    The event ID fields {self.event_ids} are necessary to perform sub-run identification \
                    (e.g. for corrections and sub-dividing data during different detector conditions),\
                    to cross-validate MC and Data (i.e. matching events for comparison), and to generate event displays. \
                    It's advised to never drop these branches from the dataformat.\n\n\
                    This error can be demoted to a warning by setting the class level variable error_missing_event_ids to False."
                raise RuntimeError(msg)
            warnings.warn(
                f"Missing event_ids : {missing_event_ids}",
                RuntimeWarning,
                stacklevel=2,
            )

        if len(missing_singletons) > 0:
            warnings.warn(
                f"Missing singletons : {missing_singletons}. [singleton-missing]",
                RuntimeWarning,
                stacklevel=2,
            )

        output = {}

        # first, register singletons (event-level, others)
        for name in {*self.event_ids, *self.singletons}:
            if name in [*missing_event_ids, *missing_singletons]:
                continue
            output[name] = awkward.to_layout(branch_arrays[name])

        # build the nominal collections
        nominal_collections = {}
        for collection_name in collections:
            collection_content = {}
            used = set()

            # subcollections with NOSYS variations
            for subname in subcollections:
                nosys_branch = f"{collection_name}_{subname}_NOSYS"
                if nosys_branch in branch_arrays:
                    collection_content[subname] = branch_arrays[nosys_branch]
                    used.add(nosys_branch)

            # non-systematic branches (like eta, phi, that do not vary)
            for k, content in branch_arrays.items():
                if (
                    k.startswith(collection_name + "_")
                    and k not in used
                    and "_NOSYS" not in k
                    and k not in systematic_branch_patterns
                ):
                    field_name = k[len(collection_name) + 1 :]
                    if field_name not in collection_content:
                        collection_content[field_name] = content

            if collection_content:
                behavior_name = self._collection_behavior(collection_name, warn=True)
                self._apply_vector_fields(behavior_name, collection_content)
                nominal_collections[collection_name] = self._zip_collection(
                    collection_content, collection_name, behavior_name
                )

        output.update(nominal_collections)

        # build the systematic event structures
        for systematic in all_systematics:
            if systematic == "NOSYS":
                continue

            systematic_collections = {}
            for collection_name in collections:
                has_systematic_data = False
                collection_content = {}
                used = set()

                for subname in subcollections:
                    prefix = f"{collection_name}_{subname}_"
                    target_branch = f"{prefix}{systematic}"
                    fallback_branch = f"{prefix}NOSYS"

                    if target_branch in branch_arrays:
                        collection_content[subname] = branch_arrays[target_branch]
                        used.add(target_branch)
                        has_systematic_data = True
                    elif fallback_branch in branch_arrays:
                        collection_content[subname] = branch_arrays[fallback_branch]
                        used.add(fallback_branch)

                for k, content in branch_arrays.items():
                    if (
                        k.startswith(collection_name + "_")
                        and k not in used
                        and "_NOSYS" not in k
                        and k not in systematic_branch_patterns
                    ):
                        field_name = k[len(collection_name) + 1 :]
                        if field_name not in collection_content:
                            collection_content[field_name] = content

                if collection_content:
                    behavior_name = self._collection_behavior(
                        collection_name, warn=False
                    )
                    if (
                        not has_systematic_data
                        and collection_name in nominal_collections
                    ):
                        # no variation for this collection: reuse the nominal one
                        systematic_collections[collection_name] = nominal_collections[
                            collection_name
                        ]
                    else:
                        self._apply_vector_fields(behavior_name, collection_content)
                        systematic_collections[collection_name] = self._zip_collection(
                            collection_content, collection_name, behavior_name
                        )

            if systematic_collections:
                _content = (*systematic_collections.values(),)
                output[systematic] = awkward.contents.RecordArray(
                    _content,
                    (*systematic_collections.keys(),),
                    length=_check_equal_lengths(_content),
                    parameters={
                        "__record__": "Systematic",
                        "metadata": {"systematic": systematic},
                    },
                )

        # handle any remaining unrecognized branches as singletons
        processed_branches = set()
        processed_branches.update(self.event_ids)
        processed_branches.update(self.singletons)
        for collection_name in collections:
            for branch_name in branch_arrays:
                if branch_name.startswith(collection_name + "_"):
                    processed_branches.add(branch_name)

        for branch_name, content in branch_arrays.items():
            if branch_name not in processed_branches:
                warnings.warn(
                    f"I identified a branch that likely does not have any leaves: "
                    f"'{branch_name}'. I will treat this as a 'singleton'. To suppress this "
                    "warning, add this branch to the singletons set. [singleton-undefined]",
                    RuntimeWarning,
                    stacklevel=2,
                )
                output[branch_name] = awkward.to_layout(content)

        discovered_systematics = sorted(s for s in all_systematics if s != "NOSYS")

        # final nanoevents (most outer) zip
        _content = (*output.values(),)
        _fields = (*output.keys(),)
        _length = int(awkward.num(array, axis=0))

        nanoevents = awkward.Array(
            awkward.contents.RecordArray(_content, _fields, length=_length),
            behavior=self.behavior(),
        )

        # fix virtual array shape generators by re-running from buffers
        nanoevents = awkward.with_name(_rewrap(nanoevents), name="NtupleEvents")
        nanoevents.layout.parameters.update(
            {
                "metadata": {
                    "version": self._version,
                    "systematics": discovered_systematics,
                },
                **array.layout.parameters,
            }
        )

        # add ref to itself in attrs
        nanoevents.attrs["@original_array"] = nanoevents

        return nanoevents

    def _discover_systematics(self, branch_arrays, collections, subcollections):
        """Extract the systematic variation names from the branch names."""
        subcoll_patterns = {f"{subcoll}_" for subcoll in subcollections}

        all_systematics = set()
        for k in branch_arrays:
            if not ("_" in k and k not in self.singletons):
                continue
            # handle the pattern collection_subcollection_systematic, where the
            # systematic can itself contain double underscores such as
            # "JET_EnergyResolution__1up"
            parts = k.split("_")
            if len(parts) < 3:
                continue
            collection = parts[0]
            if collection not in collections:
                continue
            remaining = "_".join(parts[1:])
            for pattern in subcoll_patterns:
                if remaining.startswith(pattern):
                    systematic = remaining[len(pattern) :]
                    if systematic and systematic != "NOSYS":
                        all_systematics.add(systematic)
                    break

        # always include NOSYS as the nominal case
        all_systematics.add("NOSYS")
        return all_systematics

    @classmethod
    def behavior(cls):
        """Behaviors necessary to implement this schema (dict)"""
        from awkward_zipper.behaviors import atlas

        return atlas.behavior

    @classmethod
    def suggested_behavior(cls, key, cutoff=0.4):
        """Suggest a behavior to use for a provided collection or branch name.

        If `identify_closest_behavior` is ``False``, or nothing scores above *cutoff*,
        this returns `default_behavior`.
        """
        if cls.identify_closest_behavior:
            # lowercase everything to do case-insensitive matching
            behaviors = [b for b in cls.behavior() if isinstance(b, str)]
            behaviors_l = [b.lower() for b in behaviors]
            results = difflib.get_close_matches(
                key.lower(), behaviors_l, n=1, cutoff=cutoff
            )
            if not results:
                return cls.default_behavior

            behavior = results[0]
            # need to identify the index and return the unlowered version
            return behaviors[behaviors_l.index(behavior)]
        return cls.default_behavior
