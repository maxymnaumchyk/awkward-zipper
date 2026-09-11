import collections.abc
import fnmatch
import typing as tp

import awkward

from awkward_zipper.awkward_util import (
    _non_materializing_get_field,
    _rewrap,
    _zip_arrays,
)
from awkward_zipper.layouts.base import BaseLayoutBuilder


class PDUNE(BaseLayoutBuilder):
    """ProtoDUNE-SP ``pduneana`` layout builder

    The branches (named ``<top_object>_<a>_<b>_...``) are regrouped into nested
    records by splitting on ``_``. Branches whose trailing token is a coordinate
    (``X``/``Y``/``Z`` or ``Px``/``Py``/``Pz``/``E``) are collected into
    ``ThreeVector``/``LorentzVector`` sub-records (``*3D``/``*4D``).

    This is an array-based re-implementation of coffea's ``PDUNESchema``.
    """

    mixins: tp.ClassVar = {
        "RecoBeam": "Beam",
        "Tracks": "Tracks",
        "Showers": "Showers",
        "reco_beam": "RecoBeam",
        "reco_daughter_allTrack": "Tracks",
        "reco_daughter_allShower": "Showers",
        "start3D": "ThreeVector",
        "end3D": "ThreeVector",
        "start4D": "LorentzVector",
        "end4D": "LorentzVector",
        "vtx3D": "ThreeVector",
    }

    top_objects: tp.ClassVar = {
        "reco_beam": "RecoBeam",
        "reco_daughter_allTrack": "Tracks",
        "reco_daughter_allShower": "Showers",
        "true_beam": "TrueBeam",
    }

    def __call__(self, array: awkward.Array) -> awkward.Array:
        # per-call mutable copy of mixins (3D/4D keys are added dynamically)
        self._mixins = dict(self.mixins)
        n_events = int(awkward.num(array, axis=0))
        branch_forms = {
            f: _non_materializing_get_field(array, f).layout for f in array.fields
        }

        output = self._build_collections(branch_forms)

        contents = tuple(output.values())
        names = tuple(output.keys())
        nanoevents = awkward.Array(
            awkward.contents.RecordArray(contents, names, length=n_events),
            behavior=self.behavior(),
        )
        nanoevents = awkward.with_name(_rewrap(nanoevents), name="NanoEvents")
        nanoevents.attrs["@original_array"] = nanoevents
        return nanoevents

    # --- helpers ported from coffea.nanoevents.schemas.pdune ---

    def _recursion(self, key_list, obj, key_dict, i):
        if i < len(key_list) - 1:
            curr_key = key_list[i]
            if curr_key not in key_dict:
                key_dict[curr_key] = {}
            self._recursion(key_list, obj, key_dict[curr_key], i + 1)
        elif i > 0 and key_dict.get(key_list[i]) is None:
            key_dict[key_list[i]] = obj

    def _recursive_zip(self, forms, hierarchy, key, final_zip=False):
        for k, v in hierarchy.items():
            if isinstance(v, collections.abc.Mapping):
                forms[k] = self._recursive_zip(forms.get(k, {}), v, k, True)
            else:
                name = self._mixins.get(k)
                forms[k] = _zip_arrays(forms[k], record_name=name)
        if final_zip:
            forms = _zip_arrays(forms, record_name=None)
        return forms

    def _filter_branches(self, branches, wildcard):
        return [b for b in branches if fnmatch.fnmatch(b, wildcard)]

    def _type_dictionary_builder(self, branch_forms):
        all_branches = branch_forms.keys()

        v3var = ["X", "Y", "Z"]
        v4var = ["Px", "Py", "Pz", "E"]
        v3sets = [
            {
                b.split("_")[-1][:-1]
                for b in self._filter_branches(all_branches, f"*{v}")
            }
            for v in v3var
        ]
        v3set = v3sets[0].intersection(v3sets[1], v3sets[2])

        v4sets = [
            {
                b.split("_")[-1][: -len(v)]
                for b in self._filter_branches(all_branches, f"*{v}")
            }
            for v in v4var
        ]
        v4set = v4sets[0].intersection(v4sets[1], v4sets[2])

        v3names = [s + v for s in v3set for v in v3var]
        v4names = [s + v for s in v4set for v in v4var]

        branch_behavior = {}
        for b in all_branches:
            b_end = b.split("_")[-1]
            behavior = ""
            if b_end in v3names:
                behavior = "ThreeVector"
            elif b_end in v4names:
                behavior = "FourVector"
            branch_behavior[b] = behavior
        return branch_behavior

    def _sort_branches(self, branches):
        return sorted(
            branches,
            key=lambda x: len(x.split("_")) + (self.branch_behavior_dict[x] != ""),
            reverse=True,
        )

    def _build_collections(self, branch_forms):
        self.branch_behavior_dict = self._type_dictionary_builder(branch_forms)

        key_form_dict = {}
        key_dict = {}
        obj_lists = list(self.top_objects.keys())
        branches = self._sort_branches(branch_forms.keys())

        for key in branches:
            ak_form = branch_forms[key]
            behavior = self.branch_behavior_dict[key]

            which_top_key = [t in key for t in obj_lists]
            if sum(which_top_key) == 0:
                continue

            top_key = "".join(
                t * w for t, w in zip(obj_lists, which_top_key, strict=False)
            )
            objname = self.top_objects[top_key]
            sub_keys = key.replace(top_key, objname).split("_")[1:]
            last_key = sub_keys[-1]

            if behavior != "":
                v = None
                if any(last_key.endswith(x) for x in ["X", "Y", "Z", "E"]):
                    v = last_key[-1].lower()
                    v = "energy" if v == "e" else v
                    last_key = last_key[:-1] + ("3D" if v != "energy" else "4D")
                if any(last_key.endswith(x) for x in ["Px", "Py", "Pz"]):
                    v = last_key[-2].lower()
                    last_key = last_key[:-1] + "4D"
                sub_keys[-1] = last_key
                self._mixins[last_key] = (
                    "ThreeVector" if last_key.endswith("3D") else "LorentzVector"
                )
                sub_keys.append(v)

            keys = [objname, *sub_keys]
            self._recursion(keys, ak_form, key_form_dict, 0)
            self._recursion(keys[:-1], "obj", key_dict, 0)

        return self._recursive_zip(key_form_dict, key_dict, "Events")

    @classmethod
    def behavior(cls):
        """Behaviors necessary to implement this schema (dict)"""
        from awkward_zipper.behaviors import pdune

        return pdune.behavior
