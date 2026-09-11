"""EDM4HEP data-model definitions (vendored ``edm4hep.yaml`` versions).

The EDM4HEP layout builder is driven by the official EDM4HEP yaml data model,
which describes each datatype's Members, VectorMembers, OneToOneRelations and
OneToManyRelations. The yaml files are vendored here so awkward-zipper does not
need coffea at runtime.
"""

import importlib.resources
from functools import partial

import yaml

root_dir = importlib.resources.files("awkward_zipper.assets")

# Zero-padded tags: lexical order is release order.
versions = sorted(
    p.name[len("edm4hep_v") : -len(".yaml")]
    for p in root_dir.iterdir()
    if p.name.startswith("edm4hep_v") and p.name.endswith(".yaml")
)


def _load_edm4hep_version(yamlfile):
    with yamlfile.open() as f:
        return yaml.safe_load(f)


edm4hep_ver = {
    version: partial(
        _load_edm4hep_version,
        yamlfile=root_dir / f"edm4hep_v{version}.yaml",
    )
    for version in versions
}

__all__ = ["edm4hep_ver", "versions"]
