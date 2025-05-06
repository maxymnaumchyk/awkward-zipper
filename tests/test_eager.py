import uproot

from awkward_zipper import NanoAOD


def test():
    tree = uproot.open("tests/samples/nano_dy.root")["Events"]
    array = tree.arrays(ak_add_doc=True)

    restructure = NanoAOD(version="latest")
    _ = restructure(array)

    # add test assertions here
