import uproot

from awkward_zipper import NanoAOD


def test():
    tree = uproot.open("tests/samples/nano_dy.root")["Events"]

    access_log = []
    array = tree.virtual_arrays(ak_add_doc=True, access_log=access_log)

    restructure = NanoAOD(version="latest")
    _ = restructure(array)

    # add more assertions here
    assert len(access_log) == 0
