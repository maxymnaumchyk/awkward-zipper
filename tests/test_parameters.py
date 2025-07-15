import awkward
import uproot
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

from awkward_zipper import NanoAOD

def compare_parameters(coffea_arr, zipper_arr, field=None) -> bool:
    """
    Compares parameters. Doesn't take into account parameters that zipper constructs differently from coffea.
    Args:
        coffea_arr: awkward array layout
        zipper_arr: awkward array layout
        field: internal argument to keep track of current filed name
    """
    # coffea_arr.is_list and zipper_arr.is_list
    if isinstance(
        coffea_arr, awkward.contents.listoffsetarray.ListOffsetArray
    ) and isinstance(zipper_arr, awkward.contents.listoffsetarray.ListOffsetArray):
        # in case these two lists have different parameters:
        if not awkward._parameters.parameters_are_equal(
            coffea_arr.parameters, zipper_arr.parameters
        ):
            # these doc parameters are different in coffea and zipper so don't compare those
            different_docs = "nested from "
            if zipper_arr.parameters["__doc__"].startswith(different_docs):
                return True

            # print the parameters if they are not the same
            print("(listoffsetarray) coffea params: ", coffea_arr.parameters)
            print("zipper params: ", zipper_arr.parameters)
            print(field)
            print()
        return compare_parameters(coffea_arr.content, zipper_arr.content, field)
    if coffea_arr.is_record and zipper_arr.is_record:
        # in case these two records have different parameters:
        if not awkward._parameters.parameters_are_equal(
            coffea_arr.parameters, zipper_arr.parameters
        ):
            # print the parameters if they are not the same
            print("(recordarray) coffea params: ", coffea_arr.parameters)
            print("zipper params: ", zipper_arr.parameters)
            print(field)
            print()
        return all(
            compare_parameters(coffea_arr.content(f), zipper_arr.content(f), field=f)
            for f in coffea_arr.fields
        )
    # in case these two arrays have different parameters:
    if not awkward._parameters.parameters_are_equal(
        coffea_arr.parameters, zipper_arr.parameters
    ):
        # these doc parameters are different in coffea and zipper so don't compare those
        different_docs = ("global ", "nested from ")
        if zipper_arr.parameters == {}:
            # edge case when coffea creates an empty '__doc__' parameter
            # For example, coffea.parameters:  {'__doc__': ''} and
            # zipper.parameters:  {}
            print("coffea params: ", coffea_arr.parameters)
            print("zipper params: ", zipper_arr.parameters)
            print(field)
            print()
        elif coffea_arr.parameters["typename"]:
            # coffea inherits 'typename' param from uproot automatically, but in zipper
            # you have to add it manually in tree.arrays function ('ak_add_doc' parameter)
            # don't compare these parameters for now
            pass
        elif not zipper_arr.parameters["__doc__"].startswith(different_docs):
            return False
    # otherwise, parameters are equal and return True
    return True


def parameters_test():
    # compare parameters separately
    print(compare_parameters(coffea_array.layout, zipper_array.layout))
    assert compare_parameters(coffea_array.layout, zipper_array.layout)


if __name__ == "__main__":
    # load a test root file
    file_name = "tests/samples/nano_dy.root"
    # Create a TTree from root
    tree = uproot.open(file_name)["Events"]
    # TTree -> awkward.Array[awkward.Record[str, awkward.Array]]
    array = tree.arrays(ak_add_doc=True)

    # construct an awkward array using awkward-zipper
    restructure = NanoAOD(version="latest")
    zipper_array = restructure(array)

    # construct an awkward array using coffea
    events = NanoEventsFactory.from_root(
        {file_name: "Events"},
        schemaclass=NanoAODSchema,
        mode='eager',
    )
    coffea_array = events.events()

    # run this only manually for debugging
    parameters_test()
