import functools

import awkward
import numba
import numpy as np


# function: tp.Callable[[],]
def dispatch_wrap(function):
    @functools.wraps(function)
    def _wrapper(*input_arrays, data=None, dtype=np.int64):
        """
        Calls a function and passes it input_arrays as parameters. Returns a function result in eager case.
         In virtual case returns a result from a function wrapped in a Virtual Array.
        Args:
            input_arrays: function parameters
            data: additional parameter for VirtualArray creation
            function: function to return
            dtype: additional parameter for VirtualArray creation

        Returns: function(input_arrays) or
         awkward.VirtualArray(generator=lambda: function(input_arrays))

        """
        # Virtual array
        if data is not None:
            return awkward._nplikes.virtual.VirtualArray(
                nplike=data._nplike,
                shape=(awkward._nplikes.shape.unknown_length,),
                dtype=dtype,
                generator=lambda: function(
                    *(awkward.materialize(array) for array in input_arrays)
                ),
                shape_generator=None,
            )
        # concrete array
        return function(*input_arrays)

    return _wrapper


def local2globalindex(index, counts):
    """
    Convert a jagged local index to a global index

    This is the same as local2global(index, counts2offsets(counts))
    where local2global and counts2offsets are as in coffea.nanoevents.transforms

    Example usage:
    Index array
    [[], [1], [], [2, 3]]
    Counts array
    [8, 7, 4, 7]
    Output array will be:
    [[], [9], [], [21, 22]]
    (here 21=8+7+4+2)
    """

    @dispatch_wrap
    def _local2globalindex(index, counts):
        offsets = counts2offsets(counts)
        index = index.mask[index >= 0] + offsets[:-1]
        index = index.mask[index < offsets[1:]]  # guard against out of bounds
        # workaround ValueError: can not (unsafe) zip ListOffsetArrays with non-NumpyArray contents
        # index.type is N * var * int32?
        index = awkward.fill_none(index, -1)
        # use ensure array from coffea?
        return awkward.flatten(index)

    # Check if VirtualArray
    index_data = None
    if not all(awkward.to_layout(_).is_all_materialized for _ in (index, counts)):
        index_data = index.layout.content.data

    # resulting global index will have the same offsets as local index
    index_offsets = index.layout.offsets

    index_content = _local2globalindex(index, counts, data=index_data)
    # index_content shape would be index_data.shape
    index_content = awkward.contents.numpyarray.NumpyArray(index_content)
    # create new parameters for the final array
    parameters = awkward.parameters(index)
    parameters["__doc__"] = "global " + parameters["__doc__"]
    return awkward.Array(
        awkward.contents.ListOffsetArray(
            offsets=index_offsets,
            content=index_content,
            parameters=parameters,
        )
    )


def nestedindex(indices):
    """
    Concatenate a list of indices along a new axis
    Outputs a jagged array with same outer shape as index arrays

    Example usage:
    First index array
    [[0, 2, 4],
     [8, 6]]
    Second index array
    [[1, 3, 5],
     [-1, 7]]
    Output
    [[[0, 1], [2, 3], [4, 5]],
     [[8, -1], [6, 7]]]
    """

    @dispatch_wrap
    def _nestedindex_content(indices):
        # return awkward.concatenate([idx[:, None] for idx in indexers], axis=1)
        flat_indices = []
        for idx in indices:
            # flatten the index
            flat_indices.append(awkward.Array(idx.layout.content))

        n = len(flat_indices)
        out = np.empty(n * len(flat_indices[0]), dtype="int64")
        for i, idx in enumerate(flat_indices):
            #  index arrays should all be same shape flat arrays
            out[i::n] = idx

        del flat_indices
        return out

    @dispatch_wrap
    def _get_nested_index_offsets(nested_index_content, indices):
        n = len(indices)
        return np.arange(0, len(nested_index_content) + 1, n, dtype=np.int64)

    def _combine_parameters(indices):
        parameters = {}
        for idx in indices:
            if "__doc__" in parameters:
                parameters["__doc__"] += " and "
            else:
                parameters["__doc__"] = "nested from "

            parameters["__doc__"] += awkward.parameters(idx)["__doc__"]
        return parameters

    if not all(
        isinstance(
            awkward.to_layout(index), awkward.contents.listoffsetarray.ListOffsetArray
        )
        for index in indices
    ):
        raise RuntimeError

    # Check if VirtualArray
    index_data = None
    if not all(awkward.to_layout(_).is_all_materialized for _ in indices):
        index_data = indices[0].layout.content.data

    # store offsets to later reapply them to the arrays
    offsets_stored = indices[0].layout.offsets
    nested_index_content = _nestedindex_content(
        indices, data=index_data, dtype=np.int64
    )

    nested_index_offsets = _get_nested_index_offsets(
        nested_index_content, indices, data=index_data, dtype=np.int64
    )

    # combine offsets and content
    nested_index = awkward.contents.ListOffsetArray(
        awkward.index.Index64(nested_index_offsets),
        awkward.contents.NumpyArray(nested_index_content),
    )
    # combine the parameters
    parameters = _combine_parameters(indices)
    # reapply the offsets
    return awkward.Array(
        awkward.contents.ListOffsetArray(
            offsets_stored,
            nested_index,
            parameters=parameters,
        )
    )


def counts2nestedindex(local_counts, target_offsets):
    """Turn jagged local counts into doubly-jagged global index into a target
    Outputs a jagged array with same axis-0 shape as counts axis-1

    Example usage:
    Local counts
    [[4, 3, 2],
     [4, 2]]
    Target offsets
    [9, 6]
    Target output
    [[[0, 1, 2, 3], [4, 5, 6], [7, 8]],
     [[9, 10, 11, 12], [13, 14]]]
    """

    @dispatch_wrap
    def _arange(array):
        return np.arange(array[-1], dtype=np.int64)

    @dispatch_wrap
    def _flatten(array):
        return awkward.flatten(array)

    if not isinstance(
        local_counts.layout, awkward.contents.listoffsetarray.ListOffsetArray
    ):
        raise RuntimeError
    if not isinstance(target_offsets.layout, awkward.contents.numpyarray.NumpyArray):
        raise RuntimeError

    # count offsets the same way as with counts2offsets in coffea.nanoevents.transforms
    offsets = counts2offsets(target_offsets)

    # store offsets to later reapply them to the arrays
    offsets_stored = local_counts.layout.offsets

    # Check if VirtualArray
    local_counts_data = local_counts_data_dtype = None
    if not all(
        awkward.to_layout(_).is_all_materialized for _ in (local_counts, target_offsets)
    ):
        local_counts_data = local_counts.layout.content.data
        local_counts_data_dtype = local_counts_data.dtype

    nested_index_content = _arange(offsets, data=local_counts_data)
    flat_counts = _flatten(
        local_counts, data=local_counts_data, dtype=local_counts_data_dtype
    )

    nested_index_offsets = counts2offsets(flat_counts)
    # combine offsets and content
    out = awkward.contents.ListOffsetArray(
        awkward.index.Index64(nested_index_offsets),
        awkward.contents.NumpyArray(nested_index_content),
    )

    # reapply the offsets
    return awkward.Array(
        awkward.contents.ListOffsetArray(
            offsets_stored,
            out,
        )
    )


# TODO: add dispatch_wrapper decorator for this function too
def counts2offsets(counts):
    # Cumulative sum of counts
    def _counts2offsets(counts):
        # awkward index default type is int64, so we use the same type for new arrays
        offsets = np.empty(len(counts) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        return offsets

    # VirtualArray
    # if isinstance(counts.layout.data, awkward._nplikes.virtual.VirtualArray):
    if (
        isinstance(counts, awkward._nplikes.virtual.VirtualArray)
        and not counts.is_materialized
    ):
        virtual_array = counts
    elif isinstance(counts, awkward.Array) and not counts.layout.is_all_materialized:
        virtual_array = counts.layout.data
    else:
        virtual_array = None

    if virtual_array is not None:
        return awkward._nplikes.virtual.VirtualArray(
            nplike=virtual_array._nplike,
            shape=(awkward._nplikes.shape.unknown_length,),
            dtype=np.int64,
            generator=lambda: _counts2offsets(virtual_array.materialize()),
            shape_generator=None,
        )
    # concrete array
    return _counts2offsets(counts.layout.data)


@numba.njit
def _children_kernel(offsets_in, parentidx):
    offsets1_out = np.empty(len(parentidx) + 1, dtype=np.int64)
    content1_out = np.empty(len(parentidx), dtype=np.int64)
    offsets1_out[0] = 0

    offset0 = 1
    offset1 = 0
    for record_index in range(len(offsets_in) - 1):
        start_src, stop_src = offsets_in[record_index], offsets_in[record_index + 1]

        for index in range(start_src, stop_src):
            for possible_child in range(index, stop_src):
                if parentidx[possible_child] == index:
                    if offset1 >= len(content1_out):
                        msg = "offset1 went out of bounds!"
                        raise RuntimeError(msg)
                    content1_out[offset1] = possible_child
                    offset1 = offset1 + 1
            if offset0 >= len(offsets1_out):
                msg = "offset0 went out of bounds!"
                raise RuntimeError(msg)
            offsets1_out[offset0] = offset1
            offset0 = offset0 + 1

    return offsets1_out, content1_out[:offset1]


def children(offsets, parents):
    """Compute children

    Signature: offsets,globalparents,!children
    Output will be a jagged array with same outer shape as globalparents content
    """
    coffsets, ccontent = _children_kernel(offsets, parents)
    return awkward.Array(
        awkward.contents.ListOffsetArray(
            awkward.index.Index64(coffsets),
            awkward.contents.NumpyArray(ccontent),
        )
    )


@numba.njit
def _distinct_parent_kernel(allpart_parent, allpart_pdg):
    out = np.empty(len(allpart_pdg), dtype=np.int64)
    for i in range(len(allpart_pdg)):
        parent = allpart_parent[i]
        if parent < 0:
            out[i] = -1
            continue
        thispdg = allpart_pdg[i]
        while parent >= 0 and allpart_pdg[parent] == thispdg:
            if parent >= len(allpart_pdg):
                msg = "parent index beyond length of array!"
                raise RuntimeError(msg)
            parent = allpart_parent[parent]
        out[i] = parent
    return out


def distinct_parent(parents, pdg):
    """Compute first parent with distinct PDG id

    Signature: globalparents,globalpdgs,!distinctParent
    Expects global indexes, flat arrays, which should be same length
    """
    if not isinstance(pdg.layout, awkward.contents.listoffsetarray.ListOffsetArray):
        raise RuntimeError
    if not isinstance(parents.layout, awkward.contents.listoffsetarray.ListOffsetArray):
        raise RuntimeError

    result_content = _distinct_parent_kernel(
        awkward.Array(parents.layout.content), awkward.Array(pdg.layout.content)
    )
    return awkward.Array(
        awkward.contents.ListOffsetArray(
            parents.layout.offsets,
            awkward.contents.NumpyArray(result_content),
        )
    )


@numba.njit
def _distinct_children_deep_kernel(offsets_in, global_parents, global_pdgs):
    offsets_out = np.empty(len(global_parents) + 1, dtype=np.int64)
    content_out = np.empty(len(global_parents), dtype=np.int64)
    offsets_out[0] = 0

    offset0 = 1
    offset1 = 0
    for record_index in range(len(offsets_in) - 1):
        start_src, stop_src = offsets_in[record_index], offsets_in[record_index + 1]

        for index in range(start_src, stop_src):
            this_pdg = global_pdgs[index]

            # only perform the deep lookup when this particle is not already part of a decay chain
            # otherwise, the same child indices would be repeated for every parent in the chain
            # which would require content_out to have a length that isa-priori unknown
            if (
                global_parents[index] >= 0
                and this_pdg != global_pdgs[global_parents[index]]
            ):
                # keep an index of parents with same pdg id
                parents = np.empty(stop_src - index, dtype=np.int64)
                parents[0] = index
                offset2 = 1

                # keep an additional index with parents that have at least one child
                parents_with_children = np.empty(stop_src - index, dtype=np.int64)
                offset3 = 0

                for possible_child in range(index, stop_src):
                    possible_parent = global_parents[possible_child]
                    possibe_child_pdg = global_pdgs[possible_child]

                    # compare with seen parents
                    for parent_index in range(offset2):
                        # check if we found a new child
                        if parents[parent_index] == possible_parent:
                            # first, remember that the parent has at least one child
                            if offset3 >= len(parents_with_children):
                                msg = "offset3 went out of bounds!"
                                raise RuntimeError(msg)
                            parents_with_children[offset3] = possible_parent
                            offset3 = offset3 + 1

                            # then, depending on the pdg id, add to parents or content
                            if possibe_child_pdg == this_pdg:
                                # has the same pdg id, add to parents
                                if offset2 >= len(parents):
                                    msg = "offset2 went out of bounds!"
                                    raise RuntimeError(msg)
                                parents[offset2] = possible_child
                                offset2 = offset2 + 1
                            else:
                                # has a different pdg id, add to content
                                if offset1 >= len(content_out):
                                    msg = "offset1 went out of bounds!"
                                    raise RuntimeError(msg)
                                content_out[offset1] = possible_child
                                offset1 = offset1 + 1
                            break

                # add parents with same pdg id that have no children
                for parent_index in range(1, offset2):
                    possible_child = parents[parent_index]
                    if possible_child not in parents_with_children[:offset3]:
                        if offset1 >= len(content_out):
                            msg = "offset1 went out of bounds! pt2"
                            raise RuntimeError(msg)
                        content_out[offset1] = possible_child

                        offset1 = offset1 + 1

            # finish this item by adding an offset
            if offset0 >= len(offsets_out):
                msg = "offset0 went out of bounds!"
                raise RuntimeError(msg)
            offsets_out[offset0] = offset1
            offset0 = offset0 + 1

    return offsets_out, content_out[:offset1]


def distinct_children_deep(global_pdgs, global_parents, offsets):
    """Compute all distinct children, skipping children with same pdg id in between.

    Signature: offsets,global_parents,global_pdgs,!distinctChildrenDeep
    Expects global indexes, flat arrays, which should be same length
    """
    coffsets, ccontent = _distinct_children_deep_kernel(
        offsets,
        global_parents,
        awkward.Array(global_pdgs),
    )
    return awkward.Array(
        awkward.contents.ListOffsetArray(
            awkward.index.Index64(coffsets),
            awkward.contents.NumpyArray(ccontent),
        )
    )
