import awkward
import numba
import numpy as np


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

    def _local2globalindex(index, counts, virtual=False):
        offsets = counts2offsets(counts)
        index = index.mask[index >= 0] + offsets[:-1]
        index = index.mask[index < offsets[1:]]  # guard against out of bounds
        # workaround ValueError: can not (unsafe) zip ListOffsetArrays with non-NumpyArray contents
        # index.type is N * var * int32?
        index = awkward.fill_none(index, -1)
        # use ensure array from coffea?
        if virtual:
            return awkward.flatten(index)
        return index

    # VirtualArray
    if (not index.layout.is_all_materialized) or (
        not counts.layout.is_all_materialized
    ):
        index_data = index.layout.content.data
        # resulting global index will have the same offsets as local index
        index_offsets = index.layout.offsets
        index_content = awkward._nplikes.virtual.VirtualArray(
            nplike=index_data._nplike,
            shape=(awkward._nplikes.shape.unknown_length,),
            dtype=np.int64,
            generator=lambda: _local2globalindex(
                awkward.materialize(index), awkward.materialize(counts), virtual=True
            ),
            shape_generator=index_data.shape,
        )
        index_content = awkward.contents.numpyarray.NumpyArray(index_content)
        return awkward.contents.ListOffsetArray(
            offsets=index_offsets, content=index_content
        )
    # concrete array
    return _local2globalindex(index, counts)


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
    if not all(
        isinstance(index.layout, awkward.contents.listoffsetarray.ListOffsetArray)
        for index in indices
    ):
        raise RuntimeError
    # return awkward.concatenate([idx[:, None] for idx in indexers], axis=1)

    # store offsets to later reapply them to the arrays
    offsets_stored = indices[0].layout.offsets
    # also store parameters
    parameters = {}
    for i, idx in enumerate(indices):
        if "__doc__" in parameters:
            parameters["__doc__"] += " and "
            parameters["__doc__"] += awkward.parameters(idx)["__doc__"]
        else:
            parameters["__doc__"] = "nested from "
            parameters["__doc__"] += awkward.parameters(idx)["__doc__"]
        # flatten the index
        indices[i] = awkward.Array(idx.layout.content)

    n = len(indices)
    out = np.empty(n * len(indices[0]), dtype="int64")
    for i, idx in enumerate(indices):
        #  index arrays should all be same shape flat arrays
        out[i::n] = idx
    offsets = np.arange(0, len(out) + 1, n, dtype=np.int64)
    out = awkward.Array(
        awkward.contents.ListOffsetArray(
            awkward.index.Index64(offsets),
            awkward.contents.NumpyArray(out),
        )
    )
    # reapply the offsets
    return awkward.Array(
        awkward.contents.ListOffsetArray(
            offsets_stored,
            out.layout,
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
    if not isinstance(
        local_counts.layout, awkward.contents.listoffsetarray.ListOffsetArray
    ):
        raise RuntimeError
    if not isinstance(target_offsets.layout, awkward.contents.numpyarray.NumpyArray):
        raise RuntimeError

    # count offsets the same way as with counts2offsets in coffea.nanoevents.transforms
    offsets = np.empty(len(target_offsets) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(target_offsets, out=offsets[1:])

    # store offsets to later reapply them to the arrays
    offsets_stored = local_counts.layout.offsets

    out = awkward.unflatten(
        np.arange(offsets[-1], dtype=np.int64),
        awkward.flatten(local_counts),
    )
    # reapply the offsets
    return awkward.Array(
        awkward.contents.ListOffsetArray(
            offsets_stored,
            out.layout,
        )
    )


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
    if not counts.layout.is_all_materialized:
        virtual_array = counts.layout.data
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


def distinct_parent(pdg, parents):
    """Compute first parent with distinct PDG id

    Signature: globalparents,globalpdgs,!distinctParent
    Expects global indexes, flat arrays, which should be same length
    """
    return _distinct_parent_kernel(awkward.Array(parents), awkward.Array(pdg))


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
