from collections.abc import Mapping

import awkward


def _check_equal_lengths(
    contents: list[awkward.contents.Content],
) -> int | awkward._nplikes.shape.UnknownLength:
    unknown_length = awkward._nplikes.shape.unknown_length
    length = awkward._util.maybe_length_of(contents[0])
    if length is unknown_length:
        return length
    for layout in contents:
        _length = awkward._util.maybe_length_of(layout)
        if _length is unknown_length:
            continue
        if awkward._util.maybe_length_of(layout) != length:
            msg = "all arrays must have the same length"
            raise ValueError(msg)
    return length


def _non_materializing_get_field(record, field):
    assert isinstance(field, str)
    if isinstance(record, Mapping):
        return record[field]
    if isinstance(record, awkward.Array):
        record = record.layout
    assert isinstance(record, awkward.contents.RecordArray)
    index = record.field_to_index(field)
    return awkward.Array(record._contents[index])


def _as_layout(arr):
    """Return the low-level layout of an awkward.Array (or pass a Content through)."""
    if isinstance(arr, awkward.Array):
        return arr.layout
    return arr


def _jagged_content(arr):
    """Flat content of a single-jagged (ListOffsetArray) array, without materializing."""
    return _as_layout(arr).content


def _jagged_offsets(arr):
    """Event-level offsets Index of a single-jagged (ListOffsetArray) array."""
    return _as_layout(arr).offsets


def _zip_jagged(members, offsets, record_name=None, parameters=None):
    """Build a jagged collection (``ListOffsetArray`` of ``RecordArray``).

    Parameters
    ----------
    members : dict[str, awkward.contents.Content]
        Mapping of field name to the flat content of each member (all members
        must share the same per-event ``offsets``).
    offsets : awkward.index.Index
        The shared event-level offsets.
    record_name : str, optional
        ``__record__`` parameter for the inner record.
    parameters : dict, optional
        Extra parameters for the inner record.
    """
    contents = tuple(members.values())
    fields = tuple(members.keys())
    length = _check_equal_lengths(contents)
    params = {}
    if record_name is not None:
        params["__record__"] = record_name
    if parameters:
        params.update(parameters)
    record = awkward.contents.RecordArray(
        contents, fields, length=length, parameters=params
    )
    return awkward.contents.ListOffsetArray(offsets=offsets, content=record)


def _append_record_fields(listoffset, new_members):
    """Append flat-content fields to the record inside a ``ListOffsetArray``."""
    record = listoffset.content
    contents = list(record.contents) + list(new_members.values())
    fields = list(record.fields) + list(new_members.keys())
    new_record = awkward.contents.RecordArray(
        contents, fields, length=record.length, parameters=record.parameters
    )
    return awkward.contents.ListOffsetArray(
        offsets=listoffset.offsets, content=new_record
    )


def _maybe_raw_generator(buffer):
    if isinstance(buffer, awkward._nplikes.virtual.VirtualNDArray):
        if hasattr(buffer._generator, "__awkward_raw_generator__"):
            # essentially we're forgetting all shape_generators here
            return buffer._generator.__awkward_raw_generator__
        return buffer._generator
    # maybe assert that buffer is an array-like here?
    return lambda: buffer


def _rewrap(array):
    # here we rerun `ak.from_buffers` so that `ak.from_buffers` correctly recreates dynamically all VirtualNDArrays
    # the current buffers may have 'wrong' generators on them. Not "wrong" in the sense that they would yield wrong
    # buffers, but rather they would load too much.
    form, length, buffers = awkward.to_buffers(array)
    buffers = {key: _maybe_raw_generator(buffer) for key, buffer in buffers.items()}
    return awkward.from_buffers(
        form, length, buffers, behavior=array.behavior, attrs=array.attrs
    )
