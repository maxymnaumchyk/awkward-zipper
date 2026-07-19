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


def _zip_arrays(members, record_name=None):
    """Zip a mapping of name -> array/Content into a record, array-based.

    Mirrors coffea's ``zip_forms``: if every member is a jagged
    ``ListOffsetArray`` the result is a single jagged record (``var * record``)
    sharing the first member's offsets; otherwise a flat ``RecordArray`` wrapping
    the members as-is (which may themselves be flat, jagged, or nested records).
    """
    names = list(members.keys())
    layouts = [
        m.layout if isinstance(m, awkward.Array) else m for m in members.values()
    ]
    params = {"__record__": record_name} if record_name is not None else {}

    if layouts and all(
        isinstance(layout, awkward.contents.ListOffsetArray) for layout in layouts
    ):
        offsets = layouts[0].offsets
        contents = [layout.content for layout in layouts]
        length = awkward._util.maybe_length_of(contents[0])
        record = awkward.contents.RecordArray(
            contents, names, length=length, parameters=params
        )
        return awkward.contents.ListOffsetArray(offsets=offsets, content=record)

    length = awkward._util.maybe_length_of(layouts[0]) if layouts else 0
    return awkward.contents.RecordArray(
        layouts, names, length=length, parameters=params
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
