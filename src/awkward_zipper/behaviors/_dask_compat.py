"""Minimal ``dask_property`` shim for behaviors ported from coffea.

awkward-zipper only targets eager and virtual modes, so the dask dispatch of
coffea's ``coffea.util.dask_property`` is not needed. This provides just enough
of the interface (``@dask_property`` and the ``.dask`` registration) for the
ported behavior modules to import and work eagerly. The registered dask variant
is ignored; ``_import_dask_awkward`` raises if actually reached.
"""

from typing import Any


class _DaskProperty:
    def __init__(self, func):
        self._func = func

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return self._func(instance)

    def dask(self, func):
        # eager/virtual only: keep the eager implementation, ignore dask variant
        return self


def dask_property(maybe_func=None, *, no_dispatch=False):
    def wrapper(func):
        return _DaskProperty(func)

    if maybe_func is None:
        return wrapper
    return wrapper(maybe_func)


def _isinstance(arg: Any, *class_prefixes: str) -> bool:
    """Return True if arg is an instance of a class with any given prefix."""
    for cls in type(arg).__mro__:
        class_name = f"{cls.__module__}.{cls.__qualname__}"
        if any(class_name.startswith(prefix) for prefix in class_prefixes):
            return True
    return False


def _import_dask_awkward():
    msg = "dask mode is not supported by awkward-zipper (eager/virtual only)"
    raise ModuleNotFoundError(msg)
