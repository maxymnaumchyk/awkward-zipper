class BaseLayoutBuilder:
    """Base class for all layout builders"""

    @classmethod
    def behavior(cls):
        """Behaviors necessary to implement this schema (dict)"""
        from awkward_zipper.behaviors import base

        return base.behavior
