from __future__ import annotations

from typing import TYPE_CHECKING

from bpy.types import UIList

if TYPE_CHECKING:
    from bpy.types import Context
    from sbstudio.plugin.model.storyboard import StoryboardEntry

__all__ = ("SKYBRUSH_UL_storyboard_entries",)


class SKYBRUSH_UL_storyboard_entries(UIList):
    """Storyboard entry list with split color decoration."""

    bl_idname = "SKYBRUSH_UL_storyboard_entries"

    def draw_item(
        self,
        context: Context,
        layout,
        data,
        item: StoryboardEntry,
        icon,
        active_data,
        active_propname,
        index=0,
    ):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            row.use_property_decorate = False

            if hasattr(item, "split_color"):
                swatch = row.row(align=True)
                swatch.scale_x = 0.35
                swatch.prop(item, "split_color", text="")

            row.prop(item, "name", text="", emboss=False, icon_value=icon)
            if hasattr(item, "split_id"):
                row.label(text=f"#{item.split_id}")
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text="", icon_value=icon)
