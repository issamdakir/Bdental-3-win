import bpy, os, sys
from os.path import join, dirname, exists, abspath
from .Operators.BDENTAL_Utils import *


ADDON_DIR = dirname(abspath(__file__))
RESOURCES = join(ADDON_DIR, "Resources")
Addon_Version_Path = join(RESOURCES, "BDENTAL_Version.txt")
Addon_Version_Date = "  "
if exists(Addon_Version_Path):
    with open(Addon_Version_Path, "r") as rf:
        lines = rf.readlines()
        Addon_Version_Date = lines[0].split(";")[0]


    
# Selected icons :
red_icon = "COLORSET_01_VEC"
orange_icon = "COLORSET_02_VEC"
green_icon = "COLORSET_03_VEC"
blue_icon = "COLORSET_04_VEC"
violet_icon = "COLORSET_06_VEC"
yellow_icon = "COLORSET_09_VEC"
yellow_point = "KEYTYPE_KEYFRAME_VEC"
blue_point = "KEYTYPE_BREAKDOWN_VEC"

Wmin, Wmax = -400, 3000

def get_icon_value(icon_name: str) -> int:
    icon_items = bpy.types.UILayout.bl_rna.functions["prop"].parameters["icon"].enum_items.items()
    icon_dict = {tup[1].identifier : tup[1].value for tup in icon_items}

    return icon_dict[icon_name]
class BDENTAL_PT_MainPanel(bpy.types.Panel):
    """Main Panel"""

    bl_idname = "BDENTAL_PT_MainPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"  # blender 2.7 and lower = TOOLS
    bl_category = "BDENTAL"
    bl_label = f"BDENTAL (ver. {Addon_Version_Date})"
    # bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):

        # Draw Addon UI :
        layout = self.layout
        # Box = layout.box()
        grid = layout.grid_flow(columns=2, align=True)
        grid.operator("wm.bdental_checkupdate")
        grid.operator("wm.bdental_support_telegram")

class BDENTAL_PT_GeneralPanel(bpy.types.Panel):
    """General Panel"""

    bl_idname = "BDENTAL_PT_GeneralPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"  # blender 2.7 and lower = TOOLS
    bl_category = "BDENTAL"
    bl_label = "GENERAL"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        BDENTAL_Props = context.scene.BDENTAL_Props

        # Draw Addon UI :
        layout = self.layout
        #######################
        Box = layout.box()
        grid = Box.grid_flow(columns=2, align=True)
        # grid.label(text="Project Directory")
        grid.prop(BDENTAL_Props, "UserProjectDir", text="Project Directory")
        grid = Box.grid_flow(columns=2, align=True)

        grid.operator("wm.open_mainfile", text="Open", icon="FILE_FOLDER")
        grid.operator("ed.undo", text="Undo", icon="LOOP_BACK")
        grid.operator("wm.save_mainfile", text="Save", icon="FOLDER_REDIRECT")
        grid.operator("wm.bdental_import_mesh", icon="IMPORT")

        grid.operator("wm.bdental_remove_info_footer", icon="CANCEL")
        grid.operator("ed.redo", text="Redo", icon="LOOP_FORWARDS")
        
        grid.operator("wm.save_as_mainfile", text="Save As...", icon="FILE_BLEND")
        grid.operator("wm.bdental_export_mesh", icon="EXPORT")
        
class BDENTAL_PT_DicomPanel(bpy.types.Panel):
    """Dicom Panel"""

    bl_idname = "BDENTAL_PT_DicomPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"  # blender 2.7 and lower = TOOLS
    bl_category = "BDENTAL"
    bl_label = "DICOM"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):

        BDENTAL_Props = context.scene.BDENTAL_Props
        layout = self.layout

        if BDENTAL_Props.UserProjectDir:
            box = layout.box()
            g = box.grid_flow(columns=1, align=True)
            g.prop(BDENTAL_Props, "DataType", text="Data Type")
            if BDENTAL_Props.DataType == "DICOM Series":
                g.prop(BDENTAL_Props, "UserDcmDir", text="DICOM Folder")
                if BDENTAL_Props.UserDcmDir:
                    g.prop(BDENTAL_Props, "Dicom_Series", text="DICOM Series")
                    g.prop(BDENTAL_Props, "scan_resolution")
                    g.operator("wm.bdental_organize")
                    g.operator("wm.bdental_volume_render")

            if BDENTAL_Props.DataType == "3D Image File":
                g.prop(BDENTAL_Props, "UserImageFile", text="3D IMAGE File")
                if BDENTAL_Props.UserImageFile:
                    g.prop(BDENTAL_Props, "scan_resolution")
                    g.operator("wm.bdental_organize")
                    g.operator("wm.bdental_volume_render")

        if context.object and context.object.get("bdental_type") in ["CT_Voxel"]:

            box = layout.box()

            g = box.grid_flow(columns=1, align=True)
            g.label(text=f"THTRESHOLD ({Wmin}/{Wmax})")
            g.prop(BDENTAL_Props, "TresholdMin", text="Min", slider=True)
            g.prop(BDENTAL_Props, "TresholdMax", text="Max", slider=True)
            g.label(text="DICOM TO MESH")
            g = box.grid_flow(columns=3, align=True)
            
            g.prop(BDENTAL_Props, "SoftTreshold", text="Soft Tissue")
            g.prop(BDENTAL_Props, "SoftSegmentColor", text="")
            g.prop(BDENTAL_Props, "SoftBool", text="")

            g.prop(BDENTAL_Props, "BoneTreshold", text="Bone")
            g.prop(BDENTAL_Props, "BoneSegmentColor", text="")
            g.prop(BDENTAL_Props, "BoneBool", text="")

            g.prop(BDENTAL_Props, "TeethTreshold", text="Teeth")
            g.prop(BDENTAL_Props, "TeethSegmentColor", text="")
            g.prop(BDENTAL_Props, "TeethBool", text="")

            g = box.grid_flow(columns=1, align=True)
            g.operator("wm.bdental_multitresh_segment")

class BDENTAL_PT_SlicesPanel(bpy.types.Panel):
    """Slices Panel"""

    bl_idname = "BDENTAL_PT_SlicesPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"  # blender 2.7 and lower = TOOLS
    bl_category = "BDENTAL"
    bl_label = "SLICES"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        BDENTAL_Props = context.scene.BDENTAL_Props
        layout = self.layout
        box = layout.box()
        row = box.row()
        row.operator("wm.bdental_addslices", icon="EMPTY_AXIS")
                

        row = box.row()
        row.alert = True 
        mats = [mat for mat in bpy.data.materials if "_SLICE_mat" in mat.name]
        if not mats:
            row.alert = False 
                 
        row.prop(BDENTAL_Props, "slices_brightness", text="Brightness")
        row.prop(BDENTAL_Props, "slices_contrast", text="Contrast")
                
        row = box.row()
        row.label(text="Axial Slice Flip :")

        row = box.row()
        row.operator("wm.bdental_flip_camera_axial_90_plus", icon="PLUS")
        row.operator("wm.bdental_flip_camera_axial_90_minus", icon="REMOVE")
        row.operator("wm.bdental_flip_camera_axial_up_down", icon="TRIA_UP")
        row.operator("wm.bdental_flip_camera_axial_left_right", icon="TRIA_RIGHT")

        row = box.row()
        row.label(text="Coronal Slice Flip :")

        row = box.row()
        row.operator("wm.bdental_flip_camera_coronal_90_plus", icon="PLUS")
        row.operator("wm.bdental_flip_camera_coronal_90_minus", icon="REMOVE")
        row.operator("wm.bdental_flip_camera_coronal_up_down", icon="TRIA_UP")
        row.operator("wm.bdental_flip_camera_coronal_left_right", icon="TRIA_RIGHT")

        row = box.row()
        row.label(text="Sagittal Slice Flip :")

        row = box.row()
        row.operator("wm.bdental_flip_camera_sagittal_90_plus", icon="PLUS")
        row.operator("wm.bdental_flip_camera_sagittal_90_minus", icon="REMOVE")
        row.operator("wm.bdental_flip_camera_sagittal_up_down", icon="TRIA_UP")
        row.operator("wm.bdental_flip_camera_sagittal_left_right", icon="TRIA_RIGHT")

class BDENTAL_PT_ToolsPanel(bpy.types.Panel):
    """Tools Panel"""

    bl_idname = "BDENTAL_PT_ToolsPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"  # blender 2.7 and lower = TOOLS
    bl_category = "BDENTAL"
    bl_label = "TOOLS"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        BDENTAL_Props = context.scene.BDENTAL_Props
        layout = self.layout
        ob = context.object
        
        # Model color :
        if ob and ob.type in ["MESH", "CURVE"]:
            try :
                mat = ob.active_material
                Box = layout.box()
                
                grid = Box.grid_flow(columns=2, align=True)
                
                if mat :
                    grid.prop(mat, "diffuse_color", text="")
                else :
                    grid.template_icon(get_icon_value("COLORSET_12_VEC"), scale=1.5)
                grid.label(text="COLOR")#icon=yellow_point
                

                grid = Box.grid_flow(columns=2, align=True)
                grid.operator("wm.bdental_add_color", text="Add Color", icon="MATERIAL")
                # grid.template_ID(context.object, "active_material",new="material.new",live_icon=0)
                if mat :
                    grid.operator("wm.bdental_remove_color", text="Remove Color")
                else :
                    grid.operator("wm.bdental_remove_color", text="Remove Color")
            except :
                pass

        Box = layout.box()
        grid = Box.grid_flow(columns=2, align=True)
        grid.label(text="RELATIONS")
        grid.template_icon(get_icon_value("LINKED"), scale=1.5)

        grid = Box.grid_flow(columns=2, align=True)
        grid.operator("wm.bdental_parent_object", text="Parent", icon="LINKED")
        grid.operator("wm.bdental_join_objects", text="Join", icon="SNAP_FACE")
        grid.operator("wm.bdental_lock_objects", text="Lock", icon="LOCKED")

        
        grid.operator("wm.bdental_unparent_objects", text="Un-Parent", icon="LIBRARY_DATA_OVERRIDE")
        grid.operator("wm.bdental_separate_objects", text="Separate", icon="SNAP_VERTEX")
        grid.operator("wm.bdental_unlock_objects", text="Un-Lock", icon="UNLOCKED")

        # Model Repair Tools :
        # layout.separator()
        
        Box = layout.box()
        grid = Box.grid_flow(columns=2, align=True)
        grid.label(text="REPAIR")  
        grid.template_icon(get_icon_value("TOOL_SETTINGS"), scale=1.5)

        grid = Box.grid_flow(columns=2, align=True)

        grid.operator("wm.bdental_decimate", text="Decimate", icon="MOD_DECIM")
        grid.operator("wm.bdental_clean_mesh2", text="Clean Mesh", icon="BRUSH_DATA")
        grid.operator("wm.bdental_retopo_smooth", text="Retopo-Smooth", icon="SMOOTHCURVE")
        grid.operator("wm.bdental_normals_toggle")

        grid.prop(BDENTAL_Props, "decimate_ratio", text="")
        grid.operator("wm.bdental_voxelremesh", text="Remesh", icon="MOD_REMESH")
        if ob and ob.mode == "SCULPT":
            try : grid.operator("sculpt.sample_detail_size", text="", icon="EYEDROPPER")
            except : grid.operator("wm.bdental_fill", text="Fill", icon="OUTLINER_OB_LIGHTPROBE")
        else : grid.operator("wm.bdental_fill", text="Fill", icon="OUTLINER_OB_LIGHTPROBE")
        grid.operator("wm.bdental_flip_normals")


        # Cutting Tools :
        # layout.row().separator()
        Box = layout.box()
        g = Box.grid_flow(columns=2, align=True)
        g.label(text="CUT") 
        g.template_icon(get_icon_value("COLOR"), scale=1.5)
        
        g = Box.grid_flow(columns=1, align=True)
        g.prop(BDENTAL_Props, "Cutting_Tools_Types_Prop", text="Cutters")
        if BDENTAL_Props.Cutting_Tools_Types_Prop == "Path Split" :
            g = Box.grid_flow(columns=2, align=True)
            g.operator("wm.bdental_curvecutteradd2", text="Add Cutter", icon="GP_SELECT_STROKES")
            g.operator(
                "wm.bdental_curvecutter2_cut_new",
                text="Perform Cut",
                icon="GP_MULTIFRAME_EDITING",
            )

        elif BDENTAL_Props.Cutting_Tools_Types_Prop == "Ribbon Split":
            g = Box.grid_flow(columns=2, align=True)
            g.operator("wm.bdental_curvecutter1_new", text="Add Cutter", icon="GP_SELECT_STROKES")
            
            g.operator("wm.bdental_curvecutter1_new_perform_cut", text="Perform Cut", icon="GP_MULTIFRAME_EDITING")
        

        elif BDENTAL_Props.Cutting_Tools_Types_Prop == "Ribbon Cutter":
            g = Box.grid_flow(columns=2, align=True)
            g.operator(
                "wm.bdental_ribboncutteradd", text="Add Cutter", icon="GP_SELECT_STROKES"
            )
            g.operator(
                "wm.bdental_ribboncutter_perform_cut",
                text="Perform Cut",
                icon="GP_MULTIFRAME_EDITING",
            )
        elif BDENTAL_Props.Cutting_Tools_Types_Prop == "Frame Cutter":

            g = Box.grid_flow(columns=2, align=True)
            g.prop(BDENTAL_Props, "cutting_mode", text="Cutting Mode")
            g.operator("wm.bdental_add_square_cutter", text="Frame Cutter")
            

        # elif BDENTAL_Props.Cutting_Tools_Types_Prop == "Paint Cutter":

        #     row = Box.row()
        #     row.operator("wm.bdental_paintarea_toggle", text="Add Cutter")
        #     row.operator("wm.bdental_paintarea_plus", text="", icon="ADD")
        #     row.operator("wm.bdental_paintarea_minus", text="", icon="REMOVE")
        #     row = Box.row()
        #     row.operator("wm.bdental_paint_cut", text="Perform Cut")
        
        # elif BDENTAL_Props.Cutting_Tools_Types_Prop == "Path Cutter":
        #     row = Box.row()
        #     row.operator("wm.bdental_add_path_cutter")

        if context.object and context.object.get("bdental_type"):
            
            obj = context.object
            if obj.get("bdental_type") in ["curvecutter1", "curvecutter2"] :
                g = Box.grid_flow(columns=2, align=True)
                g.prop(obj.data, "extrude", text="Extrude")
                g.prop(obj.data, "offset", text="Offset")
            elif obj.get("bdental_type") == "curvecutter3" :
                g = Box.grid_flow(columns=3, align=True)
                g.prop(obj.data, "extrude", text="Extrude")
                g.prop(obj.data, "bevel_depth", text="Bevel")
                g.prop(obj.data, "offset", text="Offset")

        # Make BaseModel, survey, Blockout :
        # layout.separator()
        
        Box = layout.box()
        grid = Box.grid_flow(columns=2, align=True)
        grid.label(text="MODEL")
        grid.template_icon(get_icon_value("FILE_VOLUME"), scale=1.5)

        grid = Box.grid_flow(columns=2, align=True)

        grid.operator("wm.bdental_model_base", text="Make Model Base")
        grid.operator("wm.bdental_undercuts_preview", text="Preview Undercuts")

        grid.operator("wm.bdental_add_offset", text="Add Offset")
        grid.operator("wm.bdental_blockout_new", text="Blocked Model")
        
        # layout.separator()
        
        # Box = layout.box()
        # grid = Box.grid_flow(columns=2, align=True)
        # grid.label(text="TEETH")
        # grid.template_icon(get_icon_value("OUTLINER_OB_LATTICE"), scale=1.5)

        # grid = Box.grid_flow(columns=2, align=True)
        # grid.prop(BDENTAL_Props, "TeethLibrary", text="")
        # grid.operator("wm.bdental_add_teeth")

        Box = layout.box()
        grid = Box.grid_flow(columns=2, align=True)
        grid.label(text="TEXTE")
        grid.template_icon(get_icon_value("SMALL_CAPS"), scale=1.5)

        grid = Box.grid_flow(columns=2, align=True)
        grid.prop(BDENTAL_Props, "text", text="")
        grid.operator("wm.bdental_add_3d_text", text="Add 3D Text")

class BDENTAL_PT_ImplantPanel(bpy.types.Panel):
    """ Implant panel"""

    bl_idname = "BDENTAL_PT_ImplantPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"  # blender 2.7 and lower = TOOLS
    bl_category = "BDENTAL"
    bl_label = "IMPLANT"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        
        layout = self.layout
        Box = layout.box()
        # Box.label(text="IMPLANT")#icon=yellow_point
        g = Box.grid_flow(columns=3, align=True)
        g.operator("wm.bdental_slices_pointer_select", text="Select Pointer", icon="EMPTY_AXIS")
        g.operator("wm.bdental_fly_previous", text="", icon="TRIA_LEFT")
        g.operator("wm.bdental_fly_next", text="", icon="TRIA_RIGHT")
        g = Box.grid_flow(columns=2, align=True)

        g.operator("wm.bdental_add_implant", text="Add Implant", icon="ADD")
        g.operator("wm.bdental_lock_to_pointer")
        g.operator("wm.bdental_implant_to_pointer")
   
        g.operator("wm.bdental_remove_implant", text="Remove Implant" , icon="REMOVE")
        g.operator("wm.bdental_unlock_from_pointer")
        g.operator("wm.bdental_pointer_to_implant")

        g = Box.grid_flow(columns=1, align=True)
        g.operator("wm.bdental_align_implants")
              
class BDENTAL_PT_Guide(bpy.types.Panel):
    """ Guide Panel"""

    bl_idname = "BDENTAL_PT_Guide"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"  # blender 2.7 and lower = TOOLS
    bl_category = "BDENTAL"
    bl_label = "SURGICAL GUIDE"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        BDENTAL_Props = context.scene.BDENTAL_Props
        layout = self.layout

        Box = layout.box()
        

        row = Box.row()

        # row.operator("wm.bdental_add_guide_splint")
        row.operator("wm.bdental_add_guide_splint_geom")

        row = Box.row()
        row.operator("wm.bdental_add_tube")
        row.prop(BDENTAL_Props, "TubeCloseMode", text="")

        if context.active_object:
            if (
                "BDENTAL_GuideTube" in context.active_object.name
                and context.active_object.type == "CURVE"
            ):
                obj = context.active_object
                row = Box.row()
                row.prop(obj.data, "bevel_depth", text="Radius")
                row.prop(obj.data, "extrude", text="Extrude")
                row.prop(obj.data, "offset", text="Offset")

        # row = Box.row()
        # row.operator("wm.bdental_add_guide_cutters_from_sleeves")

        row = Box.row()
        row.operator("wm.bdental_guide_add_component")
        row = Box.row()
        row.operator("wm.bdental_asset_browser_toggle")


        # row = Box.row()
        # row.operator("wm.bdental_set_guide_components", icon="PLUS")
        # row = Box.row()
        # row.operator("wm.bdental_guide_set_cutters", icon="REMOVE")
        

        
        
        row = Box.row()
        row.operator("wm.bdental_guide_finalise")

####################################################################
class BDENTAL_PT_Align(bpy.types.Panel):
    """ALIGN Panel"""

    bl_idname = "BDENTAL_PT_Main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"  # blender 2.7 and lower = TOOLS
    bl_category = "BDENTAL"
    bl_label = "ALIGN"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        BDENTAL_Props = context.scene.BDENTAL_Props
        layout = self.layout
        Box = layout.box()
        
        g = Box.grid_flow(columns=2, align=True)
        g.operator("wm.bdental_alignpoints", text="ALIGN")
        g.operator("wm.bdental_alignpointsinfo", text="INFO", icon="INFO")

        is_ready = context.object and context.object in context.selected_objects and len(context.selected_objects)==2
        txt = []
        if BDENTAL_Props.AlignModalState:
            txt = ["WAITING FOR ALIGNEMENT..."]

        elif is_ready :
            target_name = context.object.name
            src_name = [
                obj
                for obj in context.selected_objects
                if not obj is context.object
            ][0].name

            txt = ["READY FOR ALIGNEMENT.", f"{src_name} will be aligned to, {target_name}"]

        else:
            txt = ["STANDBY MODE"]

        #########################################
        if txt :
            b2 = Box.box()
            b2.alert = True

            for t in txt :
                b2.label(text=t)
        
        # Align Tools :
        box = layout.box()
        g = box.grid_flow(columns=3, align=True)
        g.operator("wm.bdental_align_to_front", text="Align To Me", icon="AXIS_FRONT")
        g.operator("wm.bdental_to_center", text="Move To Center", icon="SNAP_FACE_CENTER")
        g.operator("wm.bdental_align_to_active", text="Align To Active")
        
        # row.operator("wm.bdental_align_to_cursor", text="Move To Cursor", icon="AXIS_FRONT")
        
        g = box.grid_flow(columns=2, align=True)
        g.operator("wm.bdental_occlusalplane", text="OCCLUSAL PLANE")
        g.operator("wm.bdental_occlusalplaneinfo", text="INFO", icon="INFO")

        g = box.grid_flow(columns=2, align=True)
        g.operator("wm.bdental_add_reference_planes", text="Ref planes")
        #Auto align :
        # box = layout.box()
        # row = box.row()
        # row.operator("wm.bdental_auto_align_icp", text="AUTO ALIGN")


##########################################################################################
# Registration :
##########################################################################################

classes = [
    BDENTAL_PT_MainPanel,
    BDENTAL_PT_GeneralPanel,
    BDENTAL_PT_DicomPanel,
    BDENTAL_PT_SlicesPanel,
    BDENTAL_PT_ImplantPanel,
    BDENTAL_PT_Align,
    BDENTAL_PT_ToolsPanel,
    BDENTAL_PT_Guide,
    
]


def register():

    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

