import os
import shutil
import math
import threading
import pickle
from math import radians, pi, sqrt, sin, cos
import numpy as np
from time import sleep, perf_counter as tpc
from queue import Queue
from os.path import join, dirname, abspath, exists, split, basename, isdir, isfile
from glob import glob

import gpu
from gpu_extras.batch import batch_for_shader
import blf

# Blender Imports :
import bpy
import bpy_extras
from mathutils import Matrix, Vector, Euler, kdtree
from bpy.props import (
    StringProperty,
    IntProperty,
    FloatProperty,
    EnumProperty,
    FloatVectorProperty,
    BoolProperty,
)
import SimpleITK as sitk
import vtk
import cv2

from vtk.util import numpy_support
from vtk import vtkCommand

# Global Variables :

# from . import BDENTAL_Utils
from .BDENTAL_Utils import *

Addon_Enable(AddonName="mesh_looptools", Enable=True)

addon_dir = dirname(dirname(abspath(__file__)))
Addon_Version_Path = join(addon_dir, "Resources","BDENTAL_Version.txt")
DataBlendFile = join(addon_dir, "Resources", "BlendData",
                     "BDENTAL_BlendData.blend")

# "VGS_Marcos_modified_MinMax"#"VGS_Marcos_modified"  # "VGS_Marcos_01" "VGS_Dakir_01"
GpShader = "VGS_Dakir_(-400-3000)"#"VGS_Dakir_01"#"VGS_Dakir_MinMax"
bdental_volume_node_name = "bdental_volume"
ProgEvent = vtkCommand.ProgressEvent

Wmin, Wmax = -400, 3000
DRAW_HANDLERS = []
SLICES_TXT_HANDLER = []
message_queue = Queue()
FLY_IMPLANT_INDEX = None
TELEGRAM_LINK = "https://t.me/bdental_support"
VERSION_URL = "https://raw.githubusercontent.com/issamdakir/Bdental-3-win/main/Resources/BDENTAL_Version.txt"

#######################################################################################
# functions :


def gpu_info_footer(rect_color, text_list, button=False, btn_txt="", pourcentage=100):
    if pourcentage <= 0:
        pourcentage = 1
    if pourcentage > 100:
        pourcentage = 100

    def draw_callback_function():

        w = int(bpy.context.area.width * (pourcentage/100))
        for i, txt in enumerate((reversed(text_list))):

            h = 30
            # color = [0.4, 0.4, 0.8, 1.000000]
            # color =[0.9, 0.5, 0.000000, 1.000000]
            draw_gpu_rect(0, h*i, w, h, rect_color)
            blf.position(0, 10, 10 + (h*i), 0)
            blf.size(0, 40, 30)
            r, g, b, a = (0.0, 0.0, 0.0, 1.0)
            blf.color(0, r, g, b, a)
            blf.draw(0, txt)

        if button:

            h = 30
            color = [0.8, 0.258385, 0.041926, 1.0]
            draw_gpu_rect(w-110, 2, 100, h-4, color)
            blf.position(0, w-85, 10, 0)
            blf.size(0, 40, 30)
            r, g, b, a = (0.0, 0.0, 0.0, 1.0)
            blf.color(0, r, g, b, a)
            blf.draw(0, btn_txt)

    info_handler = bpy.types.SpaceView3D.draw_handler_add(
        draw_callback_function, (), "WINDOW", "POST_PIXEL"
    )
    # redraw scene
    # bpy.ops.wm.redraw_timer(type = 'DRAW_WIN_SWAP', iterations = 1)
    # for area in bpy.context.window.screen.areas:
    #     if area.type == "VIEW_3D":
    #         area.tag_redraw()

    return info_handler

def get_btn_bb(btn_index=0, btn_width=100, btn_height=26, padding_x=10, padding_y=2, safe_area=5):
    area3d = None
    area3d_check = [
        area for area in bpy.context.screen.areas if area.type == "VIEW_3D"]
    if area3d_check:
        area3d = area3d_check[0]
    if area3d:
        w = area3d.width
        x_min = w - padding_x - (btn_width*(btn_index+1)) - \
            (padding_x*btn_index) - safe_area
        x_max = w - padding_x - (btn_width*(btn_index)) - \
            (padding_x*btn_index) + safe_area
        y_min = 0
        y_max = btn_height+safe_area
        btn_bb = {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max
        }
        return btn_bb
    else:
        return None

def draw_gpu_circle(center_2d, radius, segments, color_rgba):

    x, y = center_2d
    circle_co = []
    m = (1.0 / (segments - 1)) * (pi * 2)

    for p in range(segments):
        p1 = x + cos(m * p) * radius
        p2 = y + sin(m * p) * radius
        circle_co.append((p1, p2))

    shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'TRI_FAN', {"pos":  circle_co})
    shader.bind()
    shader.uniform_float("color", color_rgba)
    batch.draw(shader)

def draw_gpu_rect(x, y, w, h, rect_color):

    vertices = (
        (x, y), (x, y + h),
        (x + w, y + h), (x + w, y))

    indices = (
        (0, 1, 2), (0, 2, 3)
    )

    gpu.state.blend_set('ALPHA')
    shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
    batch = batch_for_shader(
        shader, 'TRIS', {"pos": vertices}, indices=indices)
    shader.bind()
    shader.uniform_float("color", rect_color)
    batch.draw(shader)

def update_info(message=[], remove_handlers=True, button=False, btn_txt="", pourcentage=100, redraw_timer=True, rect_color=[0.4, 0.4, 0.8, 1.000000]):
    global DRAW_HANDLERS

    if remove_handlers:
        for _h in DRAW_HANDLERS:
            bpy.types.SpaceView3D.draw_handler_remove(_h, "WINDOW")
        DRAW_HANDLERS = []
    if message:
        info_handler = gpu_info_footer(
            text_list=message, button=False, btn_txt="", pourcentage=100, rect_color=rect_color)
        DRAW_HANDLERS.append(info_handler)
    if redraw_timer:
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

def draw_slices_text_2d():
    text_color_rgba = [0.8, 0.6, 0.0, 1.0]
    text_Thikness = (40, 30)

    def draw_callback_function():
        data = bpy.context.scene.get("bdental_implant_data")
        if data:
            loc, txt = eval(data)
            region = bpy.context.region
            region_3d = bpy.context.space_data.region_3d

            locOnScreen = view3d_utils.location_3d_to_region_2d(
                region, region_3d, loc)
            blf.position(0, locOnScreen[0]+1, locOnScreen[1]-1, 0)
            blf.size(0, text_Thikness[0], text_Thikness[1])
            r, g, b, a = text_color_rgba
            blf.color(0, r, g, b, a)
            blf.draw(0, txt)

    slices_text_handler = bpy.types.SpaceView3D.draw_handler_add(
        draw_callback_function, (), "WINDOW", "POST_PIXEL"
    )
    return slices_text_handler

def update_slices_txt(remove_handlers=True):
    global SLICES_TXT_HANDLER

    if remove_handlers:
        for _h in SLICES_TXT_HANDLER:
            bpy.types.SpaceView3D.draw_handler_remove(_h, "WINDOW")
        SLICES_TXT_HANDLER = []

    slices_text_handler = draw_slices_text_2d()
    SLICES_TXT_HANDLER.append(slices_text_handler)
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

class BDENTAL_OT_checkUpdate(bpy.types.Operator):
    """ check addon update """

    bl_idname = "wm.bdental_checkupdate"
    bl_label = "check update"
    bl_options = {"REGISTER", "UNDO"}

    txt = []
    restart = False

    def draw(self, context):
        
        layout = self.layout
        layout.alignment = "EXPAND"
        layout.alert = True
        for t in self.txt :
            layout.label(text=t)
    
    def addon_update(self, _file, addon_dir,blender_path):
        for elmt in os.listdir(_file):
            fullpath = join(addon_dir,elmt)
            new_elmt = join(_file,elmt)
            if exists(fullpath) :
                if isfile(fullpath) :
                    os.remove(fullpath)
                    shutil.move(new_elmt, fullpath)
                else :
                    if not "bdental_modules" in elmt.lower() :
                        shutil.rmtree(fullpath)
                        shutil.move(new_elmt, fullpath)
                    else :
                        resources = join(addon_dir, "Resources")
                        fullpath = join(resources,elmt)
                        shutil.move(new_elmt, fullpath)

        os.system(f'"{blender_path}"')
        self.restart = True

    def exit_blender(self):
        while True :
            sleep(1)
            if self.restart :
                sys.exit(0)
            
    def execute(self, context):
        global addon_dir
        sys.path.pop(0)
        blender_path = bpy.app.binary_path
        t1 = threading.Thread(
                target=addon_update,
                args=[self._file, addon_dir, blender_path],
                daemon=True,
                )
        
        
        t1.start()
        self.exit_blender()
        
        return{"FINISHED"}

    def invoke(self, context, event):
        if not isConnected() :
            update_info(message=["Bdental update : Please check internet connexion !"], rect_color=[1,0,0,0.7])
            sleep(3)
            update_info()
            return{"CANCELLED"}

        global Addon_Version_Path
        global VERSION_URL
        
        with open(Addon_Version_Path, "r") as rf:
            lines = rf.readlines()
            current = int(lines[0].split(";")[1])
        import requests
        success = 0
        try :
            r = requests.get(VERSION_URL)
            success = r.ok
        except Exception as er :
            print(f"request github bdental version error : {er}")
        if not success :
            update_info(message=["Bdental update : server conexion error !"], rect_color=[1,0,0,0.7])
            sleep(3)
            update_info()
            return{"CANCELLED"}

        last_txt, last_num_txt = r.text.split(";")
        last_num = int(last_num_txt)
        if last_num <= current :
            update_info(message=["Bdental is up to date."], rect_color=[0,1,0.2,0.7])
            sleep(3)
            update_info()
            return{"CANCELLED"}

        update_info(message=[f"new version availible = {last_txt}","Downloading ..."], rect_color=[0.7,0.4,0.2,1])
        self.message, self._file = addon_download()
        if self.message :
            update_info(message=self.message, rect_color=[1,0,0,0.7])
            sleep(3)
            update_info()
            return{"CANCELLED"}

        update_info()
        self.txt = [
        "Warning :",
        "Ready for update !",
        "press ok to update",
        "Blender will restart automatically" ]

        wm = context.window_manager
        return wm.invoke_props_dialog(self,width=500)

class BDENTAL_OT_SetConfig(bpy.types.Operator):
    """ set bdental config """

    bl_idname = "wm.bdental_set_config"
    bl_label = "Set Bdental Interface"
    bl_options = {"REGISTER", "UNDO"}

    def draw(self, context):
        message = ["setting bdental as startup :", "Please save your project (Blender need to restart)"]
        layout = self.layout
        layout.alignment = "EXPAND"
        # layout.alert = True
        for txt in message :
            layout.label(text=txt)

    def execute(self, context):
        context.preferences.use_preferences_save = False
        bpy.ops.wm.save_userpref()
        success = reset_config_folder()
        print(f"Reset config folder path : Success = {success}")
        if success :
            update_info(message=["Bdental Configuration Success."], rect_color=[0,1,0,1])
            sleep(1)
            update_info()
            # bpy.ops.wm.open_mainfile(filepath = path_to_startup)
            # bpy.ops.wm.save_userpref()
            t = threading.Thread(
            target=start_blender_session,
            args=[],
            daemon=True,
            )
            t.start()
            sys.exit(0)
        else :
            update_info(message=["Bdental Configuration failed."], rect_color=[1,0,0,1])
            sleep(3)
            update_info()
        return{"FINISHED"}
    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)
class BDENTAL_OT_SupportTelegram(bpy.types.Operator):
    """ open telegram bdental support link"""

    bl_idname = "wm.bdental_support_telegram"
    bl_label = "Bdental Support (Telegram)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return isConnected()



    def execute(self, context):
        global TELEGRAM_LINK
        success = browse(TELEGRAM_LINK)
        if not success :
            txt = ["Go to Bdental Support : failed!"]
            rgba = [1, 0, 0, 1]
            update_info(message=txt, rect_color=rgba)
            sleep(3)
            update_info()
            return {"CANCELLED"}
        return{"FINISHED"}


class BDENTAL_OT_AddAppTemplate(bpy.types.Operator):
    """ add bdental application template """

    bl_idname = "wm.bdental_add_app_template"
    bl_label = "Add Bdental Template"
    bl_options = {"REGISTER", "UNDO"}

    display_message: BoolProperty(default=True)

    def execute(self, context):
        success = Add_bdental_app_template()
        print(f"add bdental app-template : Success = {bool(success)}")
        if self.display_message:
            if success == 0:
                txt = ["Adding Bdental app template : failed!"]
                rgba = [1, 0, 0, 1]
            elif success == 1:
                txt = ["Adding Bdental app template ; Success./"]
                rgba = [0, 1, 0, 1]
            # else:
            #     txt = ["Bdental app template already exists!"]
            #     rgba = [1.0, 0.2, 0.0, 1.0]

            update_info(message=txt, rect_color=rgba)
            sleep(3)
            update_info()
#             # bpy.ops.wm.open_mainfile(filepath = path_to_startup)
#             # bpy.ops.wm.save_userpref()
#             t = threading.Thread(
#             target=start_blender_session,
#             args=[],
#             daemon=True,
#             )
#             t.start()
#             sys.exit(0)

        return{"FINISHED"}


#######################################################################################
########################### CT Scan Load : Operators ##############################
#######################################################################################
class BDENTAL_OT_AutoAlignIcp(bpy.types.Operator):
    """ Automatic global and icp alignement """

    bl_idname = "wm.bdental_auto_align_icp"
    bl_label = "Automatic Alignement"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        is_valid = context.object and context.object.select_get() \
            and len(context.selected_objects) == 2 \
            and all([obj.type == "MESH" for obj in context.selected_objects])
        return is_valid

    def execute(self, context):
        # message = ["Automatic Alignement Processing ..."]
        # update_info(message=message)
        to_obj = context.object
        from_obj = [
            obj for obj in context.selected_objects if obj is not to_obj][0]
        print("from_obj : ", from_obj.name)
        print("to_obj : ", to_obj.name)
        bpy.ops.object.select_all(action='DESELECT')
        # for obj in [from_obj, to_obj] :
        #     bpy.ops.object.select_all(action='DESELECT')
        #     obj.select_set(True)
        #     context.view_layer.objects.active = obj
        #     bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

        from_pcd, to_pcd = preprocess_auto_align_meshes(from_obj, to_obj)
        print("from_pcd : ", from_pcd)
        print("to_pcd : ", to_pcd)
        transform = Matrix(registration_icp(
            from_pcd, to_pcd, threshold=0.05, iterations=500))
        print("transform matrix : ", transform)
        # transform = fast_registration_with_icp(from_obj, to_obj, threshold=0.1, iterations=100)
        from_obj.matrix_world = transform @ from_obj.matrix_world

        from_obj.select_set(True)
        to_obj.select_set(True)
        context.view_layer.objects.active = to_obj
        # message = ["Automatic Alignement Done !"]
        # update_info(message=message)
        # sleep(1)
        # update_info()
        return{"FINISHED"}


class BDENTAL_OT_CleanMeshIterative(bpy.types.Operator):
    """ clean mesh iterative """

    bl_idname = "wm.bdental_clean_mesh_iterative"
    bl_label = "Clean Mesh"
    bl_options = {"REGISTER", "UNDO"}
    keep_parts: EnumProperty(
        name="Keep Parts",
        description="Keep Parts",
        items=set_enum_items(["All", "Only Big"]),
        default="All",
    )

    @classmethod
    def poll(cls, context):
        return context.object

    def execute(self, context):
        info_dict = {}
        obj = context.object
        verts_count_start, edges_count_start, polygons_count_start = mesh_count(
            obj)
        info_dict["verts_count_start"] = verts_count_start
        info_dict["edges_count_start"] = edges_count_start
        info_dict["polygons_count_start"] = polygons_count_start

        non_manifold_count = count_non_manifold_verts(obj)
        if not non_manifold_count:
            message = ["Mesh is manifold"]
            update_info(message)
            sleep(2)
            update_info()
            return {"CANCELLED"}

        merge_verts(obj, threshold=0.001, all=True)
        delete_loose(obj)
        delete_interior_faces(obj)
        fill_holes(obj, _all=True, hole_size=400)

        return{"FINISHED"}

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)


class BDENTAL_OT_ImportMesh(bpy.types.Operator):
    """ Import mesh Operator """

    bl_idname = "wm.bdental_import_mesh"
    bl_label = " Import Mesh"

    scan_extention: EnumProperty(
        items=set_enum_items(["STL", "OBJ", "PLY"]),
        name="Scan Type",
        description="Scan Extention"
    )

    def execute(self, context):

        ext = self.scan_extention

        if ext == 'STL':
            bpy.ops.import_mesh.stl("INVOKE_DEFAULT", directory=bpy.path.abspath(context.scene.BDENTAL_Props.UserProjectDir))
        if ext == 'OBJ':
            bpy.ops.import_scene.obj("INVOKE_DEFAULT", directory=bpy.path.abspath(context.scene.BDENTAL_Props.UserProjectDir))
        if ext == 'PLY':
            bpy.ops.import_mesh.ply("INVOKE_DEFAULT", directory=bpy.path.abspath(context.scene.BDENTAL_Props.UserProjectDir))

        return {"FINISHED"}

    def invoke(self, context, event):

        wm = context.window_manager
        return wm.invoke_props_dialog(self)


class BDENTAL_OT_ExportMesh(bpy.types.Operator):
    """ Export Mesh Operator """

    bl_idname = "wm.bdental_export_mesh"
    bl_label = " Export Mesh"

    scan_extention: EnumProperty(
        items=set_enum_items(["STL", "OBJ", "PLY"]),
        name="Scan Type",
        description="Scan Extention"
    )

    @classmethod
    def poll(cls, context):
        return context.object and len(context.selected_objects) == 1 and context.object.type == "MESH"

    def execute(self, context):

        ext = self.scan_extention

        if ext == 'STL':
            bpy.ops.export_mesh.stl("INVOKE_DEFAULT",  use_selection=True)
        if ext == 'OBJ':
            bpy.ops.export_scene.obj("INVOKE_DEFAULT",  use_selection=True)
        if ext == 'PLY':
            bpy.ops.export_mesh.ply("INVOKE_DEFAULT",  use_selection=True)

        return {"FINISHED"}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


class BDENTAL_OT_AlignToActive(bpy.types.Operator):
    """ Align object to active object """

    bl_idname = "wm.bdental_align_to_active"
    bl_label = "Align to active"
    bl_options = {"REGISTER", "UNDO"}
    invert_z: BoolProperty(
        name="Invert Z",
        description="Invert Z axis",
        default=False
    )

    @classmethod
    def poll(cls, context):
        return context.object and len(context.selected_objects) == 2

    def execute(self, context):
        active_object = context.object
        obj = [o for o in context.selected_objects if not o is context.object][0]
        for obj in [active_object, obj]:
            context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(
                location=False, rotation=False, scale=True)

        # obj.location = active_object.location
        # obj.rotation_euler = active_object.rotation_euler
        obj.matrix_world[:3] = active_object.matrix_world[:3]
        if self.invert_z:
            obj.rotation_euler.rotate_axis("X", math.pi)

        return{"FINISHED"}

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)


# class BDENTAL_OT_AlignToActive(bpy.types.Operator):
#     """ Align object to active object """

#     bl_idname = "wm.bdental_align_to_active"
#     bl_label = "Align to active"
#     bl_options = {"REGISTER", "UNDO"}
#     invert_z : BoolProperty(
#         name="Invert Z",
#         description="Invert Z axis",
#         default=False
#     )
#     @classmethod
#     def poll(cls, context):
#         return context.object and len(context.selected_objects) == 2
#     def execute(self, context):
#         active_object = context.object
#         obj = [o for o in context.selected_objects if not o is context.object][0]

#         active_object_constraint_targets = []
#         if active_object.constraints :
#             for c in active_object.constraints :
#                 if c.type == "CHILD_OF" :
#                     active_object_constraint_targets.append([c.target, c.use_scale_x, c.use_scale_y, c.use_scale_z])
#                     context.view_layer.objects.active = active_object
#                     bpy.ops.constraint.apply(constraint=c.name)
#         obj_constraint_targets = []
#         if obj.constraints :
#             for c in obj.constraints :
#                 if c.type == "CHILD_OF" :
#                     obj_constraint_targets.append([c.target, c.use_scale_x, c.use_scale_y, c.use_scale_z])
#                     context.view_layer.objects.active = obj
#                     bpy.ops.constraint.apply(constraint=c.name)


#         obj.location = active_object.location
#         obj.rotation_euler = active_object.rotation_euler

#         if self.invert_z :
#             obj.rotation_euler.rotate_axis("X", math.pi)

#         if active_object_constraint_targets :
#             for c_settings in active_object_constraint_targets :
#                 c = active_object.constraints.new("CHILD_OF")
#                 c.target = c_settings[0]
#                 c.use_scale_x = c_settings[1]
#                 c.use_scale_y = c_settings[2]
#                 c.use_scale_z = c_settings[3]
#         if obj_constraint_targets :
#             for c_settings in obj_constraint_targets :
#                 c = obj.constraints.new("CHILD_OF")
#                 c.target = c_settings[0]
#                 c.use_scale_x = c_settings[1]
#                 c.use_scale_y = c_settings[2]
#                 c.use_scale_z = c_settings[3]
#         return{"FINISHED"}
#     def invoke(self, context, event):
#         wm = context.window_manager
#         return wm.invoke_props_dialog(self)


class BDENTAL_OT_LockObjects(bpy.types.Operator):
    """ Lock objects transform """

    bl_idname = "wm.bdental_lock_objects"
    bl_label = "LOCK OBJECT"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if not context.object or not context.object.select_get() or not len(context.selected_objects) == 1:
            return False

        if not tuple(context.object.lock_location) == (True, True, True) or not tuple(context.object.lock_rotation) == (True, True, True) or not tuple(context.object.lock_scale) == (True, True, True):
            return True
        return False

    def execute(self, context):
        for obj in context.selected_objects:
            obj.lock_location = (True, True, True)
            obj.lock_rotation = (True, True, True)
            obj.lock_scale = (True, True, True)
        return{"FINISHED"}


class BDENTAL_OT_UnlockObjects(bpy.types.Operator):
    """ Unock objects transform """

    bl_idname = "wm.bdental_unlock_objects"
    bl_label = "UNLOCK OBJECT"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if not context.selected_objects:
            return False
        locks = []
        for obj in context.selected_objects:
            locks.extend(list(obj.lock_location))
            locks.extend(list(obj.lock_rotation))
            locks.extend(list(obj.lock_scale))
        if any(locks):
            return True

    def execute(self, context):
        for obj in context.selected_objects:
            obj.lock_location = (False, False, False)
            obj.lock_rotation = (False, False, False)
            obj.lock_scale = (False, False, False)
        return{"FINISHED"}


class BDENTAL_OT_add_3d_text(bpy.types.Operator):
    """add 3D text """

    bl_label = "Add 3D Text"
    bl_idname = "wm.bdental_add_3d_text"
    text_color = [0.0, 0.0, 1.0, 1.0]
    text = "BDental"
    font_size = 4
    text_mode: EnumProperty(
        items=set_enum_items(["Embossed", "Engraved"]),
        name="Text Mode",
        default="Embossed",
    )

    @classmethod
    def poll(cls, context):
        if not context.object:
            return False
        return context.object.select_get() and context.object.type == "MESH"

    def invoke(self, context, event):

        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def execute(self, context):
        BDENTAL_Props = context.scene.BDENTAL_Props
        self.target = context.object

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.text_add(enter_editmode=False, align="CURSOR")
        self.text_ob = context.object
        self.text_ob["bdental_type"] = "bdental_text"
        self.text_ob.data.body = BDENTAL_Props.text
        self.text_ob.name = "Text_"+BDENTAL_Props.text

        self.text_ob.data.align_x = "CENTER"
        self.text_ob.data.align_y = "CENTER"
        self.text_ob.data.size = self.font_size
        bpy.ops.view3d.view_axis(type="TOP", align_active=True)

        # change curve settings:
        self.text_ob.data.extrude = 0.5
        self.text_ob.data.bevel_depth = 0.02
        self.text_ob.data.bevel_resolution = 6

        # add SHRINKWRAP modifier :
        shrinkwrap_modif = self.text_ob.modifiers.new(
            "SHRINKWRAP", "SHRINKWRAP")
        shrinkwrap_modif.use_apply_on_spline = True
        shrinkwrap_modif.wrap_method = "PROJECT"
        shrinkwrap_modif.offset = 0
        shrinkwrap_modif.wrap_mode = "ABOVE_SURFACE"
        shrinkwrap_modif.cull_face = "OFF"
        shrinkwrap_modif.use_negative_direction = True
        shrinkwrap_modif.use_positive_direction = True
        shrinkwrap_modif.use_project_z = True
        shrinkwrap_modif.target = self.target

        mat = bpy.data.materials.get(
            "bdental_text_mat") or bpy.data.materials.new("bdental_text_mat")
        mat.diffuse_color = self.text_color
        mat.roughness = 0.6
        self.text_ob.active_material = mat

        bpy.context.scene.tool_settings.use_snap = True
        bpy.context.scene.tool_settings.snap_elements = {'FACE'}
        bpy.context.scene.tool_settings.use_snap_align_rotation = True
        bpy.context.scene.tool_settings.use_snap_rotate = True

        update_info(["Press ESC to cancel, ENTER to confirm"])

        # run modal
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if not event.type in {'ESC', 'RET'}:
            return {'PASS_THROUGH'}
        if event.type in {'ESC'}:
            try:
                bpy.data.objects.remove(self.text_ob)
            except:
                pass
            update_info(["Cancelled ./"])
            sleep(1)
            update_info()
            return {'CANCELLED'}
        if event.type in {'RET'}:
            if self.text_mode == "Embossed":
                self.embosse_text(context)
            else:
                self.engrave_text(context)

            bpy.context.scene.tool_settings.use_snap = False
            update_info(["Finished ./"])
            sleep(1)
            update_info()
            return {'FINISHED'}
        return {'RUNNING_MODAL'}

    def engrave_text(self, context):
        update_info(["Text Remesh..."])
        context.view_layer.objects.active = self.text_ob
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.text_ob.select_set(True)
        bpy.ops.object.convert(target="MESH")
        remesh_modif = self.text_ob.modifiers.new("REMESH", "REMESH")
        remesh_modif.voxel_size = 0.05
        bpy.ops.object.convert(target="MESH")

        update_info(["Target Remesh..."])
        bpy.ops.object.select_all(action="DESELECT")
        self.target.select_set(True)
        bpy.context.view_layer.objects.active = self.target

        remesh_modif = self.target.modifiers.new("REMESH", "REMESH")
        remesh_modif.mode = "SMOOTH"
        remesh_modif.octree_depth = 8

        voxel_remesh = self.target.modifiers.new("VOXEL", "REMESH")

        bpy.ops.object.convert(target="MESH")

        update_info(["Text Engraving ..."])
        difference_modif = self.target.modifiers.new("DIFFERENCE", "BOOLEAN")
        difference_modif.object = self.text_ob
        difference_modif.operation = "DIFFERENCE"
        bpy.ops.object.convert(target="MESH")

        bpy.data.objects.remove(self.text_ob)
        context.scene.BDENTAL_Props.text = "BDental"

    def embosse_text(self, context):
        update_info(["Text Remesh..."])
        context.view_layer.objects.active = self.text_ob
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.text_ob.select_set(True)
        bpy.ops.object.convert(target="MESH")
        remesh_modif = self.text_ob.modifiers.new("REMESH", "REMESH")
        remesh_modif.voxel_size = 0.05
        bpy.ops.object.convert(target="MESH")

        for mat_slot in self.text_ob.material_slots:
            bpy.ops.object.material_slot_remove({"object": self.text_ob})
        sleep(1)
        update_info(["Target Remesh..."])
        bpy.ops.object.select_all(action="DESELECT")
        self.target.select_set(True)
        context.view_layer.objects.active = self.target

        remesh_modif = self.target.modifiers.new("REMESH", "REMESH")
        remesh_modif.mode = "SMOOTH"
        remesh_modif.octree_depth = 8

        voxel_remesh = self.target.modifiers.new("VOXEL", "REMESH")

        bpy.ops.object.convert(target="MESH")

        update_info(["Text Embossing ..."])
        # join
        self.text_ob.select_set(True)
        bpy.ops.object.join()
        self.target = context.object
        voxel_remesh = self.target.modifiers.new("VOXEL", "REMESH")

        bpy.ops.object.convert(target="MESH")

        bpy.ops.object.select_all(action="DESELECT")
        self.target.select_set(True)
        context.view_layer.objects.active = self.target
        context.scene.BDENTAL_Props.text = "BDental"


class BDENTAL_OT_MPR2(bpy.types.Operator):
    """MultiView Toggle"""

    bl_idname = "wm.bdental_mpr2"
    bl_label = "MPR"
    is_update = False
    @classmethod
    def poll(cls, context):
        check_list = [
            obj for obj in context.scene.objects if obj.get("bdental_type") =="slices_pointer"]
        if not check_list:
            return False

        return True

    def execute(self, context):

        context.window.workspace = bpy.data.workspaces["Bdental Slicer"]
        slices_pointer = [
            obj for obj in context.scene.objects if obj.get("bdental_type") =="slices_pointer"][0]

        ws_slicer, scr_slicer, area_slicer_axial, area_slicer_coronal, area_slicer_sagittal, area_slicer_3d = get_slicer_areas()
        coll_names = [col.name for col in bpy.context.scene.collection.children if not (
            'SLICES' in col.name or
            'SLICES_POINTERS' in col.name or
            "GUIDE Components" in col.name)
        ]

        #####################################################
        # axial camera
        space_data_slicer_axial = area_slicer_axial.spaces.active
        region_slicer_axial = [r for r in area_slicer_axial.regions if r.type == 'WINDOW'][0]

        axial_cam = bpy.data.objects.get("1_AXIAL_SLICE_CAM")
        space_data_slicer_axial.use_local_collections = True
        space_data_slicer_axial.camera = axial_cam
        space_data_slicer_axial.use_local_camera = True

        override_slicer_axial = {
            "workspace": bpy.data.workspaces["Bdental Slicer"],
            "screen": scr_slicer,
            "area": area_slicer_axial,
            "space_data": space_data_slicer_axial,
            "region": region_slicer_axial,
        }
        bpy.ops.object.select_all(override_slicer_axial, action="DESELECT")
        slices_pointer.select_set(True)
        context.view_layer.objects.active = slices_pointer
        ################################################
        # coronal camera
        space_data_slicer_coronal = area_slicer_coronal.spaces.active
        region_slicer_coronal = [
            r for r in area_slicer_coronal.regions if r.type == 'WINDOW'][0]

        coronal_cam = bpy.data.objects.get("2_CORONAL_SLICE_CAM")
        space_data_slicer_coronal.use_local_collections = True
        space_data_slicer_coronal.camera = coronal_cam
        space_data_slicer_coronal.use_local_camera = True

        override_slicer_coronal = {
            "workspace": bpy.data.workspaces["Bdental Slicer"],
            "screen": scr_slicer,
            "area": area_slicer_coronal,
            "space_data": space_data_slicer_coronal,
            "region": region_slicer_coronal,
        }
        ################################################
        # sagittal camera
        space_data_slicer_sagittal = area_slicer_sagittal.spaces.active
        region_slicer_sagittal = [
            r for r in area_slicer_sagittal.regions if r.type == 'WINDOW'][0]
        
        sagittal_cam = bpy.data.objects.get("3_SAGITTAL_SLICE_CAM")
        space_data_slicer_sagittal.use_local_collections = True
        space_data_slicer_sagittal.camera = sagittal_cam
        space_data_slicer_sagittal.use_local_camera = True

        override_slicer_sagittal = {
            "workspace": bpy.data.workspaces["Bdental Slicer"],
            "screen": scr_slicer,
            "area": area_slicer_sagittal,
            "space_data": space_data_slicer_sagittal,
            "region": region_slicer_sagittal,
        }

        #####################################################
        # hide slices from Bdental Main Workspace
        main_ws = bpy.data.workspaces["Bdental Main"]
        scr_main = main_ws.screens[0]
        area3d_main = [a for a in scr_main.areas if a.type == 'VIEW_3D'][0]
        space_data_main = area3d_main.spaces.active
        region3d_main = [r for r in area3d_main.regions if r.type == 'WINDOW'][0]
        override_main = {
            "workspace": bpy.data.workspaces["Bdental Slicer"],
            "screen": scr_main,
            "area": area3d_main,
            "space_data": space_data_main,
            "region": region3d_main,
        }
        coll_names_main = ['SLICES']

        # if not context.scene.get("bdental_slicer_is_set") == True :
        
        space_data_main.use_local_collections = True
        for col_name in coll_names_main :
            index = getLocalCollIndex(col_name)
            if index :
                bpy.ops.object.hide_collection(
                override_main, collection_index=index, toggle=True)
                

        #####################################################
        # hide slices from Bdental Slicer Workspace (area 3d)
        space_data_slicer_3d = area_slicer_3d.spaces.active
        region_slicer_3d = [r for r in area_slicer_3d.regions if r.type == 'WINDOW'][0]
        override_slicer_3d = {
            "workspace": bpy.data.workspaces["Bdental Slicer"],
            "screen": scr_slicer,
            "area": area_slicer_3d,
            "space_data": space_data_slicer_3d,
            "region": region_slicer_3d,
            # 'active_object' : slices_pointer,
        }
        coll_names_slicer = ['SLICES']

        # if not context.scene.get("bdental_slicer_is_set") :
        
        space_data_slicer_3d.use_local_collections = True
        for col_name in coll_names_slicer :
            index = getLocalCollIndex(col_name)
            if index :
                bpy.ops.object.hide_collection(
                override_slicer_3d, collection_index=index, toggle=True)
                
        
        # bpy.ops.view3d.view_lock_to_active(override_slicer_3d)
        # bpy.ops.view3d.view_all(override_slicer_3d, center=True)
        # space_data_slicer_3d.region_3d.view_matrix = ViewMatrix
        bpy.ops.view3d.view_selected(override_slicer_3d, use_all_regions=False)
        ################################################



        
        for _override in [override_slicer_axial, override_slicer_coronal, override_slicer_sagittal]:
            bpy.ops.view3d.view_camera(_override)
            print("camera view triggred")
            if not context.scene.get("bdental_slicer_is_set")==True:
                for col_name in coll_names:
                    index = getLocalCollIndex(col_name)
                    if index :
                        bpy.ops.object.hide_collection(
                        _override, collection_index=index, toggle=True)
                            
                    
            # bpy.ops.view3d.view_selected(_override, use_all_regions=False)
            bpy.ops.wm.tool_set_by_id(_override, name="builtin.select")


       
        for space_data in [space_data_slicer_axial,
         space_data_slicer_coronal, 
         space_data_slicer_sagittal, 
         space_data_slicer_3d]:
            space_data.shading.type = "MATERIAL"

        bpy.ops.wm.tool_set_by_id(override_slicer_3d, name="builtin.select")

        # space_data_slicer_3d.shading.type = "RENDERED"
        space_data_slicer_3d.shading.use_scene_lights_render = True
        space_data_slicer_3d.shading.use_scene_world_render = True
        space_data_slicer_3d.shading.use_scene_lights = True
        space_data_slicer_3d.shading.use_scene_world = False

        ################################################
        context.scene.BDENTAL_Props.slices_brightness = 0.0
        context.scene.BDENTAL_Props.slices_contrast = 0.2
        context.scene["bdental_slicer_is_set"] = True
        return {"FINISHED"}


class BDENTAL_OT_MPR(bpy.types.Operator):
    """MultiView Toggle"""

    bl_idname = "wm.bdental_mpr"
    bl_label = "MPR"

    def execute(self, context):

        Active_Obj = bpy.context.view_layer.objects.active

        if not Active_Obj:
            message = [" Please select SLICES_POINTER "]
            icon = "COLORSET_02_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )
            return {"CANCELLED"}
        else:
            N = Active_Obj.name
            Condition = (
                N.startswith("BD")
                and "_SLICES_POINTER" in N
                and Active_Obj.select_get() == True
            )

            if not Condition:
                List = [
                    o for o in context.scene.objects if "_SLICES_POINTER" in o.name]
                if List:
                    message = [" Please select SLICES_POINTER ! "]
                else:
                    message = [
                        "To Add Multi-View Window :",
                        "1 - Please select CTVOLUME",
                        "2 - Click on < SLICE VOLUME > button",
                        "AXIAL, CORONAL and SAGITTAL slices will be added",
                        "3 - Ensure SLICES_POINTER is Selected",
                        "3 - Click <MPR> button",
                    ]
                    icon = "COLORSET_02_VEC"
                    bpy.ops.wm.bdental_message_box(
                        "INVOKE_DEFAULT", message=str(message), icon=icon
                    )
                return {"CANCELLED"}
            else:
                Preffix = Active_Obj.name[:6]
                AxialPlane = bpy.data.objects.get(f"1_{Preffix}_AXIAL_SLICE")
                CoronalPlane = bpy.data.objects.get(
                    f"2_{Preffix}_CORONAL_SLICE")
                SagittalPlane = bpy.data.objects.get(
                    f"3_{Preffix}_SAGITTAL_SLICE")
                SLICES_POINTER = Active_Obj

                if not AxialPlane or not CoronalPlane or not SagittalPlane:
                    message = [
                        "To Add Multi-View Window :",
                        "1 - Please select CTVOLUME",
                        "2 - Click on < SLICE VOLUME > button",
                        "AXIAL, CORONAL and SAGITTAL slices will be added",
                        "3 - Ensure SLICES_POINTER is Selected",
                        "3 - Click <MPR> button",
                    ]
                    icon = "COLORSET_02_VEC"
                    bpy.ops.wm.bdental_message_box(
                        "INVOKE_DEFAULT", message=str(message), icon=icon
                    )
                    return {"CANCELLED"}

                else:

                    (
                        MultiView_Window,
                        OUTLINER,
                        PROPERTIES,
                        AXIAL,
                        CORONAL,
                        SAGITTAL,
                        VIEW_3D,
                    ) = BDENTAL_MultiView_Toggle(Preffix)
                    MultiView_Screen = MultiView_Window.screen
                    AXIAL_Space3D = [
                        Space for Space in AXIAL.spaces if Space.type == "VIEW_3D"
                    ][0]
                    AXIAL_Region = [
                        reg for reg in AXIAL.regions if reg.type == "WINDOW"
                    ][0]

                    CORONAL_Space3D = [
                        Space for Space in CORONAL.spaces if Space.type == "VIEW_3D"
                    ][0]
                    CORONAL_Region = [
                        reg for reg in CORONAL.regions if reg.type == "WINDOW"
                    ][0]

                    SAGITTAL_Space3D = [
                        Space for Space in SAGITTAL.spaces if Space.type == "VIEW_3D"
                    ][0]
                    SAGITTAL_Region = [
                        reg for reg in SAGITTAL.regions if reg.type == "WINDOW"
                    ][0]
                    # AXIAL Cam view toggle :

                    AxialCam = bpy.data.objects.get(f"{AxialPlane.name}_CAM")
                    AXIAL_Space3D.use_local_collections = True
                    AXIAL_Space3D.use_local_camera = True
                    AXIAL_Space3D.camera = AxialCam
                    Override = {
                        "window": MultiView_Window,
                        "screen": MultiView_Screen,
                        "area": AXIAL,
                        "space_data": AXIAL_Space3D,
                        "region": AXIAL_Region,
                    }
                    bpy.ops.view3d.view_camera(Override)

                    # CORONAL Cam view toggle :
                    CoronalCam = bpy.data.objects.get(
                        f"{CoronalPlane.name}_CAM")
                    CORONAL_Space3D.use_local_collections = True
                    CORONAL_Space3D.use_local_camera = True
                    CORONAL_Space3D.camera = CoronalCam
                    Override = {
                        "window": MultiView_Window,
                        "screen": MultiView_Screen,
                        "area": CORONAL,
                        "space_data": CORONAL_Space3D,
                        "region": CORONAL_Region,
                    }
                    bpy.ops.view3d.view_camera(Override)

                    # AXIAL Cam view toggle :
                    SagittalCam = bpy.data.objects.get(
                        f"{SagittalPlane.name}_CAM")
                    SAGITTAL_Space3D.use_local_collections = True
                    SAGITTAL_Space3D.use_local_camera = True
                    SAGITTAL_Space3D.camera = SagittalCam
                    Override = {
                        "window": MultiView_Window,
                        "screen": MultiView_Screen,
                        "area": SAGITTAL,
                        "space_data": SAGITTAL_Space3D,
                        "region": SAGITTAL_Region,
                    }
                    bpy.ops.view3d.view_camera(Override)

                    bpy.ops.object.select_all(Override, action="DESELECT")
                    SLICES_POINTER.select_set(True)
                    bpy.context.view_layer.objects.active = SLICES_POINTER
                    for scr in bpy.data.screens:
                        for area in scr.areas:
                            if area.type == "VIEW_3D":
                                for space in area.spaces:
                                    if space.type == "VIEW_3D":
                                        space3D = space
                                        Scene_Settings(space3D)
                                        space3D.shading.type = "SOLID"
                                        space3D.shading.light = "FLAT"

        return {"FINISHED"}


class BDENTAL_OT_MessageBox(bpy.types.Operator):
    """Bdental popup message"""

    bl_idname = "wm.bdental_message_box"
    bl_label = "BDENTAL INFO"
    bl_options = {"REGISTER"}

    message: StringProperty()
    icon: StringProperty()

    def execute(self, context):
        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.alert = True
        box.alignment = "EXPAND"
        message = eval(self.message)
        for txt in message:
            row = box.row()
            row.label(text=txt)

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self, width=300)


class BDENTAL_OT_OpenManual(bpy.types.Operator):
    """Open BDENTAL Manual"""

    bl_idname = "wm.bdental_open_manual"
    bl_label = "User Manual"

    def execute(self, context):

        Manual_Path = join(addon_dir, "Resources", "BDENTAL User Manual.pdf")
        if exists(Manual_Path):
            os.startfile(Manual_Path)
            return {"FINISHED"}
        else:
            message = [" Sorry Manual not found."]
            icon = "COLORSET_02_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )
            return {"CANCELLED"}


def GetMaxSerie(UserDcmDir):

    SeriesDict = {}
    Series_reader = sitk.ImageSeriesReader()
    series_IDs = Series_reader.GetGDCMSeriesIDs(UserDcmDir)

    if not series_IDs:

        message = ["No valid DICOM Serie found in DICOM Folder ! "]
        print(message)
        icon = "COLORSET_01_VEC"
        bpy.ops.wm.bdental_message_box(
            "INVOKE_DEFAULT", message=str(message), icon=icon)
        return {"CANCELLED"}

    SeriesDict = {}
    for sID in series_IDs:
        series_file_names = Series_reader.GetGDCMSeriesFileNames(
            UserDcmDir, sID)
        count = len(series_file_names)
        SeriesDict[sID] = (count, series_file_names)

    Series_sorted_list = [
        (k, SeriesDict[k]) for k in sorted(SeriesDict, key=SeriesDict.get, reverse=True)
    ]
    max_sID, (max_count, max_series_file_names) = Series_sorted_list[0]

    return max_sID, max_count, max_series_file_names


class BDENTAL_OT_Organize(bpy.types.Operator):
    """DICOM Organize"""

    bl_idname = "wm.bdental_organize"
    bl_label = "ORGANIZE DICOM"

    def execute(self, context):
        BDENTAL_Props = context.scene.BDENTAL_Props
        if BDENTAL_Props.DataType == "DICOM Series":
            message = ["Reading DICOM Series ..."]
            update_info(message)

            UserProjectDir = AbsPath(BDENTAL_Props.UserProjectDir)
            UserDcmDir = AbsPath(BDENTAL_Props.UserDcmDir)

            DcmOrganizeDict = eval(BDENTAL_Props.DcmOrganize)

            if UserDcmDir in DcmOrganizeDict.keys():
                OrganizeReport = DcmOrganizeDict[UserDcmDir]

            else:
                if not exists(UserProjectDir):
                    txt = [
                        " Project Directory is not valid ! "]
                    update_info(message=txt, rect_color=[1,0,0,0.7])
                    sleep(2)
                    update_info()
                    return {"CANCELLED"}

                if not exists(UserDcmDir):
                    txt = [" The Selected dicom directory is not valid ! "]
                    update_info(message=txt, rect_color=[1,0,0,0.7])
                    sleep(2)
                    update_info()
                    return {"CANCELLED"}

                if not os.listdir(UserDcmDir):
                    txt = ["No files found in dicom directory ! "]
                    update_info(message=txt, rect_color=[1,0,0,0.7])
                    sleep(2)
                    update_info()
                    return {"CANCELLED"}

                Series_reader = sitk.ImageSeriesReader()

                try:
                    series_IDs_Files = []
                    for i, S_ID in enumerate(Series_reader.GetGDCMSeriesIDs(UserDcmDir)):
                        print(i)
                        series_IDs_Files.append(
                            [S_ID, Series_reader.GetGDCMSeriesFileNames(UserDcmDir, S_ID)])
                    
                except:
                    txt = ["scan data is not valid !"]
                    update_info(message=txt, rect_color=[1,0,0,0.7])
                    sleep(2)
                    update_info()
                    return {"CANCELLED"}

                tags = dict(
                    {
                        "Patient Name": "0010|0010",
                        "Series Date": "0008|0021",
                        "Series Description": "0008|103E",
                        "Patient Orientation": "0020|0020",  # 0x20, 0x20
                    }
                )
                DcmOrganizeDict[UserDcmDir] = {}
                for [S_ID, FilesList] in series_IDs_Files:
                    count = len(FilesList)
                    DcmOrganizeDict[UserDcmDir][S_ID] = {
                        "Count": count
                    }  # ,'Files':FilesList}

                    file0 = FilesList[0]
                    reader = sitk.ImageFileReader()
                    reader.SetFileName(file0)
                    reader.LoadPrivateTagsOn()
                    reader.ReadImageInformation()

                    Image = reader.Execute()
                    Spacing = Image.GetSpacing()[:2]
                    print(Spacing)
                    BDENTAL_Props.scan_resolution = max(Spacing)

                    for attribute, tag in tags.items():

                        if tag in Image.GetMetaDataKeys():
                            v = Image.GetMetaData(tag)

                        else:
                            v = "-------"

                        DcmOrganizeDict[UserDcmDir][S_ID][attribute] = v

                    DcmOrganizeDict[UserDcmDir][S_ID]["Files"] = FilesList
                    DcmOrganizeDict[UserDcmDir][S_ID]["spacing"] = Spacing

                SortedList = sorted(
                    DcmOrganizeDict[UserDcmDir].items(),
                    key=lambda x: x[1]["Count"],
                    reverse=True,
                )
                # Sorted = list(sorted(DcmOrganizeDict[UserDcmDir], reverse=True))

                SortedOrganizeDict = {}
                for i, (k, v) in enumerate(SortedList):
                    SortedOrganizeDict[
                        f"Series-{i} ({v['Count']} files)"
                    ] = DcmOrganizeDict[UserDcmDir][k]
                # for k,v in SortedOrganizeDict.items():
                #     print(k,' : ',v['Count'])

                DcmOrganizeDict[UserDcmDir] = SortedOrganizeDict
                BDENTAL_Props.DcmOrganize = str(DcmOrganizeDict)
                OrganizeReport = SortedOrganizeDict

            Message = {}
            for serie, info in OrganizeReport.items():
                Count, Name, Date, Descript, PatientOrientation = (
                    info["Count"],
                    info["Patient Name"],
                    info["Series Date"],
                    info["Series Description"],
                    info["Patient Orientation"]
                )
                Message[serie] = {
                    "Count": Count,
                    "Patient Name": Name,
                    "Series Date": Date,
                    "Series Description": Descript,
                    "Patient Orientation": PatientOrientation,
                }

            for serie, info in Message.items():
                print(serie, ":\n\t", info)

            BDENTAL_Props.OrganizeInfoProp = str(Message)
            # Save Blend File :
            bpy.ops.wm.save_mainfile()
            BDENTAL_Props.UserProjectDir = RelPath(UserProjectDir)

            # ProjectName = BDENTAL_Props.ProjectNameProp
            # BlendFile = f"{ProjectName}.blend"
            # Blendpath = join(UserProjectDir, BlendFile)
            # bpy.ops.wm.save_as_mainfile(filepath=Blendpath)
            # BDENTAL_Props.UserProjectDir = RelPath(UserProjectDir)


            n = len(series_IDs_Files)
            txt = [
                f"{n} dicom series found."]
            if n == 1 :
                txt = ["1 dicom serie found."]
            
            update_info(message=txt, rect_color=[0,1,0,0.7])

        else:
            txt = ["Reading 3D Image File ..."]
            update_info(message=txt)
            
            UserProjectDir = AbsPath(BDENTAL_Props.UserProjectDir)
            UserImageFile = AbsPath(BDENTAL_Props.UserImageFile)

            DcmOrganizeDict = eval(BDENTAL_Props.DcmOrganize)

            if UserImageFile in DcmOrganizeDict.keys():
                OrganizeReport = DcmOrganizeDict[UserImageFile]

            else:
                if not exists(UserProjectDir):
                    txt = [
                        " The Selected Project Directory is not valid ! "]
                    update_info(message=txt, rect_color=[1,0,0,0.7])
                    sleep(2)
                    update_info()
                    return {"CANCELLED"}

                if not exists(UserImageFile):
                    txt = [
                        " The Selected 3D Image filepath is not valid ! "]
                    update_info(message=txt, rect_color=[1,0,0,0.7])
                    sleep(2)
                    update_info()
                    return {"CANCELLED"}

                try:
                    reader = sitk.ImageFileReader()
                    reader.SetFileName(UserImageFile)
                    reader.LoadPrivateTagsOn()
                    reader.ReadImageInformation()

                    Image = reader.Execute()
                except Exception as e:
                    print("error reding 3D Image file :!")
                    print("path : ", abspath(UserImageFile))
                    print("error : ", e)
                    txt = [f"Can't open 3D Image file !"]
                    update_info(message=txt, rect_color=[1,0,0,0.7])
                    sleep(2)
                    update_info()
                    return {"CANCELLED"}

                tags = dict(
                    {
                        "Patient Name": "0010|0010",
                        "Patient Orientation": "0020|0020",  # 0x20, 0x20
                    }
                )
                DcmOrganizeDict[UserImageFile] = {}
                Sp = Spacing = Image.GetSpacing()
                BDENTAL_Props.scan_resolution = max(Sp)

                for attribute, tag in tags.items():

                    if tag in Image.GetMetaDataKeys():
                        v = Image.GetMetaData(tag)

                    else:
                        v = "-------"

                    DcmOrganizeDict[UserImageFile][attribute] = v

                DcmOrganizeDict[UserImageFile]["spacing"] = Sp
                BDENTAL_Props.DcmOrganize = str(DcmOrganizeDict)
                OrganizeReport = DcmOrganizeDict

                for k, v in OrganizeReport.items():
                    print(k, ":\n\t", v)

                # Save Blend File :
                bpy.ops.wm.save_mainfile()
                BDENTAL_Props.UserProjectDir = RelPath(UserProjectDir)
                txt = ["Finished."]
                update_info(message=txt, rect_color=[0,1,0,0.7])

                # Save Blend File :
                # ProjectName = BDENTAL_Props.ProjectNameProp
                # BlendFile = f"{ProjectName}.blend"
                # Blendpath = join(UserProjectDir, BlendFile)
                # bpy.ops.wm.save_as_mainfile(filepath=Blendpath)
                # BDENTAL_Props.UserProjectDir = RelPath(UserProjectDir)

        sleep(1)
        update_info()
        bpy.ops.wm.save_mainfile()

        return {"FINISHED"}


def Load_Dicom_funtion(context, series_file_names, q, voxel_mode):

    DcmInfo, message = {}, []
    ################################################################################################
    start = tpc()
    ################################################################################################
    BDENTAL_Props = context.scene.BDENTAL_Props
    UserProjectDir = AbsPath(BDENTAL_Props.UserProjectDir)
    # BDENTAL_Props.UserProjectDir = RelPath(UserProjectDir)

    # ProjectName = BDENTAL_Props.ProjectNameProp

    # Save Blend File :
    # Blendpath = join(UserProjectDir, f"{ProjectName}.blend")
    # bpy.ops.wm.save_as_mainfile(filepath=Blendpath)

    # Start Reading Dicom data :
    ######################################################################################
    try:
        Image3D = sitk.ReadImage(series_file_names)
    except Exception as Er:
        print(Er)
        message = [f"Can't read DICOM serie"]
        return DcmInfo, message

    # Get Preffix and save file :
    DcmInfoDict = eval(BDENTAL_Props.DcmInfo)
    Preffixs = list(DcmInfoDict.keys())

    for i in range(1, 100):
        Preffix = f"BD_{i:03}"
        if not Preffix in Preffixs:
            break

    # fix bad direction :
    image_center = Image3D.TransformContinuousIndexToPhysicalPoint(
        np.array(Image3D.GetSize())/2.0)

    new_transform = sitk.AffineTransform(3)
    new_transform.SetMatrix(np.array(Image3D.GetDirection()).ravel())
    new_transform.SetCenter(image_center)
    new_transform = new_transform.GetInverse()

    Image3D = sitk.Resample(Image3D, new_transform)
    Image3D.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    Image3D.SetOrigin([0, 0, 0])

    # Get Dicom Info :
    Sp = Spacing = Image3D.GetSpacing()
    Sz = Size = Image3D.GetSize()
    Dims = Dimensions = Image3D.GetDimension()
    Origin = Image3D.GetOrigin()
    Direction = Image3D.GetDirection()

    target_spacing = BDENTAL_Props.scan_resolution
    print("target_spacing", target_spacing)
    Image3D, new_size, new_spacing = ResizeImage(
        Image3D, target_spacing
    )
    print(
        f"image resized to : \nNew Size : {new_size}, New Spacing : {new_spacing}")

    # calculate Informations :
    D = Direction
    O = Origin
    DirectionMatrix_4x4 = Matrix(
        (
            (D[0], D[1], D[2], 0.0),
            (D[3], D[4], D[5], 0.0),
            (D[6], D[7], D[8], 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    TransMatrix_4x4 = Matrix(
        (
            (1.0, 0.0, 0.0, O[0]),
            (0.0, 1.0, 0.0, O[1]),
            (0.0, 0.0, 1.0, O[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    VtkTransform_4x4 = TransMatrix_4x4 @ DirectionMatrix_4x4
    P0 = Image3D.TransformContinuousIndexToPhysicalPoint((0, 0, 0))
    P_diagonal = Image3D.TransformContinuousIndexToPhysicalPoint(
        (new_size[0] - 1, new_size[1] - 1, new_size[2] - 1)
    )
    VCenter = (Vector(P0) + Vector(P_diagonal)) * 0.5

    C = VCenter

    TransformMatrix = Matrix(
        (
            (D[0], D[1], D[2], C[0]),
            (D[3], D[4], D[5], C[1]),
            (D[6], D[7], D[8], C[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    #######################################################################################
    # Add directories :
    PngDir = join(UserProjectDir, "PNG")
    AxialPngDir = join(PngDir, "Axial")
    CoronalPngDir = join(PngDir, "Coronal")
    SagittalPngDir = join(PngDir, "Sagittal")

    os.makedirs(AxialPngDir)
    os.makedirs(CoronalPngDir)
    os.makedirs(SagittalPngDir)

    Nrrd255Path = join(UserProjectDir, f"{Preffix}_Image3D255.nrrd")
    print(Nrrd255Path)
    DcmInfo["Nrrd255Path"] = RelPath(Nrrd255Path)

    #######################################################################################
    # set IntensityWindowing  :
    Array = sitk.GetArrayFromImage(Image3D)

    Image3D_255 = sitk.Cast(
        sitk.IntensityWindowing(
            Image3D,
            windowMinimum=Wmin,
            windowMaximum=Wmax,
            outputMinimum=0,
            outputMaximum=255,
        ),
        sitk.sitkUInt8,
    )

    #############################################################################################
    # MultiThreading PNG Writer:
    #########################################################################################
    def Image3DToAxialPNG(i, slices, AxialPngDir, Preffix):

        slice = slices[i]
        a = np.ones_like(slice)*(slice>=80)*255
        imgarray = np.stack([slice,slice,slice,a], axis=-1)
        img_Name = f"{Preffix}_Axial_img{i:04}.png"
        
        image_path = join(AxialPngDir, img_Name)
        cv2.imwrite(image_path, imgarray)
        image = bpy.data.images.load(image_path)
        image.pack()

    def Image3DToCoronalPNG(i, slices, CoronalPngDir, Preffix):
        img_Slice = slices[i]
        img_Name = f"{Preffix}_Coronal_img{i:04}.png"
        image_path = join(CoronalPngDir, img_Name)
        cv2.imwrite(image_path, img_Slice)
        image = bpy.data.images.load(image_path)
        image.pack()
        # print(f"{img_Name} was processed...")

    def Image3DToSagittalPNG(i, slices, SagittalPngDir, Preffix):
        img_Slice = slices[i]
        img_Name = f"{Preffix}_Sagittal_img{i:04}.png"
        image_path = join(SagittalPngDir, img_Name)
        cv2.imwrite(image_path, img_Slice)
        image = bpy.data.images.load(image_path)
        image.pack()
        # print(f"{img_Name} was processed...")

    # Convert Dicom to nrrd file :
    sitk.WriteImage(Image3D_255, Nrrd255Path)

    # make axial slices :
    Array = sitk.GetArrayFromImage(Image3D_255)
    AxialSlices = [np.flipud(Array[i, :, :]) for i in range(Array.shape[0])]
    threads = []
    Axialthreads = [
        threading.Thread(
            target=Image3DToAxialPNG,
            args=[i, AxialSlices, AxialPngDir, Preffix],
            daemon=True,
        )
        for i in range(len(AxialSlices))
    ]
    threads.extend(Axialthreads)
    for t in Axialthreads:
        t.start()
    if voxel_mode in ["OPTIMAL", "FULL"]:
        CoronalSlices = [np.flipud(Array[:, i, :])
                         for i in range(Array.shape[1])]
        Coronalthreads = [
            threading.Thread(
                target=Image3DToCoronalPNG,
                args=[i, CoronalSlices, CoronalPngDir, Preffix],
                daemon=True,
            )
            for i in range(len(CoronalSlices))
        ]
        threads.extend(Coronalthreads)
        for t in Coronalthreads:
            t.start()

    if voxel_mode == "FULL":
        SagittalSlices = [np.flipud(Array[:, :, i])
                          for i in range(Array.shape[2])]
        Sagittalthreads = [
            threading.Thread(
                target=Image3DToSagittalPNG,
                args=[i, SagittalSlices, SagittalPngDir, Preffix],
                daemon=True,
            )
            for i in range(len(SagittalSlices))
        ]
        threads.extend(Sagittalthreads)
        for t in Sagittalthreads:
            t.start()
    # slices = [Image3D_255[:, :, i] for i in range(Image3D_255.GetDepth())]

    for t in threads:
        t.join()

    shutil.rmtree(PngDir)

    # Set DcmInfo :
    DcmInfo = dict(
        {
            "Size": Sz,
            "Spacing": Sp,
            "Origin": Origin,
            "Direction": Direction,
            "Dims": Dims,
            "RenderSz": new_size,
            "RenderSp": new_spacing,
            "SlicesDir": "",
            "Nrrd255Path": RelPath(Nrrd255Path),
            "Preffix": Preffix,
            "PixelType": Image3D.GetPixelIDTypeAsString(),
            "Wmin": Wmin,
            "Wmax": Wmax,
            "TransformMatrix": TransformMatrix,
            "DirectionMatrix_4x4": DirectionMatrix_4x4,
            "TransMatrix_4x4": TransMatrix_4x4,
            "VtkTransform_4x4": VtkTransform_4x4,
            "VolumeCenter": VCenter,
            "CT_Loaded": True,
        }
    )

    # Set DcmInfo property :
    DcmInfoDict = eval(BDENTAL_Props.DcmInfo)
    DcmInfoDict[Preffix] = DcmInfo

    BDENTAL_Props.DcmInfo = str(DcmInfoDict)

    return DcmInfo, message


####### End Load_Dicom_fuction ##############


#######################################################################################
# BDENTAL CT Scan 3DImage File Load :


def Load_3DImage_function(context, q, voxel_mode):
    DcmInfo, message = {}, []
    BDENTAL_Props = context.scene.BDENTAL_Props
    UserProjectDir = AbsPath(BDENTAL_Props.UserProjectDir)
    UserImageFile = AbsPath(BDENTAL_Props.UserImageFile)
    ProjectName = BDENTAL_Props.ProjectNameProp

    # # Save Blend File :
    # BlendFile = f"{ProjectName}.blend"
    # Blendpath = join(UserProjectDir, BlendFile)

    # bpy.ops.wm.save_as_mainfile(filepath=Blendpath)
    bpy.ops.wm.save_mainfile()

    BDENTAL_Props.UserProjectDir = RelPath(UserProjectDir)

    reader = sitk.ImageFileReader()

    try:
        Image3D = sitk.ReadImage(UserImageFile)
    except Exception:
        message = [f"Can't open dicom file : {UserImageFile} "]
        return DcmInfo, message

    Depth = Image3D.GetDepth()

    if Depth <= 1:
        message = [
            "Can't Build 3D Volume from single 2D Image !",
        ]
        return DcmInfo, message

    ImgFileName = os.path.split(UserImageFile)[1]
    BDENTAL_nrrd = HU_Image = False
    if ImgFileName.startswith("BD") and ImgFileName.endswith("_Image3D255.nrrd"):
        BDENTAL_nrrd = True
    if Image3D.GetPixelIDTypeAsString() in [
        "32-bit signed integer",
        "16-bit signed integer",
    ]:
        HU_Image = True

    if not BDENTAL_nrrd or not HU_Image:
        message = [
            "Only Images with Hunsfield data or BDENTAL nrrd images are supported !"
        ]
        return DcmInfo, message
    ###########################################################################################################

    else:

        start = tpc()
        ####################################
        # Get Preffix and save file :
        DcmInfoDict = eval(BDENTAL_Props.DcmInfo)
        Preffixs = list(DcmInfoDict.keys())

        for i in range(1, 100):
            Preffix = f"BD_{i:03}"
            if not Preffix in Preffixs:
                break

        # Start Reading Dicom data :
        ######################################################################################
        # Get Dicom Info :
        del reader
        reader = sitk.ImageFileReader()
        reader.SetFileName(UserImageFile)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()

        Image3D = reader.Execute()

        # fix bad direction :
        image_center = Image3D.TransformContinuousIndexToPhysicalPoint(
            np.array(Image3D.GetSize())/2.0)

        new_transform = sitk.AffineTransform(3)
        new_transform.SetMatrix(np.array(Image3D.GetDirection()).ravel())
        new_transform.SetCenter(image_center)
        new_transform = new_transform.GetInverse()

        Image3D = sitk.Resample(Image3D, new_transform)
        Image3D.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        Image3D.SetOrigin([0, 0, 0])

        Sp = Spacing = Image3D.GetSpacing()
        Sz = Size = Image3D.GetSize()
        Dims = Dimensions = Image3D.GetDimension()
        Origin = Image3D.GetOrigin()
        Direction = Image3D.GetDirection()

        new_size, new_spacing = Sz, Sp
        
        target_spacing = BDENTAL_Props.scan_resolution
        Image3D, new_size, new_spacing = ResizeImage(
            sitkImage=Image3D, target_spacing=target_spacing
        )
        print(
            f"image resized to : \nNew Size : {new_size}, New Spacing : {new_spacing}"
        )

        # calculate Informations :
        D = Direction
        O = Origin
        DirectionMatrix_4x4 = Matrix(
            (
                (D[0], D[1], D[2], 0.0),
                (D[3], D[4], D[5], 0.0),
                (D[6], D[7], D[8], 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        )

        TransMatrix_4x4 = Matrix(
            (
                (1.0, 0.0, 0.0, O[0]),
                (0.0, 1.0, 0.0, O[1]),
                (0.0, 0.0, 1.0, O[2]),
                (0.0, 0.0, 0.0, 1.0),
            )
        )

        VtkTransform_4x4 = TransMatrix_4x4 @ DirectionMatrix_4x4
        P0 = Image3D.TransformContinuousIndexToPhysicalPoint((0, 0, 0))
        P_diagonal = Image3D.TransformContinuousIndexToPhysicalPoint(
            (new_size[0] - 1, new_size[1] - 1, new_size[2] - 1)
        )
        VCenter = (Vector(P0) + Vector(P_diagonal)) * 0.5

        C = VCenter

        TransformMatrix = Matrix(
            (
                (D[0], D[1], D[2], C[0]),
                (D[3], D[4], D[5], C[1]),
                (D[6], D[7], D[8], C[2]),
                (0.0, 0.0, 0.0, 1.0),
            )
        )

        #######################################################################################
        # Add directories :
        PngDir = join(UserProjectDir, "PNG")
        AxialPngDir = join(PngDir, "Axial")
        CoronalPngDir = join(PngDir, "Coronal")
        SagittalPngDir = join(PngDir, "Sagittal")

        os.makedirs(AxialPngDir)
        os.makedirs(CoronalPngDir)
        os.makedirs(SagittalPngDir)

        Nrrd255Path = join(AbsPath(UserProjectDir),
                           f"{Preffix}_Image3D255.nrrd")

        #######################################################################################
        if BDENTAL_nrrd:
            Image3D_255 = Image3D
        else:
            # set IntensityWindowing  :
            Image3D_255 = sitk.Cast(
                sitk.IntensityWindowing(
                    Image3D,
                    windowMinimum=Wmin,
                    windowMaximum=Wmax,
                    outputMinimum=0,
                    outputMaximum=255,
                ),
                sitk.sitkUInt8,
            )

        # Convert Dicom to nrrd file :
        sitk.WriteImage(Image3D_255, Nrrd255Path)
        # BackupArray = sitk.GetArrayFromImage(Image3D_255)
        # DcmInfo["BackupArray"] = str(BackupArray.tolist())

        #############################################################################################
        # MultiThreading PNG Writer:
        #########################################################################################
        def Image3DToAxialPNG(i, slices, AxialPngDir, Preffix):

            slice = slices[i]
            a = np.ones_like(slice)*(slice>=80)*255
            imgarray = np.stack([slice,slice,slice,a], axis=-1)
            img_Name = f"{Preffix}_Axial_img{i:04}.png"
            
            image_path = join(AxialPngDir, img_Name)
            cv2.imwrite(image_path, imgarray)
            image = bpy.data.images.load(image_path)
            image.pack()

        def Image3DToCoronalPNG(i, slices, CoronalPngDir, Preffix):
            img_Slice = slices[i]
            img_Name = f"{Preffix}_Coronal_img{i:04}.png"
            image_path = join(CoronalPngDir, img_Name)
            cv2.imwrite(image_path, img_Slice)
            image = bpy.data.images.load(image_path)
            image.pack()
            # print(f"{img_Name} was processed...")

        def Image3DToSagittalPNG(i, slices, SagittalPngDir, Preffix):
            img_Slice = slices[i]
            img_Name = f"{Preffix}_Sagittal_img{i:04}.png"
            image_path = join(SagittalPngDir, img_Name)
            cv2.imwrite(image_path, img_Slice)
            image = bpy.data.images.load(image_path)
            image.pack()
            # print(f"{img_Name} was processed...")

        # make axial slices :
        Array = sitk.GetArrayFromImage(Image3D_255)
        AxialSlices = [np.flipud(Array[i, :, :])
                       for i in range(Array.shape[0])]
        Axialthreads = [
            threading.Thread(
                target=Image3DToAxialPNG,
                args=[i, AxialSlices, AxialPngDir, Preffix],
                daemon=True,
            )
            for i in range(len(AxialSlices))
        ]
        threads = Axialthreads
        if voxel_mode in ["OPTIMAL", "FULL"]:
            CoronalSlices = [np.flipud(Array[:, i, :])
                             for i in range(Array.shape[1])]
            Coronalthreads = [
                threading.Thread(
                    target=Image3DToCoronalPNG,
                    args=[i, CoronalSlices, CoronalPngDir, Preffix],
                    daemon=True,
                )
                for i in range(len(CoronalSlices))
            ]
            threads.extend(Coronalthreads)

        if voxel_mode == "FULL":
            SagittalSlices = [np.flipud(Array[:, :, i])
                              for i in range(Array.shape[2])]
            Sagittalthreads = [
                threading.Thread(
                    target=Image3DToSagittalPNG,
                    args=[i, SagittalSlices, SagittalPngDir, Preffix],
                    daemon=True,
                )
                for i in range(len(SagittalSlices))
            ]
            threads.extend(Sagittalthreads)

        # slices = [Image3D_255[:, :, i] for i in range(Image3D_255.GetDepth())]
        for t in threads:
            t.start()

        for t in threads:
            t.join()

        shutil.rmtree(PngDir)

        # Set DcmInfo :
        DcmInfo = dict(
            {
                "Size": Sz,
                "Spacing": Sp,
                "Origin": Origin,
                "Direction": Direction,
                "Dims": Dims,
                "RenderSz": new_size,
                "RenderSp": new_spacing,
                "SlicesDir": "",
                "Nrrd255Path": RelPath(Nrrd255Path),
                "Preffix": Preffix,
                "PixelType": Image3D.GetPixelIDTypeAsString(),
                "Wmin": Wmin,
                "Wmax": Wmax,
                "TransformMatrix": TransformMatrix,
                "DirectionMatrix_4x4": DirectionMatrix_4x4,
                "TransMatrix_4x4": TransMatrix_4x4,
                "VtkTransform_4x4": VtkTransform_4x4,
                "VolumeCenter": VCenter,
                "CT_Loaded": True,
            }
        )

        # Set DcmInfo property :
        DcmInfoDict = eval(BDENTAL_Props.DcmInfo)
        DcmInfoDict[Preffix] = DcmInfo
        BDENTAL_Props.DcmInfo = str(DcmInfoDict)

        return DcmInfo, message


##########################################################################################
######################### BDENTAL Volume Render : ########################################
##########################################################################################
class BDENTAL_OT_Volume_Render(bpy.types.Operator):
    """Volume Render"""

    bl_idname = "wm.bdental_volume_render"
    bl_label = "VOXEL 3D"

    q = Queue()
    # Voxel_Modes = ["FAST", "OPTIMAL", "FULL"]
    # VoxelMode: EnumProperty(items=set_enum_items(
    #     Voxel_Modes), description="Voxel Mode", default="FAST")
    slices: BoolProperty(default=0)

    def execute(self, context):
        message = ["Loading SCAN data ..."]
        update_info(message)

        Start = tpc()
        print(message)

        global DataBlendFile
        global GpShader
        global Wmin
        global Wmax

        BDENTAL_Props = context.scene.BDENTAL_Props
        UserDcmDir = AbsPath(BDENTAL_Props.UserDcmDir)

        DataType = BDENTAL_Props.DataType

        if DataType == "DICOM Series":

            if BDENTAL_Props.Dicom_Series_mode == "Simple Mode":
                max_sID, max_count, series_file_names = GetMaxSerie(UserDcmDir)

            if BDENTAL_Props.Dicom_Series_mode == "Advanced Mode":
                Serie = BDENTAL_Props.Dicom_Series

                if not "Series" in Serie:
                    message = [" Please Organize DICOM data and retry ! "]
                    update_info(message)
                    sleep(2)
                    update_info()
                    return {"CANCELLED"}

                DcmOrganizeDict = eval(BDENTAL_Props.DcmOrganize)
                series_file_names = DcmOrganizeDict[UserDcmDir][Serie]["Files"]

            # Start Reading Dicom data :
            ######################################################################################
            DcmInfo, message = Load_Dicom_funtion(
                context, series_file_names, self.q, "FAST")

        if DataType == "3D Image File":
            UserImageFile = AbsPath(BDENTAL_Props.UserImageFile)
            if not exists(UserImageFile):
                message = [" The Selected Image File Path is not valid ! "]
                update_info(message)
                sleep(2)
                update_info()
                return {"CANCELLED"}

            DcmInfo, message = Load_3DImage_function(
                context, self.q, "FAST")

        if message:
            update_info(message)
            sleep(3)
            update_info()
            return {"CANCELLED"}
        else:

            Preffix = DcmInfo["Preffix"]
            print("\n##########################\n")
            VolumeRender(DcmInfo, GpShader, DataBlendFile, "FAST",update_info)
            print("setting volumes...")
            scn = bpy.context.scene
            scn.render.engine = "BLENDER_EEVEE"
            BDENTAL_Props.GroupNodeName = GpShader

            GpNode = bpy.data.node_groups.get(f"{Preffix}_{GpShader}")
            Low_Treshold = GpNode.nodes["Low_Treshold"].outputs[0]
            Low_Treshold.default_value = 700
            WminNode = GpNode.nodes["WminNode"].outputs[0]
            WminNode.default_value = Wmin
            WmaxNode = GpNode.nodes["WmaxNode"].outputs[0]
            WmaxNode.default_value = Wmax

            Soft, Bone, Teeth = -400, 700, 1400

            BDENTAL_Props.SoftTreshold = Soft
            BDENTAL_Props.BoneTreshold = Bone
            BDENTAL_Props.TeethTreshold = Teeth
            BDENTAL_Props.SoftBool = False
            BDENTAL_Props.BoneBool = False
            BDENTAL_Props.TeethBool = True

            BDENTAL_Props.CT_Rendered = True
            # bpy.ops.view3d.view_selected(use_all_regions=False)

            
            if self.slices :
                sleep(2)
                update_info()
                message = ["Scan Slices processing ..."]
                update_info(message)

                bpy.ops.wm.bdental_addslices()

            txt = ["Scan loaded. "]
            update_info(message=txt, rect_color=[0,1,0,0.7])
            sleep(1)
            update_info()
            # os.system("cls")

            Finish = tpc()
            print(f"Voxel rendered (Time : {Finish-Start}")
            bpy.ops.wm.save_mainfile()

            return {"FINISHED"}

    def invoke(self, context, event):

        wm = context.window_manager
        return wm.invoke_props_dialog(self)


##########################################################################################
######################### BDENTAL Add Slices : ########################################
##########################################################################################


class BDENTAL_OT_AddSlices(bpy.types.Operator):
    """Add Volume Slices"""

    bl_idname = "wm.bdental_addslices"
    bl_label = "MPR Slices Views"

    @classmethod
    def poll(cls, context):
        if not context.object:
            return False
        return context.object.get("bdental_type") == "CT_Voxel"

    def execute(self, context):
        is_update = False
        context.window.workspace = bpy.data.workspaces["Bdental Slicer"]
        Vol = context.object
        message = [
            " Generating MPR Slices views ... ",
        ]
        update_info(message)
        BDENTAL_Props = bpy.context.scene.BDENTAL_Props

        pointer_check_list = [
           obj for obj in context.scene.objects if obj.get("bdental_type") =="slices_pointer"
           ]
        if pointer_check_list:
            is_update = True
            for obj in pointer_check_list:
                bpy.data.objects.remove(obj)

        slices_col = bpy.data.collections.get("SLICES")
        if slices_col:
            is_update = True
            for obj in slices_col.objects:
                bpy.data.objects.remove(obj)
            bpy.data.collections.remove(slices_col)

        images = [img for img in bpy.data.images if "SLICE" in img.name]
        for img in images:
            bpy.data.images.remove(img)
        
            
        bpy.ops.object.empty_add(
            type="PLAIN_AXES",
            scale=(1, 1, 1),
        )

        SLICES_POINTER = bpy.context.object
        SLICES_POINTER.matrix_world = Matrix.Identity(4)
        SLICES_POINTER.empty_display_size = 20
        SLICES_POINTER.show_in_front = True
        SLICES_POINTER.name = "SLICES_POINTER"
        SLICES_POINTER["bdental_type"] = "slices_pointer"
        MoveToCollection(SLICES_POINTER, "SLICES_POINTER")


        
        Preffix = Vol.name[:6]
        context.scene["volume_preffix"] = Preffix
        print("Preffix", Preffix)
        DcmInfoDict = eval(BDENTAL_Props.DcmInfo)
        DcmInfo = DcmInfoDict[Preffix]

        Nrrd255Path = AbsPath(DcmInfo["Nrrd255Path"])
        # print("Nrrd255Path", Nrrd255Path)
        if not exists(Nrrd255Path):
            message = [
                " Can't find dicom data!",
                " Check for nrrd file in the Project Directory ! ",
            ]
            update_info(message)
            sleep(2)
            update_info()
            return {"CANCELLED"}
        
        SlicesDir = BDENTAL_Props.SlicesDir
        if not exists(SlicesDir):
            SlicesDir = tempfile.mkdtemp()
            BDENTAL_Props.SlicesDir = SlicesDir

        bpy.context.scene.render.resolution_x = 512
        bpy.context.scene.render.resolution_y = 512

        AxialPlane, CoronalPlane, SagittalPlane, AxialCam, CoronalCam, SagittalCam = AddSlices(
            Preffix, DcmInfo, SlicesDir)


        Override, _, _ = CtxOverride(bpy.context)
        bpy.ops.object.select_all(Override, action="DESELECT")
        SLICES_POINTER.select_set(True)
        bpy.context.view_layer.objects.active = SLICES_POINTER

        for c in SLICES_POINTER.constraints :
            SLICES_POINTER.constraints.remove(c)

        

        SLICES_POINTER.matrix_world[:3] = Vol.matrix_world[:3]
        child_of = SLICES_POINTER.constraints.new("CHILD_OF")
        child_of.target = Vol
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        for obj in [AxialPlane, AxialCam, CoronalPlane, CoronalCam, SagittalPlane, SagittalCam]:
            obj.matrix_world = SLICES_POINTER.matrix_world @ obj.matrix_world
            
            child_of = obj.constraints.new("CHILD_OF")
            child_of.target = SLICES_POINTER
            child_of.use_scale_x = False
            child_of.use_scale_y = False
            child_of.use_scale_z = False

            for i in range(3):
                obj.lock_location[i] = True
                obj.lock_rotation[i] = True
                obj.lock_scale[i] = True
        context.window.workspace = bpy.data.workspaces["Bdental Slicer"]
        BDENTAL_SliceUpdate(bpy.context.scene)
        # if not context.scene.get("bdental_slicer_is_set")==True :
        bpy.ops.wm.bdental_mpr2()
        
        update_info()

        return {"FINISHED"}


###############################################################################
####################### BDENTAL_FULL VOLUME to Mesh : ################################
##############################################################################
class BDENTAL_OT_MultiTreshSegment(bpy.types.Operator):
    """Add a mesh Segmentation using Treshold"""

    bl_idname = "wm.bdental_multitresh_segment"
    bl_label = "SEGMENTATION"

    TimingDict = {}
    message_queue = Queue()
    Exported = Queue()

    def ImportMeshStl(self, Segment, SegmentStlPath, SegmentColor):

        # import stl to blender scene :
        bpy.ops.import_mesh.stl(filepath=SegmentStlPath)
        obj = bpy.context.object
        obj.name = f"{self.Preffix}_{Segment}_SEGMENTATION"
        obj.data.name = f"{self.Preffix}_{Segment}_mesh"

        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")

        ############### step 8 : Add material... #########################
        mat = bpy.data.materials.get(
            obj.name) or bpy.data.materials.new(obj.name)
        mat.diffuse_color = SegmentColor
        obj.data.materials.append(mat)
        MoveToCollection(obj=obj, CollName="SEGMENTS")
        bpy.ops.object.shade_smooth()

        bpy.ops.object.modifier_add(type="CORRECTIVE_SMOOTH")
        bpy.context.object.modifiers["CorrectiveSmooth"].iterations = 2
        bpy.context.object.modifiers["CorrectiveSmooth"].use_only_smooth = True
        bpy.ops.object.modifier_apply(modifier="CorrectiveSmooth")

        print(f"{Segment} Mesh Import Finished")

        return obj

        # self.q.put(["End"])

    def DicomToStl(self, Segment, Image3D):
        message_queue = self.message_queue
        print(f"{Segment} processing ...")
        message = [f"Extracting {Segment} segment ..."]
        message_queue.put(message)
        # Load Infos :
        #########################################################################
        BDENTAL_Props = bpy.context.scene.BDENTAL_Props
        UserProjectDir = AbsPath(BDENTAL_Props.UserProjectDir)
        DcmInfo = self.DcmInfo
        Origin = DcmInfo["Origin"]
        VtkTransform_4x4 = DcmInfo["VtkTransform_4x4"]
        TransformMatrix = DcmInfo["TransformMatrix"]
        VtkMatrix_4x4 = (
            self.Vol.matrix_world @ TransformMatrix.inverted() @ VtkTransform_4x4
        )

        VtkMatrix = list(np.array(VtkMatrix_4x4).ravel())

        Thikness = 1

        SegmentTreshold = self.SegmentsDict[Segment]["Treshold"]
        SegmentColor = self.SegmentsDict[Segment]["Color"]
        SegmentStlPath = join(UserProjectDir, f"{Segment}_SEGMENTATION.stl")

        # Convert Hu treshold value to 0-255 UINT8 :
        Treshold255 = HuTo255(Hu=SegmentTreshold, Wmin=Wmin, Wmax=Wmax)
        if Treshold255 == 0:
            Treshold255 = 1
        elif Treshold255 == 255:
            Treshold255 = 254

        ############### step 2 : Extracting mesh... #########################
        # print("Extracting mesh...")

        vtkImage = sitkTovtk(sitkImage=Image3D)

        ExtractedMesh = vtk_MC_Func(vtkImage=vtkImage, Treshold=Treshold255)
        Mesh = ExtractedMesh

        self.step2 = tpc()
        self.TimingDict["Mesh Extraction Time"] = self.step2 - self.step1
        print(f"{Segment} Mesh Extraction Finished")

        ############### step 3 : mesh Smoothing 1... #########################
        message = [f"Smoothing {Segment} segment ..."]
        message_queue.put(message)
        SmthIter = 3
        SmoothedMesh1 = vtkSmoothMesh(
            q=None,
            mesh=Mesh,
            Iterations=SmthIter,
            step="Mesh Smoothing 2",
            start=0.79,
            finish=0.82,
        )
        Mesh = SmoothedMesh1

        self.step3 = tpc()
        self.TimingDict["Mesh Smoothing 1 Time"] = self.step3 - self.step1
        print(f"{Segment} Mesh Smoothing 1 Finished")

        # ############### step 4 : mesh Smoothing... #########################

        SmthIter = 20
        SmoothedMesh2 = vtkWindowedSincPolyDataFilter(
            q=None,
            mesh=Mesh,
            Iterations=SmthIter,
            step="Mesh Smoothing 1",
            start=0.76,
            finish=0.78,
        )

        self.step4 = tpc()
        self.TimingDict["Mesh Smoothing 2 Time"] = self.step4 - self.step3
        print(f"{Segment} Mesh Smoothing 2 Finished")
        Mesh = SmoothedMesh2

        ############### step 5 : mesh Reduction... #########################
        message = [f"Decimating {Segment} segment ..."]
        message_queue.put(message)
        polysCount = Mesh.GetNumberOfPolys()
        polysLimit = 1000000
        if polysCount > polysLimit:

            Reduction = round(1 - (polysLimit / polysCount), 2)
            ReductedMesh = vtkMeshReduction(
                q=None,
                mesh=Mesh,
                reduction=Reduction,
                step="Mesh Reduction",
                start=0.11,
                finish=0.75,
            )
            Mesh = ReductedMesh

        self.step5 = tpc()
        self.TimingDict["Mesh Reduction Time"] = self.step5 - self.step4
        print(f"{Segment} Mesh Reduction Finished")

        ############### step 6 : Set mesh orientation... #########################
        TransformedMesh = vtkTransformMesh(
            mesh=Mesh,
            Matrix=VtkMatrix,
        )
        self.step6 = tpc()
        self.TimingDict["Mesh Orientation"] = self.step6 - self.step5
        print(f"{Segment} Mesh Orientation Finished")
        Mesh = TransformedMesh

        ############### step 7 : exporting mesh stl... #########################
        message = [f"Exporting {Segment} segment ..."]
        message_queue.put(message)
        writer = vtk.vtkSTLWriter()
        writer.SetInputData(Mesh)
        writer.SetFileTypeToBinary()
        writer.SetFileName(SegmentStlPath)
        writer.Write()

        self.step7 = tpc()
        self.TimingDict["Mesh Export"] = self.step7 - self.step6
        print(f"{Segment} Mesh Export Finished")
        self.Exported.put([Segment, SegmentStlPath, SegmentColor])

    @classmethod
    def poll(cls, context):
        if not context.object:
            return False
        return context.object.get("bdental_type") == "CT_Voxel"

    def execute(self, context):
        global message_queue

        self.counter_start = tpc()

        BDENTAL_Props = bpy.context.scene.BDENTAL_Props

        self.Soft = BDENTAL_Props.SoftBool
        self.Bone = BDENTAL_Props.BoneBool
        self.Teeth = BDENTAL_Props.TeethBool

        self.SoftTresh = BDENTAL_Props.SoftTreshold
        self.BoneTresh = BDENTAL_Props.BoneTreshold
        self.TeethTresh = BDENTAL_Props.TeethTreshold

        self.SoftSegmentColor = BDENTAL_Props.SoftSegmentColor
        self.BoneSegmentColor = BDENTAL_Props.BoneSegmentColor
        self.TeethSegmentColor = BDENTAL_Props.TeethSegmentColor

        self.SegmentsDict = {
            "Soft": {
                "State": self.Soft,
                "Treshold": self.SoftTresh,
                "Color": self.SoftSegmentColor,
            },
            "Bone": {
                "State": self.Bone,
                "Treshold": self.BoneTresh,
                "Color": self.BoneSegmentColor,
            },
            "Teeth": {
                "State": self.Teeth,
                "Treshold": self.TeethTresh,
                "Color": self.TeethSegmentColor,
            },
        }

        ActiveSegmentsList = [
            k for k, v in self.SegmentsDict.items() if v["State"]
        ]

        if not ActiveSegmentsList:
            message = [
                " Please check at least 1 segmentation ! ",
                "(Soft - Bone - Teeth)",
            ]
            update_info(message)
            sleep(3)
            update_info()
            return {"CANCELLED"}

        message = ["Dicom segmentation processing ...",
                   f"Active Segments : {ActiveSegmentsList}"]
        update_info(message)
        sleep(1)

        self.Vol = context.object
        self.Preffix = self.Vol.name[:6]
        DcmInfoDict = eval(BDENTAL_Props.DcmInfo)
        self.DcmInfo = DcmInfoDict[self.Preffix]
        self.Nrrd255Path = AbsPath(self.DcmInfo["Nrrd255Path"])

        if not exists(self.Nrrd255Path):

            message = [" 3D Image File not Found in Project Folder ! "]
            update_info(message)
            sleep(3)
            update_info()
            return {"CANCELLED"}

        ############### step 1 : Reading DICOM #########################
        self.step1 = tpc()
        self.TimingDict["Read DICOM"] = self.step1 - self.counter_start
        print(f"step 1 : Read DICOM ({self.step1-self.counter_start})")

        Image3D = sitk.ReadImage(self.Nrrd255Path)
        target_spacing = 0.2
        Sp = Image3D.GetSpacing()
        if Sp[0] < target_spacing:
            ResizedImage, _, _ = ResizeImage(
                sitkImage=Image3D, target_spacing=target_spacing
            )
            Image3D = ResizedImage

        ############### step 2 : Dicom To Stl Threads #########################

        self.MeshesCount = len(ActiveSegmentsList)
        Imported_Meshes = []
        Threads = [
            threading.Thread(
                target=self.DicomToStl,
                args=[Segment, Image3D],
                daemon=True,
            )
            for Segment in ActiveSegmentsList
        ]
        print(f"segments list : {ActiveSegmentsList}")
        for t in Threads:
            t.start()
        count = 0
        while count < self.MeshesCount:
            if not self.message_queue.empty():
                message = self.message_queue.get()
                update_info(message=message)
                sleep(1)
            if not self.Exported.empty():
                (
                    Segment,
                    SegmentStlPath,
                    SegmentColor,
                ) = self.Exported.get()
                for i in range(10):
                    if not exists(SegmentStlPath):
                        sleep(0.1)
                    else:
                        break
                message = [f"{Segment} Mesh import ..."]
                update_info(message=message)
                obj = self.ImportMeshStl(
                    Segment, SegmentStlPath, SegmentColor
                )
                Imported_Meshes.append(obj)
                os.remove(SegmentStlPath)
                count += 1
            else:
                sleep(0.1)
        for t in Threads:
            t.join()

        for obj in Imported_Meshes:
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            for i in range(3):
                obj.lock_location[i] = True
                obj.lock_rotation[i] = True
                obj.lock_scale[i] = True

        bpy.ops.object.select_all(action="DESELECT")
        for obj in Imported_Meshes:
            child_of = obj.constraints.new("CHILD_OF")
            child_of.target = self.Vol
            child_of.use_scale_x = False
            child_of.use_scale_y = False
            child_of.use_scale_z = False

        bpy.ops.object.select_all(action="DESELECT")

        self.counter_finish = tpc()
        self.TimingDict["Total Time"] = (
            self.counter_finish - self.counter_start
        )

        print(self.TimingDict)
        override, area, space_data = CtxOverride(context)
        space_data.shading.type = "SOLID"
        space_data.shading.show_specular_highlight = False
        # space3D.shading.background_type = 'WORLD'
        space_data.shading.color_type = "TEXTURE"
        space_data.shading.light = "STUDIO"
        space_data.shading.studio_light = "paint.sl"
        self.Vol.hide_set(True)
        message = [" Dicom Segmentation Finished ! "]
        update_info(message=message)
        sleep(2)
        update_info()
        bpy.ops.wm.save_mainfile()

        return {"FINISHED"}


#######################################################################################
########################### Measurements : Operators ##############################
#######################################################################################


class BDENTAL_OT_AddReferencePlanes(bpy.types.Operator):
    """Add Reference Planes"""

    bl_idname = "wm.bdental_add_reference_planes"
    bl_label = "Add REFERENCE PLANES"
    bl_options = {"REGISTER", "UNDO"}

    def modal(self, context, event):

        if (
            event.type
            in [
                "LEFTMOUSE",
                "RIGHTMOUSE",
                "MIDDLEMOUSE",
                "WHEELUPMOUSE",
                "WHEELDOWNMOUSE",
                "N",
                "NUMPAD_2",
                "NUMPAD_4",
                "NUMPAD_6",
                "NUMPAD_8",
                "NUMPAD_1",
                "NUMPAD_3",
                "NUMPAD_5",
                "NUMPAD_7",
                "NUMPAD_9",
            ]
            and event.value == "PRESS"
        ):

            return {"PASS_THROUGH"}
        #########################################
        elif event.type == "RET":
            if event.value == ("PRESS"):

                CurrentPointsNames = [P.name for P in self.CurrentPointsList]
                P_Names = [
                    P for P in self.PointsNames if not P in CurrentPointsNames]
                if P_Names:
                    if self.MarkupVoxelMode:
                        CursorToVoxelPoint(
                            Preffix=self.Preffix, CursorMove=True)

                    loc = context.scene.cursor.location
                    P = AddMarkupPoint(
                        P_Names[0], self.Color, loc, 1, self.CollName)
                    self.CurrentPointsList.append(P)

                if not P_Names:

                    Override, area3D, space3D = CtxOverride(context)
                    RefPlanes = PointsToRefPlanes(
                        Override,
                        self.TargetObject,
                        self.CurrentPointsList,
                        color=(0.0, 0.0, 0.2, 0.7),
                        CollName=self.CollName,
                    )
                    bpy.ops.object.select_all(action="DESELECT")
                    for Plane in RefPlanes:
                        Plane.select_set(True)
                    CurrentPoints = [
                        bpy.data.objects.get(PName) for PName in CurrentPointsNames
                    ]
                    for P in CurrentPoints:
                        P.select_set(True)
                    self.TargetObject.select_set(True)
                    bpy.context.view_layer.objects.active = self.TargetObject
                    bpy.ops.object.parent_set(
                        type="OBJECT", keep_transform=True)
                    bpy.ops.object.select_all(action="DESELECT")
                    self.DcmInfo[self.Preffix]["Frankfort"] = RefPlanes[0].name
                    self.BDENTAL_Props.DcmInfo = str(self.DcmInfo)
                    ##########################################################
                    space3D.overlay.show_outline_selected = True
                    space3D.overlay.show_object_origins = True
                    space3D.overlay.show_annotation = True
                    space3D.overlay.show_text = True
                    space3D.overlay.show_extras = True
                    space3D.overlay.show_floor = True
                    space3D.overlay.show_axis_x = True
                    space3D.overlay.show_axis_y = True
                    # ###########################################################
                    bpy.ops.wm.tool_set_by_id(Override, name="builtin.select")
                    bpy.context.scene.tool_settings.use_snap = False

                    bpy.context.scene.cursor.location = (0, 0, 0)
                    # bpy.ops.screen.region_toggle(Override, region_type="UI")
                    self.BDENTAL_Props.ActiveOperator = "None"

                    return {"FINISHED"}

        #########################################

        elif event.type == ("DEL") and event.value == ("PRESS"):
            if self.CurrentPointsList:
                P = self.CurrentPointsList.pop()
                bpy.data.objects.remove(P)

        elif event.type == ("ESC"):
            if self.CurrentPointsList:
                for P in self.CurrentPointsList:
                    bpy.data.objects.remove(P)

            Override, area3D, space3D = CtxOverride(context)
            ##########################################################
            space3D.overlay.show_outline_selected = True
            space3D.overlay.show_object_origins = True
            space3D.overlay.show_annotation = True
            space3D.overlay.show_text = True
            space3D.overlay.show_extras = True
            space3D.overlay.show_floor = True
            space3D.overlay.show_axis_x = True
            space3D.overlay.show_axis_y = True
            ###########################################################
            bpy.ops.wm.tool_set_by_id(Override, name="builtin.select")
            bpy.context.scene.tool_settings.use_snap = False

            bpy.context.scene.cursor.location = (0, 0, 0)
            # bpy.ops.screen.region_toggle(Override, region_type="UI")
            self.BDENTAL_Props.ActiveOperator = "None"
            message = [
                " The Frankfort Plane Operation was Cancelled!",
            ]

            icon = "COLORSET_02_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):

        Active_Obj = bpy.context.view_layer.objects.active

        if not Active_Obj:
            message = [" Please select Target Object ! "]
            icon = "COLORSET_02_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}
        else:
            N = Active_Obj.name
            Condition = (
                CheckString(N, ["BD"])
                and CheckString(N, ["_CTVolume", "SEGMENTATION"], any)
                and Active_Obj.select_get()
            )

            if not Condition:
                message = [
                    " Please select Target Object ! ",
                    "Target Object should be a CTVolume or a Segmentation",
                ]
                icon = "COLORSET_02_VEC"
                bpy.ops.wm.bdental_message_box(
                    "INVOKE_DEFAULT", message=str(message), icon=icon
                )
                return {"CANCELLED"}

            else:
                if context.space_data.type == "VIEW_3D":

                    self.BDENTAL_Props = bpy.context.scene.BDENTAL_Props

                    # Prepare scene  :
                    ##########################################################
                    bpy.context.space_data.overlay.show_outline_selected = False
                    bpy.context.space_data.overlay.show_object_origins = False
                    bpy.context.space_data.overlay.show_annotation = False
                    bpy.context.space_data.overlay.show_text = True
                    bpy.context.space_data.overlay.show_extras = False
                    bpy.context.space_data.overlay.show_floor = False
                    bpy.context.space_data.overlay.show_axis_x = False
                    bpy.context.space_data.overlay.show_axis_y = False
                    bpy.context.scene.tool_settings.use_snap = True
                    bpy.context.scene.tool_settings.snap_elements = {"FACE"}
                    bpy.context.scene.tool_settings.transform_pivot_point = (
                        "INDIVIDUAL_ORIGINS"
                    )
                    bpy.ops.wm.tool_set_by_id(name="builtin.cursor")

                    ###########################################################
                    self.CollName = "REFERENCE PLANES"
                    self.CurrentPointsList = []
                    self.PointsNames = ["Na", "R_Or", "R_Po", "L_Or", "L_Po"]
                    self.Color = [1, 0, 0, 1]  # Red color
                    self.TargetObject = Active_Obj
                    self.visibleObjects = bpy.context.visible_objects.copy()
                    self.MarkupVoxelMode = self.TargetObject.name.endswith(
                        "_CTVolume")
                    self.Preffix = self.TargetObject.name.split("_")[0]
                    DcmInfo = self.BDENTAL_Props.DcmInfo
                    self.DcmInfo = eval(DcmInfo)
                    Override, area3D, space3D = CtxOverride(context)
                    # bpy.ops.screen.region_toggle(Override, region_type="UI")
                    bpy.ops.object.select_all(action="DESELECT")
                    # bpy.ops.object.select_all(Override, action="DESELECT")

                    context.window_manager.modal_handler_add(self)
                    self.BDENTAL_Props.ActiveOperator = "bdental.add_reference_planes"
                    return {"RUNNING_MODAL"}

                else:
                    message = [
                        "Active space must be a View3d",
                    ]

                    icon = "COLORSET_02_VEC"
                    bpy.ops.wm.bdental_message_box(
                        "INVOKE_DEFAULT", message=str(message), icon=icon
                    )

                    return {"CANCELLED"}


class BDENTAL_OT_AddMarkupPoint(bpy.types.Operator):
    """Add Markup point"""

    bl_idname = "wm.bdental_add_markup_point"
    bl_label = "ADD MARKUP POINT"

    MarkupName: StringProperty(
        name="Markup Name",
        default="Markup 01",
        description="Markup Name",
    )
    MarkupColor: FloatVectorProperty(
        name="Markup Color",
        description="Markup Color",
        default=[1.0, 0.0, 0.0, 1.0],
        size=4,
        subtype="COLOR",
    )
    Markup_Diameter: FloatProperty(
        description="Diameter", default=1, step=1, precision=2
    )

    CollName = "Markup Points"

    def execute(self, context):

        bpy.ops.object.mode_set(mode="OBJECT")

        if self.MarkupVoxelMode:
            Preffix = self.TargetObject.name.split("_")[0]
            CursorToVoxelPoint(Preffix=Preffix, CursorMove=True)

        Co = context.scene.cursor.location
        P = AddMarkupPoint(
            name=self.MarkupName,
            color=self.MarkupColor,
            loc=Co,
            Diameter=self.Markup_Diameter,
            CollName=self.CollName,
        )
        bpy.ops.object.select_all(action="DESELECT")
        self.TargetObject.select_set(True)
        bpy.context.view_layer.objects.active = self.TargetObject
        bpy.ops.object.mode_set(mode=self.mode)

        return {"FINISHED"}

    def invoke(self, context, event):

        self.BDENTAL_Props = bpy.context.scene.BDENTAL_Props

        Active_Obj = bpy.context.view_layer.objects.active

        if not Active_Obj:
            message = [" Please select Target Object ! "]
            icon = "COLORSET_02_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )
            return {"CANCELLED"}

        else:
            if Active_Obj.select_get() == False:
                message = [" Please select Target Object ! "]
                icon = "COLORSET_02_VEC"
                bpy.ops.wm.bdental_message_box(
                    "INVOKE_DEFAULT", message=str(message), icon=icon
                )
                return {"CANCELLED"}

            else:
                self.mode = Active_Obj.mode
                self.TargetObject = Active_Obj
                self.MarkupVoxelMode = CheckString(
                    self.TargetObject.name, ["BD", "_CTVolume"]
                )
                wm = context.window_manager
                return wm.invoke_props_dialog(self)


class BDENTAL_OT_CtVolumeOrientation(bpy.types.Operator):
    """CtVolume Orientation according to Frankfort Plane"""

    bl_idname = "wm.bdental_ctvolume_orientation"
    bl_label = "CTVolume Orientation"

    def execute(self, context):

        BDENTAL_Props = bpy.context.scene.BDENTAL_Props
        Active_Obj = bpy.context.view_layer.objects.active

        if not Active_Obj:
            message = [" Please select CTVOLUME for segmentation ! "]
            icon = "COLORSET_02_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )
            return {"CANCELLED"}

        else:

            Condition = CheckString(Active_Obj.name, ["BD"]) and CheckString(
                Active_Obj.name, ["_CTVolume", "Segmentation"], any
            )

            if not Condition:
                message = [
                    "CTVOLUME Orientation : ",
                    "Please select CTVOLUME or Segmentation! ",
                ]

                icon = "COLORSET_02_VEC"
                bpy.ops.wm.bdental_message_box(
                    "INVOKE_DEFAULT", message=str(message), icon=icon
                )
                return {"CANCELLED"}

            else:
                Preffix = Active_Obj.name.split("_")[0]
                DcmInfo = eval(BDENTAL_Props.DcmInfo)
                if not "Frankfort" in DcmInfo[Preffix].keys():
                    message = [
                        "CTVOLUME Orientation : ",
                        "Please Add Reference Planes before CTVOLUME Orientation ! ",
                    ]
                    icon = "COLORSET_02_VEC"
                    bpy.ops.wm.bdental_message_box(
                        "INVOKE_DEFAULT", message=str(message), icon=icon
                    )
                    return {"CANCELLED"}
                else:
                    Frankfort_Plane = bpy.data.objects.get(
                        DcmInfo[Preffix]["Frankfort"]
                    )
                    if not Frankfort_Plane:
                        message = [
                            "CTVOLUME Orientation : ",
                            "Frankfort Reference Plane has been removed",
                            "Please Add Reference Planes before CTVOLUME Orientation ! ",
                        ]
                        icon = "COLORSET_01_VEC"
                        bpy.ops.wm.bdental_message_box(
                            "INVOKE_DEFAULT", message=str(message), icon=icon
                        )
                        return {"CANCELLED"}
                    else:
                        Vol = [
                            obj
                            for obj in bpy.data.objects
                            if Preffix in obj.name and "_CTVolume" in obj.name
                        ][0]
                        Vol.matrix_world = (
                            Frankfort_Plane.matrix_world.inverted() @ Vol.matrix_world
                        )
                        bpy.ops.view3d.view_center_cursor()
                        bpy.ops.view3d.view_all(center=True)
                        return {"FINISHED"}


class BDENTAL_OT_ResetCtVolumePosition(bpy.types.Operator):
    """Reset the CtVolume to its original Patient Position"""

    bl_idname = "wm.bdental_reset_ctvolume_position"
    bl_label = "RESET CTVolume POSITION"

    def execute(self, context):

        BDENTAL_Props = bpy.context.scene.BDENTAL_Props
        Active_Obj = bpy.context.view_layer.objects.active

        if not Active_Obj:
            message = [" Please select CTVOLUME for segmentation ! "]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )
            return {"CANCELLED"}
        else:
            Condition = CheckString(Active_Obj.name, ["BD"]) and CheckString(
                Active_Obj.name, ["_CTVolume", "Segmentation"], any
            )

            if not Condition:
                message = [
                    "Reset Position : ",
                    "Please select CTVOLUME or Segmentation! ",
                ]
                icon = "COLORSET_01_VEC"
                bpy.ops.wm.bdental_message_box(
                    "INVOKE_DEFAULT", message=str(message), icon=icon
                )
                return {"CANCELLED"}

            else:
                Preffix = Active_Obj.name.split("_")[0]
                Vol = [
                    obj
                    for obj in bpy.data.objects
                    if CheckString(obj.name, [Preffix, "_CTVolume"])
                ][0]
                DcmInfoDict = eval(BDENTAL_Props.DcmInfo)
                DcmInfo = DcmInfoDict[Preffix]
                TransformMatrix = DcmInfo["TransformMatrix"]
                Vol.matrix_world = TransformMatrix

                return {"FINISHED"}


class BDENTAL_OT_AddTeeth(bpy.types.Operator):
    """Add Teeth"""

    bl_idname = "wm.bdental_add_teeth"
    bl_label = "ADD TEETH"
    bl_options = {"REGISTER", "UNDO"}

    def modal(self, context, event):

        ############################################
        if not event.type in {
            "RET",
            "ESC",
        }:
            # allow navigation

            return {"PASS_THROUGH"}

        ###########################################
        elif event.type == "RET":

            if event.value == ("PRESS"):

                Override, area3D, space3D = CtxOverride(context)

                Selected_Teeth = context.selected_objects[:]

                if len(Selected_Teeth) != 0:
                    delta = self.loc - Selected_Teeth[0].location
                    Selected_Teeth[0].location = self.loc
                    print(delta)

                    for i, t in enumerate(Selected_Teeth):
                        if i == 0:
                            continue
                        else:
                            t.location = t.location + delta
                            print(t.name, t.location)

                for obj in self.Coll.objects:
                    if not obj in Selected_Teeth:
                        bpy.data.objects.remove(obj)

                # Restore scene :
                space3D.shading.background_color = self.ViewPortColor
                space3D.shading.color_type = self.ColorType
                space3D.shading.background_type = self.SolidType
                space3D.shading.type = self.BackGroundType

                space3D.overlay.show_annotation = True
                space3D.overlay.show_extras = True
                space3D.overlay.show_floor = True
                space3D.overlay.show_axis_x = True
                space3D.overlay.show_axis_y = True

                for Name in self.visibleObjects:
                    try:
                        obj = bpy.data.objects.get(Name)
                        if obj:
                            obj.hide_set(False)
                    except:
                        pass

                bpy.ops.object.select_all(Override, action="DESELECT")
                bpy.ops.screen.screen_full_area(Override)

                # Override, area3D, space3D = CtxOverride(context)

                return {"FINISHED"}

        ###########################################
        elif event.type == ("ESC"):

            if event.value == ("PRESS"):
                Override, area3D, space3D = CtxOverride(context)
                # Restore scene :
                bpy.context.space_data.shading.background_color = self.ViewPortColor
                bpy.context.space_data.shading.color_type = self.ColorType
                bpy.context.space_data.shading.background_type = self.SolidType
                bpy.context.space_data.shading.type = self.BackGroundType

                space3D.overlay.show_annotation = True
                space3D.overlay.show_extras = True
                space3D.overlay.show_floor = True
                space3D.overlay.show_axis_x = True
                space3D.overlay.show_axis_y = True

                if self.visibleObjects:
                    for Name in self.visibleObjects:
                        obj = bpy.data.objects.get(Name)
                        if obj:
                            obj.hide_set(False)

                for obj in self.Coll.objects:
                    bpy.data.objects.remove(obj)

                bpy.data.collections.remove(self.Coll)
                bpy.ops.object.select_all(Override, action="DESELECT")
                bpy.ops.screen.screen_full_area(Override)

                message = [
                    " Add Teeth Operation was Cancelled!",
                ]

                icon = "COLORSET_02_VEC"
                bpy.ops.wm.bdental_message_box(
                    "INVOKE_DEFAULT", message=str(message), icon=icon
                )

                return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):

        if context.space_data.type == "VIEW_3D":

            BDENTAL_Props = bpy.context.scene.BDENTAL_Props
            self.loc = Vector(context.scene.cursor.location)

            bpy.ops.screen.screen_full_area()
            Override, area3D, space3D = CtxOverride(context)
            bpy.ops.object.select_all(Override, action="DESELECT")

            ###########################################################
            self.TeethLibrary = BDENTAL_Props.TeethLibrary

            self.visibleObjects = [
                obj.name for obj in bpy.context.visible_objects]

            self.BackGroundType = space3D.shading.type
            space3D.shading.type == "SOLID"

            self.SolidType = space3D.shading.background_type
            space3D.shading.background_type = "VIEWPORT"

            self.ColorType = space3D.shading.color_type
            space3D.shading.color_type = "MATERIAL"

            self.ViewPortColor = tuple(space3D.shading.background_color)
            space3D.shading.background_color = (0.0, 0.0, 0.0)

            # Prepare scene  :
            ##########################################################

            space3D.overlay.show_outline_selected = True
            space3D.overlay.show_object_origins = True
            space3D.overlay.show_annotation = False
            space3D.overlay.show_text = True
            space3D.overlay.show_extras = False
            space3D.overlay.show_floor = False
            space3D.overlay.show_axis_x = False
            space3D.overlay.show_axis_y = False

            for Name in self.visibleObjects:
                obj = bpy.data.objects.get(Name)
                if obj:
                    obj.hide_set(True)

            filename = self.TeethLibrary
            directory = join(DataBlendFile, "Collection")
            bpy.ops.wm.append(directory=directory, filename=filename)
            Coll = bpy.data.collections.get(self.TeethLibrary)

            for obj in context.selected_objects:
                MoveToCollection(obj=obj, CollName="Teeth")
            bpy.data.collections.remove(Coll)

            self.Coll = bpy.data.collections.get("Teeth")
            self.mtx = context.scene.cursor.matrix

            bpy.ops.object.select_all(Override, action="DESELECT")

            context.window_manager.modal_handler_add(self)

            return {"RUNNING_MODAL"}

        else:

            message = [
                "Active space must be a View3d",
            ]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}


class BDENTAL_OT_AddImplantSleeve(bpy.types.Operator):
    """Add Sleeve"""

    bl_idname = "wm.bdental_add_implant_sleeve"
    bl_label = "Add Implant Sleeve/Pin"

    @classmethod
    def poll(cls, context):
        if not context.object:
            return False
        return context.object.get("bdental_type") == "bdental_implant"

    def execute(self, context):

        Implant = context.active_object
        cursor = bpy.context.scene.cursor
        cursor.matrix = Implant.matrix_world
        bpy.ops.wm.bdental_add_sleeve(Orientation="AXIAL")

        Sleeve = context.object
        Implant.select_set(True)
        context.view_layer.objects.active = Implant
        constraint = Sleeve.constraints.new("CHILD_OF")
        constraint.target = Implant
        # bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)

        bpy.ops.object.select_all(action="DESELECT")
        Sleeve.select_set(True)
        context.view_layer.objects.active = Sleeve

        return {"FINISHED"}


class BDENTAL_OT_AddSleeve(bpy.types.Operator):
    """Add Sleeve"""

    bl_idname = "wm.bdental_add_sleeve"
    bl_label = "ADD SLEEVE"

    OrientationTypes = ["AXIAL", "SAGITTAL/CORONAL"]
    Orientation: EnumProperty(items=set_enum_items(
        OrientationTypes), description="Orientation", default="AXIAL")

    def execute(self, context):

        BDENTAL_Props = bpy.context.scene.BDENTAL_Props
        self.SleeveDiametre = BDENTAL_Props.SleeveDiameter
        self.SleeveHeight = BDENTAL_Props.SleeveHeight
        self.HoleDiameter = BDENTAL_Props.HoleDiameter
        self.HoleOffset = BDENTAL_Props.HoleOffset
        self.cursor = context.scene.cursor

        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=64,
            radius=self.HoleDiameter / 2 + self.HoleOffset,
            depth=30,
            align="CURSOR",
        )
        Pin = context.object
        Pin.name = "BDENTAL_Pin"
        Pin["bdental_type"] = "bdental_pin"
        if self.Orientation == "SAGITTAL/CORONAL":
            Pin.rotation_euler.rotate_axis("X", radians(-90))

        bpy.ops.mesh.primitive_cylinder_add(
            vertices=64,
            radius=self.SleeveDiametre / 2,
            depth=self.SleeveHeight,
            align="CURSOR",
        )
        Sleeve = context.object
        Sleeve.name = "BDENTAL_Sleeve"
        Sleeve["bdental_type"] = "bdental_sleeve"
        Sleeve.matrix_world[:3] = Pin.matrix_world[:3]

        Sleeve.matrix_world.translation += Sleeve.matrix_world.to_3x3() @ Vector(
            (0, 0, self.SleeveHeight / 2)
        )

        AddMaterial(
            Obj=Pin,
            matName="BDENTAL_Pin_mat",
            color=[0.0, 0.3, 0.8, 1.0],
            transparacy=None,
        )
        AddMaterial(
            Obj=Sleeve,
            matName="BDENTAL_Sleeve_mat",
            color=[1.0, 0.34, 0.0, 1.0],
            transparacy=None,
        )
        Pin.select_set(True)
        constraint = Pin.constraints.new("CHILD_OF")
        constraint.target = Sleeve
        constraint.use_scale_x = False
        constraint.use_scale_y = False
        constraint.use_scale_z = False
        context.view_layer.objects.active = Sleeve
        # bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)

        for obj in [Pin, Sleeve]:
            MoveToCollection(obj, "GUIDE Components")

        return {"FINISHED"}

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)


class BDENTAL_OT_AddImplant(bpy.types.Operator):
    """Add Implant"""

    bl_idname = "wm.bdental_add_implant"
    bl_label = "ADD IMPLANT"

    implant_diameter: FloatProperty(name="Diameter", default=4.0, min=0.0, max=7.0,
                                    step=1, precision=3, unit='LENGTH', description="Implant Diameter")
    implant_lenght: FloatProperty(name="Lenght", default=10.0, min=0.0, max=20.0,
                                  step=1, precision=3, unit='LENGTH', description="Implant Lenght")
    tooth_number: IntProperty(
        name="Tooth Number", default=11, min=11, max=48, description="Tooth Number")
    safe_zone_thikness: FloatProperty(name="Safe Zone Thikness", default=1.5, min=0.0,
                                      max=5.0, step=1, precision=3, unit='LENGTH', description="Safe Zone Thikness")
    sleeve_diameter: FloatProperty(name="Sleeve Diameter", default=8.0, min=0.0,
                                   max=20.0, step=1, precision=3, unit='LENGTH', description="Sleeve Diameter")
    sleeve_height: FloatProperty(name="Sleeve Height", default=10.0, min=0.0,
                                 max=20.0, step=1, precision=3, unit='LENGTH', description="Sleeve Height")
    pin_diameter: FloatProperty(name="Pin Diameter", default=2.0, min=0.0,
                                max=20.0, step=1, precision=3, unit='LENGTH', description="Pin Diameter")
    offset: FloatProperty(name="Offset", default=0.1, min=0.0, max=1.0,
                          step=1, precision=3, unit='LENGTH', description="Offset")

    @classmethod
    def poll(cls, context):
        return context.object and context.object.select_get() and context.object.get("bdental_type") == "slices_pointer"
        # return [obj for obj in context.scene.objects if "_SLICES_POINTER" in obj.name]

    def execute(self, context):

        name = f"BDENTAL_IMPLANT_{self.tooth_number}_{self.implant_diameter}x{self.implant_lenght}mm"
        exists = [
            obj for obj in context.scene.objects if f"BDENTAL_IMPLANT_{self.tooth_number}_" in obj.name]
        if exists:
            message = [f"implant number {self.tooth_number} already exists!"]
            update_info(message)
            sleep(3)
            update_info()
            return {"CANCELLED"}
        implant_type = "up" if self.tooth_number < 31 else "low"
        implants_coll = add_collection("Bdental Implants")
        guide_components_coll = add_collection("GUIDE Components")
        safe_zones_coll = add_collection(
            "Safe Zones", parent_collection=implants_coll)

        coll = AppendCollection("implant", parent_coll_name=implants_coll.name)

        implant = coll.objects.get("implant")
        sleeve = coll.objects.get("sleeve")
        safe_zone = coll.objects.get("safe_zone")
        pin = coll.objects.get("pin")

        implant.name = name
        implant.show_name = True
        implant.dimensions = Vector(
            (self.implant_diameter, self.implant_diameter, self.implant_lenght))
        implant["bdental_type"] = "bdental_implant"
        implant["bdental_remove_code"] = self.tooth_number

        sleeve.name = f"_ADD_SLEEVE({self.tooth_number})"
        sleeve.dimensions = Vector(
            (self.sleeve_diameter, self.sleeve_diameter, self.sleeve_height))
        sleeve["bdental_type"] = "bdental_implant_sleeve"
        sleeve["bdental_remove_code"] = self.tooth_number

        safe_zone.name = f"SAFE_ZONE({self.tooth_number})"
        safe_zone.dimensions = implant.dimensions + \
            Vector((self.safe_zone_thikness*2,
                   self.safe_zone_thikness*2, self.safe_zone_thikness))
        safe_zone["bdental_type"] = "bdental_implant_safe_zone"
        safe_zone["bdental_remove_code"] = self.tooth_number

        pin.name = f"PIN({self.tooth_number})"
        pin.dimensions = Vector((self.pin_diameter+(2*self.offset),
                                self.pin_diameter+(2*self.offset), pin.dimensions[2]))
        pin["bdental_type"] = "bdental_implant_pin"
        pin["bdental_remove_code"] = self.tooth_number

        for obj in [implant, sleeve, safe_zone, pin]:
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(
                location=False, rotation=False, scale=True)

        if implant_type == "up":
            for obj in [implant, sleeve, safe_zone, pin]:
                obj.rotation_euler.rotate_axis("X", radians(180))
                bpy.ops.object.select_all(action="DESELECT")
                obj.select_set(True)
                context.view_layer.objects.active = obj
                bpy.ops.object.transform_apply(
                    location=False, rotation=True, scale=False)

        for obj in [implant, sleeve, safe_zone, pin]:
            obj.matrix_world = self.pointer.matrix_world @ obj.matrix_world

        for obj in [sleeve, pin]:
            child_of = obj.constraints.new("CHILD_OF")
            child_of.target = implant
            child_of.use_scale_x = False
            child_of.use_scale_y = False
            child_of.use_scale_z = False

        child_of = safe_zone.constraints.new("CHILD_OF")
        child_of.target = implant

        bpy.ops.object.select_all(action="DESELECT")
        implant.select_set(True)
        bpy.context.view_layer.objects.active = implant
        bpy.ops.wm.bdental_lock_to_pointer()
        bpy.ops.object.select_all(action="DESELECT")
        self.pointer.select_set(True)
        context.view_layer.objects.active = self.pointer

        for obj in [sleeve, pin]:
            MoveToCollection(obj, guide_components_coll.name)

        MoveToCollection(implant, implants_coll.name)
        MoveToCollection(safe_zone, safe_zones_coll.name)

        bpy.data.collections.remove(coll)

        return {"FINISHED"}

        # bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        # implant = AppendObject("implant", coll_name="Bdental Implants")

        # implant.name = name
        # implant.show_name = True
        # implant.dimensions = Vector((self.implant_diameter, self.implant_diameter, self.implant_lenght))
        # implant["bdental_type"] = "bdental_implant"
        # bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        # safe_zone = AppendObject("safe_zone", coll_name="GUIDE Components")
        # safe_zone.name = f"{name}_SAFE_ZONE"
        # safe_zone.dimensions = implant.dimensions + Vector((self.safe_zone_thikness*2, self.safe_zone_thikness*2, self.safe_zone_thikness))
        # bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        # if implant_type == "up" :
        #     implant.rotation_euler.rotate_axis("X", radians(180))
        #     bpy.ops.object.select_all(action="DESELECT")
        #     implant.select_set(True)
        #     context.view_layer.objects.active = implant
        # bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

        # safe_zone.rotation_euler.rotate_axis("X", radians(180))
        # bpy.ops.object.select_all(action="DESELECT")
        # safe_zone.select_set(True)
        # context.view_layer.objects.active = safe_zone
        # bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

        # safe_zone.parent = implant

        # bpy.ops.object.select_all(action="DESELECT")
        # implant.select_set(True)
        # context.view_layer.objects.active = implant
        # safe_zone.select_set(True)
        # bpy.ops.object.join()

        # implant.matrix_world = self.pointer.matrix_world @ implant.matrix_world
        # bpy.ops.object.select_all(action="DESELECT")
        # implant.select_set(True)
        # bpy.context.view_layer.objects.active = implant
        # bpy.ops.wm.bdental_lock_to_pointer()
        # bpy.ops.object.select_all(action="DESELECT")
        # self.pointer.select_set(True)
        # context.view_layer.objects.active = self.pointer

        # return {"FINISHED"}

    def invoke(self, context, event):
        BDENTAL_Props = bpy.context.scene.BDENTAL_Props
        self.pointer = context.object
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

class BDENTAL_OT_RemoveImplant(bpy.types.Operator):
    """Remove Implant and linked components"""

    bl_idname = "wm.bdental_remove_implant"
    bl_label = "REMOVE IMPLANT"

    
    @classmethod
    def poll(cls, context):
        is_valid = context.object and \
        context.object.select_get() and \
        context.object.get("bdental_type") == "bdental_implant"
        return is_valid

    def execute(self, context):

        implt = context.object
        remove_code = implt.get("bdental_remove_code")
        for o in context.scene.objects :
            if o.get("bdental_remove_code") == remove_code :
                bpy.data.objects.remove(o)
        
        txt = [f"Bdental implant {remove_code} removed."]
        update_info(message=txt, rect_color=[0,1,0,0.7])
        sleep(1)
        update_info()
        return {"FINISHED"}



class BDENTAL_OT_LockToPointer(bpy.types.Operator):
    """add child constraint to slices pointer"""

    bl_idname = "wm.bdental_lock_to_pointer"
    bl_label = "Lock to Pointer"

    @classmethod
    def poll(cls, context):
        pointer_check_list = [
            obj for obj in context.scene.objects if  obj.get("bdental_type") == "slices_pointer"]
        if not pointer_check_list:
            return False
        pointer = pointer_check_list[0]
        if not context.object or context.object.get("bdental_type") != "bdental_implant":
            return False
        if context.object.constraints:
            cp = [c for c in context.object.constraints if c.type ==
                  "CHILD_OF" and c.target == pointer]
            if cp:
                return False
        return True

    def execute(self, context):
        implant = context.object
        mat_bdental_implant_locked = bpy.data.materials.get(
            "mat_bdental_implant_locked") or bpy.data.materials.new("mat_bdental_implant_locked")
        mat_bdental_implant_locked.diffuse_color = (1, 0, 0, 1)  # red color
        mat_bdental_implant_locked.use_nodes = True
        mat_bdental_implant_locked.node_tree.nodes["Principled BSDF"].inputs[19].default_value = (
            1, 0, 0, 1)  # red color
        mat_bdental_implant_locked.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (
            1, 0, 0, 1)  # red color
        for slot in implant.material_slots:
            bpy.ops.object.material_slot_remove({"object": implant})
        implant.active_material = mat_bdental_implant_locked

        pointer = [
            obj for obj in context.scene.objects if  obj.get("bdental_type") == "slices_pointer"][0]
        child_constraint = implant.constraints.new("CHILD_OF")
        child_constraint.target = pointer
        child_constraint.use_scale_x = False
        child_constraint.use_scale_y = False
        child_constraint.use_scale_z = False
        lock_object(context.object)
        context.view_layer.objects.active = pointer
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        pointer.select_set(True)

        return {"FINISHED"}


class BDENTAL_OT_UnlockFromPointer(bpy.types.Operator):
    """remove child constraint to slices pointer"""

    bl_idname = "wm.bdental_unlock_from_pointer"
    bl_label = "Unlock From Pointer"

    @classmethod
    def poll(cls, context):
        pointer_check_list = [
            obj for obj in context.scene.objects if  obj.get("bdental_type") == "slices_pointer"]
        if not pointer_check_list:
            return False

        pointer = pointer_check_list[0]
        if not context.object or context.object.get("bdental_type") != "bdental_implant":
            return False

        if not context.object.constraints:
            return False
        cp = [c for c in context.object.constraints if c.type ==
              "CHILD_OF" and c.target == pointer]
        if not cp:
            return False
        return True

    def execute(self, context):
        implant = context.object
        mat_bdental_implant = bpy.data.materials.get("mat_bdental_implant") or \
            bpy.data.materials.new("mat_bdental_implant")
        mat_bdental_implant.use_nodes = True
        mat_bdental_implant.node_tree.nodes["Principled BSDF"].inputs[0].default_value = [0.471736, 0.015215, 0.800000, 1.000000]
        mat_bdental_implant.node_tree.nodes["Principled BSDF"].inputs[9].default_value = 0.28


        for slot in implant.material_slots:
            bpy.ops.object.material_slot_remove({"object": implant})
        implant.active_material = mat_bdental_implant

        pointer = [
            obj for obj in context.scene.objects if  obj.get("bdental_type") == "slices_pointer"][0]
        
        for c in context.object.constraints:
            if c.type == "CHILD_OF" and c.target == pointer:
                bpy.ops.constraint.apply(constraint=c.name)
        unlock_object(context.object)

        bpy.ops.object.select_all(action="DESELECT")
        pointer.select_set(True)
        context.view_layer.objects.active = pointer

        return {"FINISHED"}


class BDENTAL_OT_ImplantToPointer(bpy.types.Operator):
    """move implant to pointer and add child constraint to slices pointer"""

    bl_idname = "wm.bdental_implant_to_pointer"
    bl_label = "Implant to Pointer"

    @classmethod
    def poll(cls, context):
        if not context.object or not context.object.get("bdental_type") == "bdental_implant" or not context.object.select_get():
            return False
        pointer_check_list = [
            obj for obj in context.scene.objects if  obj.get("bdental_type") == "slices_pointer"]
        if not pointer_check_list:
            return False
        return True

    def execute(self, context):
        active_implant = context.object
        pointer = [
            obj for obj in context.scene.objects if  obj.get("bdental_type") == "slices_pointer"][0]
        implants = [obj for obj in context.scene.objects if obj.get(
            "bdental_type") == "bdental_implant"]

        for implant in implants:
            try:
                context.view_layer.objects.active = implant
                bpy.ops.wm.bdental_unlock_from_pointer()
                break
            except:
                pass
        context.view_layer.objects.active = active_implant
        bpy.ops.object.select_all(action="DESELECT")
        active_implant.select_set(True)
        active_implant.matrix_world[:3] = pointer.matrix_world[:3]
        bpy.ops.wm.bdental_lock_to_pointer()

        return {"FINISHED"}


class BDENTAL_OT_PointerToImplant(bpy.types.Operator):
    """move pointer to implant and add child constraint to slices pointer"""

    bl_idname = "wm.bdental_pointer_to_implant"
    bl_label = "Pointer to Implant"

    @classmethod
    def poll(cls, context):
        if not context.object or not context.object.get("bdental_type") == "bdental_implant" or not context.object.select_get():
            return False
        pointer_check_list = [
            obj for obj in context.scene.objects if  obj.get("bdental_type") == "slices_pointer"]
        if not pointer_check_list:
            return False
        return True

    def execute(self, context):
        active_implant = context.object
        pointer = [
            obj for obj in context.scene.objects if  obj.get("bdental_type") == "slices_pointer"][0]

        # check all implants and apply constraints if exists and unlock them
        implants = [obj for obj in context.scene.objects if obj.get(
            "bdental_type") == "bdental_implant"]
        for implant in implants:
            try:
                context.view_layer.objects.active = implant
                bpy.ops.wm.bdental_unlock_from_pointer()
                break
            except:
                pass

        # Move pointer to implant
        pointer.matrix_world[:3] = active_implant.matrix_world[:3]
        context.view_layer.objects.active = active_implant

        # Lock implant to pointer
        # bpy.ops.wm.bdental_lock_to_pointer()

        # Deselect all but pointer
        context.view_layer.objects.active = pointer
        bpy.ops.object.select_all(action="DESELECT")
        pointer.select_set(True)

        return {"FINISHED"}


class BDENTAL_OT_FlyNext(bpy.types.Operator):
    """move pointer to implants iteratively"""

    bl_idname = "wm.bdental_fly_next"
    bl_label = "Fly Next"

    @classmethod
    def poll(cls, context):
        pointer_check_list = [
            obj for obj in context.scene.objects if  obj.get("bdental_type") == "slices_pointer"]
        
        implant_check_list = [obj for obj in context.scene.objects if obj.get(
            "bdental_type") == "bdental_implant"]
        if not pointer_check_list or not implant_check_list:
            return False
        if len(implant_check_list) < 2:
            return False
        return True

    def execute(self, context):
        global FLY_IMPLANT_INDEX
        pointer_check_list = [
            obj for obj in context.scene.objects if  obj.get("bdental_type") == "slices_pointer"]
        implant_check_list = [obj for obj in context.scene.objects if obj.get(
            "bdental_type") == "bdental_implant"]
        pointer = pointer_check_list[0]

        # check all implants and apply constraints if exists and unlock them
        for implant in implant_check_list:
            try:
                context.view_layer.objects.active = implant
                bpy.ops.wm.bdental_unlock_from_pointer()
                FLY_IMPLANT_INDEX = implant_check_list.index(implant)
                break
            except:
                pass
        # for implant in implant_check_list:
        #     if implant.constraints :
        #         context.view_layer.objects.active = implant
        #         for c in implant.constraints :
        #             if c.type == "CHILD_OF" and c.target == pointer :
        #                 FLY_IMPLANT_INDEX = implant_check_list.index(implant)
        #                 mat_bdental_implant = bpy.data.materials.get("mat_bdental_implant")
        #                 for slot in implant.material_slots:
        #                     bpy.ops.object.material_slot_remove({"object" : implant})
        #                 implant.active_material = mat_bdental_implant
        #             bpy.ops.constraint.apply(constraint=c.name)
        if not FLY_IMPLANT_INDEX:
            FLY_IMPLANT_INDEX = 0
        next_implant_index = FLY_IMPLANT_INDEX + 1
        if next_implant_index >= len(implant_check_list):
            next_implant_index = 0
        FLY_IMPLANT_INDEX = next_implant_index
        active_implant = implant_check_list[next_implant_index]
        # Move pointer to implant
        pointer.matrix_world[:3] = active_implant.matrix_world[:3]

        # Deselect all but pointer
        context.view_layer.objects.active = pointer
        bpy.ops.object.select_all(action="DESELECT")
        pointer.select_set(True)

        tooth_number = active_implant.name.split("_")[2].split("_")[0]
        message = [f"Pointer at implant ({tooth_number})"]
        update_info(message, rect_color=[0.8, 0.6, 0.000000, 1.000000])

        return {"FINISHED"}


class BDENTAL_OT_FlyPrevious(bpy.types.Operator):
    """move pointer to implants iteratively"""

    bl_idname = "wm.bdental_fly_previous"
    bl_label = "Fly Previous"

    @classmethod
    def poll(cls, context):
        pointer_check_list = [
            obj for obj in context.scene.objects if  obj.get("bdental_type") == "slices_pointer"]
        implant_check_list = [obj for obj in context.scene.objects if obj.get(
            "bdental_type") == "bdental_implant"]
        if not pointer_check_list or not implant_check_list:
            return False
        if len(implant_check_list) < 2:
            return False
        return True

    def execute(self, context):
        global FLY_IMPLANT_INDEX
        pointer_check_list = [
            obj for obj in context.scene.objects if  obj.get("bdental_type") == "slices_pointer"]
        implant_check_list = [obj for obj in context.scene.objects if obj.get(
            "bdental_type") == "bdental_implant"]
        pointer = pointer_check_list[0]

        # check all implants and apply constraints if exists and unlock them
        for implant in implant_check_list:
            try:
                context.view_layer.objects.active = implant
                bpy.ops.wm.bdental_unlock_from_pointer()
                FLY_IMPLANT_INDEX = implant_check_list.index(implant)
                break
            except:
                pass
        # for implant in implant_check_list:
        #     if implant.constraints :
        #         context.view_layer.objects.active = implant
        #         for c in implant.constraints :
        #             if c.type == "CHILD_OF" and c.target == pointer :
        #                 FLY_IMPLANT_INDEX = implant_check_list.index(implant)
        #                 mat_bdental_implant = bpy.data.materials.get("mat_bdental_implant")
        #                 for slot in implant.material_slots:
        #                     bpy.ops.object.material_slot_remove({"object" : implant})
        #                 implant.active_material = mat_bdental_implant
        #             bpy.ops.constraint.apply(constraint=c.name)
        if not FLY_IMPLANT_INDEX:
            FLY_IMPLANT_INDEX = 0
        previous_implant_index = FLY_IMPLANT_INDEX - 1
        if previous_implant_index < 0:
            previous_implant_index = len(implant_check_list) - 1
        FLY_IMPLANT_INDEX = previous_implant_index
        active_implant = implant_check_list[previous_implant_index]
        # Move pointer to implant
        pointer.matrix_world[:3] = active_implant.matrix_world[:3]

        # Deselect all but pointer
        context.view_layer.objects.active = pointer
        bpy.ops.object.select_all(action="DESELECT")
        pointer.select_set(True)

        tooth_number = active_implant.name.split("_")[2].split("_")[0]
        message = [f"Pointer at implant ({tooth_number})"]
        update_info(message, rect_color=[0.8, 0.6, 0.000000, 1.000000])

        return {"FINISHED"}


class BDENTAL_OT_RemoveInfoFooter(bpy.types.Operator):
    """Remove Info Footer"""
    bl_idname = "wm.bdental_remove_info_footer"
    bl_label = "Hide Info Footer"

    @classmethod
    def poll(cls, context):
        return DRAW_HANDLERS != []

    def execute(self, context):
        update_info()
        return {"FINISHED"}


class BDENTAL_OT_AlignImplants(bpy.types.Operator):
    """Align Implants"""

    bl_idname = "wm.bdental_align_implants"
    bl_label = "ALIGN IMPLANTS AXES"

    AlignModes = ["To Active", "Averrage Axes"]
    items = []
    for i in range(len(AlignModes)):
        item = (str(AlignModes[i]), str(AlignModes[i]), str(""), int(i))
        items.append(item)

    AlignMode: EnumProperty(
        items=items, description="Implant Align Mode", default="To Active"
    )

    @classmethod
    def poll(cls, context):
        implant_check_list = [obj for obj in context.selected_objects if obj.get(
            "bdental_type") == "bdental_implant"]
        if not implant_check_list:
            return False
        if len(implant_check_list) < 2:
            return False

        if not context.object or not context.object.select_get():
            return False

        return True

    def execute(self, context):
        for implant in self.Implants:
            try:
                context.view_layer.objects.active = implant
                bpy.ops.wm.bdental_unlock_from_pointer()
                break
            except:
                pass

        if self.AlignMode == "Averrage Axes":
            MeanRot = np.mean(
                np.array([Impt.rotation_euler for Impt in self.Implants]), axis=0
            )
            for Impt in self.Implants:
                Impt.rotation_euler = MeanRot

        elif self.AlignMode == "To Active":
            for Impt in self.Implants:
                Impt.rotation_euler = self.Active_Imp.rotation_euler

        # slices_pointer_checklist = [obj for obj in context.scene.objects if "_SLICES_POINTER" in obj.name]
        # if slices_pointer_checklist:
        #     slices_pointer = slices_pointer_checklist[0]
        #     bpy.ops.object.select_all(action="DESELECT")
        #     self.Active_Imp.select_set(True)
        #     context.view_layer.objects.active = self.Active_Imp
        #     bpy.ops.wm.bdental_pointer_to_implant()
        #     bpy.ops.object.select_all(action="DESELECT")
        #     slices_pointer.select_set(True)
        #     context.view_layer.objects.active = slices_pointer
        return {"FINISHED"}

    def invoke(self, context, event):
        self.Active_Imp = context.object
        self.Implants = context.selected_objects[:]
        wm = context.window_manager
        return wm.invoke_props_dialog(self)


class BDENTAL_OT_GuideSetComponents(bpy.types.Operator):
    """set guide components"""
    bl_idname = "wm.bdental_set_guide_components"
    bl_label = "Set Guide Components"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if not context.object:
            return False
        if not context.selected_objects:
            return False
        return ([obj.type == 'MESH' for obj in context.selected_objects] and
                context.object.mode == 'OBJECT' and
                context.object.select_get()
                )

    def execute(self, context):

        context.scene["bdental_guide_components"] = [
            obj.name for obj in context.selected_objects]
        return {"FINISHED"}


class BDENTAL_OT_GuideSetCutters(bpy.types.Operator):
    """set Guide cutters"""
    bl_idname = "wm.bdental_guide_set_cutters"
    bl_label = "Set Guide Cutters"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if not context.object:
            return False
        if not context.selected_objects:
            return False
        return ([obj.type == 'MESH' for obj in context.selected_objects] and
                context.object.mode == 'OBJECT' and
                context.object.select_get()
                )

    def execute(self, context):
        context.scene["bdental_guide_cutters"] = [
            obj.name for obj in context.selected_objects]

        return {"FINISHED"}


##################################################################
class BDENTAL_OT_SplintGuide(bpy.types.Operator):
    """Splint Guide"""

    bl_idname = "wm.bdental_add_guide_splint"
    bl_label = "Add Guide Splint"
    bl_options = {"REGISTER", "UNDO"}

    guide_thikness: FloatProperty(name="Guide Thikness", default=2, min=0.1,
                                  max=10.0, step=1, precision=2, description="MASK BASE 3D PRINTING THIKNESS")
    splint_type: EnumProperty(
        items=set_enum_items(["Splint Up", "Splint Low"]),
        name="Splint Type",
        description="Splint Type (Up,Low)"
    )

    def draw(self, context):
        layout = self.layout
        layout.alignment = "EXPAND"
        row = layout.row()
        row.prop(self, "splint_type")
        row = layout.row()
        row.prop(self, "guide_thikness")

    def add_cutter_point(self):
        Override, area3D, space3D = CtxOverride(bpy.context)

        bpy.ops.object.mode_set(Override, mode="EDIT")
        bpy.ops.curve.extrude(Override, mode="INIT")
        bpy.ops.view3d.snap_selected_to_cursor(Override, use_offset=False)
        bpy.ops.curve.select_all(Override, action="SELECT")
        bpy.ops.curve.handle_type_set(Override, type="AUTOMATIC")
        bpy.ops.curve.select_all(Override, action="DESELECT")
        points = self.cutter.data.splines[0].bezier_points[:]
        points[-1].select_control_point = True
        bpy.ops.object.mode_set(Override, mode="OBJECT")

    def del_cutter_point(self):
        Override, area3D, space3D = CtxOverride(bpy.context)
        try:
            bpy.ops.object.mode_set(Override, mode="EDIT")
            bpy.ops.curve.select_all(Override, action="DESELECT")
            points = self.cutter.data.splines[0].bezier_points[:]
            points[-1].select_control_point = True
            points = self.cutter.data.splines[0].bezier_points[:]
            if len(points) > 1:

                bpy.ops.curve.delete(Override, type="VERT")
                points = self.cutter.data.splines[0].bezier_points[:]
                bpy.ops.curve.select_all(Override, action="SELECT")
                bpy.ops.curve.handle_type_set(Override, type="AUTOMATIC")
                bpy.ops.curve.select_all(Override, action="DESELECT")
                points = self.cutter.data.splines[0].bezier_points[:]
                points[-1].select_control_point = True

            bpy.ops.object.mode_set(Override, mode="OBJECT")

        except Exception:
            pass

    def cut_mesh(self, context):
        Override, area3D, space3D = CtxOverride(bpy.context)

        for obj in [self.cutter, self.base_mesh_duplicate]:
            obj.hide_select = False
            obj.hide_set(False)
            obj.hide_viewport = False
        context.view_layer.objects.active = self.cutter

        bpy.ops.object.mode_set(Override, mode="EDIT")
        bpy.ops.curve.cyclic_toggle()
        bpy.ops.object.mode_set(Override, mode="OBJECT")
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

        self.cutter.data.bevel_depth = 0
        self.cutter.data.resolution_u = 3
        bpy.ops.object.select_all(Override, action="DESELECT")
        self.cutter.select_set(True)
        bpy.ops.object.convert(Override, target="MESH")
        self.cutter = context.object

        bpy.ops.object.modifier_add(Override, type="SHRINKWRAP")
        self.cutter.modifiers["Shrinkwrap"].target = self.base_mesh_duplicate
        bpy.ops.object.convert(Override, target='MESH')

        bpy.ops.object.select_all(Override, action="DESELECT")
        self.base_mesh_duplicate.select_set(True)
        bpy.context.view_layer.objects.active = self.base_mesh_duplicate

        self.base_mesh_duplicate.vertex_groups.clear()
        me = self.base_mesh_duplicate.data

        # initiate a KDTree :
        size = len(me.vertices)
        kd = kdtree.KDTree(size)

        for v_id, v in enumerate(me.vertices):
            kd.insert(v.co, v_id)

        kd.balance()

        CutterCoList = [self.base_mesh_duplicate.matrix_world.inverted(
        ) @ self.cutter.matrix_world @ v.co for v in self.cutter.data.vertices]
        Closest_VIDs = [kd.find(CutterCoList[i])[1]
                        for i in range(len(CutterCoList))]
        CloseState = True
        Loop = ShortestPath(self.base_mesh_duplicate,
                            Closest_VIDs, close=CloseState)

        bpy.ops.object.mode_set(Override, mode="EDIT")
        bpy.ops.mesh.select_all(Override, action='DESELECT')
        bpy.ops.object.mode_set(Override, mode="OBJECT")
        for idx in Loop:
            me.vertices[idx].select = True

        bpy.ops.object.mode_set(Override, mode="EDIT")
        bpy.ops.mesh.looptools_relax(Override,
                                     input="selected", interpolation="cubic", iterations="3", regular=True
                                     )

        # perform cut :
        bpy.ops.object.mode_set(Override, mode="EDIT")
        bpy.ops.mesh.hide()
        bpy.ops.mesh.select_all(Override, action='DESELECT')
        bpy.ops.object.mode_set(Override, mode="OBJECT")

        colist = [(self.base_mesh_duplicate.matrix_world @ v.co)[2]
                  for v in self.base_mesh_duplicate.data.vertices]
        if "Up" in self.splint_type:
            z = max(colist)
        elif "Low" in self.splint_type:
            z = min(colist)

        id = colist.index(z)
        self.base_mesh_duplicate.data.vertices[id].select = True

        bpy.ops.object.mode_set(Override, mode="EDIT")
        bpy.ops.mesh.select_linked(Override)
        # bpy.ops.mesh.select_all(action='INVERT')
        bpy.ops.mesh.delete(Override, type='VERT')
        bpy.ops.mesh.reveal(Override)
        bpy.ops.object.mode_set(Override, mode="OBJECT")

        bpy.data.objects.remove(self.cutter)
        col = bpy.data.collections['Bdental Cutters']
        bpy.data.collections.remove(col)

        bpy.context.scene.tool_settings.use_snap = False
        bpy.ops.wm.tool_set_by_id(Override, name="builtin.select")
        space3D.overlay.show_outline_selected = True

    def splint(self, context):
        self.splint = context.object
        smooth_corrective = self.splint.modifiers.new(
            name="Smooth Corrective", type="CORRECTIVE_SMOOTH")
        smooth_corrective.iterations = 10
        smooth_corrective.use_only_smooth = True

        solidify = self.splint.modifiers.new(name="Solidify", type="SOLIDIFY")
        solidify.thickness = 1
        solidify.offset = 0

        smooth_corrective = self.splint.modifiers.new(
            name="Smooth Corrective", type="CORRECTIVE_SMOOTH")
        smooth_corrective.iterations = 3
        smooth_corrective.use_only_smooth = True

        remesh = self.splint.modifiers.new(name="Remesh", type="REMESH")
        remesh.voxel_size = 0.2

        displace = self.splint.modifiers.new(name="Displace", type="DISPLACE")
        displace.strength = self.guide_thikness - 0.5
        displace.mid_level = 0

        remesh = self.splint.modifiers.new(name="Remesh", type="REMESH")
        remesh.voxel_size = 0.2

        smooth_corrective = self.splint.modifiers.new(
            name="Smooth Corrective", type="CORRECTIVE_SMOOTH")
        smooth_corrective.iterations = 3
        smooth_corrective.use_only_smooth = True

        bpy.ops.object.convert(target='MESH', keep_original=False)

        for slot in self.splint.material_slots:
            bpy.ops.object.material_slot_remove({'object': self.splint})

        mat = bpy.data.materials.get(
            "Splint_mat") or bpy.data.materials.new("Splint_mat")
        mat.diffuse_color = [0.1, 0.4, 1.0, 1.0]
        mat.roughness = 0.3
        self.splint.active_material = mat

        override, _, _ = CtxOverride(context)
        bpy.ops.view3d.view_all(override, center=True)
        bpy.ops.view3d.view_axis(override, type='FRONT')

    def make_boolean(self, context):
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.splint.select_set(True)
        bpy.context.view_layer.objects.active = self.splint
        remesh = self.splint.modifiers.new(name="Remesh", type="REMESH")
        remesh.voxel_size = 0.1
        bpy.ops.object.convert(target='MESH', keep_original=False)

        bool = self.splint.modifiers.new(name="Bool", type="BOOLEAN")
        bool.operation = "DIFFERENCE"
        bool.object = self.bool_model
        bpy.ops.object.convert(target='MESH', keep_original=False)
        bpy.data.objects.remove(self.bool_model)
        for obj in self.start_visible_objects:
            try:
                obj.hide_set(False)
            except:
                pass
        # bpy.ops.object.select_all(action="DESELECT")

    def add_splint_cutter(self, context):
        Override, area3D, space3D = CtxOverride(context)
        # Prepare scene settings :
        bpy.context.scene.tool_settings.use_snap = True
        bpy.context.scene.tool_settings.snap_elements = {"FACE"}

        # ....Add Curve ....... :
        bpy.ops.curve.primitive_bezier_curve_add(Override,
                                                 radius=1, enter_editmode=False, align="CURSOR"
                                                 )
        # Set cutting_tool name :
        self.cutter = bpy.context.view_layer.objects.active

        # CurveCutter settings :
        bpy.ops.object.mode_set(Override, mode="EDIT")
        bpy.ops.curve.select_all(Override, action="DESELECT")
        self.cutter.data.splines[0].bezier_points[-1].select_control_point = True
        bpy.ops.curve.dissolve_verts(Override)
        bpy.ops.curve.select_all(Override, action="SELECT")
        bpy.ops.view3d.snap_selected_to_cursor(Override, use_offset=False)

        bpy.context.object.data.dimensions = "3D"
        bpy.context.object.data.twist_smooth = 3
        bpy.ops.curve.handle_type_set(Override, type="AUTOMATIC")
        bpy.context.object.data.bevel_depth = 0.1
        bpy.context.object.data.bevel_resolution = 6
        bpy.context.scene.tool_settings.curve_paint_settings.error_threshold = 1
        bpy.context.scene.tool_settings.curve_paint_settings.corner_angle = 0.785398
        bpy.context.scene.tool_settings.curve_paint_settings.depth_mode = "SURFACE"
        bpy.context.scene.tool_settings.curve_paint_settings.surface_offset = 0
        bpy.context.scene.tool_settings.curve_paint_settings.use_offset_absolute = True

        # Add color material :
        mat = bpy.data.materials.get(
            "Bdental_curve_cutter_mat"
        ) or bpy.data.materials.new("Bdental_curve_cutter_mat")
        mat.diffuse_color = [0.1, 0.4, 1.0, 1.0]
        mat.roughness = 0.3
        bpy.ops.object.mode_set(mode="OBJECT")
        self.cutter.active_material = mat

        bpy.ops.wm.tool_set_by_id(Override, name="builtin.cursor")
        space3D.overlay.show_outline_selected = False

        shrinkwrap = self.cutter.modifiers.new(
            name="Shrinkwrap", type="SHRINKWRAP")
        shrinkwrap.target = self.base_mesh
        shrinkwrap.wrap_mode = "ABOVE_SURFACE"
        shrinkwrap.use_apply_on_spline = True

        MoveToCollection(self.cutter, "Bdental Cutters")

    @classmethod
    def poll(cls, context):

        base_mesh = context.object and context.object.select_get(
        ) and context.object.type == "MESH"
        return base_mesh

    def modal(self, context, event):
        if not event.type in ["RET", "ESC", "LEFTMOUSE", "DEL"]:
            return {"PASS_THROUGH"}

        elif event.type in ["LEFTMOUSE", "DEL"] and not self.counter in (0, 1):
            return {"PASS_THROUGH"}

        elif event.type == "ESC":
            if event.value == ("PRESS"):

                for obj in bpy.data.objects:
                    if not obj in self.start_objects:
                        bpy.data.objects.remove(obj)
                for col in bpy.data.collections:
                    if not col in self.start_collections:
                        bpy.data.collections.remove(col)

                for obj in context.visible_objects:
                    obj.hide_set(True)
                for obj in self.start_visible_objects:
                    try:
                        obj.hide_set(False)
                    except:
                        pass

                Override, area3D, space3D = CtxOverride(context)
                bpy.ops.wm.tool_set_by_id(Override, name="builtin.select")
                self.scn.tool_settings.use_snap = False
                space3D.overlay.show_outline_selected = True

                message = ["CANCELLED"]
                update_info(message)
                sleep(2)
                update_info()
                return {"CANCELLED"}

        elif event.type == "RET" and self.counter == 2:
            if event.value == ("PRESS"):

                message = ["Guide Splint Remeshing ..."]
                update_info(message)
                context.view_layer.objects.active = self.splint
                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.select_all(action="DESELECT")
                self.splint.select_set(True)
                remesh = self.splint.modifiers.new(
                    name="Remesh", type="REMESH")
                bpy.ops.object.convert(target='MESH', keep_original=False)

                message = ["FINISHED ./"]
                update_info(message)
                sleep(1)
                update_info()
                return {"FINISHED"}

        elif event.type == "RET" and self.counter == 1:
            if event.value == ("PRESS"):
                self.counter += 1
                message = ["Cutting Mesh..."]
                update_info(message)
                self.cut_mesh(context)

                message = [f"Creating Guide Splint {self.splint_suffix} ..."]
                update_info(message)
                self.splint(context)

                bpy.ops.object.mode_set(mode='SCULPT')

                # Set the active brush to Smooth
                bpy.context.tool_settings.sculpt.brush = bpy.data.brushes['Smooth']

                # Optionally, set the brush strength (default is 0.5)
                bpy.context.tool_settings.sculpt.brush.strength = 0.8

                message = [
                    f"(Optional) : Please smooth Guide Splint and press ENTER ..."]
                update_info(message)
                return {"RUNNING_MODAL"}

        elif event.type == ("LEFTMOUSE") and self.counter == 1:
            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}
            if event.value == ("RELEASE"):
                self.add_cutter_point()
                return {"RUNNING_MODAL"}

        elif event.type == ("LEFTMOUSE") and self.counter == 0:
            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}
            if event.value == ("RELEASE"):
                self.add_splint_cutter(context)
                self.counter += 1
                return {"RUNNING_MODAL"}

        elif event.type == ("DEL") and self.counter == 1:
            if event.value == ("PRESS"):
                self.del_cutter_point()
                return {"RUNNING_MODAL"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        if context.space_data.type == "VIEW_3D":
            self.base_mesh = context.object
            wm = context.window_manager
            return wm.invoke_props_dialog(self, width=500)

        else:

            message = ["Active space must be a View3d"]
            icon = "COLORSET_02_VEC"
            bpy.ops.bdental.message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon)
            return {"CANCELLED"}

    def execute(self, context):
        self.scn = context.scene
        self.counter = 0

        self.start_objects = bpy.data.objects[:]
        self.start_collections = bpy.data.collections[:]
        self.start_visible_objects = bpy.context.visible_objects[:]

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.base_mesh.select_set(True)
        context.view_layer.objects.active = self.base_mesh
        bpy.ops.object.duplicate_move()
        self.base_mesh_duplicate = context.object

        smooth_corrective = self.base_mesh_duplicate.modifiers.new(
            name="Smooth Corrective", type="CORRECTIVE_SMOOTH")
        smooth_corrective.iterations = 10
        smooth_corrective.use_only_smooth = True
        bpy.ops.object.convert(target='MESH', keep_original=False)
        self.base_mesh_duplicate.name = "Guide Splint"
        self.base_mesh_duplicate["bdental_type"] = "bdental_splint"
        self.base_mesh_duplicate.hide_set(True)

        bpy.ops.object.select_all(action="DESELECT")
        self.base_mesh.select_set(True)
        context.view_layer.objects.active = self.base_mesh

        self.splint_suffix = "Up" if "up" in self.splint_type.lower() else "Low"

        override, _, _ = CtxOverride(context)
        bpy.ops.wm.tool_set_by_id(override, name="builtin.cursor")
        context.window_manager.modal_handler_add(self)
        message = ["please draw Guide border", "when done press ENTER"]
        update_info(message)
        return {"RUNNING_MODAL"}


class BDENTAL_OT_SplintGuideGeom(bpy.types.Operator):
    """Splint Guide Geo Nodes"""

    bl_idname = "wm.bdental_add_guide_splint_geom"
    bl_label = "Add Guide Splint"
    bl_options = {"REGISTER", "UNDO"}

    guide_thikness: FloatProperty(name="Guide Thikness", default=2, min=0.1,
                                  max=10.0, step=1, precision=2, description="MASK BASE 3D PRINTING THIKNESS")
    splint_type: EnumProperty(
        items=set_enum_items(["Splint Up", "Splint Low"]),
        name="Splint Type",
        description="Splint Type (Up,Low)"
    )

    def draw(self, context):
        layout = self.layout
        layout.alignment = "EXPAND"
        row = layout.row()
        row.prop(self, "splint_type")
        row = layout.row()
        row.prop(self, "guide_thikness")

    def add_cutter_point(self):
        Override, area3D, space3D = CtxOverride(bpy.context)

        bpy.ops.object.mode_set(Override, mode="EDIT")
        bpy.ops.curve.extrude(Override, mode="INIT")
        bpy.ops.view3d.snap_selected_to_cursor(Override, use_offset=False)
        bpy.ops.curve.select_all(Override, action="SELECT")
        bpy.ops.curve.handle_type_set(Override, type="AUTOMATIC")
        bpy.ops.curve.select_all(Override, action="DESELECT")
        points = self.cutter.data.splines[0].bezier_points[:]
        points[-1].select_control_point = True
        bpy.ops.object.mode_set(Override, mode="OBJECT")

    def del_cutter_point(self):
        Override, area3D, space3D = CtxOverride(bpy.context)
        try:
            bpy.ops.object.mode_set(Override, mode="EDIT")
            bpy.ops.curve.select_all(Override, action="DESELECT")
            points = self.cutter.data.splines[0].bezier_points[:]
            points[-1].select_control_point = True
            points = self.cutter.data.splines[0].bezier_points[:]
            if len(points) > 1:

                bpy.ops.curve.delete(Override, type="VERT")
                points = self.cutter.data.splines[0].bezier_points[:]
                bpy.ops.curve.select_all(Override, action="SELECT")
                bpy.ops.curve.handle_type_set(Override, type="AUTOMATIC")
                bpy.ops.curve.select_all(Override, action="DESELECT")
                points = self.cutter.data.splines[0].bezier_points[:]
                points[-1].select_control_point = True

            bpy.ops.object.mode_set(Override, mode="OBJECT")

        except Exception:
            pass

    def cut_mesh(self, context):
        Override, area3D, space3D = CtxOverride(bpy.context)

        for obj in [self.cutter, self.base_mesh_duplicate]:
            obj.hide_select = False
            obj.hide_set(False)
            obj.hide_viewport = False
        context.view_layer.objects.active = self.cutter

        bpy.ops.object.mode_set(Override, mode="EDIT")
        bpy.ops.curve.cyclic_toggle()
        bpy.ops.object.mode_set(Override, mode="OBJECT")
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

        self.cutter.data.bevel_depth = 0
        self.cutter.data.resolution_u = 3
        bpy.ops.object.select_all(Override, action="DESELECT")
        self.cutter.select_set(True)
        bpy.ops.object.convert(Override, target="MESH")
        self.cutter = context.object

        bpy.ops.object.modifier_add(Override, type="SHRINKWRAP")
        self.cutter.modifiers["Shrinkwrap"].target = self.base_mesh_duplicate
        bpy.ops.object.convert(Override, target='MESH')

        bpy.ops.object.select_all(Override, action="DESELECT")
        self.base_mesh_duplicate.select_set(True)
        bpy.context.view_layer.objects.active = self.base_mesh_duplicate

        self.base_mesh_duplicate.vertex_groups.clear()
        me = self.base_mesh_duplicate.data

        # initiate a KDTree :
        size = len(me.vertices)
        kd = kdtree.KDTree(size)

        for v_id, v in enumerate(me.vertices):
            kd.insert(v.co, v_id)

        kd.balance()

        CutterCoList = [self.base_mesh_duplicate.matrix_world.inverted(
        ) @ self.cutter.matrix_world @ v.co for v in self.cutter.data.vertices]
        Closest_VIDs = [kd.find(CutterCoList[i])[1]
                        for i in range(len(CutterCoList))]
        CloseState = True
        Loop = ShortestPath(self.base_mesh_duplicate,
                            Closest_VIDs, close=CloseState)

        bpy.ops.object.mode_set(Override, mode="EDIT")
        bpy.ops.mesh.select_all(Override, action='DESELECT')
        bpy.ops.object.mode_set(Override, mode="OBJECT")
        for idx in Loop:
            me.vertices[idx].select = True

        bpy.ops.object.mode_set(Override, mode="EDIT")
        bpy.ops.mesh.looptools_relax(Override,
                                     input="selected", interpolation="cubic", iterations="3", regular=True
                                     )

        # perform cut :
        bpy.ops.object.mode_set(Override, mode="EDIT")
        bpy.ops.mesh.hide()
        bpy.ops.mesh.select_all(Override, action='DESELECT')
        bpy.ops.object.mode_set(Override, mode="OBJECT")

        colist = [(self.base_mesh_duplicate.matrix_world @ v.co)[2]
                  for v in self.base_mesh_duplicate.data.vertices]
        if "Up" in self.splint_type:
            z = max(colist)
        elif "Low" in self.splint_type:
            z = min(colist)

        id = colist.index(z)
        self.base_mesh_duplicate.data.vertices[id].select = True

        bpy.ops.object.mode_set(Override, mode="EDIT")
        bpy.ops.mesh.select_linked(Override)
        # bpy.ops.mesh.select_all(action='INVERT')
        bpy.ops.mesh.delete(Override, type='VERT')
        bpy.ops.mesh.reveal(Override)
        bpy.ops.object.mode_set(Override, mode="OBJECT")

        bpy.data.objects.remove(self.cutter)
        col = bpy.data.collections['Bdental Cutters']
        bpy.data.collections.remove(col)

        bpy.context.scene.tool_settings.use_snap = False
        bpy.ops.wm.tool_set_by_id(Override, name="builtin.select")
        space3D.overlay.show_outline_selected = True

    def splint(self, context):
        global bdental_volume_node_name
        self.splint = context.object
        smooth_corrective = self.splint.modifiers.new(
            name="Smooth Corrective", type="CORRECTIVE_SMOOTH")
        smooth_corrective.iterations = 5
        smooth_corrective.use_only_smooth = True
        smooth_corrective.use_pin_boundary = True

        gn = append_group_nodes(bdental_volume_node_name)
        mesh_to_volume(self.splint, gn,
                       offset_out=self.guide_thikness, offset_in=-1)

        remesh = self.splint.modifiers.new(name="Remesh", type="REMESH")
        remesh.voxel_size = 0.2

        smooth_corrective = self.splint.modifiers.new(
            name="Smooth Corrective", type="CORRECTIVE_SMOOTH")
        smooth_corrective.iterations = 10
        smooth_corrective.use_only_smooth = True
        smooth_corrective.use_pin_boundary = True

        bpy.ops.object.select_all(action='DESELECT')
        self.splint.select_set(True)
        bpy.context.view_layer.objects.active = self.splint
        bpy.ops.object.convert(target='MESH', keep_original=False)

        for slot in self.splint.material_slots:
            bpy.ops.object.material_slot_remove({'object': self.splint})

        mat = bpy.data.materials.get(
            "Splint_mat") or bpy.data.materials.new("Splint_mat")
        mat.diffuse_color = [0.1, 0.4, 1.0, 1.0]
        mat.roughness = 0.3
        self.splint.active_material = mat

        override, _, _ = CtxOverride(context)
        bpy.ops.view3d.view_all(override, center=True)
        bpy.ops.view3d.view_axis(override, type='FRONT')

    def add_splint_cutter(self, context):
        Override, area3D, space3D = CtxOverride(context)
        # Prepare scene settings :
        bpy.context.scene.tool_settings.use_snap = True
        bpy.context.scene.tool_settings.snap_elements = {"FACE"}

        # ....Add Curve ....... :
        bpy.ops.curve.primitive_bezier_curve_add(Override,
                                                 radius=1, enter_editmode=False, align="CURSOR"
                                                 )
        # Set cutting_tool name :
        self.cutter = bpy.context.view_layer.objects.active

        # CurveCutter settings :
        bpy.ops.object.mode_set(Override, mode="EDIT")
        bpy.ops.curve.select_all(Override, action="DESELECT")
        self.cutter.data.splines[0].bezier_points[-1].select_control_point = True
        bpy.ops.curve.dissolve_verts(Override)
        bpy.ops.curve.select_all(Override, action="SELECT")
        bpy.ops.view3d.snap_selected_to_cursor(Override, use_offset=False)

        bpy.context.object.data.dimensions = "3D"
        bpy.context.object.data.twist_smooth = 3
        bpy.ops.curve.handle_type_set(Override, type="AUTOMATIC")
        bpy.context.object.data.bevel_depth = 0.1
        bpy.context.object.data.bevel_resolution = 6
        bpy.context.scene.tool_settings.curve_paint_settings.error_threshold = 1
        bpy.context.scene.tool_settings.curve_paint_settings.corner_angle = 0.785398
        bpy.context.scene.tool_settings.curve_paint_settings.depth_mode = "SURFACE"
        bpy.context.scene.tool_settings.curve_paint_settings.surface_offset = 0
        bpy.context.scene.tool_settings.curve_paint_settings.use_offset_absolute = True

        # Add color material :
        mat = bpy.data.materials.get(
            "Bdental_curve_cutter_mat"
        ) or bpy.data.materials.new("Bdental_curve_cutter_mat")
        mat.diffuse_color = [0.1, 0.4, 1.0, 1.0]
        mat.roughness = 0.3
        bpy.ops.object.mode_set(mode="OBJECT")
        self.cutter.active_material = mat

        bpy.ops.wm.tool_set_by_id(Override, name="builtin.cursor")
        space3D.overlay.show_outline_selected = False

        shrinkwrap = self.cutter.modifiers.new(
            name="Shrinkwrap", type="SHRINKWRAP")
        shrinkwrap.target = self.base_mesh
        shrinkwrap.wrap_mode = "ABOVE_SURFACE"
        shrinkwrap.use_apply_on_spline = True

        MoveToCollection(self.cutter, "Bdental Cutters")

    @classmethod
    def poll(cls, context):

        base_mesh = context.object and context.object.select_get(
        ) and context.object.type == "MESH"
        return base_mesh

    def modal(self, context, event):
        if not event.type in ["RET", "ESC", "LEFTMOUSE", "DEL"]:
            return {"PASS_THROUGH"}

        elif event.type in ["LEFTMOUSE", "DEL"] and not self.counter in (0, 1):
            return {"PASS_THROUGH"}

        elif event.type == "ESC":
            if event.value == ("PRESS"):

                for obj in bpy.data.objects:
                    if not obj in self.start_objects:
                        bpy.data.objects.remove(obj)
                for col in bpy.data.collections:
                    if not col in self.start_collections:
                        bpy.data.collections.remove(col)

                for obj in context.visible_objects:
                    obj.hide_set(True)
                for obj in self.start_visible_objects:
                    try:
                        obj.hide_set(False)
                    except:
                        pass

                Override, area3D, space3D = CtxOverride(context)
                bpy.ops.wm.tool_set_by_id(Override, name="builtin.select")
                self.scn.tool_settings.use_snap = False
                space3D.overlay.show_outline_selected = True

                message = ["CANCELLED"]
                update_info(message)
                sleep(2)
                update_info()
                return {"CANCELLED"}

        elif event.type == "RET" and self.counter == 2:
            if event.value == ("PRESS"):

                message = ["Guide Splint Remeshing ..."]
                update_info(message)
                context.view_layer.objects.active = self.splint
                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.select_all(action="DESELECT")
                self.splint.select_set(True)
                remesh = self.splint.modifiers.new(
                    name="Remesh", type="REMESH")
                remesh.voxel_size = 0.3
                bpy.ops.object.convert(target='MESH', keep_original=False)

                message = ["FINISHED ./"]
                update_info(message)
                sleep(1)
                update_info()
                return {"FINISHED"}

        elif event.type == "RET" and self.counter == 1:
            if event.value == ("PRESS"):
                self.counter += 1
                message = ["Cutting Mesh..."]
                update_info(message)
                self.cut_mesh(context)

                message = [f"Creating Guide Splint {self.splint_suffix} ..."]
                update_info(message)
                self.splint(context)

                bpy.ops.object.mode_set(mode='SCULPT')

                # Set the active brush to Smooth
                bpy.context.tool_settings.sculpt.brush = bpy.data.brushes['Smooth']

                # Optionally, set the brush strength (default is 0.5)
                bpy.context.tool_settings.sculpt.brush.strength = 0.8

                message = [
                    f"(Optional) : Please smooth Guide Splint and press ENTER ..."]
                update_info(message)
                return {"RUNNING_MODAL"}

        elif event.type == ("LEFTMOUSE") and self.counter == 1:
            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}
            if event.value == ("RELEASE"):
                self.add_cutter_point()
                return {"RUNNING_MODAL"}

        elif event.type == ("LEFTMOUSE") and self.counter == 0:
            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}
            if event.value == ("RELEASE"):
                self.add_splint_cutter(context)
                self.counter += 1
                return {"RUNNING_MODAL"}

        elif event.type == ("DEL") and self.counter == 1:
            if event.value == ("PRESS"):
                self.del_cutter_point()
                return {"RUNNING_MODAL"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        if context.space_data.type == "VIEW_3D":
            self.base_mesh = context.object
            wm = context.window_manager
            return wm.invoke_props_dialog(self, width=500)

        else:

            message = ["Active space must be a View3d"]
            icon = "COLORSET_02_VEC"
            bpy.ops.bdental.message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon)
            return {"CANCELLED"}

    def execute(self, context):
        guide_components_coll = add_collection("GUIDE Components")
        self.scn = context.scene
        self.counter = 0

        self.start_objects = bpy.data.objects[:]
        self.start_collections = bpy.data.collections[:]
        self.start_visible_objects = bpy.context.visible_objects[:]

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.base_mesh.select_set(True)
        context.view_layer.objects.active = self.base_mesh
        bpy.ops.object.duplicate_move()
        self.base_mesh_duplicate = context.object

        smooth_corrective = self.base_mesh_duplicate.modifiers.new(
            name="Smooth Corrective", type="CORRECTIVE_SMOOTH")
        smooth_corrective.iterations = 10
        smooth_corrective.use_only_smooth = True
        bpy.ops.object.convert(target='MESH', keep_original=False)
        self.base_mesh_duplicate.name = "_ADD_Guide_Splint"
        self.base_mesh_duplicate["bdental_type"] = "bdental_splint"
        MoveToCollection(self.base_mesh_duplicate, guide_components_coll.name)
        self.base_mesh_duplicate.hide_set(True)

        bpy.ops.object.select_all(action="DESELECT")
        self.base_mesh.select_set(True)
        context.view_layer.objects.active = self.base_mesh

        self.splint_suffix = "Up" if "up" in self.splint_type.lower() else "Low"

        override, _, _ = CtxOverride(context)
        bpy.ops.wm.tool_set_by_id(override, name="builtin.cursor")
        context.window_manager.modal_handler_add(self)
        message = ["please draw Guide border", "when done press ENTER"]
        update_info(message)
        return {"RUNNING_MODAL"}


class BDENTAL_OT_GuideFinalise(bpy.types.Operator):
    """add Guide Cutters From Sleeves"""

    bl_idname = "wm.bdental_guide_finalise"
    bl_label = "Guide Finalise"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        coll = bpy.data.collections.get("GUIDE Components")
        if coll and coll.objects:
            return True
        else:
            return False

    def execute(self, context):
        # guide_components_names = context.scene.get("bdental_guide_components")
        # guide_cutters_names = context.scene.get("bdental_guide_cutters")

        # if not guide_components_names :
        #     message = ["Cancelled : Can't find Guide Components !"]
        #     update_info(message)
        #     sleep(3)
        #     update_info()
        #     return {"CANCELLED"}

        # guide_components = []
        # for name in guide_components_names :
        #     obj = bpy.data.objects.get(name)
        #     if obj :
        #         guide_components.append(obj)

        guide, guide_cutter = None, None
        coll = bpy.data.collections.get("GUIDE Components")
        guide_components = coll.objects
        add_components, cut_components = [], []
        for obj in guide_components:
            if obj.name.startswith("_ADD_"):
                add_components.append(obj.name)
            else:
                cut_components.append(obj.name)
        if not add_components:
            message = ["Cancelled : Can't find Guide _ADD_ Components !"]
            update_info(message)
            sleep(3)
            update_info()
            return {"CANCELLED"}

        message = ["Preparing Guide _ADD_ Components ..."]
        update_info(message)
        context.view_layer.objects.active = bpy.data.objects[add_components[0]]
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action='DESELECT')

        for name in add_components:
            obj = bpy.data.objects.get(name)
            obj.hide_set(False)
            obj.hide_viewport = False
            context.view_layer.objects.active = obj
            obj.select_set(True)

        bpy.ops.object.duplicate_move()
        add_components_dup = context.selected_objects[:]
        for obj in add_components_dup:
            bpy.ops.object.select_all(action='DESELECT')
            context.view_layer.objects.active = obj
            obj.select_set(True)
            if obj.constraints:
                for c in obj.constraints:
                    bpy.ops.constraint.apply(constraint=c.name)
            elif obj.modifiers or obj.type == "CURVE":
                bpy.ops.object.convert(target='MESH', keep_original=False)

            # check non manifold :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.context.tool_settings.mesh_select_mode = (True, False, False)
            bpy.ops.mesh.select_non_manifold()
            bpy.ops.object.mode_set(mode="OBJECT")

            if bpy.context.object.data.total_edge_sel > 0:
                remesh = guide.modifiers.new(name="Remesh", type="REMESH")
                remesh.mode = "SHARP"
                remesh.octree_depth = 8

        for name in add_components:
            obj = bpy.data.objects.get(name)
            obj.hide_set(True)

        for obj in add_components_dup:
            context.view_layer.objects.active = obj
            obj.select_set(True)
        if len(add_components) > 1:
            bpy.ops.object.join()

        guide = context.object
        mat = bpy.data.materials.get(
            "Splint_mat") or bpy.data.materials.new(name="Splint_mat")
        mat.diffuse_color = [0.000000, 0.227144, 0.275441, 1.000000]
        mat.use_nodes = True
        mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = [
            0.000000, 0.227144, 0.275441, 1.000000]
        for slot in guide.material_slots:
            bpy.ops.object.material_slot_remove({"object": guide})
        guide.active_material = mat

        # remesh :
        remesh = guide.modifiers.new(name="Remesh", type="REMESH")
        remesh.voxel_size = 0.1
        decimate = guide.modifiers.new(name="Decimate", type="DECIMATE")
        decimate.ratio = 0.2
        bpy.ops.object.convert(target='MESH', keep_original=False)

        # cutters :
        if not cut_components:
            guide.name = "Bdental Guide"
            guide["bdental_type"] = "bdental_guide"
            bpy.context.scene.collection.objects.link(guide)
            for col in guide.users_collection:
                col.objects.unlink(guide)

            message = ["Warning : Can't find Guide _CUT_ Components !",
                       "Guide will be exported without cutting !"]
            update_info(message)
            sleep(3)
            update_info()
            return {"FINISHED"}

        bpy.ops.object.select_all(action='DESELECT')
        message = ["Preaparing Cutters ..."]
        update_info(message)
        for i, name in enumerate(cut_components):
            obj = bpy.data.objects.get(name)
            obj.hide_set(False)
            obj.hide_viewport = False
            print(obj)
            context.view_layer.objects.active = obj
            obj.select_set(True)
            if len(obj.data.vertices) > 50000:
                decimate = obj.modifiers.new(name="Decimate", type="DECIMATE")
                ratio = 50000 / len(obj.data.vertices)
                decimate.ratio = ratio
            bpy.ops.object.convert(target='MESH', keep_original=False)

            # if obj.constraints :
            #     for c in obj.constraints :
            #         bpy.ops.constraint.apply(constraint=c.name)
            # elif obj.modifiers or obj.type == "CURVE" :
            #     bpy.ops.object.convert(target='MESH', keep_original=False)

            # # check non manifold :
            # bpy.ops.object.mode_set(mode="EDIT")
            # bpy.ops.mesh.select_all(action='DESELECT')
            # bpy.context.tool_settings.mesh_select_mode = (True, False, False)
            # bpy.ops.mesh.select_non_manifold()
            # bpy.ops.object.mode_set(mode="OBJECT")

            # if bpy.context.object.data.total_edge_sel > 0 :
            #     remesh = guide_cutter.modifiers.new(name="Remesh", type="REMESH")
            #     remesh.voxel_size = 0.1
            #     bpy.ops.object.convert(target='MESH', keep_original=False)
            message = [f"Making Cuts {i+1}/{len(cut_components)}..."]
            update_info(message)

            bpy.ops.object.select_all(action='DESELECT')
            context.view_layer.objects.active = guide
            guide.select_set(True)
            bool_modif = guide.modifiers.new(name="Bool", type="BOOLEAN")
            bool_modif.operation = "DIFFERENCE"
            bool_modif.solver = "FAST"
            bool_modif.object = obj
            bpy.ops.object.convert(target='MESH', keep_original=False)
            obj.hide_set(True)

        # cutter = context.object
        # bpy.ops.object.select_all(action='DESELECT')
        # context.view_layer.objects.active = guide
        # guide.select_set(True)
        # try :
        #     bool_modif = guide.modifiers.new(name="Bool", type="BOOLEAN")
        #     bool_modif.object = cutter
        #     bool_modif.operation = "DIFFERENCE"

        #     bpy.ops.object.convert(target='MESH', keep_original=False)
        #     bpy.data.objects.remove(cutter, do_unlink=True)
        # except Exception as e:
        #     print(e)
        #     pass

            # if len(guide_cutters_dup) > 1 :
            #     bpy.ops.object.join()
            # guide_cutter = context.object

            # check non manifold :
            # bpy.ops.object.mode_set(mode="EDIT")
            # bpy.ops.mesh.select_all(action='DESELECT')
            # bpy.context.tool_settings.mesh_select_mode = (True, False, False)
            # bpy.ops.mesh.select_non_manifold()
            # bpy.ops.object.mode_set(mode="OBJECT")

            # if bpy.context.object.data.total_edge_sel > 0 :
            #     remesh = guide.modifiers.new(name="Remesh", type="REMESH")
            #     remesh.mode = "SHARP"
            #     remesh.octree_depth = 8

            # remesh = guide_cutter.modifiers.new(name="Remesh", type="REMESH")
            # remesh.voxel_size = 0.1
            # bpy.ops.object.convert(target='MESH', keep_original=False)

        # if guide and guide_cutter :
        #     message = ["Guide Processing -> Boolean operation ..."]
        #     update_info(message)

        #     bpy.ops.object.select_all(action='DESELECT')
        #     guide.select_set(True)
        #     context.view_layer.objects.active = guide

        #     bool_modif = guide.modifiers.new(name="Bool", type="BOOLEAN")
        #     bool_modif.object = guide_cutter
        #     bool_modif.operation = "DIFFERENCE"

        #     bpy.ops.object.convert(target='MESH', keep_original=False)
        #     bpy.data.objects.remove(guide_cutter, do_unlink=True)
        guide.name = "Bdental Guide"
        guide["bdental_type"] = "bdental_guide"

        for col in guide.users_collection:
            col.objects.unlink(guide)

        bpy.context.scene.collection.objects.link(guide)
        os.system("cls")

        message = ["Finished ."]
        update_info(message)
        sleep(2)
        update_info()
        return {"FINISHED"}

        # for name in guide_cutters_names :
        #     obj = bpy.data.objects.get(name)
        #     if obj :
        #         context.view_layer.objects.active = obj
        #         bpy.ops.object.mode_set(mode="OBJECT")
        #         bpy.ops.object.select_all(action='DESELECT')
        #         obj.select_set(True)
        #         bpy.ops.object.duplicate_move()
        #         bpy.ops.object.convert(target='MESH', keep_original=False)
        #         guide_cutters.append(obj)
        # splint = context.object
        # bool_model = context.scene.BDENTAL_Props.splint_target_model
        # sleeves_checklist = [obj for obj in bpy.data.objects if obj.get("bdental_type") and "bdental_sleeve" in obj.get("bdental_type")]
        # pins_checklist = [obj for obj in bpy.data.objects if obj.get("bdental_type") and "bdental_pin" in obj.get("bdental_type")]
        # message = []
        # if not bool_model :
        #     message.extend(["Operation Cancelled !","Can't find Passive Blocked Model"])
        # if not sleeves_checklist :
        #     message.extend([", Sleeves"])
        # if not pins_checklist :
        #     message.extend([", Pins"])
        # if message :
        #     update_info(message)
        #     sleep(3)
        #     update_info()
        #     return {"CANCELLED"}

        # message = ["Guide Processing -> Prepare Blocked model..."]
        # update_info(message)

        # bpy.ops.object.mode_set(mode="OBJECT")
        # bool_model.hide_set(False)
        # bool_model.hide_viewport = False
        # bpy.ops.object.select_all(action="DESELECT")
        # bool_model.select_set(True)
        # context.view_layer.objects.active = bool_model
        # bpy.ops.object.duplicate_move()
        # bool_model_duplicate = context.object

        # displace_modif = bool_model_duplicate.modifiers.new(name="Displace", type="DISPLACE")
        # displace_modif.strength = context.scene.BDENTAL_Props.splint_offset
        # displace_modif.mid_level = 0

        # # remesh = bool_model_duplicate.modifiers.new(name="Remesh", type="REMESH")
        # bpy.ops.object.convert(target='MESH', keep_original=False)

        # message = ["Guide Processing -> Combine sleeves..."]
        # update_info(message)

        # splint.hide_set(False)
        # splint.hide_viewport = False
        # context.view_layer.objects.active = splint
        # bpy.ops.object.mode_set(mode="OBJECT")

        # for sleeve in sleeves_checklist :
        #     sleeve.hide_set(False)
        #     sleeve.hide_viewport = False
        #     bpy.ops.object.select_all(action="DESELECT")
        #     sleeve.select_set(True)
        #     context.view_layer.objects.active = sleeve
        #     for constraint in sleeve.constraints :
        #         bpy.ops.constraint.apply(constraint="Child Of", owner='OBJECT')

        # bpy.ops.object.select_all(action="DESELECT")
        # splint.select_set(True)
        # context.view_layer.objects.active = splint
        # for sleeve in sleeves_checklist :
        #     sleeve.select_set(True)

        # bpy.ops.object.join()
        # joined_splint = context.object
        # remesh = joined_splint.modifiers.new(name="Remesh", type="REMESH")

        # bpy.ops.object.convert(target='MESH', keep_original=False)

        # bpy.ops.object.select_all(action="DESELECT")

        # for pin in pins_checklist :
        #     pin.hide_set(False)
        #     pin.hide_viewport = False
        #     bpy.ops.object.select_all(action="DESELECT")
        #     pin.select_set(True)
        #     context.view_layer.objects.active = pin
        #     for constraint in pin.constraints :
        #         bpy.ops.constraint.apply(constraint="Child Of", owner='OBJECT')
        # bpy.ops.object.select_all(action="DESELECT")
        # for pin in pins_checklist :
        #     pin.select_set(True)
        #     context.view_layer.objects.active = pin

        # if len(pins_checklist) > 1 :
        #     bpy.ops.object.join()
        # pins_all = context.object

        # # pins_all.display_type = 'WIRE'
        # context.view_layer.objects.active = joined_splint
        # bpy.ops.object.select_all(action="DESELECT")
        # joined_splint.select_set(True)

        # bool_modif_1 = joined_splint.modifiers.new(name="bool1", type="BOOLEAN")
        # bool_modif_1.object = pins_all
        # bool_modif_1.operation = "DIFFERENCE"

        # message = ["Guide Processing -> Please wait while performing Booleans :) ..."]
        # update_info(message)

        # bool_modif_2 = joined_splint.modifiers.new(name="bool2", type="BOOLEAN")
        # bool_modif_2.object = bool_model_duplicate
        # bool_modif_2.operation = "DIFFERENCE"

        # bpy.ops.object.convert(target='MESH', keep_original=False)
        # joined_splint["bdental_type"] = "Guide Splint"

        # for obj in [pins_all, bool_model_duplicate] :
        #     bpy.data.objects.remove(obj, do_unlink=True)
        # os.system("cls")

        # message = ["Finished ."]
        # update_info(message)
        # sleep(2)
        # update_info()
        # return {"FINISHED"}

    # def modal(self, context, event):
    #     if not event.type in {"ESC", "RET"}:
    #         return {"PASS_THROUGH"}
    #     if event.type == "ESC":
    #         return {"CANCELLED"}

    #     if event.type == "RET" :
    #         if event.value == ("PRESS"):
    #             bool_model = context.object
    #             if not bool_model or bool_model.type != "MESH" :
    #                 message = ["Operation Cancelled !","Select Passive Blocked Model and try again"]
    #             # message = ["Guide Processing ..."]
    #             # update_info(message)
    #             # splint_checklist = [ obj for obj in context.scene.objects if obj.get("bdental_type") and "bdental_splint" in obj.get("bdental_type")]
    #             # sleeves_checklist = [ obj for obj in context.scene.objects if obj.get("bdental_type") and "bdental_sleeve" in obj.get("bdental_type") ]
    #             # pins_checklist = [ obj for obj in context.scene.objects if obj.get("bdental_type") and "bdental_pin" in obj.get("bdental_type") ]

    #             # print("splint_checklist",splint_checklist)
    #             # print("sleeves_checklist",sleeves_checklist)
    #             # print("pins_checklist",pins_checklist)
    #             # splint = splint_checklist[0]
    #             # splint.hide_set(False)
    #             # splint.hide_viewport = False
    #             # context.view_layer.objects.active = splint
    #             # bpy.ops.object.mode_set(mode="OBJECT")
    #             # bpy.ops.object.select_all(action="DESELECT")
    #             # splint.select_set(True)

    #             # for sleeve in sleeves_checklist :
    #             #     sleeve.hide_set(False)
    #             #     sleeve.hide_viewport = False
    #             #     sleeve.select_set(True)

    #             # bpy.ops.object.join()
    #             # self.joined_splint = context.object
    #             # remesh = self.joined_splint.modifiers.new(name="Remesh", type="REMESH")
    #             # bpy.ops.object.convert(target='MESH', keep_original=False)

    #             # bpy.ops.object.select_all(action="DESELECT")
    #             if len(pins_checklist) > 1 :
    #                 for pin in pins_checklist :
    #                     pin.hide_set(False)
    #                     pin.hide_viewport = False
    #                     pin.select_set(True)
    #                     context.view_layer.objects.active = pin

    #                 bpy.ops.object.join()
    #                 self.pins_all = context.object
    #                 self.pins_all.select_set(True)

    #             else :
    #                 pins_checklist[0].hide_set(False)
    #                 pins_checklist[0].hide_viewport = False
    #                 pins_checklist[0].select_set(True)
    #                 context.view_layer.objects.active = pins_checklist[0]
    #                 self.pins_all = context.object

    #             self.pins_all.display_type = 'WIRE'

    #             bool_modif = self.joined_splint.modifiers.new(name="bool", type="BOOLEAN")
    #             bool_modif.object = self.pins_all
    #             bool_modif.operation = "DIFFERENCE"

    #             message = ["Finished ."]
    #             update_info(message)
    #             sleep(2)
    #             update_info()
    #             return {"FINISHED"}

    #     return {"RUNNING_MODAL"}

    # def invoke(self, context, event):
    #     self.splint = context.object
    #     bool_model_pointer = int(self.splint["bool_model_pointer"])
    #     sleeves_pointers = eval(self.splint["sleeves_pointers"] )
    #     pins_pointers = eval(self.splint["pins_pointers"] )

    #     bool_model_checklist = [obj for obj in context.scene.objects if obj.as_pointer() == bool_model_pointer]
    #     sleeves_checklist = [obj for obj in context.scene.objects if obj.as_pointer() in sleeves_pointers]
    #     pins_checklist = [obj for obj in context.scene.objects if obj.as_pointer() in pins_pointers]
    #     message = []
    #     if not bool_model_checklist :
    #         message.extend(["Operation Cancelled !","Can't find Passive Blocked Model"])
    #     if not sleeves_checklist :
    #         message.extend([", Sleeves"])
    #     if not pins_checklist :
    #         message.extend([", Pins"])
    #     if message :
    #         update_info(message)
    #         sleep(3)
    #         update_info()
    #         return {"CANCELLED"}

    #     self.bool_model = bool_model_checklist[0]
    #     # modal operator
    #     context.window_manager.modal_handler_add(self)
    #     update_info(message)
    #     return {"RUNNING_MODAL"}


class BDENTAL_OT_AddGuideCuttersFromSleeves(bpy.types.Operator):
    """add Guide Cutters From Sleeves"""

    bl_idname = "wm.bdental_add_guide_cutters_from_sleeves"
    bl_label = "Guide Cutters From Sleeves"
    bl_options = {"REGISTER", "UNDO"}

    guide_cutters = []

    @classmethod
    def poll(cls, context):
        target = context.object and context.object.select_get(
        ) and context.object.type == "MESH"
        sleeves_checklist = [obj for obj in context.scene.objects if obj.get(
            "bdental_type") and "bdental_sleeve" in obj.get("bdental_type")]
        if not target or not sleeves_checklist:
            return False
        return True

    def modal(self, context, event):
        if not event.type in {"ESC", "RET"}:
            return {"PASS_THROUGH"}
        if event.type == "ESC":
            for sleeve in self.sleeves_checklist:
                sleeve.hide_set(False)
            for cutter in self.guide_cutters:
                bpy.data.objects.remove(cutter, do_unlink=True)
            return {"CANCELLED"}

        if event.type == "RET" and self.counter == 1:
            if event.value == ("PRESS"):
                message = ["Cutting Apply ..."]
                update_info(message)
                self.target.hide_set(False)
                self.target.hide_viewport = False
                context.view_layer.objects.active = self.target
                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.select_all(action="DESELECT")
                self.target.select_set(True)
                bpy.ops.object.convert(target='MESH', keep_original=False)
                bpy.data.objects.remove(self.joined_cutter, do_unlink=True)
                override, _, _ = CtxOverride(context)
                bpy.ops.wm.tool_set_by_id(override, name="builtin.select")

                for sleeve in self.sleeves_checklist:
                    sleeve.hide_set(False)

                message = ["Finished ."]
                update_info(message)
                sleep(2)
                update_info()
                return {"FINISHED"}

        elif event.type == "RET" and self.counter == 0:
            if event.value == ("PRESS"):
                self.counter += 1
                message = ["making Boolean ..."]
                update_info(message)
                self.target.hide_set(False)
                self.target.hide_viewport = False
                context.view_layer.objects.active = self.target
                bpy.ops.object.mode_set(mode="OBJECT")
                self.target.select_set(True)

                remesh = self.target.modifiers.new(
                    name="remesh", type="REMESH")
                remesh.voxel_size = 0.1
                bpy.ops.object.convert(target='MESH', keep_original=False)
                bpy.ops.object.select_all(action="DESELECT")

                for cutter in self.guide_cutters:
                    cutter.hide_set(False)
                    cutter.hide_viewport = False
                    cutter.select_set(True)
                    context.view_layer.objects.active = cutter

                bpy.ops.object.join()
                self.joined_cutter = context.object
                self.joined_cutter.display_type = 'WIRE'

                bool = self.target.modifiers.new(name="bool", type="BOOLEAN")
                bool.object = self.joined_cutter
                bool.operation = "DIFFERENCE"

                message = ["Please Control cuttings", "when done press ENTER"]
                update_info(message)
                return {"RUNNING_MODAL"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        if context.space_data.type == "VIEW_3D":
            self.counter = 0
            self.sleeves_checklist = [obj for obj in context.scene.objects if obj.get(
                "bdental_type") and "bdental_sleeve" in obj.get("bdental_type")]

            self.target = context.object
            self.target.hide_set(False)
            self.target.hide_viewport = False
            context.view_layer.objects.active = self.target
            bpy.ops.object.mode_set(mode="OBJECT")
            for sleeve in self.sleeves_checklist:
                sleeve.hide_set(False)
                sleeve.hide_viewport = False
                bpy.ops.object.select_all(action="DESELECT")
                sleeve.select_set(True)
                context.view_layer.objects.active = sleeve
                bpy.ops.object.duplicate_move()
                cutter = context.object
                cutter["bdental_type"] = "bdental_guide_cutter"
                cutter.display_type = 'WIRE'

                cutter.scale *= Vector((1.2, 1.2, 1))
                z_loc = cutter.dimensions.z/2
                # cutter.location += cutter.matrix_world.to_3x3() @ Vector((0,0,z_loc))
                mat = cutter.material_slots[0].material = bpy.data.materials.get(
                    "bdental_guide_cutter_material") or bpy.data.materials.new(name="bdental_guide_cutter_material")
                mat.diffuse_color = (0.8, 0, 0, 1)
                self.guide_cutters.append(context.object)
                # sleeve.hide_set(True)
            override, _, _ = CtxOverride(context)
            bpy.ops.wm.tool_set_by_id(override, name="builtin.transform")
            # modal operator
            context.window_manager.modal_handler_add(self)
            message = ["please set cutters scale and position",
                       "when done press ENTER"]
            update_info(message)
            return {"RUNNING_MODAL"}


class BDENTAL_OT_guide_3d_text(bpy.types.Operator):
    """add guide 3D text """

    bl_label = "3D Text"
    bl_idname = "wm.bdental_guide_3d_text"
    text_color = [0.0, 0.0, 1.0, 1.0]
    font_size = 4
    add: bpy.props.BoolProperty(default=True)

    def invoke(self, context, event):
        if not context.object or not context.object.select_get() or not context.object.type == "MESH":
            message = ["Please select the target mesh object"]
            update_info(message)
            sleep(2)
            update_info()
            return {"CANCELLED"}

        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def execute(self, context):
        update_info(
            ["Press <TAB> to edit text, <ESC> to cancel, <T> to confirm"])
        BDENTAL_Props = context.scene.BDENTAL_Props
        self.target = context.object

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.text_add(enter_editmode=False, align="CURSOR")
        self.text_ob = context.object

        self.text_ob["bdental_type"] = "bdental_text"
        self.text_ob.data.body = "Guide_"
        self.text_ob.name = "Guide_3d_text"
        if self.add:
            self.text_ob.name = "_ADD_"+self.text_ob.name
        guide_components_coll = add_collection("GUIDE Components")
        MoveToCollection(self.text_ob, guide_components_coll.name)

        self.text_ob.data.align_x = "CENTER"
        self.text_ob.data.align_y = "CENTER"
        self.text_ob.data.size = self.font_size
        override, _, _ = CtxOverride(context)
        self.text_ob.location = context.scene.cursor.location
        bpy.ops.view3d.view_axis(override, type="TOP", align_active=True)

        # change curve settings:
        self.text_ob.data.extrude = 0.5
        self.text_ob.data.bevel_depth = 0.1
        self.text_ob.data.bevel_resolution = 3

        # add SHRINKWRAP modifier :
        shrinkwrap_modif = self.text_ob.modifiers.new(
            "SHRINKWRAP", "SHRINKWRAP")
        shrinkwrap_modif.use_apply_on_spline = True
        shrinkwrap_modif.wrap_method = "PROJECT"
        shrinkwrap_modif.offset = 0
        shrinkwrap_modif.wrap_mode = "ABOVE_SURFACE"
        shrinkwrap_modif.cull_face = "OFF"
        shrinkwrap_modif.use_negative_direction = True
        shrinkwrap_modif.use_positive_direction = True
        shrinkwrap_modif.use_project_z = True
        shrinkwrap_modif.target = self.target

        mat = bpy.data.materials.get(
            "bdental_text_mat") or bpy.data.materials.new("bdental_text_mat")
        mat.diffuse_color = self.text_color
        mat.roughness = 0.6
        self.text_ob.active_material = mat

        bpy.context.scene.tool_settings.use_snap = True
        bpy.context.scene.tool_settings.snap_elements = {'FACE'}
        bpy.context.scene.tool_settings.use_snap_align_rotation = True
        bpy.context.scene.tool_settings.use_snap_rotate = True

        # bpy.ops.object.mode_set(override, mode="EDIT")

        # run modal
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if not event.type in {'ESC', 'T'}:
            return {'PASS_THROUGH'}
        if event.type in {'ESC'}:
            try:
                bpy.data.objects.remove(self.text_ob)
            except:
                pass
            bpy.context.scene.tool_settings.use_snap = False
            bpy.ops.wm.tool_set_by_id(name="builtin.select")
            update_info(["Cancelled"])
            sleep(1)
            update_info()
            return {'CANCELLED'}
        if event.type in {'T'}:
            if event.value == 'PRESS':
                if self.text_ob.mode == "EDIT":
                    return {'PASS_THROUGH'}

                self.text_to_mesh(context)

                bpy.context.scene.tool_settings.use_snap = False
                bpy.ops.wm.tool_set_by_id(name="builtin.select")
                update_info(["Finished"])
                sleep(1)
                update_info()
                return {'FINISHED'}
        return {'RUNNING_MODAL'}

    def text_to_mesh(self, context):
        update_info(["Text Remesh..."])
        context.view_layer.objects.active = self.text_ob
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.text_ob.select_set(True)
        bpy.ops.object.convert(target="MESH")
        remesh_modif = self.text_ob.modifiers.new("REMESH", "REMESH")
        remesh_modif.voxel_size = 0.05
        bpy.ops.object.convert(target="MESH")


class BDENTAL_OT_GuideAddComponent(bpy.types.Operator):
    """set Guide cutters"""
    bl_idname = "wm.bdental_guide_add_component"
    bl_label = "Add Guide Component"
    bl_options = {'REGISTER', 'UNDO'}

    sleeve_diameter: FloatProperty(name="Sleeve Diameter", default=4.0, min=0.0,
                                   max=20.0, step=1, precision=3, unit='LENGTH', description="Sleeve Diameter")
    sleeve_height: FloatProperty(name="Sleeve Height", default=10.0, min=0.0,
                                 max=20.0, step=1, precision=3, unit='LENGTH', description="Sleeve Height")
    pin_diameter: FloatProperty(name="Pin Diameter", default=2.0, min=0.0,
                                max=20.0, step=1, precision=3, unit='LENGTH', description="Pin Diameter")
    offset: FloatProperty(name="Offset", default=0.1, min=0.0, max=1.0,
                          step=1, precision=3, unit='LENGTH', description="Offset")

    guide_component: EnumProperty(
        name="Guide Component",
        description="Guide Component",
        items=set_enum_items(
            ["Cube", "Sphere", "Cylinder", "Fixing Sleeve/Pin", "3D Text"]),
        default="Cube",
    )
    component_type: EnumProperty(
        name="Component Type",
        description="Component Type",
        items=set_enum_items(["ADD", "CUT"]),
        default="ADD",
    )
    component_size: FloatProperty(
        description="Component Size", default=2, step=1, precision=2
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "guide_component")
        if self.guide_component == "3D Text":
            layout.prop(self, "component_type")
        if self.guide_component in ["Cube", "Sphere", "Cylinder"]:
            layout.prop(self, "component_type")
            layout.prop(self, "component_size")
        elif self.guide_component == "Fixing Sleeve/Pin":
            layout.prop(self, "sleeve_diameter")
            layout.prop(self, "sleeve_height")
            layout.prop(self, "pin_diameter")
            layout.prop(self, "offset")

    # @classmethod
    # def poll(cls, context):
    #     if not context.object :
    #         return False
    #     if not context.selected_objects :
    #         return False
    #     return ([obj.type == 'MESH' for obj in context.selected_objects] and
    #             context.object.mode == 'OBJECT' and
    #             context.object.select_get()
    #     )

    def modal(self, context, event):
        if not event.type in {'ESC', 'RET'}:
            return {'PASS_THROUGH'}

        elif event.type in {'ESC'}:
            if event.value == 'PRESS':
                override, _, _ = CtxOverride(context)
                bpy.ops.wm.tool_set_by_id(override, name="builtin.select")
                update_info(["Cancelled"])
                sleep(1)
                update_info()
                return {'CANCELLED'}
        elif event.type in {'RET'}:
            if event.value == 'RELEASE':
                bpy.ops.object.select_all(action="DESELECT")
                if self.guide_component == "Cube":
                    bpy.ops.mesh.primitive_cube_add(
                        size=self.component_size, align='CURSOR')
                    cube = context.object
                    n = len(
                        [o for o in bpy.data.objects if "Cube_Component" in o.name])
                    cube.name = f"{self.preffix}Cube_Component({n+1})"
                    bevel = context.object.modifiers.new(
                        name="bevel", type="BEVEL")
                    bevel.width = 0.3
                    bevel.segments = 3
                    bpy.ops.object.convert(target='MESH', keep_original=False)

                    MoveToCollection(cube, "GUIDE Components")
                    cube.active_material = self.mat_add if self.component_type == "ADD" else self.mat_cut
                    override, _, _ = CtxOverride(context)
                    bpy.ops.wm.tool_set_by_id(
                        override, name="builtin.transform")
                    update_info()

                elif self.guide_component == "Sphere":
                    bpy.ops.mesh.primitive_uv_sphere_add(
                        radius=self.component_size, align='CURSOR')
                    sphere = context.object
                    n = len(
                        [o for o in bpy.data.objects if "Sphere_Component" in o.name])
                    sphere.name = f"{self.preffix}Sphere_Component({n+1})"

                    subsurf = context.object.modifiers.new(
                        name="subsurf", type="SUBSURF")
                    subsurf.levels = 1
                    bpy.ops.object.convert(target='MESH', keep_original=False)

                    MoveToCollection(sphere, "GUIDE Components")
                    sphere.active_material = self.mat_add if self.component_type == "ADD" else self.mat_cut
                    override, _, _ = CtxOverride(context)
                    bpy.ops.wm.tool_set_by_id(
                        override, name="builtin.transform")
                    update_info()

                elif self.guide_component == "Cylinder":
                    bpy.ops.mesh.primitive_cylinder_add(
                        radius=self.component_size/2, depth=self.component_size*2, align='CURSOR')
                    cylinder = context.object
                    n = len(
                        [o for o in bpy.data.objects if "Cylinder_Component" in o.name])
                    cylinder.name = f"{self.preffix}Cylinder_Component({n+1})"

                    bevel = context.object.modifiers.new(
                        name="bevel", type="BEVEL")
                    bevel.width = 0.3
                    bevel.segments = 3
                    bpy.ops.object.convert(target='MESH', keep_original=False)

                    MoveToCollection(cylinder, "GUIDE Components")
                    cylinder.active_material = self.mat_add if self.component_type == "ADD" else self.mat_cut
                    override, _, _ = CtxOverride(context)
                    bpy.ops.wm.tool_set_by_id(
                        override, name="builtin.transform")
                    update_info()

                elif self.guide_component == "Fixing Sleeve/Pin":
                    n = len(
                        [o for o in bpy.data.objects if o.name.startswith("Fixing_Pin")])

                    bpy.ops.object.select_all(action="DESELECT")
                    pin = AppendObject("pin", coll_name="GUIDE Components")

                    bpy.ops.object.select_all(action="DESELECT")
                    pin.select_set(True)
                    context.view_layer.objects.active = pin

                    pin.active_material = self.mat_cut
                    pin.matrix_world[:3] = context.scene.cursor.matrix[:3]
                    pin.dimensions = [self.pin_diameter,
                                      self.pin_diameter, pin.dimensions[2]]
                    # pin.location = context.scene.cursor.location
                    # pin.rotation_euler = context.scene.cursor.rotation_euler
                    pin.name = f"Fixing_Pin({n+1})"

                    # bpy.ops.wm.bdental_lock_objects()

                    bpy.ops.object.select_all(action="DESELECT")
                    sleeve = AppendObject(
                        "sleeve", coll_name="GUIDE Components")

                    bpy.ops.object.select_all(action="DESELECT")
                    sleeve.select_set(True)
                    context.view_layer.objects.active = sleeve

                    sleeve.active_material = self.mat_add
                    sleeve.matrix_world[:3] = context.scene.cursor.matrix[:3]
                    sleeve.dimensions = [self.sleeve_diameter,
                                         self.sleeve_diameter, self.sleeve_height]

                    child_of = pin.constraints.new(type="CHILD_OF")
                    child_of.target = sleeve
                    child_of.use_scale_x = False
                    child_of.use_scale_y = False
                    child_of.use_scale_z = False

                    sleeve.name = f"_ADD_Fixing_Sleeve({n+1})"

                    override, _, _ = CtxOverride(context)
                    bpy.ops.wm.tool_set_by_id(
                        override, name="builtin.transform")
                    update_info()

                elif self.guide_component == "3D Text":
                    if self.component_type == "CUT":
                        bpy.ops.wm.bdental_guide_3d_text(
                            "EXEC_DEFAULT", add=False)
                    else:
                        bpy.ops.wm.bdental_guide_3d_text(
                            "EXEC_DEFAULT", add=True)

                return {'FINISHED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        if context.space_data.type == "VIEW_3D":
            wm = context.window_manager
            return wm.invoke_props_dialog(self, width=500)
        else:
            self.report({"WARNING"}, "Active space must be a View3d")
            return {"CANCELLED"}

    def execute(self, context):
        self.mat_add = bpy.data.materials.get(
            "mat_component_add") or bpy.data.materials.new(name="mat_component_add")
        self.mat_add.diffuse_color = [0.0, 1.0, 0.0, 1.0]
        self.mat_cut = bpy.data.materials.get(
            "mat_component_cut") or bpy.data.materials.new(name="mat_component_cut")
        self.mat_cut.diffuse_color = [1.0, 0.0, 0.0, 1.0]
        if self.component_type == "ADD":
            self.preffix = "_ADD_"

        elif self.component_type == "CUT":
            self.preffix = ""

        elif self.guide_component == "Fixing Sleeve/Pin":
            self.preffix = ""

        override, _, _ = CtxOverride(context)
        bpy.ops.wm.tool_set_by_id(override, name="builtin.cursor")
        message = [
            "Please left click to set the component position, and prees ENTER when done"]
        update_info(message)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}


class BDENTAL_OT_AddSplint(bpy.types.Operator):
    """Add Splint"""

    bl_idname = "wm.bdental_add_splint"
    bl_label = "Splint"
    bl_options = {"REGISTER", "UNDO"}

    thikness: FloatProperty(
        description="SPLINT thikness", default=2, step=1, precision=2
    )

    def execute(self, context):

        Splint = Metaball_Splint(self.BaseMesh, self.thikness)

        return {"FINISHED"}

    def invoke(self, context, event):

        if not context.object:
            message = ["Please select a base mesh!"]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        if not context.object.select_get() or context.object.type != "MESH":
            message = ["Please select a base mesh!"]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        else:
            self.BaseMesh = context.object
            wm = context.window_manager
            return wm.invoke_props_dialog(self)


class BDENTAL_OT_Survey(bpy.types.Operator):
    "Survey the model from view top"

    bl_idname = "wm.bdental_survey"
    bl_label = "Survey Model"

    SurveyColor: FloatVectorProperty(
        name="Survey Color",
        description="Survey Color",
        default=[0.2, 0.12, 0.17, 1.0],
        soft_min=0.0,
        soft_max=1.0,
        size=4,
        subtype="COLOR",
    )

    def execute(self, context):
        BDENTAL_Props = bpy.context.scene.BDENTAL_Props
        bpy.ops.object.mode_set(mode="OBJECT")
        Old_Survey_mat = bpy.data.materials.get("BDENTAL_survey_mat")
        if Old_Survey_mat:
            OldmatSlotsIds = [
                i
                for i in range(len(self.Model.material_slots))
                if self.Model.material_slots[i].material == Old_Survey_mat
            ]
            if OldmatSlotsIds:
                for idx in OldmatSlotsIds:
                    self.Model.active_material_index = idx
                    bpy.ops.object.material_slot_remove()

        Override, area3D, space3D = CtxOverride(context)
        view_mtx = space3D.region_3d.view_matrix.copy()
        if not self.Model.data.materials[:]:
            ModelMat = bpy.data.materials.get(
                "BDENTAL_Neutral_mat"
            ) or bpy.data.materials.new("BDENTAL_Neutral_mat")
            ModelMat.diffuse_color = (0.8, 0.8, 0.8, 1.0)
            self.Model.active_material = ModelMat

        Survey_mat = bpy.data.materials.get(
            "BDENTAL_survey_mat"
        ) or bpy.data.materials.new("BDENTAL_survey_mat")
        Survey_mat.diffuse_color = self.SurveyColor
        self.Model.data.materials.append(Survey_mat)
        self.Model.active_material_index = len(self.Model.material_slots) - 1

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")

        # #############################____Surveying____###############################
        survey_faces_index_list = []

        obj = self.Model
        View_Local_Z = obj.matrix_world.inverted().to_quaternion() @ (
            space3D.region_3d.view_rotation @ Vector((0, 0, 1))
        )

        survey_faces_Idx = [
            f.index for f in obj.data.polygons if f.normal.dot(View_Local_Z) < -0.000001
        ]

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.mesh.select_all(action="DESELECT")

        bpy.ops.object.mode_set(mode="OBJECT")

        for i in survey_faces_Idx:
            f = obj.data.polygons[i]
            f.select = True

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        Survey_Vg = obj.vertex_groups.get("BDENTAL_survey_vg") or obj.vertex_groups.new(
            name="BDENTAL_survey_vg"
        )
        # obj.vertex_groups.active_index = Survey_Vg.index
        bpy.ops.object.vertex_group_assign()
        bpy.ops.object.material_slot_assign()
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)

        # Store Survey direction :
        SurveyInfo_Dict = eval(BDENTAL_Props.SurveyInfo)
        SurveyInfo_Dict[obj.as_pointer()] = (View_Local_Z, Survey_mat)
        BDENTAL_Props.SurveyInfo = str(SurveyInfo_Dict)

        return {"FINISHED"}

    def invoke(self, context, event):

        if not context.active_object:
            message = ["Please select Model to survey!"]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        if (
            not context.active_object.select_get()
            or context.active_object.type != "MESH"
        ):
            message = ["Please select Model to survey!"]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        else:
            self.Model = context.active_object
            wm = context.window_manager
            return wm.invoke_props_dialog(self)


class BDENTAL_OT_ModelBase(bpy.types.Operator):
    """Make a model base from top user view prspective"""

    bl_idname = "wm.bdental_model_base"
    bl_label = "Model Base"
    bl_options = {"REGISTER", "UNDO"}

    ModelType: EnumProperty(
        items=set_enum_items(["Upper Model", "Lower Model"]), description="Model Type", default="Upper Model"
    )
    BaseHeight: FloatProperty(
        description="Base Height ", default=10, step=1, precision=2
    )
    HollowModel: BoolProperty(
        name="Make Hollow Model",
        description="Add Hollow Model",
        default=False,
    )

    @classmethod
    def poll(cls, context):
        return context.object and context.object.select_get() and context.object.type == "MESH"

    def execute(self, context):
        
        # Check base boarder :
        TargetMesh = context.object
        NonManifoldVerts = count_non_manifold_verts(TargetMesh) #mode = OBJECT


        # bpy.ops.object.mode_set(mode="EDIT")
        # bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        # bpy.ops.mesh.select_all(action="DESELECT")
        # bpy.ops.mesh.select_non_manifold()
        # bpy.ops.object.mode_set(mode="OBJECT")
        # NonManifoldVerts = [
        #     v for v in TargetMesh.data.vertices if v.select]

        if not NonManifoldVerts:
            txt = [
                "Operation cancelled !",
                "Can't make model base from Closed mesh.",
            ]
            update_info(message=txt, rect_color=[1,0,0,0.7])
            sleep(3)
            update_info()

            return {"CANCELLED"}

        else:
            txt = ["Processing ..."]
            update_info(message=txt)

            BaseHeight = self.BaseHeight

            ####### Duplicate Target Mesh #######
            bpy.ops.object.select_all(action="DESELECT")
            TargetMesh.select_set(True)
            bpy.context.view_layer.objects.active = TargetMesh
            bpy.ops.object.duplicate_move()

            ModelBase = context.object
            ModelBase.name = f"(BASE MODEL){TargetMesh.name}"
            ModelBase.data.name = ModelBase.name
            obj = ModelBase
            # Relax border loop :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.mesh.select_non_manifold()
            bpy.ops.mesh.looptools_relax(
                input="selected",
                interpolation="cubic",
                iterations="3",
                regular=True,
            )
            bpy.ops.mesh.remove_doubles(threshold=0.1)
            

            # Make some calcul of average z_cordinate of border vertices :

            bpy.ops.object.mode_set(mode="OBJECT")

            obj_mx = obj.matrix_world.copy()
            verts = obj.data.vertices
            global_z_cords = [(obj_mx @ v.co)[2] for v in verts]

            HollowOffset = 0
            if self.ModelType == "Upper Model":
                Extrem_z = max(global_z_cords)
                Delta = BaseHeight
                if self.HollowModel:
                    HollowOffset = 4
                    BisectPlaneLoc = Vector((0, 0, Extrem_z+Delta))
                    BisectPlaneNormal = Vector((0, 0, 1))

            if self.ModelType == "Lower Model":
                Extrem_z = min(global_z_cords)
                Delta = -BaseHeight
                if self.HollowModel:
                    HollowOffset = -4
                    BisectPlaneLoc = Vector((0, 0, Extrem_z+Delta))
                    BisectPlaneNormal = Vector((0, 0, -1))

            # Border_2 = Extrude 1st border loop no translation :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.extrude_region_move()

            # change Border2 vertices zco to min_z - base_height  :

            bpy.ops.object.mode_set(mode="OBJECT")
            selected_verts = [v for v in verts if v.select]

            for v in selected_verts:
                global_v_co = obj_mx @ v.co
                v.co = obj_mx.inverted() @ Vector(
                    (
                        global_v_co[0],
                        global_v_co[1],
                        Extrem_z + Delta + HollowOffset,
                    )
                )

            # fill base :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.edge_face_add()
            bpy.ops.mesh.dissolve_limited()

            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.fill_holes(sides=100)

            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.object.mode_set(mode="OBJECT")

            for slot in obj.material_slots:
                bpy.ops.object.material_slot_remove()
                
            mat = bpy.data.materials.get(
                "blue stone mat") or bpy.data.materials.new("blue stone mat")
            mat.diffuse_color = [0.411693, 0.600872, 0.8, 1.0]
            obj.active_material = mat

            NonManifoldVerts = count_non_manifold_verts(obj)
            if NonManifoldVerts :
                txt = ["Base model have bad geometry remeshing ..."]
                update_info(message=txt, rect_color=[0.8,0.2,0,0.7])
                remesh = obj.modifiers.new("Remesh", "REMESH")
                remesh.mode = "SHARP"
                remesh.octree_depth = 8
                bpy.ops.object.convert(target="MESH")
                bpy.ops.wm.bdental_voxelremesh(VoxelSize=0.1)

            
            if self.HollowModel:
                txt = ["Processing Hollowed Model ..."]
                update_info(message=txt, rect_color=[0.8,0.2,0,0.7])
                bpy.ops.wm.bdental_hollow_model(thikness=2)
                HollowModel = context.object
                for slot in HollowModel.material_slots:
                    bpy.ops.object.material_slot_remove()
                
                mat = bpy.data.materials.get(
                    "yellow stone mat") or bpy.data.materials.new("yellow stone mat")
                mat.diffuse_color = [0.8, 0.652387, 0.435523, 1.0]
                HollowModel.active_material = mat

                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action="SELECT")

                bpy.ops.mesh.bisect(
                    plane_co=BisectPlaneLoc,
                    plane_no=BisectPlaneNormal,
                    use_fill=True,
                    clear_inner=False,
                    clear_outer=True,
                )
                bpy.ops.mesh.select_all(action="DESELECT")
                bpy.ops.object.mode_set(mode="OBJECT")

                bpy.ops.object.select_all(action="DESELECT")
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.bisect(
                    plane_co=BisectPlaneLoc,
                    plane_no=BisectPlaneNormal,
                    use_fill=True,
                    clear_inner=False,
                    clear_outer=True,
                )
                bpy.ops.mesh.select_all(action="DESELECT")
                bpy.ops.object.mode_set(mode="OBJECT")

            txt = ["Base model created successfully"]
            if self.HollowModel :
                txt = ["Base and hollowed models created successfully"]
            
            update_info(message=txt, rect_color=[0,1,0,0.7])
            sleep(3)
            update_info()
            return {"FINISHED"}

    def invoke(self, context, event):

        wm = context.window_manager
        return wm.invoke_props_dialog(self)


class BDENTAL_OT_hollow_model(bpy.types.Operator):
    """Create a hollow Dental Model from closed Model"""

    bl_idname = "wm.bdental_hollow_model"
    bl_label = "Hollow Model"
    bl_options = {"REGISTER", "UNDO"}

    thikness: FloatProperty(description="OFFSET",
                            default=2, step=1, precision=2)
    display_info : BoolProperty(
        name="display info",
        description="display info footer",
        default=False,
    )
    @classmethod
    def poll(cls, context):
        return context.object and context.object.select_get() and context.object.type == "MESH"

    def execute(self, context):
        Model = context.object
        # NonManifoldVerts = count_non_manifold_verts(Model) #mode = OBJECT

        # if NonManifoldVerts:
        #     txt = [
        #         "Operation cancelled !",
        #         "the mesh is not tight.",
        #     ]
        #     print(txt)
        #     if self.display_info :
        #         update_info(message=txt, rect_color=[1,0,0,0.7])
        #         sleep(3)
        #         update_info()

        #     return {"CANCELLED"}

        # Prepare scene settings :
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.object.mode_set(mode="OBJECT")

        # Duplicate Model to Model_hollow:

        bpy.ops.object.select_all(action="DESELECT")
        Model.select_set(True)
        bpy.context.view_layer.objects.active = Model
        bpy.ops.object.duplicate_move()

        # Rename Model_hollow....

        Model_hollow = context.object
        Model_hollow.name = f"(Model hollowed){Model.name}"

        # Duplicate Model_hollow and make a low resolution duplicate :

        bpy.ops.object.duplicate_move()
        Model_lowres = context.object

        
        bpy.ops.wm.bdental_voxelremesh(VoxelSize=0.5)

        # Add Metaballs :

        obj = Model_lowres

        loc, rot, scale = obj.matrix_world.decompose()

        verts = obj.data.vertices
        vcords = [rot @ v.co + loc for v in verts]
        mball_elements_cords = [vco - vcords[0] for vco in vcords[1:]]

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")

        thikness = self.thikness
        radius = thikness * 5 / 8

        bpy.ops.object.metaball_add(
            type="BALL", radius=radius, enter_editmode=False, location=vcords[0]
        )

        Mball_object = context.object
        mball = Mball_object.data
        mball.resolution = 0.6
        context.object.data.update_method = "FAST"

        for i in range(len(mball_elements_cords)):
            element = mball.elements.new()
            element.co = mball_elements_cords[i]
            element.radius = radius * 2

        bpy.ops.object.convert(target="MESH")
        Mball_object = context.object
        
        # Make boolean intersect operation :
        bpy.ops.object.select_all(action="DESELECT")
        Model_hollow.select_set(True)
        bpy.context.view_layer.objects.active = Model_hollow
        bool_modif = Model_hollow.modifiers.new("bool", "BOOLEAN")
        bool_modif.object = Mball_object
        bool_modif.operation = "INTERSECT"
        bpy.ops.object.convert(target="MESH")

        # Delet Model_lowres and Mball_object:
        bpy.data.objects.remove(Model_lowres)
        bpy.data.objects.remove(Mball_object)

        return {"FINISHED"}

    def invoke(self, context, event):

        Active_Obj = context.active_object

        if not Active_Obj:
            message = [" Please select Target mesh Object ! "]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )
            return {"CANCELLED"}

        if Active_Obj.select_get() == False or Active_Obj.type != "MESH":
            message = [" Please select Target mesh Object ! "]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )
            return {"CANCELLED"}

        else:
            wm = context.window_manager
            return wm.invoke_props_dialog(self)


class BDENTAL_OT_BlockModel(bpy.types.Operator):
    "Blockout Model (Remove Undercuts)"

    bl_idname = "wm.bdental_block_model"
    bl_label = "BLOCK Model"

    printing_offset: FloatProperty(name="Printing Offset", default=0.1, min=0.0,
                                   max=1.0, step=1, precision=3, unit='LENGTH', description="Implant Diameter")

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == "MESH"

    def invoke(self, context, event):
        surveyed_model = context.object
        Pointer = surveyed_model.as_pointer()
        BDENTAL_Props = bpy.context.scene.BDENTAL_Props
        SurveyInfo_Dict = eval(BDENTAL_Props.SurveyInfo)
        if not Pointer in SurveyInfo_Dict.keys():
            message = ["Please Survey Model before Blockout !"]
            update_info(message)
            sleep(2)
            update_info()
            return {"CANCELLED"}
        else:
            wm = context.window_manager
            return wm.invoke_props_dialog(self)

    def execute(self, context):
        surveyed_model = context.object
        surveyed_model_pointer = surveyed_model.as_pointer()
        BDENTAL_Props = bpy.context.scene.BDENTAL_Props
        SurveyInfo_Dict = eval(BDENTAL_Props.SurveyInfo)

        View_Local_Z, Survey_mat = SurveyInfo_Dict[surveyed_model_pointer]
        ExtrudeVector = -20 * (
            surveyed_model.matrix_world.to_quaternion() @ View_Local_Z
        )

        # print(ExtrudeVector)

        # duplicate Model :
        bpy.ops.object.select_all(action="DESELECT")
        surveyed_model.select_set(True)
        bpy.context.view_layer.objects.active = surveyed_model

        bpy.ops.object.duplicate_move()
        blocked_model = bpy.context.view_layer.objects.active
        blocked_model.name = f"(BLOCKED)_{blocked_model.name}"

        for _ in blocked_model.material_slots:
            bpy.ops.object.material_slot_remove()

        blocked_model.active_material = Survey_mat
        bpy.ops.object.mode_set(mode="EDIT")

        bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.mesh.extrude_region_move()
        bpy.ops.transform.translate(value=ExtrudeVector)

        bpy.ops.object.mode_set(mode="OBJECT")
        blocked_model.data.remesh_mode = "VOXEL"
        blocked_model.data.remesh_voxel_size = 0.2
        blocked_model.data.use_remesh_fix_poles = True
        blocked_model.data.use_remesh_preserve_volume = True

        bpy.ops.object.voxel_remesh()

        return {"FINISHED"}


class BDENTAL_OT_UnderctsPreview(bpy.types.Operator):
    " Survey the model from view"

    bl_idname = "wm.bdental_undercuts_preview"
    bl_label = "Preview Undercuts"

    undercuts_color = [0.54, 0.13, 0.5, 1.0]

    @classmethod
    def poll(cls, context):
        if not context.object:
            return False
        return context.object.type == "MESH" and context.object.mode == "OBJECT" and context.object.select_get()

    def execute(self, context):
        obj = context.object
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        if not obj.material_slots:
            undercuts_mat_white = bpy.data.materials.get(
                "undercuts_preview_mat_white") or bpy.data.materials.new("undercuts_preview_mat_white")
            undercuts_mat_white.diffuse_color = (0.8, 0.8, 0.8, 1.0)
            obj.active_material = undercuts_mat_white

        for i, slot in enumerate(obj.material_slots):
            if slot.material.name == "undercuts_preview_mat_color":
                obj.active_material_index = i
                bpy.ops.object.material_slot_remove()

        undercuts_mat_color = bpy.data.materials.get(
            "undercuts_preview_mat_color") or bpy.data.materials.new("undercuts_preview_mat_color")
        undercuts_mat_color.diffuse_color = self.undercuts_color
        obj.data.materials.append(undercuts_mat_color)
        obj.active_material_index = len(obj.material_slots) - 1

        # bpy.ops.object.mode_set(mode="EDIT")
        # bpy.ops.mesh.select_all(action="SELECT")
        # bpy.ops.mesh.normals_make_consistent(inside=False)
        # bpy.ops.mesh.select_all(action="DESELECT")
        # bpy.ops.object.mode_set(mode="OBJECT")

        # #############################____Surveying____###############################
        Override, area3D, space3D = CtxOverride(context)
        view_rotation_mtx = space3D.region_3d.view_rotation.to_matrix().to_4x4()

        view_z_local = obj.matrix_world.inverted().to_quaternion(
        ) @ (space3D.region_3d.view_rotation @ Vector((0, 0, 1)))

        survey_faces_index_list = [
            f.index for f in obj.data.polygons if f.normal.dot(view_z_local) < -0.000001
        ]

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.mesh.select_all(action="DESELECT")

        bpy.ops.object.mode_set(mode="OBJECT")

        for i in survey_faces_index_list:
            obj.data.polygons[i].select = True

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        undercuts_vg = obj.vertex_groups.get(
            "undercuts_vg"
        ) or obj.vertex_groups.new(name="undercuts_vg")
        obj.vertex_groups.active_index = undercuts_vg.index
        bpy.ops.object.vertex_group_assign()
        bpy.ops.object.material_slot_assign()
        bpy.ops.object.mode_set(mode="OBJECT")

        obj["undercuts_vector"] = view_z_local
        obj["undercuts_view_rotation_mtx"] = list(view_rotation_mtx)

        return {"FINISHED"}


class BDENTAL_OT_BlockoutNew(bpy.types.Operator):
    " Create a blockout model for undercuts"

    bl_idname = "wm.bdental_blockout_new"
    bl_label = "Blockout Model"

    undercuts_color = [0.54, 0.13, 0.5, 1.0]
    printing_offset: FloatProperty(name="Offset", default=0.1, min=0.0, max=1.0,
                                   step=1, precision=3, unit='LENGTH', description="Implant Diameter")

    @classmethod
    def poll(cls, context):
        if not context.object:
            return False
        return context.object.get("undercuts_vector")

    def make_blocked(self, context, obj):
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)

        # check non manifold
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_non_manifold()
        bpy.ops.object.mode_set(mode="OBJECT")

        non_manifold = [v for v in obj.data.vertices if v.select]

        if non_manifold:
            message = ["Adding mesh base ..."]
            update_info(message)
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.extrude_region_move()
            bpy.ops.object.mode_set(mode="OBJECT")
            extruded_verts = [v for v in obj.data.vertices if v.select]

            min_z, min_z_id = sorted(
                [(self.undercuts_vector.dot(v.co), v.index) for v in obj.data.vertices])[0]
            min_co_local = obj.data.vertices[min_z_id].co

            for v in extruded_verts:
                offset_z = self.undercuts_vector.normalized()@(min_co_local-v.co)
                v.co = v.co + ((offset_z-5) *
                               self.undercuts_vector.normalized())

            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.remove_doubles(threshold=0.1)
            bpy.ops.mesh.looptools_relax(
                input="selected",
                interpolation="cubic",
                iterations="5",
                regular=True,
            )

            bpy.ops.mesh.edge_face_add()
            bpy.ops.mesh.dissolve_limited()
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.fill_holes(sides=100)

            # check non manifold
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.mesh.select_non_manifold()
            bpy.ops.object.mode_set(mode="OBJECT")

            non_manifold = [v for v in obj.data.vertices if v.select]
            if non_manifold:
                message = ["Remeshing mesh base ..."]
                update_info(message)
                remesh = obj.modifiers.new("Remesh", "REMESH")
                remesh.octree_depth = 8
                remesh.mode = "SHARP"
                bpy.ops.object.convert(target="MESH")

        survey_faces_index_list = [
            f.index for f in obj.data.polygons if f.normal.dot(self.undercuts_vector) < -0.000001
        ]

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.object.mode_set(mode="OBJECT")

        for i in survey_faces_index_list:
            obj.data.polygons[i].select = True

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.extrude_region_move()
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.object.mode_set(mode="OBJECT")

        selected_verts = [v for v in obj.data.vertices if v.select]
        min_z, min_z_id = sorted(
            [(self.undercuts_vector.dot(v.co), v.index) for v in obj.data.vertices])[0]
        min_co_local = obj.data.vertices[min_z_id].co
        for v in selected_verts:
            offset_z = self.undercuts_vector.normalized()@(min_co_local-v.co)
            v.co = v.co + (offset_z * self.undercuts_vector.normalized())

        message = ["Remeshing Blocked model ..."]
        update_info(message)

        modif_remesh = self.blocked.modifiers.new("remesh", type="REMESH")
        modif_remesh.voxel_size = 0.2
        bpy.ops.object.convert(target="MESH")

    def invoke(self, context, event):
        self.target = context.object
        if not self.target.get("undercuts_vector") or not self.target.get("undercuts_view_rotation_mtx"):
            message = ["Operation Cancelled",
                       "Please survey the model from view first"]
            update_info(message)
            sleep(2)
            update_info()
            return {"CANCELLED"}
        self.undercuts_vector = Vector(self.target["undercuts_vector"])
        self.view_rotation_mtx = Matrix(
            self.target["undercuts_view_rotation_mtx"])
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action='DESELECT')
        self.target.select_set(True)
        wm = context.window_manager
        return wm.invoke_props_dialog(self, width=500)

    def execute(self, context):

        message = ["Making duplicate tight mesh..."]
        update_info(message)

        offset = self.printing_offset
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action='DESELECT')
        self.target.select_set(True)
        bpy.context.view_layer.objects.active = self.target
        bpy.ops.object.duplicate_move()

        self.blocked = bpy.context.object
        # context.view_layer.objects.active = self.blocked
        for _ in self.blocked.material_slots:
            bpy.ops.object.material_slot_remove()

        mat = bpy.data.materials.get("undercuts_preview_mat_color") or bpy.data.materials.new(
            "undercuts_preview_mat_color")
        mat.diffuse_color = self.undercuts_color
        self.blocked.active_material = mat

        # #check for non manifold
        # bpy.ops.object.mode_set(mode="EDIT")
        # bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        # bpy.ops.mesh.select_all(action="DESELECT")
        # bpy.ops.mesh.select_non_manifold()
        # bpy.ops.object.mode_set(mode="OBJECT")
        # non_manifold_verts = [v for v in self.blocked.data.vertices if v.select]

        # if non_manifold_verts :
        #     modif_remesh = self.blocked.modifiers.new("remesh", type="REMESH")
        #     modif_remesh.octree_depth = 9
        #     modif_remesh.mode = "SHARP"

        if offset:

            displace = self.blocked.modifiers.new("displace", type="DISPLACE")
            displace.mid_level = 0
            displace.strength = offset
            message = ["displacement applied = " + str(round(offset, 2))]
            update_info(message)

        bpy.ops.object.convert(target="MESH")

        message = ["Making Blocked Model..."]
        update_info(message)

        self.make_blocked(context, self.blocked)
        context.view_layer.objects.active = self.blocked

        self.blocked.name = f"Blocked_{self.target.name}"
        guide_components_coll = add_collection("GUIDE Components")
        # MoveToCollection(self.blocked,guide_components_coll.name)
        for col in self.blocked.users_collection:
            col.objects.unlink(self.blocked)
        guide_components_coll.objects.link(self.blocked)

        message = ["Finished."]
        update_info(message)
        sleep(1)
        update_info()

        return {"FINISHED"}


class BDENTAL_OT_add_offset(bpy.types.Operator):
    """Add offset to mesh"""

    bl_idname = "wm.bdental_add_offset"
    bl_label = "Add Offset"
    bl_options = {"REGISTER", "UNDO"}

    Offset: FloatProperty(description="OFFSET",
                          default=0.1, step=1, precision=2)

    def execute(self, context):

        offset = round(self.Offset, 2)

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.modifier_add(type="DISPLACE")
        bpy.context.object.modifiers["Displace"].mid_level = 0
        bpy.context.object.modifiers["Displace"].strength = offset
        bpy.ops.object.modifier_apply(modifier="Displace")

        return {"FINISHED"}

    def invoke(self, context, event):

        Active_Obj = context.active_object

        if not Active_Obj:
            message = [" Please select Target mesh Object ! "]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )
            return {"CANCELLED"}

        if Active_Obj.select_get() == False or Active_Obj.type != "MESH":
            message = [" Please select Target mesh Object ! "]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )
            return {"CANCELLED"}

        else:
            wm = context.window_manager
            return wm.invoke_props_dialog(self)


#######################################################################
# Align Operators
#######################################################################

############################################################################
class BDENTAL_OT_AlignPoints(bpy.types.Operator):
    """Add Align Refference points"""

    bl_idname = "wm.bdental_alignpoints"
    bl_label = "ALIGN POINTS"
    bl_options = {"REGISTER", "UNDO"}

    TargetColor = (0, 1, 0, 1)  # Green
    SourceColor = (1, 0, 0, 1)  # Red
    CollName = "ALIGN POINTS"
    TargetChar = "B"
    SourceChar = "A"

    def IcpPipline(
        self,
        SourceObj,
        TargetObj,
        SourceVidList,
        TargetVidList,
        VertsLimite,
        Iterations,
        Precision,
    ):

        MaxDist = 0.0
        for i in range(Iterations):

            SourceVcoList = [
                SourceObj.matrix_world @ SourceObj.data.vertices[idx].co
                for idx in SourceVidList
            ]
            TargetVcoList = [
                TargetObj.matrix_world @ TargetObj.data.vertices[idx].co
                for idx in TargetVidList
            ]

            (
                SourceKdList,
                TargetKdList,
                DistList,
                SourceIndexList,
                TargetIndexList,
            ) = KdIcpPairs(SourceVcoList, TargetVcoList, VertsLimite=VertsLimite)

            TransformMatrix = KdIcpPairsToTransformMatrix(
                TargetKdList=TargetKdList, SourceKdList=SourceKdList
            )
            SourceObj.matrix_world = TransformMatrix @ SourceObj.matrix_world
            for RefP in self.SourceRefPoints:
                RefP.matrix_world = TransformMatrix @ RefP.matrix_world
            # Update scene :
            SourceObj.update_tag()
            bpy.context.view_layer.update()

            SourceObj = self.SourceObject

            SourceVcoList = [
                SourceObj.matrix_world @ SourceObj.data.vertices[idx].co
                for idx in SourceVidList
            ]
            _, _, DistList, _, _ = KdIcpPairs(
                SourceVcoList, TargetVcoList, VertsLimite=VertsLimite
            )
            MaxDist = max(DistList)
            Override, area3D, space3D = CtxOverride(bpy.context)
            bpy.ops.wm.redraw_timer(
                Override, type="DRAW_WIN_SWAP", iterations=1)
            #######################################################
            if MaxDist <= Precision:
                self.ResultMessage = [
                    "Allignement Done !",
                    f"Max Distance < or = {Precision} mm",
                ]
                print(f"Number of iterations = {i}")
                print(f"Precision of {Precision} mm reached.")
                print(f"Max Distance = {round(MaxDist, 6)} mm")
                break

        if MaxDist > Precision:
            print(f"Number of iterations = {i}")
            print(f"Max Distance = {round(MaxDist, 6)} mm")
            self.ResultMessage = [
                "Allignement Done !",
                f"Max Distance = {round(MaxDist, 6)} mm",
            ]

    def modal(self, context, event):

        ############################################
        if not event.type in {
            self.TargetChar,
            self.SourceChar,
            "DEL",
            "RET",
            "ESC",
        }:
            # allow navigation

            return {"PASS_THROUGH"}
        #########################################
        if event.type == self.TargetChar:
            # Add Target Refference point :
            if event.value == ("PRESS"):
                if self.TargetVoxelMode:

                    CursorToVoxelPoint(
                        Volume=self.TargetObject, CursorMove=True)

                color = self.TargetColor
                CollName = self.CollName
                self.Targetpc += 1
                name = f"B{self.Targetpc}"
                RefP = AddRefPoint(name, color, CollName)
                self.TargetRefPoints.append(RefP)
                self.TotalRefPoints.append(RefP)
                bpy.ops.object.select_all(action="DESELECT")

        #########################################
        if event.type == self.SourceChar:
            # Add Source Refference point :
            if event.value == ("PRESS"):
                if self.SourceVoxelMode:

                    CursorToVoxelPoint(
                        Volume=self.SourceObject, CursorMove=True)

                color = self.SourceColor
                CollName = self.CollName
                self.SourceCounter += 1
                name = f"M{self.SourceCounter}"
                RefP = AddRefPoint(name, color, CollName)
                self.SourceRefPoints.append(RefP)
                self.TotalRefPoints.append(RefP)
                bpy.ops.object.select_all(action="DESELECT")

        ###########################################
        elif event.type == ("DEL"):
            if event.value == ("PRESS"):
                if self.TotalRefPoints:
                    obj = self.TotalRefPoints.pop()
                    name = obj.name
                    if name.startswith("B"):
                        self.Targetpc -= 1
                        self.TargetRefPoints.pop()
                    if name.startswith("M"):
                        self.SourceCounter -= 1
                        self.SourceRefPoints.pop()
                    bpy.data.objects.remove(obj)
                    bpy.ops.object.select_all(action="DESELECT")

        ###########################################
        elif event.type == "RET":

            if event.value == ("PRESS"):

                start = tpc()

                TargetObj = self.TargetObject
                SourceObj = self.SourceObject

                #############################################
                condition = (
                    len(self.TargetRefPoints) == len(self.SourceRefPoints)
                    and len(self.TargetRefPoints) >= 3
                )
                if not condition:
                    message = [
                        "          Please check the following :",
                        "   - The number of Base Refference points and,",
                        "       Align Refference points should match!",
                        "   - The number of Base Refference points ,",
                        "         and Align Refference points,",
                        "       should be superior or equal to 3",
                        "        <<Please check and retry !>>",
                    ]
                    icon = "COLORSET_01_VEC"
                    bpy.ops.wm.bdental_message_box(
                        "INVOKE_DEFAULT", message=str(message), icon=icon
                    )

                else:

                    TransformMatrix = RefPointsToTransformMatrix(
                        self.TargetRefPoints, self.SourceRefPoints
                    )

                    SourceObj.matrix_world = TransformMatrix @ SourceObj.matrix_world
                    for SourceRefP in self.SourceRefPoints:
                        SourceRefP.matrix_world = (
                            TransformMatrix @ SourceRefP.matrix_world
                        )

                    for i, SP in enumerate(self.SourceRefPoints):
                        TP = self.TargetRefPoints[i]
                        MidLoc = (SP.location + TP.location) / 2
                        SP.location = TP.location = MidLoc

                    # Update scene :
                    context.view_layer.update()
                    for obj in [TargetObj, SourceObj]:
                        obj.update_tag()
                    bpy.ops.wm.redraw_timer(
                        self.FullOverride, type="DRAW_WIN_SWAP", iterations=1
                    )

                    self.ResultMessage = []
                    if not self.TargetVoxelMode and not self.SourceVoxelMode:
                        #########################################################
                        # ICP alignement :
                        print("ICP Align processing...")
                        IcpVidDict = VidDictFromPoints(
                            TargetRefPoints=self.TargetRefPoints,
                            SourceRefPoints=self.SourceRefPoints,
                            TargetObj=TargetObj,
                            SourceObj=SourceObj,
                            radius=3,
                        )
                        BDENTAL_Props = bpy.context.scene.BDENTAL_Props
                        BDENTAL_Props.IcpVidDict = str(IcpVidDict)

                        SourceVidList, TargetVidList = (
                            IcpVidDict[SourceObj],
                            IcpVidDict[TargetObj],
                        )

                        self.IcpPipline(
                            SourceObj=SourceObj,
                            TargetObj=TargetObj,
                            SourceVidList=SourceVidList,
                            TargetVidList=TargetVidList,
                            VertsLimite=10000,
                            Iterations=30,
                            Precision=0.0001,
                        )

                    ##########################################################
                    self.FullSpace3D.overlay.show_outline_selected = True
                    self.FullSpace3D.overlay.show_object_origins = True
                    self.FullSpace3D.overlay.show_annotation = True
                    self.FullSpace3D.overlay.show_text = True
                    self.FullSpace3D.overlay.show_extras = True
                    self.FullSpace3D.overlay.show_floor = True
                    self.FullSpace3D.overlay.show_axis_x = True
                    self.FullSpace3D.overlay.show_axis_y = True
                    ###########################################################
                    for Name in self.visibleObjects:
                        obj = bpy.data.objects.get(Name)
                        if obj:
                            obj.hide_set(False)

                    bpy.ops.object.select_all(
                        self.FullOverride, action="DESELECT")
                    bpy.ops.wm.tool_set_by_id(
                        self.FullOverride, name="builtin.select")
                    bpy.context.scene.tool_settings.use_snap = False
                    bpy.context.scene.cursor.location = (0, 0, 0)
                    bpy.ops.screen.region_toggle(
                        self.FullOverride, region_type="UI")

                    if self.Solid:
                        self.FullSpace3D.shading.background_color = (
                            self.background_color
                        )
                        self.FullSpace3D.shading.background_type = self.background_type

                    TargetObj = self.TargetObject
                    SourceObj = self.SourceObject

                    if self.TotalRefPoints:
                        for RefP in self.TotalRefPoints:
                            bpy.data.objects.remove(RefP)

                    AlignColl = bpy.data.collections.get("ALIGN POINTS")
                    if AlignColl:
                        bpy.data.collections.remove(AlignColl)

                    BDENTAL_Props = context.scene.BDENTAL_Props
                    BDENTAL_Props.AlignModalState = False

                    bpy.ops.screen.screen_full_area(self.FullOverride)

                    if self.ResultMessage:
                        message = self.ResultMessage
                        icon = "COLORSET_02_VEC"
                        bpy.ops.wm.bdental_message_box(
                            "INVOKE_DEFAULT", message=str(message), icon=icon
                        )

                    ##########################################################

                    finish = tpc()
                    print(f"Alignement finshed in {finish-start} secondes")

                    return {"FINISHED"}

        ###########################################
        elif event.type == ("ESC"):

            if event.value == ("PRESS"):

                ##########################################################
                self.FullSpace3D.overlay.show_outline_selected = True
                self.FullSpace3D.overlay.show_object_origins = True
                self.FullSpace3D.overlay.show_annotation = True
                self.FullSpace3D.overlay.show_text = True
                self.FullSpace3D.overlay.show_extras = True
                self.FullSpace3D.overlay.show_floor = True
                self.FullSpace3D.overlay.show_axis_x = True
                self.FullSpace3D.overlay.show_axis_y = True
                ###########################################################
                for Name in self.visibleObjects:
                    obj = bpy.data.objects.get(Name)
                    if obj:
                        obj.hide_set(False)

                bpy.ops.object.select_all(self.FullOverride, action="DESELECT")
                bpy.ops.wm.tool_set_by_id(
                    self.FullOverride, name="builtin.select")
                bpy.context.scene.tool_settings.use_snap = False
                bpy.context.scene.cursor.location = (0, 0, 0)
                bpy.ops.screen.region_toggle(
                    self.FullOverride, region_type="UI")

                if self.Solid:
                    self.FullSpace3D.shading.background_color = self.background_color
                    self.FullSpace3D.shading.background_type = self.background_type

                TargetObj = self.TargetObject
                SourceObj = self.SourceObject

                if self.TotalRefPoints:
                    for RefP in self.TotalRefPoints:
                        bpy.data.objects.remove(RefP)

                AlignColl = bpy.data.collections.get("ALIGN POINTS")
                if AlignColl:
                    bpy.data.collections.remove(AlignColl)

                BDENTAL_Props = context.scene.BDENTAL_Props
                BDENTAL_Props.AlignModalState = False

                bpy.ops.screen.screen_full_area(self.FullOverride)

                message = [
                    " The Align Operation was Cancelled!",
                ]

                icon = "COLORSET_02_VEC"
                bpy.ops.wm.bdental_message_box(
                    "INVOKE_DEFAULT", message=str(message), icon=icon
                )
                return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        Condition_1 = len(bpy.context.selected_objects) != 2
        Condition_2 = bpy.context.selected_objects and not bpy.context.active_object
        Condition_3 = bpy.context.selected_objects and not (
            bpy.context.active_object in bpy.context.selected_objects
        )

        if Condition_1 or Condition_2 or Condition_3:

            message = [
                "Selection is invalid !",
                "Please Deselect all objects,",
                "Select the Object to Align and ,",
                "<SHIFT + Select> the Base Object.",
                "Click info button for more info.",
            ]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        else:

            if context.space_data.type == "VIEW_3D":
                BDENTAL_Props = context.scene.BDENTAL_Props
                BDENTAL_Props.AlignModalState = True
                # Prepare scene  :
                ##########################################################

                bpy.context.space_data.overlay.show_outline_selected = False
                bpy.context.space_data.overlay.show_object_origins = False
                bpy.context.space_data.overlay.show_annotation = False
                bpy.context.space_data.overlay.show_text = False
                bpy.context.space_data.overlay.show_extras = False
                bpy.context.space_data.overlay.show_floor = False
                bpy.context.space_data.overlay.show_axis_x = False
                bpy.context.space_data.overlay.show_axis_y = False
                bpy.context.scene.tool_settings.use_snap = True
                bpy.context.scene.tool_settings.snap_elements = {"FACE"}
                bpy.context.scene.tool_settings.transform_pivot_point = (
                    "INDIVIDUAL_ORIGINS"
                )
                bpy.ops.wm.tool_set_by_id(name="builtin.cursor")

                ###########################################################
                self.TargetObject = bpy.context.active_object
                self.SourceObject = [
                    obj
                    for obj in bpy.context.selected_objects
                    if not obj is self.TargetObject
                ][0]

                VisObj = bpy.context.visible_objects
                self.visibleObjects = [obj.name for obj in VisObj]
                for obj in VisObj:
                    if not obj in [self.TargetObject, self.SourceObject]:
                        obj.hide_set(True)

                self.Solid = False
                if bpy.context.space_data.shading.type == "SOLID":
                    self.Solid = True
                    self.background_type = (
                        bpy.context.space_data.shading.background_type
                    )
                    bpy.context.space_data.shading.background_type = "VIEWPORT"
                    self.background_color = tuple(
                        bpy.context.space_data.shading.background_color
                    )
                    bpy.context.space_data.shading.background_color = (
                        0.0, 0.0, 0.0)

                self.TargetVoxelMode = self.TargetObject.name.startswith(
                    "BD"
                ) and self.TargetObject.name.endswith("_CTVolume")
                self.SourceVoxelMode = self.SourceObject.name.startswith(
                    "BD"
                ) and self.SourceObject.name.endswith("_CTVolume")
                self.TargetRefPoints = []
                self.SourceRefPoints = []
                self.TotalRefPoints = []

                self.Targetpc = 0
                self.SourceCounter = 0

                bpy.ops.screen.screen_full_area()
                self.FullOverride, self.FullArea3D, self.FullSpace3D = CtxOverride(
                    context
                )

                context.window_manager.modal_handler_add(self)

                return {"RUNNING_MODAL"}

            else:

                self.report({"WARNING"}, "Active space must be a View3d")

                return {"CANCELLED"}


############################################################################
class BDENTAL_OT_AlignPointsInfo(bpy.types.Operator):
    """Add Align Refference points"""

    bl_idname = "wm.bdental_alignpointsinfo"
    bl_label = "INFO"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        message = [
            "\u2588 Deselect all objects,",
            "\u2588 Select the Object to Align,",
            "\u2588 Press <SHIFT + Click> to select the Base Object,",
            "\u2588 Click <ALIGN> button,",
            f"      Press <Left Click> to Place Cursor,",
            f"      Press <'B'> to Add Green Point (Base),",
            f"      Press <'A'> to Add Red Point (Align),",
            f"      Press <'DEL'> to delete Point,",
            f"      Press <'ESC'> to Cancel Operation,",
            f"      Press <'ENTER'> to execute Alignement.",
            "\u2588 NOTE :",
            "3 Green Points and 3 Red Points,",
            "are the minimum required for Alignement!",
        ]

        icon = "COLORSET_02_VEC"
        bpy.ops.wm.bdental_message_box(
            "INVOKE_DEFAULT", message=str(message), icon=icon)

        return {"FINISHED"}


########################################################################
# Mesh Tools Operators
########################################################################
class BDENTAL_OT_AddColor(bpy.types.Operator):
    """Add color material"""

    bl_idname = "wm.bdental_add_color"
    bl_label = "Add Color"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type in ["MESH", "CURVE"]

    def execute(self, context):
        obj = context.active_object
        if not obj:

            message = ["Please select the target object !"]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )
            return {"CANCELLED"}

        if not obj.select_get() or not obj.type in ["MESH", "CURVE"]:
            message = ["Please select the target object !"]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        else:

            matName = f"{obj.name}_Mat"
            mat = bpy.data.materials.get(
                matName) or bpy.data.materials.new(matName)
            mat.use_nodes = False
            mat.diffuse_color = [0.8, 0.8, 0.8, 1.0]

            obj.active_material = mat

        return {"FINISHED"}


class BDENTAL_OT_RemoveColor(bpy.types.Operator):
    """Remove color material"""

    bl_idname = "wm.bdental_remove_color"
    bl_label = "Remove Color"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type in ["MESH", "CURVE"] and context.object.active_material

    def execute(self, context):
        obj = context.active_object
        if not obj:

            message = ["Please select the target object !"]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )
            return {"CANCELLED"}

        if not obj.select_get() or not obj.type in ["MESH", "CURVE"]:
            message = ["Please select the target object !"]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        else:

            if obj.material_slots:
                for _ in obj.material_slots:
                    bpy.ops.object.material_slot_remove()

        return {"FINISHED"}


class BDENTAL_OT_JoinObjects(bpy.types.Operator):
    "Join Objects"

    bl_idname = "wm.bdental_join_objects"
    bl_label = "JOIN :"

    @classmethod
    def poll(cls, context):
        return context.object and len(context.selected_objects) >= 2

    def execute(self, context):

        ActiveObj = context.active_object
        condition = (
            ActiveObj
            and ActiveObj in context.selected_objects
            and len(context.selected_objects) >= 2
        )

        if not condition:

            message = [" Please select objects to join !"]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        else:

            bpy.ops.object.join()

            return {"FINISHED"}


#######################################################################################
# Separate models operator :


class BDENTAL_OT_SeparateObjects(bpy.types.Operator):
    "Separate Objects"

    bl_idname = "wm.bdental_separate_objects"
    bl_label = "SEPARATE :"

    items = ["Selection", "Loose Parts", ""]
    SeparateMode: EnumProperty(
        items= set_enum_items(items), description="SeparateMode", default="Loose Parts"
    )
    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == "MESH" and len(context.selected_objects) == 1

    def execute(self, context):

        if self.SeparateMode == "Loose Parts":
            bpy.ops.mesh.separate(type="LOOSE")

        if self.SeparateMode == "Selection":
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.separate(type="SELECTED")

        bpy.ops.object.mode_set(mode="OBJECT")

        Parts = list(context.selected_objects)

        if Parts and len(Parts) > 1:
            for obj in Parts:
                bpy.ops.object.select_all(action="DESELECT")
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.origin_set(
                    type="ORIGIN_GEOMETRY", center="MEDIAN")

            bpy.ops.object.select_all(action="DESELECT")
            Parts[-1].select_set(True)
            bpy.context.view_layer.objects.active = Parts[-1]

        return {"FINISHED"}

    def invoke(self, context, event):

        self.ActiveObj = context.active_object
        condition = (
            self.ActiveObj
            and self.ActiveObj.type == "MESH"
            and self.ActiveObj in bpy.context.selected_objects
        )

        if not condition:

            message = ["Please select the target object !"]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        else:
            wm = context.window_manager
            return wm.invoke_props_dialog(self)


#######################################################################################
# Parent model operator :


class BDENTAL_OT_Parent(bpy.types.Operator):
    "Parent Object"

    bl_idname = "wm.bdental_parent_object"
    bl_label = "PARENT"

    @classmethod
    def poll(cls, context):
        return context.object and len(context.selected_objects) >= 2

    def execute(self, context):
        active_cobject = context.object
        objects = [
            obj for obj in context.selected_objects if obj is not context.object]
        for obj in objects:
            if obj.constraints:
                for c in obj.constraints:
                    if c.type == "CHILD_OF":
                        context.view_layer.objects.active = obj
                        bpy.ops.constraint.apply(constraint=c.name)

            child_constraint = obj.constraints.new("CHILD_OF")
            child_constraint.target = active_cobject

        if len(context.selected_objects) > 2:
            message = ["{} selected objects parented to {}".format(
                len(context.selected_objects)-1, context.object.name)]
        else:
            message = ["selected object parented to {}".format(
                context.object.name)]
        update_info(message)
        sleep(2)
        update_info()
        os.system("cls")
        return {"FINISHED"}


#######################################################################################
# Unparent model operator :


class BDENTAL_OT_Unparent(bpy.types.Operator):
    "Un-Parent objects"

    bl_idname = "wm.bdental_unparent_objects"
    bl_label = "UnParent"

    @classmethod
    def poll(cls, context):
        if context.selected_objects:
            return True

    def execute(self, context):
        for obj in context.selected_objects:
            if obj.constraints:
                for c in obj.constraints:
                    if c.type == "CHILD_OF":
                        context.view_layer.objects.active = obj
                        bpy.ops.constraint.apply(constraint=c.name)

        message = ["Selected object(s) unparented "]
        update_info(message)
        sleep(2)
        update_info()
        os.system("cls")
        return {"FINISHED"}


#######################################################################################
# Align model to front operator :
class BDENTAL_OT_align_to_cursor(bpy.types.Operator):
    """Align Model To Front view"""

    bl_idname = "wm.bdental_align_to_cursor"
    bl_label = "Align to Cursor"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.object

    def execute(self, context):
        context.object.matrix_world[:3] = context.scene.cursor.matrix[:3]

        return {"FINISHED"}


class BDENTAL_OT_align_to_front(bpy.types.Operator):
    """Align Model To Front view"""

    bl_idname = "wm.bdental_align_to_front"
    bl_label = "Align to Front"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if not context.object or not context.object.select_get():
            return False
        if not context.object.type in ["MESH", "CURVE"]:
            return False
        return True

    def execute(self, context):
        bpy.ops.object.transform_apply(
            location=False, rotation=False, scale=True)
        override, area, space_data = CtxOverride(context)
        obj = context.object
        view3d_rot_matrix = space_data.region_3d.view_rotation.to_matrix().to_4x4()
        Eul_90x_matrix = Euler((radians(90), 0, 0), "XYZ").to_matrix().to_4x4()

        # Rotate Model :
        obj.matrix_world = (
            Eul_90x_matrix @ view3d_rot_matrix.inverted() @ obj.matrix_world
        )

        # view3d_rot_matrix = context.space_data.region_3d.view_rotation.to_matrix().to_4x4()
        # obj.matrix_world = view3d_rot_matrix.inverted() @ obj.matrix_world
        # obj.rotation_euler.rotate_axis("X", math.pi)
        bpy.ops.view3d.view_all(override, center=True)
        bpy.ops.view3d.view_axis(override, type="FRONT")
        bpy.ops.wm.tool_set_by_id(override, name="builtin.select")

        return {"FINISHED"}


#######################################################################################
# Center model modal operator :


class BDENTAL_OT_to_center(bpy.types.Operator):
    "Center Model to world origin"

    bl_idname = "wm.bdental_to_center"
    bl_label = "TO CENTER"

    yellow_stone = [1.0, 0.36, 0.06, 1.0]

    @classmethod
    def poll(cls, context):
        if not context.object or not context.object.select_get():
            return False
        if not context.object.type in ["MESH", "CURVE"]:
            return False
        return True

    def modal(self, context, event):

        if not event.type in {"RET", "ESC"}:
            # allow navigation

            return {"PASS_THROUGH"}

        elif event.type == "RET":

            if event.value == ("PRESS"):
                self.target.location -= context.scene.cursor.location
                override, _, _ = CtxOverride(context)
                context.view_layer.objects.active = self.target
                bpy.ops.view3d.snap_cursor_to_center(override)

                bpy.ops.view3d.view_all(override, center=True)
                bpy.ops.wm.tool_set_by_id(override, name="builtin.select")
                update_info()

            return {"FINISHED"}

        elif event.type == ("ESC"):

            if event.value == ("PRESS"):
                override, _, _ = CtxOverride(context)
                bpy.ops.wm.tool_set_by_id(override, name="builtin.select")
                update_info()

                return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):

        if context.space_data.type == "VIEW_3D":

            bpy.ops.object.mode_set(mode="OBJECT")
            self.target = context.view_layer.objects.active
            bpy.ops.wm.tool_set_by_id(name="builtin.cursor")

            message = [
                "Left-Click : to place cursor",
                "Enter : to center model",
            ]

            update_info(message)
            wm = context.window_manager
            wm.modal_handler_add(self)
            return {"RUNNING_MODAL"}

        else:

            self.report({"WARNING"}, "Active space must be a View3d")

            return {"CANCELLED"}


#######################################################################################
# Cursor to world origin operator :


class BDENTAL_OT_center_cursor(bpy.types.Operator):
    """Cursor to World Origin"""

    bl_idname = "wm.bdental_center_cursor"
    bl_label = "Center Cursor"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        bpy.ops.view3d.snap_cursor_to_center()

        return {"FINISHED"}


#######################################################################################
# Add Occlusal Plane :
class BDENTAL_OT_OcclusalPlane(bpy.types.Operator):
    """Add Occlusal Plane"""

    bl_idname = "wm.bdental_occlusalplane"
    bl_label = "OCCLUSAL PLANE"
    bl_options = {"REGISTER", "UNDO"}

    CollName = "Occlusal Points"
    OcclusalPoints = []

    def modal(self, context, event):

        if (
            event.type
            in [
                "LEFTMOUSE",
                "RIGHTMOUSE",
                "MIDDLEMOUSE",
                "WHEELUPMOUSE",
                "WHEELDOWNMOUSE",
                "N",
                "NUMPAD_2",
                "NUMPAD_4",
                "NUMPAD_6",
                "NUMPAD_8",
                "NUMPAD_1",
                "NUMPAD_3",
                "NUMPAD_5",
                "NUMPAD_7",
                "NUMPAD_9",
            ]
            and event.value == "PRESS"
        ):

            return {"PASS_THROUGH"}
        #########################################
        if event.type == "R":
            # Add Right point :
            if event.value == ("PRESS"):
                Override, area3D, space3D = CtxOverride(context)
                color = (1, 0, 0, 1)  # red
                CollName = self.CollName
                name = "Right_Occlusal_Point"
                OldPoint = bpy.data.objects.get(name)
                if OldPoint:
                    bpy.data.objects.remove(OldPoint)
                loc = context.scene.cursor.location
                NewPoint = AddMarkupPoint(name, color, loc, 1.2, CollName)
                self.RightPoint = NewPoint
                bpy.ops.object.select_all(Override, action="DESELECT")
                self.OcclusalPoints = [
                    name
                    for name in self.OcclusalPoints
                    if not name == "Right_Occlusal_Point"
                ]
                self.OcclusalPoints.append(self.RightPoint.name)

        #########################################
        if event.type == "A":
            # Add Right point :
            if event.value == ("PRESS"):
                Override, area3D, space3D = CtxOverride(context)
                color = (0, 1, 0, 1)  # green
                CollName = self.CollName
                name = "Anterior_Occlusal_Point"
                OldPoint = bpy.data.objects.get(name)
                if OldPoint:
                    bpy.data.objects.remove(OldPoint)
                loc = context.scene.cursor.location
                NewPoint = AddMarkupPoint(name, color, loc, 1.2, CollName)
                self.AnteriorPoint = NewPoint
                bpy.ops.object.select_all(Override, action="DESELECT")

                self.OcclusalPoints = [
                    name
                    for name in self.OcclusalPoints
                    if not name == "Anterior_Occlusal_Point"
                ]
                self.OcclusalPoints.append(self.AnteriorPoint.name)
        #########################################
        if event.type == "L":
            # Add Right point :
            if event.value == ("PRESS"):
                Override, area3D, space3D = CtxOverride(context)
                color = (0, 0, 1, 1)  # blue
                CollName = self.CollName
                name = "Left_Occlusal_Point"
                OldPoint = bpy.data.objects.get(name)
                if OldPoint:
                    bpy.data.objects.remove(OldPoint)
                loc = context.scene.cursor.location
                NewPoint = AddMarkupPoint(name, color, loc, 1.2, CollName)
                self.LeftPoint = NewPoint
                bpy.ops.object.select_all(Override, action="DESELECT")
                self.OcclusalPoints = [
                    name
                    for name in self.OcclusalPoints
                    if not name == "Left_Occlusal_Point"
                ]
                self.OcclusalPoints.append(self.LeftPoint.name)
        #########################################

        elif event.type == ("DEL") and event.value == ("PRESS"):

            if self.OcclusalPoints:
                name = self.OcclusalPoints.pop()
                bpy.data.objects.remove(bpy.data.objects.get(name))

        elif event.type == "RET":
            if event.value == ("PRESS"):

                if not len(self.OcclusalPoints) == 3:
                    message = ["3 points needed",
                               "Please check Info and retry"]
                    icon = "COLORSET_01_VEC"
                    bpy.ops.wm.bdental_message_box(
                        "INVOKE_DEFAULT", message=str(message), icon=icon
                    )

                else:
                    OcclusalPlane = PointsToOcclusalPlane(
                        self.FullOverride,
                        self.TargetObject,
                        self.RightPoint,
                        self.AnteriorPoint,
                        self.LeftPoint,
                        color=(0.0, 0.0, 0.2, 0.7),
                        subdiv=50,
                    )

                    #########################################################
                    self.FullSpace3D.overlay.show_outline_selected = True
                    self.FullSpace3D.overlay.show_object_origins = True
                    self.FullSpace3D.overlay.show_annotation = True
                    self.FullSpace3D.overlay.show_text = True
                    self.FullSpace3D.overlay.show_extras = True
                    self.FullSpace3D.overlay.show_floor = True
                    self.FullSpace3D.overlay.show_axis_x = True
                    self.FullSpace3D.overlay.show_axis_y = True
                    ##########################################################
                    for Name in self.visibleObjects:
                        obj = bpy.data.objects.get(Name)
                        if obj:
                            obj.hide_set(False)

                    bpy.ops.object.select_all(
                        self.FullOverride, action="DESELECT")
                    bpy.ops.wm.tool_set_by_id(
                        self.FullOverride, name="builtin.select")
                    bpy.context.scene.tool_settings.use_snap = False
                    bpy.context.scene.cursor.location = (0, 0, 0)
                    bpy.ops.screen.region_toggle(
                        self.FullOverride, region_type="UI")

                    self.FullSpace3D.shading.background_color = self.background_color
                    self.FullSpace3D.shading.background_type = self.background_type

                    bpy.ops.screen.screen_full_area(self.FullOverride)

                    if self.OcclusalPoints:
                        for name in self.OcclusalPoints:
                            P = bpy.data.objects.get(name)
                            if P:
                                bpy.data.objects.remove(P)
                    Coll = bpy.data.collections.get(self.CollName)
                    if Coll:
                        bpy.data.collections.remove(Coll)
                    ##########################################################
                    return {"FINISHED"}

        elif event.type == ("ESC"):

            ##########################################################
            self.FullSpace3D.overlay.show_outline_selected = True
            self.FullSpace3D.overlay.show_object_origins = True
            self.FullSpace3D.overlay.show_annotation = True
            self.FullSpace3D.overlay.show_text = True
            self.FullSpace3D.overlay.show_extras = True
            self.FullSpace3D.overlay.show_floor = True
            self.FullSpace3D.overlay.show_axis_x = True
            self.FullSpace3D.overlay.show_axis_y = True
            ###########################################################
            for Name in self.visibleObjects:
                obj = bpy.data.objects.get(Name)
                if obj:
                    obj.hide_set(False)

            bpy.ops.object.select_all(self.FullOverride, action="DESELECT")
            bpy.ops.wm.tool_set_by_id(self.FullOverride, name="builtin.select")
            bpy.context.scene.tool_settings.use_snap = False
            bpy.context.scene.cursor.location = (0, 0, 0)
            bpy.ops.screen.region_toggle(self.FullOverride, region_type="UI")

            self.FullSpace3D.shading.background_color = self.background_color
            self.FullSpace3D.shading.background_type = self.background_type

            bpy.ops.screen.screen_full_area(self.FullOverride)

            if self.OcclusalPoints:
                for name in self.OcclusalPoints:
                    P = bpy.data.objects.get(name)
                    if P:
                        bpy.data.objects.remove(P)
            Coll = bpy.data.collections.get(self.CollName)
            if Coll:
                bpy.data.collections.remove(Coll)

            message = [
                " The Occlusal Plane Operation was Cancelled!",
            ]

            icon = "COLORSET_02_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        Condition_1 = bpy.context.selected_objects and bpy.context.active_object

        if not Condition_1:

            message = [
                "Please select Target object",
            ]
            icon = "COLORSET_01_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        else:

            if context.space_data.type == "VIEW_3D":

                # Prepare scene  :
                ##########################################################

                bpy.context.space_data.overlay.show_outline_selected = False
                bpy.context.space_data.overlay.show_object_origins = False
                bpy.context.space_data.overlay.show_annotation = False
                bpy.context.space_data.overlay.show_text = False
                bpy.context.space_data.overlay.show_extras = False
                bpy.context.space_data.overlay.show_floor = False
                bpy.context.space_data.overlay.show_axis_x = False
                bpy.context.space_data.overlay.show_axis_y = False
                bpy.context.scene.tool_settings.use_snap = True
                bpy.context.scene.tool_settings.snap_elements = {"FACE"}
                bpy.context.scene.tool_settings.transform_pivot_point = (
                    "INDIVIDUAL_ORIGINS"
                )
                bpy.ops.wm.tool_set_by_id(name="builtin.cursor")

                ###########################################################
                self.TargetObject = bpy.context.active_object
                VisObj = bpy.context.visible_objects
                self.visibleObjects = [obj.name for obj in VisObj]

                for obj in VisObj:
                    if obj is not self.TargetObject:
                        obj.hide_set(True)
                self.Background = bpy.context.space_data.shading.type
                bpy.context.space_data.shading.type = "SOLID"
                self.background_type = bpy.context.space_data.shading.background_type
                bpy.context.space_data.shading.background_type = "VIEWPORT"
                self.background_color = tuple(
                    bpy.context.space_data.shading.background_color
                )
                bpy.context.space_data.shading.background_color = (
                    0.0, 0.0, 0.0)
                bpy.ops.screen.region_toggle(region_type="UI")
                bpy.ops.object.select_all(action="DESELECT")
                bpy.ops.screen.screen_full_area()
                self.FullOverride, self.FullArea3D, self.FullSpace3D = CtxOverride(
                    context
                )

                context.window_manager.modal_handler_add(self)

                return {"RUNNING_MODAL"}

            else:

                self.report({"WARNING"}, "Active space must be a View3d")

                return {"CANCELLED"}


############################################################################
class BDENTAL_OT_OcclusalPlaneInfo(bpy.types.Operator):
    """Add Align Refference points"""

    bl_idname = "wm.bdental_occlusalplaneinfo"
    bl_label = "INFO"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        message = [
            "\u2588 Deselect all objects,",
            "\u2588 Select the target Object,",
            "\u2588 Click < OCCLUSAL PLANE > button,",
            f"      Press <Left Click> to Place Cursor,",
            f"      <'R'> to Add Right Point,",
            f"      <'A'> to Add Anterior median Point,",
            f"      <'L'> to Add Left Point,",
            f"      Press <'DEL'> to delete Point,",
            f"      Press <'ESC'> to Cancel Operation,",
            f"      Press <'ENTER'> to Add Occlusal Plane.",
        ]

        icon = "COLORSET_02_VEC"
        bpy.ops.wm.bdental_message_box(
            "INVOKE_DEFAULT", message=str(message), icon=icon)

        return {"FINISHED"}


#######################################################################################
# Decimate model operator :


class BDENTAL_OT_decimate(bpy.types.Operator):
    """Decimate to ratio"""

    bl_idname = "wm.bdental_decimate"
    bl_label = "Decimate Model"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        
        return context.object and context.object.select_get() and context.object.type in ["MESH", "CURVE"] 


    def execute(self, context):
        BDENTAL_Props = context.scene.BDENTAL_Props
        decimate_ratio = round(BDENTAL_Props.decimate_ratio, 2)

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.modifier_add(type="DECIMATE")
        bpy.context.object.modifiers["Decimate"].ratio = decimate_ratio
        bpy.ops.object.modifier_apply(modifier="Decimate")

        return {"FINISHED"}


#######################################################################################
# Fill holes operator :


class BDENTAL_OT_fill(bpy.types.Operator):
    """fill edge or face"""

    bl_idname = "wm.bdental_fill"
    bl_label = "FILL"
    bl_options = {"REGISTER", "UNDO"}

    Fill_treshold: IntProperty(
        name="Hole Fill Treshold",
        description="Hole Fill Treshold",
        default=400,
    )
    @classmethod
    def poll(cls, context):
        
        return context.object and context.object.select_get() and context.object.type in ["MESH"] 


    def execute(self, context):

        Mode = context.object.mode
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (False, True, False)
        
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_non_manifold()
        
        bpy.ops.mesh.fill_holes(sides=self.Fill_treshold)
        bpy.ops.mesh.quads_convert_to_tris(
            quad_method="BEAUTY", ngon_method="BEAUTY")
        bpy.ops.mesh.select_all(action="DESELECT")

        bpy.ops.object.mode_set(mode=Mode)

        return {"FINISHED"}

    def invoke(self, context, event):
        
        if context.object.mode == "EDIT":
            bpy.ops.mesh.edge_face_add()
            return {"FINISHED"}

        else:
            wm = context.window_manager
            return wm.invoke_props_dialog(self)


#######################################################################################
# Retopo smooth operator :


class BDENTAL_OT_retopo_smooth(bpy.types.Operator):
    """Retopo sculpt for filled holes"""

    bl_idname = "wm.bdental_retopo_smooth"
    bl_label = "Retopo Smooth"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        
        return context.object and context.object.select_get() and context.object.type in ["MESH"] 


    def execute(self, context):

        # Prepare scene settings :
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)

        bpy.ops.object.mode_set(mode="SCULPT")

        obj = context.object

        context.scene.tool_settings.sculpt.use_symmetry_x = False
        context.scene.tool_settings.unified_paint_settings.size = 50

        bpy.ops.wm.tool_set_by_id(name="builtin_brush.Simplify")
        bpy.data.brushes["Simplify"].cursor_color_add = (
            0.3, 0.0, 0.7, 0.4)
        bpy.data.brushes["Simplify"].strength = 0.5
        bpy.data.brushes["Simplify"].auto_smooth_factor = 0.5
        bpy.data.brushes["Simplify"].use_automasking_topology = True
        bpy.data.brushes["Simplify"].use_frontface = True

        if not obj.use_dynamic_topology_sculpting:
            bpy.ops.sculpt.dynamic_topology_toggle()

        context.scene.tool_settings.sculpt.detail_type_method = "CONSTANT"
        context.scene.tool_settings.sculpt.constant_detail_resolution = 16
        bpy.ops.sculpt.sample_detail_size(mode="DYNTOPO")

        return {"FINISHED"}


#######################################################################################
# clean model operator :
class BDENTAL_OT_clean_mesh2(bpy.types.Operator):
    """Fill small and medium holes and remove small parts"""

    bl_idname = "wm.bdental_clean_mesh2"
    bl_label = "CLEAN MESH"
    bl_options = {"REGISTER", "UNDO"}

    Fill_treshold: IntProperty(
        name="Holes Fill Treshold",
        description="Hole Fill Treshold",
        default=100,
    )
    @classmethod
    def poll(cls, context):
        
        return context.object and context.object.select_get() and context.object.type in ["MESH"] 


    def execute(self, context):

        obj = context.object

        ####### Get model to clean #######
        bpy.ops.object.mode_set(mode="OBJECT")
        # bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
        
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)

        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=0.001)
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.fill_holes(sides=self.Fill_treshold)
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.delete_loose(use_faces=True)

        ############ clean non_manifold borders ##############
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_non_manifold()
        bpy.ops.mesh.select_mode(type="FACE")
        bpy.ops.mesh.delete(type="FACE")
        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.delete_loose(use_faces=True)

        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_non_manifold()
        bpy.ops.mesh.select_less()
        bpy.ops.mesh.delete(type="VERT")

        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.fill_holes(sides=self.Fill_treshold)
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_non_manifold()

        bpy.ops.mesh.looptools_relax(
            input="selected",
            interpolation="cubic",
            iterations="3",
            regular=True,
        )

        bpy.ops.object.mode_set(mode="OBJECT")

        print("Clean Mesh finished.")

        return {"FINISHED"}

    def invoke(self, context, event):
        
        wm = context.window_manager
        return wm.invoke_props_dialog(self)


# clean model operator :
class BDENTAL_OT_clean_mesh(bpy.types.Operator):
    """Fill small and medium holes and remove small parts"""

    bl_idname = "wm.bdental_clean_mesh"
    bl_label = "CLEAN MESH"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        ActiveObj = context.active_object

        if not ActiveObj:
            message = [" Invalid Selection ", "Please select Target mesh ! "]
            icon = "COLORSET_02_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}
        else:
            Conditions = [
                not ActiveObj.select_set,
                not ActiveObj.type == "MESH",
            ]

            if Conditions[0] or Conditions[1]:
                message = [" Invalid Selection ",
                           "Please select Target mesh ! "]
                icon = "COLORSET_01_VEC"
                bpy.ops.wm.bdental_message_box(
                    "INVOKE_DEFAULT", message=str(message), icon=icon
                )
                return {"CANCELLED"}

            else:

                ####### Get model to clean #######
                bpy.ops.object.mode_set(mode="OBJECT")
                # bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
                Obj = ActiveObj
                bpy.ops.object.select_all(action="DESELECT")
                Obj.select_set(True)
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.context.tool_settings.mesh_select_mode = (
                    True, False, False)

                ####### Remove doubles, Make mesh consistent (face normals) #######
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.remove_doubles(threshold=0.1)
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.normals_make_consistent(inside=False)

                ############ clean non_manifold borders ##############
                bpy.ops.mesh.select_all(action="DESELECT")
                bpy.ops.mesh.select_non_manifold()
                bpy.ops.mesh.delete(type="VERT")

                bpy.ops.mesh.select_non_manifold()
                bpy.ops.mesh.select_more()
                bpy.ops.mesh.select_less()
                bpy.ops.mesh.delete(type="VERT")

                ####### Fill Holes #######

                bpy.ops.mesh.select_all(action="DESELECT")
                bpy.ops.mesh.select_non_manifold()
                bpy.ops.mesh.fill_holes(sides=100)
                bpy.ops.mesh.quads_convert_to_tris(
                    quad_method="BEAUTY", ngon_method="BEAUTY"
                )

                ####### Relax borders #######
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action="DESELECT")
                bpy.ops.mesh.select_non_manifold()
                bpy.ops.mesh.remove_doubles(threshold=0.1)

                bpy.ops.mesh.looptools_relax(
                    input="selected",
                    interpolation="cubic",
                    iterations="1",
                    regular=True,
                )

                bpy.ops.mesh.select_all(action="DESELECT")
                bpy.ops.object.mode_set(mode="OBJECT")
                Obj.select_set(True)
                bpy.context.view_layer.objects.active = Obj

                print("Clean Mesh finished.")

                return {"FINISHED"}


class BDENTAL_OT_VoxelRemesh(bpy.types.Operator):
    """Voxel Remesh Operator"""

    bl_idname = "wm.bdental_voxelremesh"
    bl_label = "REMESH"
    bl_options = {"REGISTER", "UNDO"}

    VoxelSize: FloatProperty(
        name="Voxel Size",
        description="Remesh Voxel Size",
        default=0.1,
        min=0.0,
        max=100.0,
        soft_min=0.0,
        soft_max=100.0,
        step=10,
        precision=1,
    )
    @classmethod
    def poll(cls, context):
        
        return context.object and context.object.select_get() and context.object.type in ["MESH"] 


    def execute(self, context):
        obj = context.object
        remesh = obj.modifiers.new(name="Remesh", type="REMESH")
        remesh.voxel_size = self.VoxelSize
        bpy.ops.object.convert(target='MESH', keep_original=False)
        
        return {"FINISHED"}

    def invoke(self, context, event):

        self.VoxelSize = 0.1
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

class BDENTAL_OT_RibbonCutterAdd(bpy.types.Operator):
    """mesh curve cutter tool"""

    bl_idname = "wm.bdental_ribboncutteradd"
    bl_label = "DRAW CURVE"
    bl_options = {"REGISTER", "UNDO"}

    closeCurve : BoolProperty(name="close cutting curve", default=True)

    @classmethod
    def poll(cls, context):

        base_mesh = context.object and context.object.select_get(
        ) and context.object.type == "MESH"
        return base_mesh

    def add_curve_cutter(self, context):
        Override, area3D, space3D = CtxOverride(context)
        # Prepare scene settings :
        bpy.context.scene.tool_settings.use_snap = True
        bpy.context.scene.tool_settings.snap_elements = {"FACE"}

        # ....Add Curve ....... :
        bpy.ops.curve.primitive_bezier_curve_add(Override,
                                                 radius=1, enter_editmode=False, align="CURSOR"
                                                 )
        # Set cutting_tool name :
        self.cutter = context.object
        self.cutter.name = "BDENTAL_Ribbon_Cutter"
        self.cutter["bdental_type"] = "curvecutter3"
        self.cutter["bdental_target"] = self.base_mesh.name
        

        # CurveCutter settings :
        bpy.ops.object.mode_set(Override, mode="EDIT")
        bpy.ops.curve.select_all(Override, action="DESELECT")
        self.cutter.data.splines[0].bezier_points[-1].select_control_point = True
        bpy.ops.curve.dissolve_verts(Override)
        bpy.ops.curve.select_all(Override, action="SELECT")
        bpy.ops.view3d.snap_selected_to_cursor(Override, use_offset=False)
        bpy.ops.curve.handle_type_set(Override, type="AUTOMATIC")
        self.cutter.data.dimensions = "3D"
        self.cutter.data.twist_smooth = 3
        self.cutter.data.bevel_resolution = 6
        self.cutter.data.use_fill_caps = True
        self.cutter.data.bevel_depth = 0.3
        self.cutter.data.extrude = 4
        self.cutter.data.offset = -0.3
        context.scene.tool_settings.curve_paint_settings.error_threshold = 1
        context.scene.tool_settings.curve_paint_settings.corner_angle = 0.785398
        context.scene.tool_settings.curve_paint_settings.depth_mode = "SURFACE"
        context.scene.tool_settings.curve_paint_settings.surface_offset = 0
        context.scene.tool_settings.curve_paint_settings.use_offset_absolute = True
        
        
        # Add color material :
        mat = bpy.data.materials.get(
            "Bdental_curve_cutter_mat"
        ) or bpy.data.materials.new("Bdental_curve_cutter_mat")
        mat.diffuse_color = [1, 0, 0, 1]
        mat.roughness = 0.3
        bpy.ops.object.mode_set(mode="OBJECT")
        self.cutter.active_material = mat

        bpy.ops.wm.tool_set_by_id(Override, name="builtin.cursor")
        space3D.overlay.show_outline_selected = False

        shrinkwrap = self.cutter.modifiers.new(
            name="Shrinkwrap", type="SHRINKWRAP")
        shrinkwrap.target = self.base_mesh
        shrinkwrap.wrap_mode = "ABOVE_SURFACE"
        shrinkwrap.use_apply_on_spline = True

        MoveToCollection(self.cutter, "Bdental Cutters")

    def add_cutter_point(self):
        Override, area3D, space3D = CtxOverride(bpy.context)

        bpy.ops.object.mode_set(Override, mode="EDIT")
        bpy.ops.curve.extrude(Override, mode="INIT")
        bpy.ops.view3d.snap_selected_to_cursor(Override, use_offset=False)
        bpy.ops.curve.select_all(Override, action="SELECT")
        bpy.ops.curve.handle_type_set(Override, type="AUTOMATIC")
        bpy.ops.curve.select_all(Override, action="DESELECT")
        points = self.cutter.data.splines[0].bezier_points[:]
        points[-1].select_control_point = True
        bpy.ops.object.mode_set(Override, mode="OBJECT")

    def del_cutter_point(self):
        Override, area3D, space3D = CtxOverride(bpy.context)
        try:
            bpy.ops.object.mode_set(Override, mode="EDIT")
            bpy.ops.curve.select_all(Override, action="DESELECT")
            points = self.cutter.data.splines[0].bezier_points[:]
            points[-1].select_control_point = True
            points = self.cutter.data.splines[0].bezier_points[:]
            if len(points) > 1:

                bpy.ops.curve.delete(Override, type="VERT")
                points = self.cutter.data.splines[0].bezier_points[:]
                bpy.ops.curve.select_all(Override, action="SELECT")
                bpy.ops.curve.handle_type_set(Override, type="AUTOMATIC")
                bpy.ops.curve.select_all(Override, action="DESELECT")
                points = self.cutter.data.splines[0].bezier_points[:]
                points[-1].select_control_point = True

            bpy.ops.object.mode_set(Override, mode="OBJECT")

        except Exception:
            pass

    def modal(self, context, event):
        
        if not event.type in ["RET", "ESC", "LEFTMOUSE", "DEL"]:
            return {"PASS_THROUGH"}
        elif event.type == "RET" and self.counter == 0:
            return {"PASS_THROUGH"}
        elif event.type == "DEL" and self.counter == 0:
            return {"PASS_THROUGH"}

        elif event.type == "ESC":
            if event.value == ("PRESS"):

                for obj in bpy.data.objects:
                    if not obj in self.start_objects:
                        bpy.data.objects.remove(obj)
                for col in bpy.data.collections:
                    if not col in self.start_collections:
                        bpy.data.collections.remove(col)

                for obj in context.visible_objects:
                    obj.hide_set(True)
                for obj in self.start_visible_objects:
                    try:
                        obj.hide_set(False)
                    except:
                        pass

                Override, area3D, space3D = CtxOverride(context)
                bpy.ops.wm.tool_set_by_id(Override, name="builtin.select")
                self.scn.tool_settings.use_snap = False
                space3D.overlay.show_outline_selected = True

                message = ["CANCELLED"]
                update_info(message)
                sleep(2)
                update_info()
                return {"CANCELLED"}

        elif event.type == ("LEFTMOUSE") and self.counter == 1:
            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}

            if event.value == ("RELEASE"):
                _is_valid = click_is_in_view3d(context, event)
                print(_is_valid)
                if _is_valid :
                    self.add_cutter_point()
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

        elif event.type == ("LEFTMOUSE") and self.counter == 0:
            
            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}

            if event.value == ("RELEASE"):
                _is_valid = click_is_in_view3d(context, event)
                print(_is_valid)
                if _is_valid :
                    self.add_curve_cutter(context)
                    self.counter += 1
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

        elif event.type == ("DEL") and self.counter == 1:
            if event.value == ("PRESS"):
                self.del_cutter_point()
                return {"RUNNING_MODAL"}

        elif event.type == "RET" and self.counter == 1:

            if event.value == ("PRESS"):
                override, area, space_data = CtxOverride(context)
                
                bpy.context.view_layer.objects.active = self.cutter
                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.select_all(action='DESELECT')
                self.cutter.select_set(True)

                if self.closeCurve :
                    bpy.ops.object.mode_set(mode="EDIT")
                    bpy.ops.curve.cyclic_toggle()
                    bpy.ops.object.mode_set(mode="OBJECT")

                bpy.ops.object.modifier_apply( modifier="Shrinkwrap")
                bpy.ops.wm.tool_set_by_id(override, name="builtin.select")
                bpy.context.scene.tool_settings.use_snap = False
                space_data.overlay.show_outline_selected = True
                update_info()
                return {"FINISHED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        if context.space_data.type == "VIEW_3D":
            self.base_mesh = context.object
            wm = context.window_manager
            return wm.invoke_props_dialog(self, width=500)

        else:

            message = ["Active space must be a View3d"]
            icon = "COLORSET_02_VEC"
            bpy.ops.bdental.message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon)
            return {"CANCELLED"}

    def execute(self, context):
        self.scn = context.scene
        self.counter = 0

        self.start_objects = bpy.data.objects[:]
        self.start_collections = bpy.data.collections[:]
        self.start_visible_objects = bpy.context.visible_objects[:]

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.base_mesh.select_set(True)
        context.view_layer.objects.active = self.base_mesh

        override, _, _ = CtxOverride(context)
        bpy.ops.wm.tool_set_by_id(override, name="builtin.cursor")
        context.window_manager.modal_handler_add(self)
        txt = ["Left click : draw curve | DEL : roll back | ESC : to cancell operation", "ENTER : to finalise"]
        update_info(txt)
        return {"RUNNING_MODAL"}

class BDENTAL_OT_RibbonCutter_Perform_Cut(bpy.types.Operator):
    "Performe CurveCutter1 Operation"

    bl_idname = "wm.bdental_ribboncutter_perform_cut"
    bl_label = "CUT"
    items = ["Remove Small Part", "Remove Big Part", "Keep All"]

    CutMode: EnumProperty(
        name="Splint Cut Mode",
        items=set_enum_items(items),
        description="Splint Cut Mode",
        default="Keep All",
    )
    @classmethod
    def poll(cls, context):
        cutters = [obj for obj in context.scene.objects if obj.get("bdental_type") == "curvecutter3"]
        return cutters

    def execute(self, context):

        start_unvis_objects = [obj for obj in context.scene.objects if not obj in context.visible_objects]

        txt = ["Processing ..."]
        update_info(message=txt, rect_color=[0.8,0.2,0,0.7])
        override, _, _ = CtxOverride(context)
        bpy.context.scene.tool_settings.use_snap = False
        bpy.ops.view3d.snap_cursor_to_center(override)

        datadict = {}

        allcutters = [obj for obj in context.scene.objects if obj.get("bdental_type") == "curvecutter3"]
        for c in allcutters :
            target_name = c["bdental_target"]
            if not datadict.get(target_name):
                datadict.update({target_name : [c.name]})
            else :
                datadict[target_name].append(c.name)
        
        for target_name, cutter_names in datadict.items() :
            if not bpy.data.objects.get(target_name) :
                continue
            target = bpy.data.objects.get(target_name)
            target.hide_set(False)
            target.hide_viewport = False
            target.hide_select = False
            cutters = [bpy.data.objects.get(n) for n in cutter_names if bpy.data.objects.get(n)]
            cutters_list = []
            for cc in cutters :
                bpy.context.view_layer.objects.active = cc
                cc.hide_set(False)
                cc.hide_viewport = False
                cc.hide_select = False
                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.select_all(action="DESELECT")
                cc.select_set(True)
                bpy.ops.object.duplicate_move()
                cc = context.object

                # remove material :
                for mat_slot in cc.material_slots:
                    bpy.ops.object.material_slot_remove()

                # convert CurveCutter to mesh :
                bpy.ops.object.convert(target="MESH")
                cutter = context.object
                cutters_list.append(cutter)
            

            bpy.ops.object.select_all(action="DESELECT")
            for obj in cutters_list:
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
            if len(cutters_list) > 1:
                bpy.ops.object.join()

            cutter = context.object
            # select CuttingTarget :
            bpy.ops.object.select_all(action="DESELECT")
            target.select_set(True)
            bpy.context.view_layer.objects.active = target

            # boolean modifier difference :
            bool = target.modifiers.new(name="Bool", type="BOOLEAN")
            bool.operation = "DIFFERENCE"
            bool.object = cutter
            bpy.ops.object.convert(target='MESH', keep_original=False)



            bpy.data.objects.remove(cutter)

            VisObj = [obj.name for obj in context.visible_objects]
            for obj in context.visible_objects :
                if not obj is target :
                    obj.hide_set(True)
            

            bpy.ops.wm.bdental_separate_objects(SeparateMode="Loose Parts")

            if not self.CutMode == "Keep All":

                target_Max = (
                    max(
                        [
                            [len(obj.data.polygons), obj.name]
                            for obj in context.visible_objects
                        ]
                    )
                )[1]
                target_min = (
                    min(
                        [
                            [len(obj.data.polygons), obj.name]
                            for obj in context.visible_objects
                        ]
                    )
                )[1]

                if self.CutMode == "Remove Small Part":
                    result = bpy.data.objects.get(target_Max)
                    for obj in context.visible_objects:
                        if not obj is result:
                            bpy.data.objects.remove(obj)

                if self.CutMode == "Remove Big Part":
                    result = bpy.data.objects.get(target_min)
                    for obj in context.visible_objects:
                        if not obj is result:
                            bpy.data.objects.remove(obj)

                result.select_set(True)
                bpy.context.view_layer.objects.active = result
                # bpy.ops.object.shade_flat()

            if self.CutMode == "Keep All":
                bpy.ops.object.select_all(action="DESELECT")

            for objname in VisObj:
                obj = bpy.data.objects.get(objname)
                if obj:
                    obj.hide_set(False)
        Override, area3D, space3D = CtxOverride(context)
        bpy.context.scene.tool_settings.use_snap = False
        bpy.ops.view3d.snap_cursor_to_center(Override)
            
        col = bpy.data.collections.get("Bdental Cutters")
        if col :
            for obj in col.objects :
                bpy.data.objects.remove(obj)

            bpy.data.collections.remove(col)

        for obj in context.scene.objects :
            if obj in start_unvis_objects :
                obj.hide_set(True)
            else:
                obj.hide_set(False)
                obj.hide_viewport = False
                obj.hide_select = False

        
        txt = ["Done."]
        update_info(message=txt, rect_color=[0.0,1.0,0,0.7])
        sleep(1)
        update_info()

        return {"FINISHED"}

   
    def invoke(self, context, event):

        wm = context.window_manager
        return wm.invoke_props_dialog(self)

# class BDENTAL_OT_SplintCutterAdd(bpy.types.Operator):
#     """ """

#     bl_idname = "wm.bdental_splintcutteradd"
#     bl_label = "CURVE CUTTER ADD"
#     bl_options = {"REGISTER", "UNDO"}

#     @classmethod
#     def poll(cls, context):
        
#         return context.object and context.object.select_get() and context.object.type in ["MESH"] 


#     def modal(self, context, event):

#         BDENTAL_Props = context.scene.BDENTAL_Props

#         if not event.type in {
#             "DEL",
#             "LEFTMOUSE",
#             "RET",
#             "ESC",
#         }:
#             # allow navigation

#             return {"PASS_THROUGH"}

#         elif event.type == ("DEL"):
#             if event.value == ("PRESS"):

#                 DeleteLastCurvePoint()

#             return {"RUNNING_MODAL"}

#         elif event.type == ("LEFTMOUSE"):

#             if event.value == ("PRESS"):

#                 return {"PASS_THROUGH"}

#             if event.value == ("RELEASE"):

#                 ExtrudeCurvePointToCursor(context, event)

#         elif event.type == "RET":

#             if event.value == ("PRESS"):
#                 override, area, space_data = CtxOverride(context)
#                 CurveCutterName = BDENTAL_Props.CurveCutterNameProp
#                 CurveCutter = bpy.data.objects[CurveCutterName]
#                 CurveCutter.select_set(True)
#                 bpy.context.view_layer.objects.active = CurveCutter
#                 bpy.ops.object.mode_set(mode="OBJECT")

#                 if BDENTAL_Props.CurveCutCloseMode == "Close Curve":
#                     bpy.ops.object.mode_set(mode="EDIT")
#                     bpy.ops.curve.cyclic_toggle()
#                     bpy.ops.object.mode_set(mode="OBJECT")

#                 bpy.ops.wm.tool_set_by_id(override, name="builtin.select")
#                 bpy.context.scene.tool_settings.use_snap = False
#                 space_data.overlay.show_outline_selected = True

#                 return {"FINISHED"}

#         elif event.type == ("ESC"):

#             if event.value == ("PRESS"):
#                 override, area, space_data = CtxOverride(context)
#                 CurveCutterName = bpy.context.scene.BDENTAL_Props.CurveCutterNameProp
#                 CurveCutter = bpy.data.objects[CurveCutterName]
#                 bpy.ops.object.mode_set(mode="OBJECT")
#                 bpy.data.objects.remove(CurveCutter)

#                 CuttingTargetName = context.scene.BDENTAL_Props.CuttingTargetNameProp
#                 CuttingTarget = bpy.data.objects[CuttingTargetName]

#                 bpy.ops.object.select_all(action="DESELECT")
#                 CuttingTarget.select_set(True)
#                 bpy.context.view_layer.objects.active = CuttingTarget

#                 bpy.ops.wm.tool_set_by_id(override, name="builtin.select")
#                 bpy.context.scene.tool_settings.use_snap = False
#                 space_data.overlay.show_outline_selected = True

#                 return {"CANCELLED"}

#         return {"RUNNING_MODAL"}

#     def invoke(self, context, event):

#         if context.space_data.type == "VIEW_3D":
#             BDENTAL_Props = context.scene.BDENTAL_Props
#             # Assign Model name to CuttingTarget property :
#             CuttingTarget = context.object
#             BDENTAL_Props.CuttingTargetNameProp = CuttingTarget.name

#             bpy.ops.object.mode_set(mode="OBJECT")
#             # bpy.ops.object.hide_view_set(unselected=True)

#             CuttingCurveAdd()
#             SplintCutter = context.object
#             SplintCutter.name = "BDENTAL_Ribbon_Cutter"
#             SplintCutter["bdental_type"] = "curvecutter3"
#             BDENTAL_Props.CurveCutterNameProp = "BDENTAL_Ribbon_Cutter"
#             SplintCutter.data.use_fill_caps = True
#             SplintCutter.data.bevel_depth = 0.3
#             SplintCutter.data.extrude = 4
#             SplintCutter.data.offset = -0.3
#             SplintCutter.active_material.diffuse_color = [1, 0, 0, 1]

#             context.window_manager.modal_handler_add(self)

#             return {"RUNNING_MODAL"}

#         else:

#             self.report({"WARNING"}, "Active space must be a View3d")

#             return {"CANCELLED"}


# class BDENTAL_OT_SplintCutterCut(bpy.types.Operator):
#     """ """

#     bl_idname = "wm.bdental_splintcuttercut"
#     bl_label = "CURVE CUTTER ADD"
#     bl_options = {"REGISTER", "UNDO"}

#     Cut_Modes_List = ["Remove Small Part", "Remove Big Part", "Keep All"]

#     CutMode: EnumProperty(
#         name="Splint Cut Mode",
#         items=set_enum_items(Cut_Modes_List),
#         description="Splint Cut Mode",
#         default="Keep All",
#     )

#     def execute(self, context):

#         # Get CurveCutter :
#         bpy.ops.object.mode_set(mode="OBJECT")
#         bpy.ops.object.select_all(action="DESELECT")

#         CurveMeshesList = []
#         for CurveCutter in self.CurveCuttersList:
#             bpy.ops.object.select_all(action="DESELECT")
#             CurveCutter.select_set(True)
#             bpy.context.view_layer.objects.active = CurveCutter

#             # remove material :
#             for mat_slot in CurveCutter.material_slots:
#                 bpy.ops.object.material_slot_remove()

#             # convert CurveCutter to mesh :
#             bpy.ops.object.convert(target="MESH")
#             CurveMesh = context.object
#             CurveMeshesList.append(CurveMesh)

#         bpy.ops.object.select_all(action="DESELECT")
#         for obj in CurveMeshesList:
#             obj.select_set(True)
#             bpy.context.view_layer.objects.active = obj
#         if len(CurveMeshesList) > 1 :
#             bpy.ops.object.join()
#         CurveCutter = context.object
#         bpy.ops.wm.bdental_voxelremesh(VoxelSize=0.2)

#         bpy.ops.object.select_all(action="DESELECT")
#         self.CuttingTarget.select_set(True)
#         bpy.context.view_layer.objects.active = self.CuttingTarget

#         bpy.ops.object.modifier_add(type="BOOLEAN")
#         bpy.context.object.modifiers["Boolean"].show_viewport = False
#         bpy.context.object.modifiers["Boolean"].operation = "DIFFERENCE"
#         bpy.context.object.modifiers["Boolean"].object = CurveCutter
#         bpy.context.object.modifiers["Boolean"].solver = "EXACT"
#         bpy.ops.object.modifier_apply(modifier="Boolean")

#         bpy.data.objects.remove(CurveCutter)

#         VisObj = [obj.name for obj in context.visible_objects]
#         bpy.ops.object.select_all(action="DESELECT")
#         self.CuttingTarget.select_set(True)
#         bpy.ops.object.hide_view_set(unselected=True)

#         bpy.ops.wm.bdental_separate_objects(SeparateMode="Loose Parts")

#         if not self.CutMode == "Keep All":

#             Splint_Max = (
#                 max(
#                     [
#                         [len(obj.data.polygons), obj.name]
#                         for obj in context.visible_objects
#                     ]
#                 )
#             )[1]
#             Splint_min = (
#                 min(
#                     [
#                         [len(obj.data.polygons), obj.name]
#                         for obj in context.visible_objects
#                     ]
#                 )
#             )[1]

#             if self.CutMode == "Remove Small Part":
#                 Splint = bpy.data.objects.get(Splint_Max)
#                 for obj in context.visible_objects:
#                     if not obj is Splint:
#                         bpy.data.objects.remove(obj)

#             if self.CutMode == "Remove Big Part":
#                 Splint = bpy.data.objects.get(Splint_min)
#                 for obj in context.visible_objects:
#                     if not obj is Splint:
#                         bpy.data.objects.remove(obj)

#             Splint.select_set(True)
#             bpy.context.view_layer.objects.active = Splint
#             bpy.ops.object.shade_flat()

#         if self.CutMode == "Keep All":
#             bpy.ops.object.select_all(action="DESELECT")

#         for objname in VisObj:
#             obj = bpy.data.objects.get(objname)
#             if obj:
#                 obj.hide_set(False)

#         bpy.context.scene.tool_settings.use_snap = False
#         bpy.ops.view3d.snap_cursor_to_center()
#         return {"FINISHED"}

#     def invoke(self, context, event):

#         self.BDENTAL_Props = context.scene.BDENTAL_Props

#         # Get CuttingTarget :
#         CuttingTargetName = self.BDENTAL_Props.CuttingTargetNameProp
#         self.CuttingTarget = bpy.data.objects.get(CuttingTargetName)
#         self.CurveCuttersList = [
#             obj
#             for obj in context.scene.objects
#             if obj.type == "CURVE" and obj.name.startswith("BDENTAL_Splint_Cut")
#         ]

#         if not self.CurveCuttersList or not self.CuttingTarget:

#             message = [" Please Add Splint Cutters first !"]
#             update_info(message)
#             sleep(2)
#             update_info()
#             return {"CANCELLED"}

#         else:
#             wm = context.window_manager
#             return wm.invoke_props_dialog(self)


#######################################################################################
###################################### Cutters ########################################
#######################################################################################
# CurveCutter_01
class BDENTAL_OT_CurveCutterAdd(bpy.types.Operator):
    """description of this Operator"""

    bl_idname = "wm.bdental_curvecutteradd"
    bl_label = "CURVE CUTTER ADD"
    bl_options = {"REGISTER", "UNDO"}

    def modal(self, context, event):

        BDENTAL_Props = context.scene.BDENTAL_Props

        if not event.type in {
            "DEL",
            "LEFTMOUSE",
            "RET",
            "ESC",
        }:
            # allow navigation

            return {"PASS_THROUGH"}

        elif event.type == ("DEL"):
            if event.value == ("PRESS"):

                DeleteLastCurvePoint()

            return {"RUNNING_MODAL"}

        elif event.type == ("LEFTMOUSE"):

            if event.value == ("PRESS"):

                return {"PASS_THROUGH"}

            if event.value == ("RELEASE"):

                ExtrudeCurvePointToCursor(context, event)

        elif event.type == "RET":

            if event.value == ("PRESS"):
                override, area, space_data = CtxOverride(context)
                CurveCutterName = BDENTAL_Props.CurveCutterNameProp
                CurveCutter = bpy.data.objects[CurveCutterName]
                CurveCutter.select_set(True)
                bpy.context.view_layer.objects.active = CurveCutter
                bpy.ops.object.mode_set(mode="OBJECT")

                if BDENTAL_Props.CurveCutCloseMode == "Close Curve":
                    bpy.ops.object.mode_set(mode="EDIT")
                    bpy.ops.curve.cyclic_toggle()
                    bpy.ops.object.mode_set(mode="OBJECT")

                # bpy.ops.object.modifier_apply(apply_as="DATA", modifier="Shrinkwrap")

                bpy.context.object.data.bevel_depth = 0
                bpy.context.object.data.extrude = 2
                bpy.context.object.data.offset = 0

                bpy.ops.wm.tool_set_by_id(override, name="builtin.select")
                bpy.context.scene.tool_settings.use_snap = False
                space_data.overlay.show_outline_selected = True

                return {"FINISHED"}

        elif event.type == ("ESC"):

            if event.value == ("PRESS"):

                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.data.objects.remove(self.Cutter)
                Coll = bpy.data.collections.get("BDENTAL-4D Cutters")
                if Coll:
                    Hooks = [obj for obj in Coll.objects if "Hook" in obj.name]
                    if Hooks:
                        for obj in Hooks:
                            bpy.data.objects.remove(obj)
                CuttingTargetName = context.scene.BDENTAL_Props.CuttingTargetNameProp
                CuttingTarget = bpy.data.objects[CuttingTargetName]

                bpy.ops.object.select_all(action="DESELECT")
                CuttingTarget.select_set(True)
                bpy.context.view_layer.objects.active = CuttingTarget

                bpy.ops.wm.tool_set_by_id(name="builtin.select")
                bpy.context.scene.tool_settings.use_snap = False
                bpy.context.space_data.overlay.show_outline_selected = True
                return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):

        if not context.object or not context.object.type == "MESH":

            message = ["Please select the target mesh !"]
            update_info(message)
            sleep(2)
            update_info()
            return {"CANCELLED"}

        else:

            if context.space_data.type == "VIEW_3D":

                # Assign Model name to CuttingTarget property :
                CuttingTarget = bpy.context.object
                bpy.context.scene.BDENTAL_Props.CuttingTargetNameProp = (
                    CuttingTarget.name
                )

                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.select_all(action="DESELECT")
                CuttingTarget.select_set(True)
                bpy.context.view_layer.objects.active = CuttingTarget
                # Hide everything but model :
                # bpy.ops.object.hide_view_set(unselected=True)

                CuttingCurveAdd()
                self.Cutter = context.active_object

                context.window_manager.modal_handler_add(self)

                return {"RUNNING_MODAL"}

            else:

                self.report({"WARNING"}, "Active space must be a View3d")

                return {"CANCELLED"}


class BDENTAL_OT_CurveCutterCut(bpy.types.Operator):
    "Performe Curve Cutting Operation"

    bl_idname = "wm.bdental_curvecuttercut"
    bl_label = "CURVE CUTTER CUT"

    def execute(self, context):

        BDENTAL_Props = context.scene.BDENTAL_Props

        # Get CuttingTarget :
        CuttingTargetName = BDENTAL_Props.CuttingTargetNameProp
        CuttingTarget = bpy.data.objects.get(CuttingTargetName)
        CurveCuttersList = [
            obj
            for obj in context.visible_objects
            if obj.type == "CURVE" and obj.name.startswith("BDENTAL_Curve_Cut")
        ]

        if not CurveCuttersList or not CuttingTarget:

            message = [" Please Add Curve Cutters first !"]
            update_info(message)
            sleep(2)
            update_info()
            return {"CANCELLED"}

        else:

            # Get CurveCutter :
            bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.select_all(action="DESELECT")

            CurveMeshesList = []
            for CurveCutter in CurveCuttersList:
                bpy.ops.object.select_all(action="DESELECT")
                CurveCutter.select_set(True)
                bpy.context.view_layer.objects.active = CurveCutter

                # remove material :
                for mat_slot in CurveCutter.material_slots:
                    bpy.ops.object.material_slot_remove()

                # convert CurveCutter to mesh :
                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.convert(target="MESH")
                CurveMesh = context.object
                CurveMeshesList.append(CurveMesh)

            bpy.ops.object.select_all(action="DESELECT")
            for obj in CurveMeshesList:
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
            if len(CurveMeshesList) > 1:
                bpy.ops.object.join()

            CurveCutter = context.object

            # CurveCutter.select_set(True)
            # bpy.context.view_layer.objects.active = CurveCutter

            bpy.context.scene.tool_settings.use_snap = False
            bpy.ops.view3d.snap_cursor_to_center()

            # # Make vertex group :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.select_all(action="SELECT")
            curve_vgroup = CurveCutter.vertex_groups.new(name="curve_vgroup")
            bpy.ops.object.vertex_group_assign()
            bpy.ops.object.mode_set(mode="OBJECT")

            # select CuttingTarget :
            bpy.ops.object.select_all(action="DESELECT")
            CuttingTarget.select_set(True)
            bpy.context.view_layer.objects.active = CuttingTarget

            # delete old vertex groups :
            CuttingTarget.vertex_groups.clear()

            # deselect all vertices :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.object.mode_set(mode="OBJECT")

            # Join CurveCutter to CuttingTarget :
            CurveCutter.select_set(True)
            bpy.ops.object.join()
            CuttingTarget = context.object
            bpy.ops.object.hide_view_set(unselected=True)

            # intersect make vertex group :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.intersect()

            intersect_vgroup = CuttingTarget.vertex_groups.new(
                name="intersect_vgroup")
            CuttingTarget.vertex_groups.active_index = intersect_vgroup.index
            bpy.ops.object.vertex_group_assign()

            # OtherObjList = [obj for obj in bpy.data.objects if obj!= CuttingTarget]
            # hide all but object
            # bpy.ops.object.mode_set(mode="OBJECT")
            # bpy.ops.object.hide_view_set(unselected=True)

            # delete curve_vgroup :
            # bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="DESELECT")
            curve_vgroup = CuttingTarget.vertex_groups["curve_vgroup"]

            CuttingTarget.vertex_groups.active_index = curve_vgroup.index
            bpy.ops.object.vertex_group_select()
            bpy.ops.mesh.select_more()

            bpy.ops.mesh.delete(type="VERT")

            # # 1st methode :
            SplitSeparator(CuttingTarget=CuttingTarget)
            bpy.data.collections.remove(
                bpy.data.collections["BDENTAL-4D Cutters"])

            return {"FINISHED"}

class BDENTAL_OT_CurveCutter1_New(bpy.types.Operator):
    """mesh curve cutter tool"""

    bl_idname = "wm.bdental_curvecutter1_new"
    bl_label = "DRAW CURVE"
    bl_options = {"REGISTER", "UNDO"}

    closeCurve : BoolProperty(name="close cutting curve", default=True)

    @classmethod
    def poll(cls, context):

        base_mesh = context.object and context.object.select_get(
        ) and context.object.type == "MESH"
        return base_mesh

    def add_curve_cutter(self, context):
        Override, area3D, space3D = CtxOverride(context)
        # Prepare scene settings :
        bpy.context.scene.tool_settings.use_snap = True
        bpy.context.scene.tool_settings.snap_elements = {"FACE"}

        # ....Add Curve ....... :
        bpy.ops.curve.primitive_bezier_curve_add(Override,
                                                 radius=1, enter_editmode=False, align="CURSOR"
                                                 )
        # Set cutting_tool name :
        self.cutter = bpy.context.view_layer.objects.active
        self.cutter.name = "BDENTAL_Cutter"
        self.cutter["bdental_type"] = "curvecutter1"
        self.cutter["bdental_target"] = self.base_mesh.name

        # CurveCutter settings :
        bpy.ops.object.mode_set(Override, mode="EDIT")
        bpy.ops.curve.select_all(Override, action="DESELECT")
        self.cutter.data.splines[0].bezier_points[-1].select_control_point = True
        bpy.ops.curve.dissolve_verts(Override)
        bpy.ops.curve.select_all(Override, action="SELECT")
        bpy.ops.view3d.snap_selected_to_cursor(Override, use_offset=False)

        bpy.context.object.data.dimensions = "3D"
        bpy.context.object.data.twist_smooth = 3
        bpy.ops.curve.handle_type_set(Override, type="AUTOMATIC")
        bpy.context.object.data.bevel_depth = 0.1
        bpy.context.object.data.bevel_resolution = 6
        bpy.context.scene.tool_settings.curve_paint_settings.error_threshold = 1
        bpy.context.scene.tool_settings.curve_paint_settings.corner_angle = 0.785398
        bpy.context.scene.tool_settings.curve_paint_settings.depth_mode = "SURFACE"
        bpy.context.scene.tool_settings.curve_paint_settings.surface_offset = 0
        bpy.context.scene.tool_settings.curve_paint_settings.use_offset_absolute = True

        # Add color material :
        mat = bpy.data.materials.get(
            "Bdental_curve_cutter_mat"
        ) or bpy.data.materials.new("Bdental_curve_cutter_mat")
        mat.diffuse_color = [0.1, 1, 0.4, 1.0]
        mat.roughness = 0.3
        bpy.ops.object.mode_set(mode="OBJECT")
        self.cutter.active_material = mat

        bpy.ops.wm.tool_set_by_id(Override, name="builtin.cursor")
        space3D.overlay.show_outline_selected = False

        shrinkwrap = self.cutter.modifiers.new(
            name="Shrinkwrap", type="SHRINKWRAP")
        shrinkwrap.target = self.base_mesh
        shrinkwrap.wrap_mode = "ABOVE_SURFACE"
        shrinkwrap.use_apply_on_spline = True

        MoveToCollection(self.cutter, "Bdental Cutters")

    def add_cutter_point(self):
        Override, area3D, space3D = CtxOverride(bpy.context)

        bpy.ops.object.mode_set(Override, mode="EDIT")
        bpy.ops.curve.extrude(Override, mode="INIT")
        bpy.ops.view3d.snap_selected_to_cursor(Override, use_offset=False)
        bpy.ops.curve.select_all(Override, action="SELECT")
        bpy.ops.curve.handle_type_set(Override, type="AUTOMATIC")
        bpy.ops.curve.select_all(Override, action="DESELECT")
        points = self.cutter.data.splines[0].bezier_points[:]
        points[-1].select_control_point = True
        bpy.ops.object.mode_set(Override, mode="OBJECT")

    def del_cutter_point(self):
        Override, area3D, space3D = CtxOverride(bpy.context)
        try:
            bpy.ops.object.mode_set(Override, mode="EDIT")
            bpy.ops.curve.select_all(Override, action="DESELECT")
            points = self.cutter.data.splines[0].bezier_points[:]
            points[-1].select_control_point = True
            points = self.cutter.data.splines[0].bezier_points[:]
            if len(points) > 1:

                bpy.ops.curve.delete(Override, type="VERT")
                points = self.cutter.data.splines[0].bezier_points[:]
                bpy.ops.curve.select_all(Override, action="SELECT")
                bpy.ops.curve.handle_type_set(Override, type="AUTOMATIC")
                bpy.ops.curve.select_all(Override, action="DESELECT")
                points = self.cutter.data.splines[0].bezier_points[:]
                points[-1].select_control_point = True

            bpy.ops.object.mode_set(Override, mode="OBJECT")

        except Exception:
            pass

    def modal(self, context, event):
        
        if not event.type in ["RET", "ESC", "LEFTMOUSE", "DEL"]:
            return {"PASS_THROUGH"}
        elif event.type == "RET" and self.counter == 0:
            return {"PASS_THROUGH"}
        elif event.type == "DEL" and self.counter == 0:
            return {"PASS_THROUGH"}

        elif event.type == "ESC":
            if event.value == ("PRESS"):

                for obj in bpy.data.objects:
                    if not obj in self.start_objects:
                        bpy.data.objects.remove(obj)
                for col in bpy.data.collections:
                    if not col in self.start_collections:
                        bpy.data.collections.remove(col)

                for obj in context.visible_objects:
                    obj.hide_set(True)
                for obj in self.start_visible_objects:
                    try:
                        obj.hide_set(False)
                    except:
                        pass

                Override, area3D, space3D = CtxOverride(context)
                bpy.ops.wm.tool_set_by_id(Override, name="builtin.select")
                self.scn.tool_settings.use_snap = False
                space3D.overlay.show_outline_selected = True

                message = ["CANCELLED"]
                update_info(message)
                sleep(2)
                update_info()
                return {"CANCELLED"}

        elif event.type == ("LEFTMOUSE") and self.counter == 1:
            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}

            if event.value == ("RELEASE"):
                _is_valid = click_is_in_view3d(context, event)
                print(_is_valid)
                if _is_valid :
                    self.add_cutter_point()
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

        elif event.type == ("LEFTMOUSE") and self.counter == 0:
            
            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}

            if event.value == ("RELEASE"):
                _is_valid = click_is_in_view3d(context, event)
                print(_is_valid)
                if _is_valid :
                    self.add_curve_cutter(context)
                    self.counter += 1
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

        elif event.type == ("DEL") and self.counter == 1:
            if event.value == ("PRESS"):
                self.del_cutter_point()
                return {"RUNNING_MODAL"}

        elif event.type == "RET" and self.counter == 1:

            if event.value == ("PRESS"):
                override, area, space_data = CtxOverride(context)
                
                bpy.context.view_layer.objects.active = self.cutter
                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.select_all(action='DESELECT')
                self.cutter.select_set(True)

                if self.closeCurve :
                    bpy.ops.object.mode_set(mode="EDIT")
                    bpy.ops.curve.cyclic_toggle()
                    bpy.ops.object.mode_set(mode="OBJECT")

                bpy.ops.object.modifier_apply( modifier="Shrinkwrap")

                bpy.context.object.data.bevel_depth = 0
                bpy.context.object.data.extrude = 2
                bpy.context.object.data.offset = 0

                bpy.ops.wm.tool_set_by_id(override, name="builtin.select")
                bpy.context.scene.tool_settings.use_snap = False
                space_data.overlay.show_outline_selected = True
                update_info()
                return {"FINISHED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        if context.space_data.type == "VIEW_3D":
            self.base_mesh = context.object
            wm = context.window_manager
            return wm.invoke_props_dialog(self, width=500)

        else:

            message = ["Active space must be a View3d"]
            icon = "COLORSET_02_VEC"
            bpy.ops.bdental.message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon)
            return {"CANCELLED"}

    def execute(self, context):
        self.scn = context.scene
        self.counter = 0

        self.start_objects = bpy.data.objects[:]
        self.start_collections = bpy.data.collections[:]
        self.start_visible_objects = bpy.context.visible_objects[:]

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.base_mesh.select_set(True)
        context.view_layer.objects.active = self.base_mesh

        override, _, _ = CtxOverride(context)
        bpy.ops.wm.tool_set_by_id(override, name="builtin.cursor")
        context.window_manager.modal_handler_add(self)
        txt = ["Left click : draw curve | DEL : roll back | ESC : to cancell operation", "ENTER : to finalise"]
        update_info(txt)
        return {"RUNNING_MODAL"}
    
class BDENTAL_OT_CurveCutter1_New_Perform_Cut(bpy.types.Operator):
    "Performe CurveCutter1 Operation"

    bl_idname = "wm.bdental_curvecutter1_new_perform_cut"
    bl_label = "CUT"

    @classmethod
    def poll(cls, context):
        cutters = [obj for obj in context.scene.objects if obj.get("bdental_type") == "curvecutter1"]
        return cutters


    def execute(self, context):

        start_unvis_objects = [obj for obj in context.scene.objects if not obj in context.visible_objects]

        txt = ["Processing ..."]
        update_info(message=txt, rect_color=[0.8,0.2,0,0.7])
        override, _, _ = CtxOverride(context)
        bpy.context.scene.tool_settings.use_snap = False
        bpy.ops.view3d.snap_cursor_to_center(override)

        datadict = {}

        allcurvecutters1 = [obj for obj in context.scene.objects if obj.get("bdental_type") == "curvecutter1"]
        for c in allcurvecutters1 :
            target_name = c["bdental_target"]
            if not datadict.get(target_name):
                datadict.update({target_name : [c.name]})
            else :
                datadict[target_name].append(c.name)
        
        for target_name, curvecutters1_names in datadict.items() :
            if not bpy.data.objects.get(target_name) :
                continue
            target = bpy.data.objects.get(target_name)
            target.hide_set(False)
            target.hide_viewport = False
            target.hide_select = False
            curvecutters1 = [bpy.data.objects.get(n) for n in curvecutters1_names if bpy.data.objects.get(n)]
            cutters1_list = []
            for cc in curvecutters1 :
                bpy.context.view_layer.objects.active = cc
                cc.hide_set(False)
                cc.hide_viewport = False
                cc.hide_select = False
                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.select_all(action="DESELECT")
                cc.select_set(True)

                # remove material :
                for mat_slot in cc.material_slots:
                    bpy.ops.object.material_slot_remove()

                # convert CurveCutter to mesh :
                bpy.ops.object.convert(target="MESH")
                cutter = context.object
                cutters1_list.append(cutter)
            

            bpy.ops.object.select_all(action="DESELECT")
            for obj in cutters1_list:
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
            if len(cutters1_list) > 1:
                bpy.ops.object.join()

            cutter = context.object

            # # Make vertex group :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.select_all(action="SELECT")
            curve_vgroup = cutter.vertex_groups.new(name="curve_vgroup")
            bpy.ops.object.vertex_group_assign()
            bpy.ops.object.mode_set(mode="OBJECT")

            # select CuttingTarget :
            bpy.ops.object.select_all(action="DESELECT")
            target.select_set(True)
            bpy.context.view_layer.objects.active = target

            # delete old vertex groups :
            target.vertex_groups.clear()

            # deselect all vertices :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.object.mode_set(mode="OBJECT")

            # Join CurveCutter to CuttingTarget :
            cutter.select_set(True)
            bpy.ops.object.join()
            target = context.object
            # bpy.ops.object.hide_view_set(override, unselected=True)

            # intersect make vertex group :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.intersect()

            intersect_vgroup = target.vertex_groups.new(
                name="intersect_vgroup")
            target.vertex_groups.active_index = intersect_vgroup.index
            bpy.ops.object.vertex_group_assign()

            # OtherObjList = [obj for obj in bpy.data.objects if obj!= CuttingTarget]
            # hide all but object
            # bpy.ops.object.mode_set(mode="OBJECT")
            # bpy.ops.object.hide_view_set(unselected=True)

            # delete curve_vgroup :
            # bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="DESELECT")
            curve_vgroup = target.vertex_groups["curve_vgroup"]

            target.vertex_groups.active_index = curve_vgroup.index
            bpy.ops.object.vertex_group_select()
            bpy.ops.mesh.select_more()

            bpy.ops.mesh.delete(type="VERT")
            bpy.ops.object.hide_view_set(override, unselected=True)
            # # 1st methode :
            SplitSeparator(CuttingTarget=target)
            bpy.ops.object.mode_set(mode="OBJECT")
            

        bpy.data.collections.remove(
            bpy.data.collections["Bdental Cutters"])

        for obj in context.scene.objects :
            if obj in start_unvis_objects :
                obj.hide_set(True)
            else:
                obj.hide_set(False)
                obj.hide_viewport = False
                obj.hide_select = False

        
        txt = ["Done."]
        update_info(message=txt, rect_color=[0.0,1.0,0,0.7])
        sleep(1)
        update_info()

        return {"FINISHED"}


#######################################################################################

##################################################################


class BDENTAL_OT_AddTube(bpy.types.Operator):
    """Add Curve Tube"""

    bl_idname = "wm.bdental_add_tube"
    bl_label = "ADD TUBE"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        target = context.object and context.object.select_get(
        ) and context.object.type == "MESH"
        return target

    # def invoke(self, context, event):
    #     if context.space_data.type == "VIEW_3D":
    #         self.target = context.object
    #         bpy.ops.object.mode_set(mode="OBJECT")
    #         bpy.ops.object.select_all(action="DESELECT")
    #         self.target.select_set(True)

    #         wm = context.window_manager
    #         return wm.invoke_props_dialog(self, width=500)

    #     else:
    #         print("Must use in 3D View")
    #         return {"CANCELLED"}

    def execute(self, context):
        self.bdental_props = context.scene.BDENTAL_Props
        self.target = context.object
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.target.select_set(True)

        self.scn = context.scene
        self.counter = 0
        self.target = context.object
        override, _, _ = CtxOverride(context)
        bpy.ops.wm.tool_set_by_id(override, name="builtin.cursor")
        context.window_manager.modal_handler_add(self)
        message = ["Please draw Tube, and Press <ENTER>"]
        update_info(message)
        return {"RUNNING_MODAL"}

    def add_tube(self, context):

        # Prepare scene settings :
        bpy.ops.transform.select_orientation(orientation="GLOBAL")
        bpy.context.scene.tool_settings.use_snap = True
        bpy.context.scene.tool_settings.snap_elements = {"FACE"}
        bpy.context.scene.tool_settings.transform_pivot_point = "INDIVIDUAL_ORIGINS"

        # ....Add Curve ....... :
        bpy.ops.curve.primitive_bezier_curve_add(
            radius=1, enter_editmode=False, align="CURSOR"
        )

        self.tube = context.view_layer.objects.active
        self.tube.name = "_ADD_BDENTAL_GuideTube"
        curve = self.tube.data
        curve.name = "BDENTAL_GuideTube"

        guide_components_coll = add_collection("GUIDE Components")
        MoveToCollection(self.tube, guide_components_coll.name)

        # Tube settings :
        # dg = context.evaluated_depsgraph_get()
        # temp_obj = self.tube.evaluated_get(dg).copy()
        # temp_obj.data.

        # obj.data = temp_obj.data
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.curve.select_all(action="DESELECT")
        curve.splines[0].bezier_points[-1].select_control_point = True

        bpy.ops.curve.delete(type='VERT')

        print("tube debug")
        bpy.ops.curve.select_all(action="SELECT")
        bpy.ops.view3d.snap_selected_to_cursor()
        bpy.ops.curve.handle_type_set(type="AUTOMATIC")
        bpy.ops.object.mode_set(mode="OBJECT")

        curve.dimensions = "3D"
        curve.twist_smooth = 3
        curve.use_fill_caps = True

        curve.bevel_depth = self.bdental_props.TubeWidth
        curve.bevel_resolution = 10
        bpy.context.scene.tool_settings.curve_paint_settings.error_threshold = 1
        bpy.context.scene.tool_settings.curve_paint_settings.corner_angle = 0.785398
        bpy.context.scene.tool_settings.curve_paint_settings.depth_mode = "SURFACE"
        bpy.context.scene.tool_settings.curve_paint_settings.surface_offset = 0
        bpy.context.scene.tool_settings.curve_paint_settings.use_offset_absolute = True

        # Add color material :
        mat_tube = bpy.data.materials.get(
            "mat_BDENTAL_guide_tube") or bpy.data.materials.new("mat_BDENTAL_guide_tube")
        mat_tube.diffuse_color = [0.03, 0.20,
                                  0.14, 1.0]  # [0.1, 0.4, 1.0, 1.0]
        mat_tube.roughness = 0.3
        mat_tube.use_nodes = True
        mat_tube.node_tree.nodes["Principled BSDF"].inputs[0].default_value = [
            0.03, 0.20, 0.14, 1.0]

        self.tube.active_material = mat_tube

        bpy.ops.wm.tool_set_by_id(name="builtin.cursor")
        bpy.context.space_data.overlay.show_outline_selected = False

        bpy.ops.object.modifier_add(type="SHRINKWRAP")
        bpy.context.object.modifiers["Shrinkwrap"].target = self.target
        bpy.context.object.modifiers["Shrinkwrap"].wrap_mode = "ABOVE_SURFACE"
        bpy.context.object.modifiers["Shrinkwrap"].use_apply_on_spline = True

        os.system("cls")

    def add_tube_point(self, context, obj):
        Override, area3D, space3D = CtxOverride(bpy.context)
        context.view_layer.objects.active = obj

        bpy.ops.object.mode_set(Override, mode="EDIT")
        bpy.ops.curve.extrude(Override, mode="INIT")
        bpy.ops.view3d.snap_selected_to_cursor(Override, use_offset=False)
        bpy.ops.curve.select_all(Override, action="SELECT")
        bpy.ops.curve.handle_type_set(Override, type="AUTOMATIC")
        bpy.ops.curve.select_all(Override, action="DESELECT")
        points = obj.data.splines[0].bezier_points[:]
        points[-1].select_control_point = True
        bpy.ops.object.mode_set(Override, mode="OBJECT")

    def del_tube_point(self, context, obj):
        Override, area3D, space3D = CtxOverride(bpy.context)
        try:
            bpy.ops.object.mode_set(Override, mode="EDIT")
            bpy.ops.curve.select_all(Override, action="DESELECT")
            points = obj.data.splines[0].bezier_points[:]
            points[-1].select_control_point = True
            points = obj.data.splines[0].bezier_points[:]
            if len(points) > 1:

                bpy.ops.curve.delete(Override, type="VERT")
                points = obj.data.splines[0].bezier_points[:]
                bpy.ops.curve.select_all(Override, action="SELECT")
                bpy.ops.curve.handle_type_set(Override, type="AUTOMATIC")
                bpy.ops.curve.select_all(Override, action="DESELECT")
                points = obj.data.splines[0].bezier_points[:]
                points[-1].select_control_point = True

            bpy.ops.object.mode_set(Override, mode="OBJECT")

        except Exception:
            pass

    def cancell(self, context):
        try:
            bpy.data.objects.remove(self.tube, do_unlink=True)
        except Exception:
            pass
        Override, area3D, space3D = CtxOverride(context)
        bpy.ops.wm.tool_set_by_id(Override, name="builtin.select")
        self.scn.tool_settings.use_snap = False
        space3D.overlay.show_outline_selected = True
        bpy.ops.wm.tool_set_by_id(Override, name="builtin.select")

    def modal(self, context, event):

        if not event.type in ["RET", "ESC", "LEFTMOUSE", "DEL"]:
            return {"PASS_THROUGH"}

        elif event.type == "ESC":
            if event.value == ("PRESS"):

                self.cancell(context)

                message = ["CANCELLED"]
                update_info(message)
                sleep(2)
                update_info()
                return {"CANCELLED"}

        elif event.type == "RET" and self.counter == 0:
            if event.value == ("PRESS"):

                message = [
                    "Warning : Use left click to draw Tube, and Press <ENTER>"]
                update_info(message)

                return {"RUNNING_MODAL"}

        elif event.type == "RET" and self.counter == 1:
            if event.value == ("PRESS"):
                n = len(self.tube.data.splines[0].bezier_points[:])
                if n <= 1:
                    message = [
                        "Warning : Please draw at least 2 Tube points, and Press <ENTER>"]
                    update_info(message)

                    return {"RUNNING_MODAL"}

                else:
                    bpy.context.view_layer.objects.active = self.tube
                    bpy.ops.object.mode_set(mode="OBJECT")
                    self.tube.select_set(True)
                    if self.bdental_props.TubeCloseMode == "Close Tube":
                        bpy.ops.object.mode_set(mode="EDIT")
                        bpy.ops.curve.cyclic_toggle()
                        bpy.ops.object.mode_set(mode="OBJECT")

                    Override, area3D, space3D = CtxOverride(context)
                    bpy.ops.wm.tool_set_by_id(Override, name="builtin.select")
                    self.scn.tool_settings.use_snap = False
                    space3D.overlay.show_outline_selected = True
                    bpy.ops.wm.tool_set_by_id(Override, name="builtin.select")
                    os.system("cls")

                    message = ["Tube created"]
                    update_info(message)
                    sleep(2)
                    update_info()

                    return {"FINISHED"}

        elif event.type == ("LEFTMOUSE") and self.counter == 1:
            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}
            if event.value == ("RELEASE"):
                override, area, space_data = CtxOverride(context)
                if space_data.type == "VIEW_3D":
                    self.add_tube_point(context, self.tube)
                    os.system("cls")
                    return {"RUNNING_MODAL"}
                else:
                    return {"PASS_THROUGH"}

        elif event.type == ("LEFTMOUSE") and self.counter == 0:
            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}
            if event.value == ("RELEASE"):
                override, area, space_data = CtxOverride(context)
                if space_data.type == "VIEW_3D":
                    self.add_tube(context)
                    self.counter += 1
                    return {"RUNNING_MODAL"}
                else:
                    return {"PASS_THROUGH"}

        elif event.type == ("DEL") and self.counter == 1:
            if event.value == ("PRESS"):
                self.del_tube_point(context, self.tube)
                os.system("cls")

            return {"RUNNING_MODAL"}

        return {"RUNNING_MODAL"}


#########################################################################################
# CurveCutter_02
class BDENTAL_OT_CurveCutterAdd2(bpy.types.Operator):
    """add curve cutter v2"""

    bl_idname = "wm.bdental_curvecutteradd2"
    bl_label = "CURVE CUTTER ADD"
    bl_options = {"REGISTER", "UNDO"}

    closeCurve : BoolProperty(name="close cutting curve", default=True)

    @classmethod
    def poll(cls, context):

        base_mesh = context.object and context.object.select_get(
        ) and context.object.type == "MESH"
        return base_mesh
    
    def add_curve_cutter(self, context):
        Override, area3D, space3D = CtxOverride(context)
        # Prepare scene settings :
        bpy.context.scene.tool_settings.use_snap = True
        bpy.context.scene.tool_settings.snap_elements = {"FACE"}

        # ....Add Curve ....... :
        bpy.ops.curve.primitive_bezier_curve_add(
            radius=1, enter_editmode=False, align="CURSOR"
        )
        # Set cutting_tool name :
        self.cutter = bpy.context.view_layer.objects.active
        self.cutter.name = "BDENTAL_Cutter"
        self.cutter["bdental_type"] = "curvecutter2"
        self.cutter["bdental_target"] = self.base_mesh.name
        self.cutter["bdental_close_curve"] = self.closeCurve

        # CurveCutter settings :
        bpy.ops.object.mode_set(Override, mode="EDIT")
        bpy.ops.curve.select_all(Override, action="DESELECT")
        self.cutter.data.splines[0].bezier_points[-1].select_control_point = True
        bpy.ops.curve.dissolve_verts(Override)
        bpy.ops.curve.select_all(Override, action="SELECT")
        bpy.ops.view3d.snap_selected_to_cursor(Override, use_offset=False)

        self.cutter.data.dimensions = "3D"
        self.cutter.data.twist_smooth = 3
        bpy.ops.curve.handle_type_set(Override, type="AUTOMATIC")
        self.cutter.data.bevel_depth = 0.1
        self.cutter.data.bevel_resolution = 2
        bpy.context.scene.tool_settings.curve_paint_settings.error_threshold = 1
        bpy.context.scene.tool_settings.curve_paint_settings.corner_angle = 0.785398
        bpy.context.scene.tool_settings.curve_paint_settings.depth_mode = "SURFACE"
        bpy.context.scene.tool_settings.curve_paint_settings.surface_offset = 0
        bpy.context.scene.tool_settings.curve_paint_settings.use_offset_absolute = True

        # Add color material :
        mat = bpy.data.materials.get(
            "Bdental_curve_cutter_mat"
        ) or bpy.data.materials.new("Bdental_curve_cutter_mat")
        mat.diffuse_color = [0.1, 0.4, 1.0, 1.0]
        mat.roughness = 0.3
        bpy.ops.object.mode_set(mode="OBJECT")
        self.cutter.active_material = mat

        bpy.ops.wm.tool_set_by_id(Override, name="builtin.cursor")
        space3D.overlay.show_outline_selected = False

        shrinkwrap = self.cutter.modifiers.new(
            name="Shrinkwrap", type="SHRINKWRAP")
        shrinkwrap.target = self.base_mesh
        shrinkwrap.wrap_mode = "ABOVE_SURFACE"
        shrinkwrap.use_apply_on_spline = True

        MoveToCollection(self.cutter, "Bdental Cutters")
        self.cutters_collection = bpy.data.collections.get("Bdental Cutters")

    def add_cutter_point(self):
        Override, area3D, space3D = CtxOverride(bpy.context)

        bpy.ops.object.mode_set(Override, mode="EDIT")
        bpy.ops.curve.extrude(Override, mode="INIT")
        bpy.ops.view3d.snap_selected_to_cursor(Override, use_offset=False)
        bpy.ops.curve.select_all(Override, action="SELECT")
        bpy.ops.curve.handle_type_set(Override, type="AUTOMATIC")
        bpy.ops.curve.select_all(Override, action="DESELECT")
        points = self.cutter.data.splines[0].bezier_points[:]
        points[-1].select_control_point = True
        bpy.ops.object.mode_set(Override, mode="OBJECT")

    def del_cutter_point(self):
        Override, area3D, space3D = CtxOverride(bpy.context)
        try:
            bpy.ops.object.mode_set(Override, mode="EDIT")
            bpy.ops.curve.select_all(Override, action="DESELECT")
            points = self.cutter.data.splines[0].bezier_points[:]
            points[-1].select_control_point = True
            points = self.cutter.data.splines[0].bezier_points[:]
            if len(points) > 1:

                bpy.ops.curve.delete(Override, type="VERT")
                points = self.cutter.data.splines[0].bezier_points[:]
                bpy.ops.curve.select_all(Override, action="SELECT")
                bpy.ops.curve.handle_type_set(Override, type="AUTOMATIC")
                bpy.ops.curve.select_all(Override, action="DESELECT")
                points = self.cutter.data.splines[0].bezier_points[:]
                points[-1].select_control_point = True

            bpy.ops.object.mode_set(Override, mode="OBJECT")

        except Exception:
            pass

    def AddCurveSphere(self, context, Name, i, CollName):
        bpy.ops.object.select_all(action="DESELECT")
        bezier_points = self.cutter.data.splines[0].bezier_points[:]
        Bpt = bezier_points[i]
        loc = self.cutter.matrix_world @ Bpt.co
        AddMarkupPoint(
            name=Name, color=(0, 1, 0, 1), loc=loc, Diameter=0.5, CollName=CollName
        )
        Hook = context.object
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        Hook.select_set(True)
        self.cutter.select_set(True)
        bpy.context.view_layer.objects.active = self.cutter
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.curve.select_all(action="DESELECT")
        bezier_points = self.cutter.data.splines[0].bezier_points[:]
        Bpt = bezier_points[i]
        Bpt.select_control_point = True
        bpy.ops.object.hook_add_selob(use_bone=False)
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.cutter.select_set(True)
        bpy.context.view_layer.objects.active = self.cutter

        return Hook

    def modal(self, context, event):
        
        if not event.type in ["RET", "ESC", "LEFTMOUSE", "DEL"]:
            return {"PASS_THROUGH"}
        elif event.type == "RET" and self.counter == 0:
            return {"PASS_THROUGH"}
        elif event.type == "DEL" and self.counter == 0:
            return {"PASS_THROUGH"}

        elif event.type == "ESC":
            if event.value == ("PRESS"):

                for obj in bpy.data.objects:
                    if not obj in self.start_objects:
                        bpy.data.objects.remove(obj)
                for col in bpy.data.collections:
                    if not col in self.start_collections:
                        bpy.data.collections.remove(col)

                for obj in context.visible_objects:
                    obj.hide_set(True)
                for obj in self.start_visible_objects:
                    try:
                        obj.hide_set(False)
                    except:
                        pass

                Override, area3D, space3D = CtxOverride(context)
                bpy.ops.wm.tool_set_by_id(Override, name="builtin.select")
                self.scn.tool_settings.use_snap = False
                space3D.overlay.show_outline_selected = True

                message = ["CANCELLED"]
                update_info(message)
                sleep(2)
                update_info()
                return {"CANCELLED"}

        elif event.type == ("LEFTMOUSE") and self.counter == 1:
            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}

            if event.value == ("RELEASE"):
                _is_valid = click_is_in_view3d(context, event)
                print(_is_valid)
                if _is_valid :
                    self.add_cutter_point()
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

        elif event.type == ("LEFTMOUSE") and self.counter == 0:
            
            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}

            if event.value == ("RELEASE"):
                _is_valid = click_is_in_view3d(context, event)
                print(_is_valid)
                if _is_valid :
                    self.add_curve_cutter(context)
                    self.counter += 1
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

        elif event.type == ("DEL") and self.counter == 1:
            if event.value == ("PRESS"):
                self.del_cutter_point()
                return {"RUNNING_MODAL"}

        elif event.type == "RET" and self.counter == 1:

            if event.value == ("PRESS"):
                override, area, space_data = CtxOverride(context)
                
                bpy.context.view_layer.objects.active = self.cutter
                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.select_all(action='DESELECT')
                self.cutter.select_set(True)

                if self.closeCurve :
                    bpy.ops.object.mode_set(mode="EDIT")
                    bpy.ops.curve.cyclic_toggle()
                    bpy.ops.object.mode_set(mode="OBJECT")

                bpy.ops.object.modifier_apply( modifier="Shrinkwrap")

                bpy.ops.wm.tool_set_by_id(override, name="builtin.select")
                bpy.context.scene.tool_settings.use_snap = False
                space_data.overlay.show_outline_selected = True

                bezier_points = self.cutter.data.splines[0].bezier_points[:]
                Hooks = [obj for obj in self.cutters_collection.objects if "Hook" in obj.name]
                for i in range(len(bezier_points)):
                    Hook = self.AddCurveSphere(
                        context,
                        Name=f"Hook_{i}",
                        i=i,
                        CollName="Bdental Cutters",
                    )
                    Hooks.append(Hook)
                # print(Hooks)
                for h in Hooks:
                    for o in Hooks:
                        if not o is h:
                            delta = o.location - h.location

                            distance = sqrt(
                                delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2
                            )
                            if distance <= 0.5:
                                # center = h.location + (delta/2)
                                o.location = h.location #= center
                bpy.context.space_data.overlay.show_relationship_lines = False
                bpy.context.scene.tool_settings.use_snap = True
                bpy.context.scene.tool_settings.snap_elements = {"FACE"}
                bpy.context.scene.tool_settings.snap_target = "CENTER"
                bpy.ops.object.select_all(action="DESELECT")

                self.cutter.hide_select = True
                update_info()
                return {"FINISHED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        if context.space_data.type == "VIEW_3D":
            self.base_mesh = context.object
            wm = context.window_manager
            return wm.invoke_props_dialog(self, width=500)

        else:

            message = ["Active space must be a View3d"]
            icon = "COLORSET_02_VEC"
            bpy.ops.bdental.message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon)
            return {"CANCELLED"}

    def execute(self, context):
        self.scn = context.scene
        self.counter = 0

        self.start_objects = bpy.data.objects[:]
        self.start_collections = bpy.data.collections[:]
        self.start_visible_objects = bpy.context.visible_objects[:]

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.base_mesh.select_set(True)
        context.view_layer.objects.active = self.base_mesh

        override, _, _ = CtxOverride(context)
        bpy.ops.wm.tool_set_by_id(override, name="builtin.cursor")
        context.window_manager.modal_handler_add(self)
        txt = ["Left click : draw curve | DEL : roll back | ESC : to cancell operation", "ENTER : to finalise"]
        update_info(txt)
        return {"RUNNING_MODAL"}
################################################################################

class BDENTAL_OT_CurveCutter2_Cut_New(bpy.types.Operator):
    "Performe CurveCutter2 Operation"

    bl_idname = "wm.bdental_curvecutter2_cut_new"
    bl_label = "Cut"

    Resolution: IntProperty(
        name="Cut Resolution",
        description="Cutting curve Resolution",
        default=2,
    )

    @classmethod
    def poll(cls, context):
        cutters = [obj for obj in context.scene.objects if obj.get("bdental_type") == "curvecutter2"]
        return cutters

    def split_cut(self, context):
        bpy.context.view_layer.objects.active = self.target
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        intersect_vgroup = self.target.vertex_groups["intersect_vgroup"]
        self.target.vertex_groups.active_index = intersect_vgroup.index
        bpy.ops.object.vertex_group_select()

        bpy.ops.mesh.edge_split(type='VERT')

        # Separate by loose parts :
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.mesh.separate(type="LOOSE")

        for obj in bpy.context.visible_objects:
            if not obj.data.polygons or len(obj.data.polygons) < 10:
                bpy.data.objects.remove(obj)
            else :
                context.view_layer.objects.active = obj

                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.remove_doubles()
                bpy.ops.mesh.select_all(action="DESELECT")

   
    def execute(self, context):

        start = tpc()
        start_unvis_objects = [obj for obj in context.scene.objects if not obj in context.visible_objects]

        txt = ["Processing ..."]
        update_info(message=txt, rect_color=[0.8,0.2,0,0.7])
        override, _, _ = CtxOverride(context)
        

        datadict = {}

        allcurvecutters2 = [obj for obj in context.scene.objects if obj.get("bdental_type") == "curvecutter2"]
        for c in allcurvecutters2 :
            target_name = c["bdental_target"]
            if not datadict.get(target_name):
                datadict.update({target_name : [c.name]})
            else :
                datadict[target_name].append(c.name)
        
        for target_name, curvecutters2_names in datadict.items() :
            if not bpy.data.objects.get(target_name) :
                continue
            self.target = target = bpy.data.objects.get(target_name)
            target.hide_set(False)
            target.hide_viewport = False
            target.hide_select = False
            curvecutters2 = [bpy.data.objects.get(n) for n in curvecutters2_names if bpy.data.objects.get(n)]
            cutters2_list = []
            for cc in curvecutters2 :
                context.view_layer.objects.active = cc
                cc.data.bevel_depth = 0
                cc.data.resolution_u = self.Resolution
                cc.hide_set(False)
                cc.hide_viewport = False
                cc.hide_select = False
                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.select_all(action="DESELECT")
                cc.select_set(True)
                
                # hook_modifiers = [
                # mod.name for mod in cc.modifiers if "Hook" in mod.name
                # ]
                # for mod in hook_modifiers:
                #     bpy.ops.object.modifier_apply(modifier=mod)

                # cc.data.bevel_depth = 0
                # cc.data.resolution_u = self.Resolution
                # bpy.ops.object.modifier_apply(modifier="Shrinkwrap")

                # convert CurveCutter to mesh :
                bdental_close_curve = cc["bdental_close_curve"]
                bpy.ops.object.convert(target="MESH")
                cutter = context.object
                cutter["bdental_close_curve"] = bdental_close_curve
                cutters2_list.append(cutter)
            

           
            bpy.ops.object.select_all(action="DESELECT")
            target.select_set(True)
            context.view_layer.objects.active = target
            me = target.data
            # initiate a KDTree :
            size = len(me.vertices)
            kd = kdtree.KDTree(size)

            for v_id, v in enumerate(me.vertices):
                kd.insert(v.co, v_id)

            kd.balance()

            Loop = []
            for cc_mesh in cutters2_list:

                CutterCoList = [
                    target.matrix_world.inverted() @ cc_mesh.matrix_world @ v.co
                    for v in cc_mesh.data.vertices
                ]
                Closest_VIDs = [
                    kd.find(CutterCoList[i])[1] for i in range(len(CutterCoList))
                ]
                if cc_mesh["bdental_close_curve"]:
                    CloseState = True
                else:
                    CloseState = False
                
                # CutLine = ShortestPath(
                #     target, Closest_VIDs, close=CloseState)
                CutLine = ConnectPath(
                    target, Closest_VIDs, close=CloseState)

                
                Loop.extend(CutLine)

            # print(Loop)                                                                                                                                                                                                       
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="DESELECT")

            bpy.ops.object.mode_set(mode="OBJECT")
            for Id in Loop:
                me.vertices[Id].select = True

            
            for vg in target.vertex_groups :
                target.vertex_groups.remove(vg)

            bpy.ops.object.mode_set(mode="EDIT")
            vg = target.vertex_groups.new(name="intersect_vgroup")
            bpy.ops.object.vertex_group_assign()
            bpy.ops.object.mode_set(mode="OBJECT")
            

            # print("Split Cut Line...")

            # Split :
            self.split_cut(context)
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="DESELECT")


        print("Remove Cutter tool...")
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
        for obj in context.visible_objects:
            if obj.type == "MESH" and len(obj.data.polygons) <= 10:
                bpy.data.objects.remove(obj)
        col = bpy.data.collections["Bdental Cutters"]
        for obj in col.objects:
            bpy.data.objects.remove(obj)
        bpy.data.collections.remove(col)

        finish = tpc()

        context.scene.tool_settings.use_snap = False
        bpy.ops.view3d.snap_cursor_to_center(override)

        print("finished in : ", finish - start, "secondes")
        update_info()
        return {"FINISHED"}

    def invoke(self, context, event):

        wm = context.window_manager
        return wm.invoke_props_dialog(self)


class BDENTAL_OT_CurveCutter2_ShortPath(bpy.types.Operator):
    "Shortpath Curve Cutting tool"

    bl_idname = "wm.bdental_curvecutter2_shortpath"
    bl_label = "ShortPath"

    Resolution: IntProperty(
        name="Cut Resolution",
        description="Cutting curve Resolution",
        default=3,
    )

    def execute(self, context):

        start = tpc()
        BDENTAL_Props = bpy.context.scene.BDENTAL_Props
        ###########################################################################

        CuttingTarget = self.CuttingTarget
        CurveCuttersList = self.CurveCuttersList

        if bpy.context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        CuttingTarget.hide_select = False
        # delete old vertex groups :
        CuttingTarget.vertex_groups.clear()

        CurveMeshesList = []
        for CurveCutter in CurveCuttersList:
            CurveCutter.hide_select = False
            bpy.ops.object.select_all(action="DESELECT")
            CurveCutter.select_set(True)
            bpy.context.view_layer.objects.active = CurveCutter

            HookModifiers = [
                mod.name for mod in CurveCutter.modifiers if "Hook" in mod.name
            ]
            for mod in HookModifiers:
                bpy.ops.object.modifier_apply(modifier=mod)

            CurveCutter.data.bevel_depth = 0
            CurveCutter.data.resolution_u = self.Resolution
            bpy.ops.object.modifier_apply(modifier="Shrinkwrap")

            bpy.ops.object.convert(target="MESH")
            CurveCutter = context.object
            bpy.ops.object.modifier_add(type="SHRINKWRAP")
            CurveCutter.modifiers["Shrinkwrap"].target = CuttingTarget
            bpy.ops.object.convert(target="MESH")

            CurveMesh = context.object
            CurveMeshesList.append(CurveMesh)

        bpy.ops.object.select_all(action="DESELECT")
        CuttingTarget.select_set(True)
        bpy.context.view_layer.objects.active = CuttingTarget
        me = CuttingTarget.data
        # initiate a KDTree :
        size = len(me.vertices)
        kd = kdtree.KDTree(size)

        for v_id, v in enumerate(me.vertices):
            kd.insert(v.co, v_id)

        kd.balance()
        Loop = []
        for CurveCutter in CurveMeshesList:

            CutterCoList = [
                CuttingTarget.matrix_world.inverted() @ CurveCutter.matrix_world @ v.co
                for v in CurveCutter.data.vertices
            ]
            Closest_VIDs = [
                kd.find(CutterCoList[i])[1] for i in range(len(CutterCoList))
            ]
            print("Get closest verts list done")
            if BDENTAL_Props.CurveCutCloseMode == "Close Curve":
                CloseState = True
            else:
                CloseState = False
            CutLine = ShortestPath(
                CuttingTarget, Closest_VIDs, close=CloseState)
            Loop.extend(CutLine)

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")

        bpy.ops.object.mode_set(mode="OBJECT")
        for Id in Loop:
            me.vertices[Id].select = True

        bpy.ops.object.mode_set(mode="EDIT")
        vg = CuttingTarget.vertex_groups.new(name="intersect_vgroup")
        bpy.ops.object.vertex_group_assign()

        print("Shrinkwrap Modifier...")
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        for CurveCutter in CurveMeshesList:
            CurveCutter.select_set(True)
            bpy.context.view_layer.objects.active = CurveCutter

        if len(CurveMeshesList) > 1:
            bpy.ops.object.join()

        CurveCutter = context.object
        print("CurveCutter", CurveCutter)
        print("CuttingTarget", CuttingTarget)
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        CuttingTarget.select_set(True)
        bpy.context.view_layer.objects.active = CuttingTarget

        bpy.ops.object.modifier_add(type="SHRINKWRAP")
        CuttingTarget.modifiers["Shrinkwrap"].wrap_method = "NEAREST_VERTEX"
        CuttingTarget.modifiers["Shrinkwrap"].vertex_group = vg.name
        CuttingTarget.modifiers["Shrinkwrap"].target = CurveCutter
        bpy.ops.object.modifier_apply(modifier="Shrinkwrap")

        print("Relax Cut Line...")
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.looptools_relax(
            input="selected", interpolation="cubic", iterations="3", regular=True
        )

        print("Split Cut Line...")

        # Split :
        SplitSeparator(CuttingTarget=CuttingTarget)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")

        print("Remove Cutter tool...")
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
        for obj in context.visible_objects:
            if obj.type == "MESH" and len(obj.data.polygons) <= 10:
                bpy.data.objects.remove(obj)
        col = bpy.data.collections["BDENTAL-4D Cutters"]
        for obj in col.objects:
            bpy.data.objects.remove(obj)
        bpy.data.collections.remove(col)

        finish = tpc()
        print("finished in : ", finish - start, "secondes")

        return {"FINISHED"}

    def invoke(self, context, event):

        BDENTAL_Props = bpy.context.scene.BDENTAL_Props

        # Get CuttingTarget :
        CuttingTargetName = BDENTAL_Props.CuttingTargetNameProp
        self.CuttingTarget = bpy.data.objects.get(CuttingTargetName)

        # Get CurveCutter :
        self.CurveCuttersList = [
            obj for obj in bpy.data.objects if "BDENTAL_Curve_Cut2" in obj.name
        ]

        if not self.CurveCuttersList or not self.CuttingTarget:

            message = [" Please Add Curve Cutters first !"]
            update_info(message)
            sleep(2)
            update_info()
            return {"CANCELLED"}
        else:

            wm = context.window_manager
            return wm.invoke_props_dialog(self)


# CurveCutter_03
class BDENTAL_OT_CurveCutterAdd3(bpy.types.Operator):
    """description of this Operator"""

    bl_idname = "wm.bdental_curvecutteradd3"
    bl_label = "CURVE CUTTER ADD"
    bl_options = {"REGISTER", "UNDO"}

    def modal(self, context, event):

        BDENTAL_Props = context.scene.BDENTAL_Props

        if not event.type in {
            "DEL",
            "LEFTMOUSE",
            "RET",
            "ESC",
        }:
            # allow navigation

            return {"PASS_THROUGH"}

        elif event.type == ("DEL"):
            if event.value == ("PRESS"):

                DeleteLastCurvePoint()

            return {"RUNNING_MODAL"}

        elif event.type == ("LEFTMOUSE"):

            if event.value == ("PRESS"):

                return {"PASS_THROUGH"}

            if event.value == ("RELEASE"):

                ExtrudeCurvePointToCursor(context, event)

        elif event.type == "RET":

            if event.value == ("PRESS"):
                CurveCutterName = BDENTAL_Props.CurveCutterNameProp
                CurveCutter = bpy.data.objects[CurveCutterName]
                CurveCutter.select_set(True)
                bpy.context.view_layer.objects.active = CurveCutter

                bpy.ops.object.mode_set(mode="OBJECT")

                if BDENTAL_Props.CurveCutCloseMode == "Close Curve":
                    bpy.ops.object.mode_set(mode="EDIT")
                    bpy.ops.curve.cyclic_toggle()
                    bpy.ops.object.mode_set(mode="OBJECT")

                # bpy.ops.object.modifier_apply(apply_as="DATA", modifier="Shrinkwrap")

                bpy.ops.wm.tool_set_by_id(name="builtin.select")
                # bpy.context.scene.tool_settings.use_snap = False
                bpy.context.space_data.overlay.show_outline_selected = True

                bezier_points = CurveCutter.data.splines[0].bezier_points[:]
                for i in range(len(bezier_points)):
                    AddCurveSphere(
                        Name=f"Hook_{i}",
                        Curve=CurveCutter,
                        i=i,
                        CollName="BDENTAL-4D Cutters",
                    )
                bpy.context.space_data.overlay.show_relationship_lines = False

                return {"FINISHED"}

        elif event.type == ("ESC"):

            if event.value == ("PRESS"):

                CurveCutterName = bpy.context.scene.BDENTAL_Props.CurveCutterNameProp
                CurveCutter = bpy.data.objects[CurveCutterName]
                bpy.ops.object.mode_set(mode="OBJECT")

                bpy.ops.object.select_all(action="DESELECT")
                CurveCutter.select_set(True)
                bpy.context.view_layer.objects.active = CurveCutter
                bpy.ops.object.delete(use_global=False, confirm=False)

                CuttingTargetName = context.scene.BDENTAL_Props.CuttingTargetNameProp
                CuttingTarget = bpy.data.objects[CuttingTargetName]

                bpy.ops.object.select_all(action="DESELECT")
                CuttingTarget.select_set(True)
                bpy.context.view_layer.objects.active = CuttingTarget

                bpy.ops.wm.tool_set_by_id(name="builtin.select")
                bpy.context.scene.tool_settings.use_snap = True
                bpy.context.space_data.overlay.show_outline_selected = True
                bpy.context.scene.tool_settings.snap_target = "CENTER"
                bpy.context.scene.tool_settings.snap_elements = {"FACE"}
                bpy.context.space_data.overlay.show_relationship_lines = False

                return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):

        if bpy.context.selected_objects == []:

            message = ["Please select the target object !"]
            icon = "COLORSET_02_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        else:

            if context.space_data.type == "VIEW_3D":

                # Assign Model name to CuttingTarget property :
                CuttingTarget = bpy.context.view_layer.objects.active
                bpy.context.scene.BDENTAL_Props.CuttingTargetNameProp = (
                    CuttingTarget.name
                )

                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.hide_view_clear()
                bpy.ops.object.select_all(action="DESELECT")

                # for obj in bpy.data.objects:
                #     if "CuttingCurve" in obj.name:
                #         obj.select_set(True)
                #         bpy.ops.object.delete(use_global=False, confirm=False)

                bpy.ops.object.select_all(action="DESELECT")
                CuttingTarget.select_set(True)
                bpy.context.view_layer.objects.active = CuttingTarget
                # Hide everything but model :
                bpy.ops.object.hide_view_set(unselected=True)

                CuttingCurveAdd2()

                context.window_manager.modal_handler_add(self)

                return {"RUNNING_MODAL"}

            else:

                self.report({"WARNING"}, "Active space must be a View3d")

                return {"CANCELLED"}


class BDENTAL_OT_CurveCutterCut3(bpy.types.Operator):
    "Performe Curve Cutting Operation"

    bl_idname = "wm.bdental_curvecuttercut3"
    bl_label = "CURVE CUTTER CUT"

    def execute(self, context):

        # Get CuttingTarget :
        CuttingTargetName = bpy.context.scene.BDENTAL_Props.CuttingTargetNameProp
        CuttingTarget = bpy.data.objects.get(CuttingTargetName)

        # Get CurveCutter :
        bpy.ops.object.select_all(action="DESELECT")

        CurveCuttersList = [
            obj
            for obj in context.visible_objects
            if obj.type == "CURVE" and obj.name.startswith("BDENTAL_Curve_Cut")
        ]

        if not CurveCuttersList:

            message = [
                " Can't find curve Cutters ",
                "Please ensure curve Cutters are not hiden !",
            ]

            icon = "COLORSET_02_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        if CurveCuttersList:
            CurveMeshesList = []
            for CurveCutter in CurveCuttersList:
                bpy.ops.object.select_all(action="DESELECT")
                CurveCutter.select_set(True)
                bpy.context.view_layer.objects.active = CurveCutter

                # remove material :
                for _ in CurveCutter.material_slots:
                    bpy.ops.object.material_slot_remove()

                # Change CurveCutter setting   :
                CurveCutter.data.bevel_depth = 0
                CurveCutter.data.resolution_u = 6

                # Add shrinkwrap modif outside :
                bpy.ops.object.modifier_add(type="SHRINKWRAP")
                CurveCutter.modifiers["Shrinkwrap"].use_apply_on_spline = True
                CurveCutter.modifiers["Shrinkwrap"].target = CuttingTarget
                CurveCutter.modifiers["Shrinkwrap"].offset = 0.5
                CurveCutter.modifiers["Shrinkwrap"].wrap_mode = "OUTSIDE"

                # duplicate curve :
                bpy.ops.object.duplicate_move()
                CurveCutterDupli = context.object
                CurveCutterDupli.modifiers["Shrinkwrap"].wrap_mode = "INSIDE"
                CurveCutterDupli.modifiers["Shrinkwrap"].offset = 0.8

                IntOut = []
                for obj in [CurveCutter, CurveCutterDupli]:
                    # convert CurveCutter to mesh :
                    bpy.ops.object.mode_set(mode="OBJECT")
                    bpy.ops.object.select_all(action="DESELECT")
                    obj.select_set(True)
                    bpy.context.view_layer.objects.active = obj
                    bpy.ops.object.convert(target="MESH")
                    CurveMesh = context.object
                    IntOut.append(CurveMesh)

                bpy.ops.object.select_all(action="DESELECT")
                for obj in IntOut:
                    obj.select_set(True)
                    bpy.context.view_layer.objects.active = obj
                bpy.ops.object.join()
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.bridge_edge_loops()
                bpy.ops.object.mode_set(mode="OBJECT")
                CurveMeshesList.append(context.object)

            bpy.ops.object.select_all(action="DESELECT")
            for obj in CurveMeshesList:
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj

            if len(CurveMeshesList) > 1:
                bpy.ops.object.join()

            CurveCutter = context.object

            CurveCutter.select_set(True)
            bpy.context.view_layer.objects.active = CurveCutter

            bpy.context.scene.tool_settings.use_snap = False
            bpy.ops.view3d.snap_cursor_to_center()

            # # Make vertex group :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.context.tool_settings.mesh_select_mode = (True, False, False)
            bpy.ops.mesh.select_all(action="SELECT")
            curve_vgroup = CurveCutter.vertex_groups.new(name="curve_vgroup")
            bpy.ops.object.vertex_group_assign()
            bpy.ops.object.mode_set(mode="OBJECT")

            # select CuttingTarget :
            bpy.ops.object.select_all(action="DESELECT")
            CuttingTarget.select_set(True)
            bpy.context.view_layer.objects.active = CuttingTarget

            # delete old vertex groups :
            CuttingTarget.vertex_groups.clear()

            # deselect all vertices :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.object.mode_set(mode="OBJECT")

            ###############################################################

            # Join CurveCutter to CuttingTarget :
            CurveCutter.select_set(True)
            bpy.ops.object.join()
            bpy.ops.object.hide_view_set(unselected=True)

            # intersect make vertex group :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.intersect()

            intersect_vgroup = CuttingTarget.vertex_groups.new(
                name="intersect_vgroup")
            CuttingTarget.vertex_groups.active_index = intersect_vgroup.index
            bpy.ops.object.vertex_group_assign()

            # # delete curve_vgroup :
            # bpy.ops.object.mode_set(mode="EDIT")
            # bpy.ops.mesh.select_all(action="DESELECT")
            # curve_vgroup = CuttingTarget.vertex_groups["curve_vgroup"]

            # CuttingTarget.vertex_groups.active_index = curve_vgroup.index
            # bpy.ops.object.vertex_group_select()
            # bpy.ops.mesh.delete(type="FACE")

            # bpy.ops.ed.undo_push()
            # # 1st methode :
            # SplitSeparator(CuttingTarget=CuttingTarget)

            # for obj in context.visible_objects:
            #     if len(obj.data.polygons) <= 10:
            #         bpy.data.objects.remove(obj)
            # for obj in context.visible_objects:
            #     if obj.name.startswith("Hook"):
            #         bpy.data.objects.remove(obj)

            # print("Cutting done with first method")

            return {"FINISHED"}


#######################################################################################
# Square cut modal operator :


class BDENTAL_OT_AddSquareCutter(bpy.types.Operator):
    """Square Cutting Tool add"""

    bl_idname = "wm.bdental_add_square_cutter"
    bl_label = "Square Cut"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == "MESH" and context.object.select_get()

    def add_square_cutter(self, context):

        override, area, space_data = CtxOverride(context)
        context.view_layer.objects.active = self.target
        bpy.ops.object.mode_set(mode="OBJECT")

        mesh_center = np.mean(
            [self.target.matrix_world @ v.co for v in self.target.data.vertices], axis=0)
        view_rotation_4x4 = space_data.region_3d.view_rotation.to_matrix().to_4x4()
        dim = max(self.target.dimensions) * 1.5
        # Add cube :
        bpy.ops.mesh.primitive_cube_add(size=dim)

        square_cutter = context.object
        for obj in bpy.data.objects:
            if "SquareCutter" in obj.name:
                bpy.data.objects.remove(obj)
        square_cutter.name = "SquareCutter"

        # Reshape and align cube :

        square_cutter.matrix_world = view_rotation_4x4

        square_cutter.location = mesh_center

        square_cutter.display_type = "WIRE"
        square_cutter.scale[1] = 0.5
        square_cutter.scale[2] = 2

        # Subdivide cube 10 iterations 3 times :
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.subdivide(number_cuts=100)

        # Make cube normals consistent :
        bpy.ops.object.mode_set(mode="OBJECT")

        return square_cutter.name

    def cut(self, context):
        cutting_mode = context.scene.BDENTAL_Props.cutting_mode
        bpy.context.view_layer.objects.active = self.target
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.target.select_set(True)

        # Add Boolean Modifier :
        bool_modif = self.target.modifiers.new(name="Boolean", type="BOOLEAN")
        bool_modif.object = self.square_cutter

        # Apply boolean modifier :
        if cutting_mode == "Cut inner":
            bool_modif.operation = "DIFFERENCE"

        if cutting_mode == "Keep inner":
            bool_modif.operation = "INTERSECT"

        bpy.ops.object.convert(target="MESH")

        bpy.ops.object.select_all(action="DESELECT")
        self.square_cutter.select_set(True)
        bpy.context.view_layer.objects.active = self.square_cutter

    def modal(self, context, event):
        if not event.type in {"RET", "ESC"}:
            return {"PASS_THROUGH"}
        if event.type == "RET":
            if event.value == ("PRESS"):
                self.square_cutter = bpy.data.objects.get(
                    self.square_cutter_name)
                if not self.square_cutter:
                    message = ["Cancelled, No Square Cutter Found ..."]
                    update_info(message)
                    sleep(2)
                    update_info()
                    return {"CANCELLED"}
                else:
                    txt = ["Processing ..."]
                    update_info(message=txt, rect_color=[0.8,0.2,0,0.7])
                    self.cut(context)
                    txt = ["Square Cut Done ..."]
                    update_info(message=txt, rect_color=[0,1,0,0.7])
                    sleep(2)
                    txt = ["ENTER : to make another cut / ESC : to resume."]
                    update_info(message=txt)
                    return {"RUNNING_MODAL"}

        elif event.type == ("ESC"):
            message = ["Finished"]
            try:
                bpy.data.objects.remove(self.square_cutter)
            except:
                pass

            for obj in self.start_visible_objects:
                obj.hide_set(False)
            update_info(message)
            sleep(2)
            update_info()
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def execute(self, context):

        # if context.space_data.type == "VIEW_3D":
        self.target = context.object
        bpy.ops.object.mode_set(mode="OBJECT")
        self.start_visible_objects = context.visible_objects[:]

        for obj in self.start_visible_objects:
            if not obj is self.target:
                obj.hide_set(True)

        self.square_cutter_name = self.add_square_cutter(context)

        message = [
            " Press <ENTER> to Cut, <ESC> to exit",
        ]
        update_info(message)

        context.window_manager.modal_handler_add(self)

        return {"RUNNING_MODAL"}

        # else:

        #     self.report({"WARNING"}, "Active space must be a View3d")

        #     return {"CANCELLED"}


#######################################################################################
# Square cut confirm operator :


class BDENTAL_OT_square_cut_confirm(bpy.types.Operator):
    """confirm Square Cut operation"""

    bl_idname = "wm.bdental_square_cut_confirm"
    bl_label = "Tirm"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        if not context.scene["square_cutter_target"]:

            message = ["Please add square cutter first !"]
            update_info(message)
            sleep(2)
            update_info()
            return {"CANCELLED"}

        elif not context.scene["square_cutter"]:
            message = ["Cancelled, can't find the square cutter !"]
            update_info(message)
            sleep(2)
            update_info()

        else:

            cutting_mode = context.scene.BDENTAL_Props.cutting_mode
            target = bpy.data.objects.get(
                context.scene["square_cutter_target"])
            if not target:
                message = ["Cancelled, can't find the target !"]
                update_info(message)
                sleep(2)
                update_info()
                return {"CANCELLED"}
            cutter = bpy.data.objects.get(context.scene["square_cutter"])
            if not cutter:
                message = ["Cancelled, can't find the cutter !"]
                update_info(message)
                sleep(2)
                update_info()
                return {"CANCELLED"}

            bpy.context.tool_settings.mesh_select_mode = (True, False, False)
            bpy.ops.wm.tool_set_by_id(name="builtin.select")
            bpy.ops.object.mode_set(mode="OBJECT")

            bpy.ops.object.select_all(action="DESELECT")
            target.select_set(True)
            bpy.context.view_layer.objects.active = target

            # Make Model normals consitent :

            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.object.mode_set(mode="OBJECT")

            # Add Boolean Modifier :
            bool_modif = target.modifiers.new(name="Boolean", type="BOOLEAN")
            bool_modif.object = cutter

            # Apply boolean modifier :
            if cutting_mode == "Cut inner":
                bool_modif.operation = "DIFFERENCE"
                bpy.ops.object.modifier_apply(modifier="Boolean")

            if cutting_mode == "Keep inner":
                bool_modif.operation = "INTERSECT"
            bpy.ops.object.convert(target="MESH")

            # Delete resulting loose geometry :
            bpy.data.objects.remove(cutter, do_unlink=True)

            bpy.ops.object.select_all(action="DESELECT")
            target.select_set(True)
            bpy.context.view_layer.objects.active = target

            return {"FINISHED"}


#######################################################################################
# Square cut exit operator :


class BDENTAL_OT_square_cut_exit(bpy.types.Operator):
    """Square Cutting Tool Exit"""

    bl_idname = "wm.bdental_square_cut_exit"
    bl_label = "Exit"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        # Delete frame :
        try:

            frame = bpy.data.objects["my_frame_cutter"]
            bpy.ops.object.select_all(action="DESELECT")
            frame.select_set(True)

            bpy.ops.object.select_all(action="INVERT")
            Model = bpy.context.selected_objects[0]

            bpy.ops.object.select_all(action="DESELECT")
            frame.select_set(True)
            bpy.context.view_layer.objects.active = frame

            bpy.ops.object.delete(use_global=False, confirm=False)

            bpy.ops.object.select_all(action="DESELECT")
            Model.select_set(True)
            bpy.context.view_layer.objects.active = Model

        except Exception:
            pass

        return {"FINISHED"}


class BDENTAL_OT_PaintArea(bpy.types.Operator):
    """Vertex paint area context toggle"""

    bl_idname = "wm.bdental_paintarea_toggle"
    bl_label = "PAINT AREA"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        ActiveObj = context.active_object
        if not ActiveObj:

            message = ["Please select the target object !"]
            icon = "COLORSET_02_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        else:

            condition = ActiveObj.type == "MESH" and ActiveObj.select_get() == True

            if not condition:

                message = ["Please select the target object !"]
                icon = "COLORSET_02_VEC"
                bpy.ops.wm.bdental_message_box(
                    "INVOKE_DEFAULT", message=str(message), icon=icon
                )

                return {"CANCELLED"}

            else:

                Override, area3D, space3D = CtxOverride(context)
                bpy.ops.object.mode_set(mode="VERTEX_PAINT")
                bpy.ops.wm.tool_set_by_id(name="builtin_brush.Draw")

                DrawBrush = bpy.data.brushes.get("Draw")
                DrawBrush.blend = "MIX"
                DrawBrush.color = (0.0, 1.0, 0.0)
                DrawBrush.strength = 1.0
                DrawBrush.use_frontface = True
                DrawBrush.use_alpha = True
                DrawBrush.stroke_method = "SPACE"
                DrawBrush.curve_preset = "CUSTOM"
                DrawBrush.cursor_color_add = (0.0, 0.0, 1.0, 0.9)
                DrawBrush.use_cursor_overlay = True

                bpy.context.tool_settings.vertex_paint.tool_slots[0].brush = DrawBrush

                for vg in ActiveObj.vertex_groups:
                    ActiveObj.vertex_groups.remove(vg)

                for VC in ActiveObj.data.vertex_colors:
                    ActiveObj.data.vertex_colors.remove(VC)

                ActiveObj.data.vertex_colors.new(name="BDENTAL_PaintCutter_VC")

                return {"FINISHED"}


class BDENTAL_OT_PaintAreaPlus(bpy.types.Operator):
    """Vertex paint area Paint Plus toggle"""

    bl_idname = "wm.bdental_paintarea_plus"
    bl_label = "PLUS"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        if not context.mode == "PAINT_VERTEX":

            message = [
                " Please select the target object ",
                "and activate Vertex Paint mode !",
            ]
            icon = "COLORSET_02_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )
            return {"CANCELLED"}

        else:
            Override, area3D, space3D = CtxOverride(context)
            bpy.ops.wm.tool_set_by_id(name="builtin_brush.Draw")
            DrawBrush = bpy.data.brushes.get("Draw")
            context.tool_settings.vertex_paint.tool_slots[0].brush = DrawBrush
            DrawBrush.blend = "MIX"
            DrawBrush.color = (0.0, 1.0, 0.0)
            DrawBrush.strength = 1.0
            DrawBrush.use_frontface = True
            DrawBrush.use_alpha = True
            DrawBrush.stroke_method = "SPACE"
            DrawBrush.curve_preset = "CUSTOM"
            DrawBrush.cursor_color_add = (0.0, 0.0, 1.0, 0.9)
            DrawBrush.use_cursor_overlay = True
            space3D.show_region_header = False
            space3D.show_region_header = True

            return {"FINISHED"}


class BDENTAL_OT_PaintAreaMinus(bpy.types.Operator):
    """Vertex paint area Paint Minus toggle"""

    bl_idname = "wm.bdental_paintarea_minus"
    bl_label = "MINUS"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        if not context.mode == "PAINT_VERTEX":

            message = [
                " Please select the target object ",
                "and activate Vertex Paint mode !",
            ]
            icon = "COLORSET_02_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        else:
            Override, area3D, space3D = CtxOverride(context)
            bpy.ops.wm.tool_set_by_id(name="builtin_brush.Draw")
            LightenBrush = bpy.data.brushes.get("Lighten")
            context.tool_settings.vertex_paint.tool_slots[0].brush = LightenBrush
            LightenBrush.blend = "MIX"
            LightenBrush.color = (1.0, 1.0, 1.0)
            LightenBrush.strength = 1.0
            LightenBrush.use_frontface = True
            LightenBrush.use_alpha = True
            LightenBrush.stroke_method = "SPACE"
            LightenBrush.curve_preset = "CUSTOM"
            LightenBrush.cursor_color_add = (1, 0.0, 0.0, 0.9)
            LightenBrush.use_cursor_overlay = True
            space3D.show_region_header = False
            space3D.show_region_header = True

            return {"FINISHED"}


class BDENTAL_OT_PaintCut(bpy.types.Operator):
    """Vertex paint Cut"""

    bl_idname = "wm.bdental_paint_cut"
    bl_label = "CUT"

    Cut_Modes_List = [
        "Cut", "Make Copy (Shell)", "Remove Painted", "Keep Painted"]
    items = []
    for i in range(len(Cut_Modes_List)):
        item = (str(Cut_Modes_List[i]), str(
            Cut_Modes_List[i]), str(""), int(i))
        items.append(item)

    Cut_Mode_Prop: EnumProperty(
        name="Cut Mode", items=items, description="Cut Mode", default="Cut"
    )

    def execute(self, context):

        VertexPaintCut(mode=self.Cut_Mode_Prop)
        bpy.ops.ed.undo_push(message="BDENTAL Paint Cutter")

        return {"FINISHED"}

    def invoke(self, context, event):

        if not context.mode == "PAINT_VERTEX":

            message = [
                " Please select the target object ",
                "and activate Vertex Paint mode !",
            ]
            icon = "COLORSET_02_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        else:

            wm = context.window_manager
            return wm.invoke_props_dialog(self)


###########################################################################
# DSD Camera
###########################################################################
# ---------------------------------------------------------------
# 3x4 P matrix from Blender camera
# ---------------------------------------------------------------

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
def get_calibration_matrix_K_from_blender(DsdCam):
    f_in_mm = DsdCam.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = DsdCam.sensor_width
    sensor_height_in_mm = DsdCam.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if DsdCam.sensor_fit == "VERTICAL":
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0  # only use rectangular pixels

    K = np.array(((alpha_u, skew, u_0), (0, alpha_v, v_0), (0, 0, 1)))
    return K


# Returns camera rotation and translation matrices from Blender.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(DsdCam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = DsdCam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = DsdCam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*DsdCam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    # NOTE: Use * instead of @ here for older versions of Blender
    # TODO: detect Blender version
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    # put into 3x4 matrix
    RT = Matrix(
        (
            R_world2cv[0][:] + (T_world2cv[0],),
            R_world2cv[1][:] + (T_world2cv[1],),
            R_world2cv[2][:] + (T_world2cv[2],),
        )
    )
    return RT


def get_3x4_P_matrix_from_blender(DsdCam):
    K = get_calibration_matrix_K_from_blender(DsdCam.data)
    RT = get_3x4_RT_matrix_from_blender(DsdCam)
    return K @ RT, K, RT


# ----------------------------------------------------------
# Alternate 3D coordinates to 2D pixel coordinate projection code
# adapted from https://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex?lq=1
# to have the y axes pointing up and origin at the top-left corner
def project_by_object_utils(cam, point):
    scene = bpy.context.scene
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
    )
    return Vector((co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]))


#     p1 = P @ e1
#     p1 /= p1[2]
#     print("Projected e1")
#     print(p1)
#     print("proj by object_utils")
#     print(project_by_object_utils(cam, Vector(e1[0:3])))


def CamIntrisics(CalibFile):
    with open(CalibFile, "rb") as rf:
        (K, distCoeffs, _, _) = pickle.load(rf)
    fx, fy = K[0, 0], K[1, 1]
    return fx, fy, K, distCoeffs


def Undistort(DistImage, K, distCoeffs):
    img = cv2.imread(DistImage)
    h, w = img.shape[:2]
    if w < h:
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        K = np.array([[fy, 0, cy], [0, fx, cx], [0, 0, 1]], dtype=np.float32)

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        K, distCoeffs, (w, h), 1, (w, h))
    # undistort
    UndistImage = cv2.undistort(img, K, distCoeffs, None, newcameramtx)
    # crop the image
    # x, y, w, h = roi
    # UndistImage = UndistImage[y:y+h, x:x+w]
    Split = split(DistImage)
    UndistImagePath = join(Split[0], f"Undistorted_{Split[1]}")
    cv2.imwrite(UndistImagePath, UndistImage)

    return UndistImagePath


def DsdCam_from_CalibMatrix(fx, fy, cx, cy):
    if cx > cy:
        print("Horizontal mode")
        sensor_width_in_mm = fy * cx / (fx * cy)
        sensor_height_in_mm = 1  # doesn't matter
        s_u = cx * 2 / sensor_width_in_mm
        f_in_mm = fx / s_u
    if cx < cy:
        print("Vertical mode")
        sensor_width_in_mm = fy * cy / (fx * cx)
        sensor_height_in_mm = 1  # doesn't matter
        s_u = cy * 2 / sensor_width_in_mm
        f_in_mm = fx / s_u

    return sensor_width_in_mm, sensor_height_in_mm, f_in_mm


def Focal_lengh_To_K(f_in_mm, w, h):
    if w > h:
        cx, cy = w / 2, h / 2
        fx = fy = f_in_mm * h
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        sensor_width_in_mm = fy * cx / (fx * cy)
    if w < h:
        cx, cy = w / 2, h / 2
        fx = fy = f_in_mm * w
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        sensor_width_in_mm = fy * cy / (fx * cx)

    return K, sensor_width_in_mm, 1


def DsdCam_Orientation(ObjPoints3D, Undistorted_Image_Points2D, K, cx, cy):

    K[0, 2] = cx
    K[1, 2] = cy
    distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    ret, rvec, tvec = cv2.solvePnP(
        ObjPoints3D, Undistorted_Image_Points2D, K, distCoeffs
    )

    mat = mathutils.Matrix(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    ViewCam_RotM = Matrix(cv2.Rodrigues(rvec)[0])
    ViewCam_Matrix = ViewCam_RotM.to_4x4()
    ViewCam_Matrix.translation = Vector(tvec)
    Cam_Matrix = ViewCam_Matrix.inverted() @ mat

    return Cam_Matrix


class BDENTAL_OT_XrayToggle(bpy.types.Operator):
    """ """

    bl_idname = "wm.bdental_xray_toggle"
    bl_label = "2D Image to 3D Matching"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        bpy.ops.view3d.toggle_xray()

        return {"FINISHED"}


class BDENTAL_OT_Matching2D3D(bpy.types.Operator):
    """ """

    bl_idname = "wm.bdental_matching_2d3d"
    bl_label = "2D Image to 3D Matching"
    bl_options = {"REGISTER", "UNDO"}

    MP2D_List = []
    MP3D_List = []
    MP_All_List = []

    counter2D, counter3D = 1, 1
    color2D = (1, 0, 1, 1)
    color3D = (0, 1, 0, 1)

    def MouseActionZone(self, context, event):
        if event.mouse_x >= self.Right_A3D.x:
            return "2D"
        else:
            return "3D"

    def modal(self, context, event):

        if not event.type in {
            "DEL",
            "LEFTMOUSE",
            "RET",
            "ESC",
        }:
            # allow navigation

            return {"PASS_THROUGH"}

        elif event.type == ("DEL"):

            if event.value == ("PRESS"):

                if self.MP_All_List:
                    obj = self.MP_All_List.pop()
                    bpy.data.objects.remove(obj)

                    if obj in self.MP2D_List:
                        self.counter2D -= 1
                        self.MP2D_List.pop()
                        self.Matchs["2D"].pop()
                    if obj in self.MP3D_List:
                        self.counter3D -= 1
                        self.MP3D_List.pop()
                        self.Matchs["3D"].pop()

            return {"RUNNING_MODAL"}

        elif event.type == ("LEFTMOUSE"):

            if event.value == ("PRESS"):

                bpy.ops.wm.tool_set_by_id(name="builtin.cursor")
                return {"PASS_THROUGH"}

            if event.value == ("RELEASE"):

                Cursor = context.scene.cursor.location
                CursorLocal = self.Mtx @ Cursor
                u = (CursorLocal[0] + self.Dims[0] / 2) / self.Dims[0]
                v = (CursorLocal[1] + self.Dims[1] / 2) / self.Dims[1]
                w = CursorLocal[2]

                e = 0.0001

                MAZ = self.MouseActionZone(context, event)

                if MAZ == "2D":
                    if not 0 <= u + e <= 1 or not 0 <= v + e <= 1 or w > e:
                        self.report(
                            {"INFO"},
                            f"Outside the Image : (u : {u} , v : {v}, w : {w})",
                        )
                    else:
                        name = f"MP2D_{self.counter2D}"
                        MP2D = AddMarkupPoint(
                            name,
                            self.color2D,
                            Cursor,
                            Diameter=0.01,
                            CollName="CAM_DSD",
                            show_name=False,
                        )
                        self.MP2D_List.append(MP2D)
                        self.MP_All_List.append(MP2D)

                        bpy.ops.object.select_all(action="DESELECT")

                        size_x, size_y = self.size

                        px, py = int((size_x - 1) *
                                     u), int((size_y - 1) * (1 - v))

                        self.Matchs["2D"].append([px, py])
                        self.counter2D += 1

                if MAZ == "3D":
                    name = f"MP3D_{self.counter3D}"
                    MP3D = AddMarkupPoint(
                        name, self.color3D, Cursor, Diameter=1, show_name=False
                    )
                    self.MP3D_List.append(MP3D)
                    self.MP_All_List.append(MP3D)

                    bpy.ops.object.select_all(action="DESELECT")
                    self.Matchs["3D"].append(list(MP3D.location))
                    self.counter3D += 1

                bpy.ops.wm.tool_set_by_id(name="builtin.select")
                self.ImagePlane.select_set(True)
                context.view_layer.objects.active = self.ImagePlane

        elif event.type == "RET":
            bpy.ops.object.hide_collection(
                self.Right_override, collection_index=1, toggle=True
            )
            Points = [
                p for p in bpy.data.objects if "MP2D" in p.name or "MP3D" in p.name
            ]
            Points2D = [p for p in bpy.data.objects if "MP2D" in p.name]
            Points3D = [p for p in bpy.data.objects if "MP3D" in p.name]
            if len(Points2D) != len(Points3D):
                message = [
                    "Number of Points is not equal !",
                    "Please check and retry !",
                ]
                icon = "COLORSET_02_VEC"
                bpy.ops.wm.bdental_message_box(
                    "INVOKE_DEFAULT", message=str(message), icon=icon
                )
                return {"RUNNING_MODAL"}

            else:
                if Points:
                    for p in Points:
                        bpy.data.objects.remove(p)

                arr3D = np.array(self.Matchs["3D"], dtype=np.float32)
                arr2D = np.array(self.Matchs["2D"], dtype=np.float32)

                invMtx = self.Cam_obj.matrix_world.inverted().copy()
                Cam_Matrix = DsdCam_Orientation(
                    arr3D, arr2D, self.K, self.cx, self.cy)
                self.Cam_obj.matrix_world = Cam_Matrix
                self.ImagePlane.matrix_world = (
                    Cam_Matrix @ invMtx @ self.ImagePlane.matrix_world
                )
                self.ImagePlane.hide_set(True)
                bpy.ops.view3d.toggle_xray(self.Right_override)
                self.Right_S3D.shading.xray_alpha = 0.1

                return {"FINISHED"}

        elif event.type == ("ESC"):
            Points = [
                p for p in bpy.data.objects if "MP2D" in p.name or "MP3D" in p.name
            ]
            if Points:
                for obj in Points:
                    bpy.data.objects.remove(obj)

            col = bpy.data.collections.get("Matching Points")
            if col:
                bpy.data.collections.remove(col)
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):

        BDENTAL_Props = context.scene.BDENTAL_Props
        ImagePath = BDENTAL_Props.Back_ImageFile
        CalibFile = AbsPath(BDENTAL_Props.DSD_CalibFile)

        if not exists(ImagePath):
            message = [
                "Please check Image path and retry !",
            ]

            icon = "COLORSET_02_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        self.Target = context.object
        if not self.Target:

            message = ["Please select the target object !"]
            icon = "COLORSET_02_VEC"
            bpy.ops.wm.bdental_message_box(
                "INVOKE_DEFAULT", message=str(message), icon=icon
            )

            return {"CANCELLED"}

        else:

            if context.space_data.type == "VIEW_3D":
                #######################################################
                if not CalibFile:

                    ImgName = (
                        os.path.split(
                            ImagePath)[-1] or os.path.split(ImagePath)[-2]
                    )
                    self.Suffix = ImgName.split(".")[0]

                    ImageName = f"DSD_Image({self.Suffix})"
                    self.Dsd_Image = Image = bpy.data.images.get(
                        ImageName
                    ) or bpy.data.images.load(ImagePath, check_existing=False)

                    Image.name = ImageName
                    Image.colorspace_settings.name = "Non-Color"

                    # Add Camera :
                    bpy.ops.object.camera_add(
                        location=(0, 0, 0), rotation=(pi, 0, 0), scale=(1, 1, 1)
                    )
                    self.Cam_obj = context.object
                    self.Cam_obj.name = f"DSD_Camera({self.Suffix})"
                    MoveToCollection(self.Cam_obj, "CAM_DSD")
                    Cam = self.Cam_obj.data
                    Cam.name = self.Cam_obj.name
                    Cam.type = "PERSP"
                    Cam.lens_unit = "MILLIMETERS"
                    Cam.display_size = 10
                    Cam.show_background_images = True

                    # Make background Image :
                    self.Cam_obj.data.background_images.new()
                    bckg_Image = self.Cam_obj.data.background_images[0]
                    bckg_Image.image = self.Dsd_Image
                    bckg_Image.display_depth = "BACK"
                    bckg_Image.alpha = 0.9

                    ######################################
                    # sensor_width_in_mm = Cam.sensor_width
                    # sensor_height_in_mm = Cam.sensor_height
                    # f_in_mm = Cam.lens
                    W, H = Image.size[:]
                    render = context.scene.render
                    render.resolution_percentage = 100
                    render.resolution_x = W
                    render.resolution_y = H

                    Cam.sensor_fit = "AUTO"
                    f_in_mm = 3.80
                    # W, H = Image.size[:]
                    # self.K = get_calibration_matrix_K_from_blender(DsdCam=Cam)
                    self.K, sensor_width_in_mm, sensor_height_in_mm = Focal_lengh_To_K(
                        f_in_mm, W, H
                    )
                    self.cx, self.cy = W / 2, H / 2

                # #######################################################
                if CalibFile:
                    if not exists(CalibFile):
                        message = [
                            "Please check Camera Calibration file and retry !",
                        ]

                        icon = "COLORSET_02_VEC"
                        bpy.ops.wm.bdental_message_box(
                            "INVOKE_DEFAULT", message=str(message), icon=icon
                        )

                        return {"CANCELLED"}

                    fx, fy, self.K, distCoeffs = CamIntrisics(CalibFile)

                    UndistImagePath = Undistort(ImagePath, self.K, distCoeffs)

                    ImgName = (
                        os.path.split(
                            ImagePath)[-1] or os.path.split(ImagePath)[-2]
                    )
                    self.Suffix = ImgName.split(".")[0]

                    ImageName = f"DSD_Image({self.Suffix})"
                    self.Dsd_Image = Image = bpy.data.images.get(
                        ImageName
                    ) or bpy.data.images.load(UndistImagePath, check_existing=False)
                    # self.Dsd_Image = Image = bpy.data.images.get(ImageName) or bpy.data.images.load(ImagePath, check_existing=False)

                    Image.name = ImageName
                    Image.colorspace_settings.name = "Non-Color"

                    # Add Camera :
                    bpy.ops.object.camera_add(
                        location=(0, 0, 0), rotation=(pi, 0, 0), scale=(1, 1, 1)
                    )
                    self.Cam_obj = context.object
                    self.Cam_obj.name = f"DSD_Camera({self.Suffix})"
                    MoveToCollection(self.Cam_obj, "CAM_DSD")
                    Cam = self.Cam_obj.data
                    Cam.name = self.Cam_obj.name
                    Cam.type = "PERSP"
                    Cam.lens_unit = "MILLIMETERS"
                    Cam.display_size = 10
                    Cam.show_background_images = True

                    # Make background Image :
                    self.Cam_obj.data.background_images.new()
                    bckg_Image = self.Cam_obj.data.background_images[0]
                    bckg_Image.image = self.Dsd_Image
                    bckg_Image.display_depth = "BACK"
                    bckg_Image.alpha = 0.9

                    ######################################
                    W, H = Image.size[:]
                    render = context.scene.render
                    render.resolution_percentage = 100
                    render.resolution_x = W
                    render.resolution_y = H

                    Cam.sensor_fit = "AUTO"

                    self.cx, self.cy = W / 2, H / 2

                    (
                        sensor_width_in_mm,
                        sensor_height_in_mm,
                        f_in_mm,
                    ) = DsdCam_from_CalibMatrix(fx, fy, self.cx, self.cy)

                # render = context.scene.render
                # render.resolution_percentage = 100
                # render.resolution_x = W
                # render.resolution_y = H

                # Cam.sensor_fit = 'AUTO'

                Cam.sensor_width = sensor_width_in_mm
                Cam.sensor_height = sensor_height_in_mm
                Cam.lens = f_in_mm

                frame = Cam.view_frame()
                Cam_frame_World = [
                    self.Cam_obj.matrix_world @ co for co in frame]
                Plane_loc = (Cam_frame_World[0] + Cam_frame_World[2]) / 2
                Plane_Dims = [W / max([W, H]), H / max([W, H]), 0]
                # Plane_Dims = [1, H/W,0]

                bpy.ops.mesh.primitive_plane_add(
                    location=Plane_loc, rotation=self.Cam_obj.rotation_euler
                )
                self.ImagePlane = bpy.context.object
                self.DSD_Coll = MoveToCollection(
                    self.ImagePlane, "BDENTAL_DSD")
                self.ImagePlane.name = f"DSD_Plane_{self.Suffix}"
                self.ImagePlane.dimensions = Plane_Dims
                bpy.ops.object.transform_apply(
                    location=False, rotation=False, scale=True
                )
                bpy.ops.object.origin_set(
                    type="ORIGIN_GEOMETRY", center="MEDIAN")

                mat = bpy.data.materials.new(f"DSD_Mat_{self.Suffix}")
                mat.use_nodes = True
                node_tree = mat.node_tree
                nodes = node_tree.nodes
                links = node_tree.links

                for node in nodes:
                    if node.type != "OUTPUT_MATERIAL":
                        nodes.remove(node)

                TextureCoord = AddNode(
                    nodes, type="ShaderNodeTexCoord", name="TextureCoord"
                )
                ImageTexture = AddNode(
                    nodes, type="ShaderNodeTexImage", name="Image Texture"
                )

                ImageTexture.image = Image

                materialOutput = nodes["Material Output"]

                links.new(TextureCoord.outputs[0], ImageTexture.inputs[0])
                links.new(
                    ImageTexture.outputs["Color"], materialOutput.inputs["Surface"]
                )
                for slot in self.ImagePlane.material_slots:
                    bpy.ops.object.material_slot_remove()

                self.ImagePlane.active_material = mat

                mat.blend_method = "BLEND"
                context.space_data.shading.type = "SOLID"
                context.space_data.shading.color_type = "TEXTURE"
                context.space_data.shading.show_specular_highlight = False

                ##############################################################
                # Split area :
                WM = bpy.context.window_manager
                Window = WM.windows[-1]
                Screen = Window.screen

                bpy.ops.screen.area_split(direction="VERTICAL", factor=1 / 2)
                Areas3D = [
                    area for area in Screen.areas if area.type == "VIEW_3D"]
                for area in Areas3D:
                    area.type = "CONSOLE"
                    area.type = "VIEW_3D"

                    if area.x == 0:
                        print("Area left found")
                        self.Left_A3D = Left_A3D = area
                        self.Left_S3D = Left_S3D = [
                            space
                            for space in Left_A3D.spaces
                            if space.type == "VIEW_3D"
                        ][0]
                        self.Left_R3D = Left_R3D = [
                            reg for reg in Left_A3D.regions if reg.type == "WINDOW"
                        ][0]
                        self.Left_override = Left_override = {
                            "area": Left_A3D,
                            "space_data": Left_S3D,
                            "region": Left_R3D,
                        }
                        Left_S3D.show_region_ui = False
                        Left_S3D.use_local_collections = True

                    else:
                        print("Area right found")

                        self.Right_A3D = Right_A3D = area
                        self.Right_S3D = Right_S3D = [
                            space
                            for space in Right_A3D.spaces
                            if space.type == "VIEW_3D"
                        ][0]
                        self.Right_R3D = Right_R3D = [
                            reg for reg in Right_A3D.regions if reg.type == "WINDOW"
                        ][0]

                        self.Right_override = Right_override = {
                            "area": Right_A3D,
                            "space_data": Right_S3D,
                            "region": Right_R3D,
                        }
                        Right_S3D.show_region_ui = False
                        Right_S3D.use_local_collections = True

                bpy.ops.view3d.view_camera(Right_override)
                bpy.ops.view3d.view_center_camera(Right_override)

                for i in range(2, len(bpy.data.collections) + 1):
                    bpy.ops.object.hide_collection(
                        Left_override, collection_index=i, toggle=True
                    )
                bpy.ops.object.hide_collection(
                    Right_override, collection_index=1, toggle=True
                )
                #######################################################

                self.Mtx = self.ImagePlane.matrix_world.inverted()
                self.size = Image.size
                self.Dims = self.ImagePlane.dimensions
                self.Matchs = {"2D": [], "3D": []}

                bpy.ops.wm.tool_set_by_id(name="builtin.select")
                self.ImagePlane.select_set(True)
                context.view_layer.objects.active = self.ImagePlane

                context.window_manager.modal_handler_add(self)

                return {"RUNNING_MODAL"}

            else:

                self.report({"WARNING"}, "Active space must be a View3d")

                return {"CANCELLED"}

# axial camera flip


class BDENTAL_OT_FilpCameraAxial90Plus(bpy.types.Operator):
    bl_idname = "wm.bdental_flip_camera_axial_90_plus"
    bl_label = "Axial 90"

    @classmethod
    def poll(cls, context):
        camera_axial_checklist = [
            obj for obj in bpy.data.objects if "_AXIAL_SLICE_CAM" in obj.name]
        slices_pointer_checklist = [
            obj for obj in bpy.data.objects if "_SLICES_POINTER" in obj.name]
        return camera_axial_checklist and slices_pointer_checklist

    def execute(self, context):
        obj = context.object
        selected_objects = context.selected_objects
        camera_axial = [
            obj for obj in bpy.data.objects if "_AXIAL_SLICE_CAM" in obj.name][0]
        slices_pointer = [
            obj for obj in bpy.data.objects if "_SLICES_POINTER" in obj.name][0]
        bpy.ops.object.select_all(action='DESELECT')
        camera_axial.select_set(True)
        context.view_layer.objects.active = camera_axial
        for c in camera_axial.constraints:
            bpy.ops.constraint.apply(constraint=c.name)

        rotate_local(
            target=slices_pointer,
            obj=camera_axial,
            axis="Z",
            angle=90
        )
        child_of = camera_axial.constraints.new(type='CHILD_OF')
        child_of.target = slices_pointer
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_objects:
            obj.select_set(True)

        context.view_layer.objects.active = obj
        return {"FINISHED"}


class BDENTAL_OT_FilpCameraAxial90Minus(bpy.types.Operator):
    bl_idname = "wm.bdental_flip_camera_axial_90_minus"
    bl_label = "Axial 90"

    @classmethod
    def poll(cls, context):
        camera_axial_checklist = [
            obj for obj in bpy.data.objects if "_AXIAL_SLICE_CAM" in obj.name]
        slices_pointer_checklist = [
            obj for obj in bpy.data.objects if "_SLICES_POINTER" in obj.name]
        return camera_axial_checklist and slices_pointer_checklist

    def execute(self, context):
        obj = context.object
        selected_objects = context.selected_objects
        camera_axial = [
            obj for obj in bpy.data.objects if "_AXIAL_SLICE_CAM" in obj.name][0]
        slices_pointer = [
            obj for obj in bpy.data.objects if "_SLICES_POINTER" in obj.name][0]
        bpy.ops.object.select_all(action='DESELECT')
        camera_axial.select_set(True)
        context.view_layer.objects.active = camera_axial
        for c in camera_axial.constraints:
            bpy.ops.constraint.apply(constraint=c.name)

        rotate_local(
            target=slices_pointer,
            obj=camera_axial,
            axis="Z",
            angle=-90
        )
        child_of = camera_axial.constraints.new(type='CHILD_OF')
        child_of.target = slices_pointer
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_objects:
            obj.select_set(True)

        context.view_layer.objects.active = obj
        return {"FINISHED"}


class BDENTAL_OT_FilpCameraAxialUpDown(bpy.types.Operator):
    bl_idname = "wm.bdental_flip_camera_axial_up_down"
    bl_label = "Axial Up/Down"

    @classmethod
    def poll(cls, context):
        camera_axial_checklist = [
            obj for obj in bpy.data.objects if "_AXIAL_SLICE_CAM" in obj.name]
        slices_pointer_checklist = [
            obj for obj in bpy.data.objects if "_SLICES_POINTER" in obj.name]
        return camera_axial_checklist and slices_pointer_checklist

    def execute(self, context):
        obj = context.object
        selected_objects = context.selected_objects
        camera_axial = [
            obj for obj in bpy.data.objects if "_AXIAL_SLICE_CAM" in obj.name][0]
        slices_pointer = [
            obj for obj in bpy.data.objects if "_SLICES_POINTER" in obj.name][0]
        bpy.ops.object.select_all(action='DESELECT')
        camera_axial.select_set(True)
        context.view_layer.objects.active = camera_axial
        for c in camera_axial.constraints:
            bpy.ops.constraint.apply(constraint=c.name)

        rotate_local(
            target=slices_pointer,
            obj=camera_axial,
            axis="X",
            angle=180
        )
        child_of = camera_axial.constraints.new(type='CHILD_OF')
        child_of.target = slices_pointer
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_objects:
            obj.select_set(True)

        context.view_layer.objects.active = obj
        return {"FINISHED"}


class BDENTAL_OT_FilpCameraAxialLeftRight(bpy.types.Operator):
    bl_idname = "wm.bdental_flip_camera_axial_left_right"
    bl_label = "Axial L/R"

    @classmethod
    def poll(cls, context):
        camera_axial_checklist = [
            obj for obj in bpy.data.objects if "_AXIAL_SLICE_CAM" in obj.name]
        slices_pointer_checklist = [
            obj for obj in bpy.data.objects if "_SLICES_POINTER" in obj.name]
        return camera_axial_checklist and slices_pointer_checklist

    def execute(self, context):
        obj = context.object
        selected_objects = context.selected_objects
        camera_axial = [
            obj for obj in bpy.data.objects if "_AXIAL_SLICE_CAM" in obj.name][0]
        slices_pointer = [
            obj for obj in bpy.data.objects if "_SLICES_POINTER" in obj.name][0]
        bpy.ops.object.select_all(action='DESELECT')
        camera_axial.select_set(True)
        context.view_layer.objects.active = camera_axial
        for c in camera_axial.constraints:
            bpy.ops.constraint.apply(constraint=c.name)

        rotate_local(
            target=slices_pointer,
            obj=camera_axial,
            axis="Y",
            angle=180
        )
        child_of = camera_axial.constraints.new(type='CHILD_OF')
        child_of.target = slices_pointer
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_objects:
            obj.select_set(True)

        context.view_layer.objects.active = obj
        return {"FINISHED"}

# coronal camera flip


class BDENTAL_OT_FilpCameraCoronal90Plus(bpy.types.Operator):
    bl_idname = "wm.bdental_flip_camera_coronal_90_plus"
    bl_label = "Coronal 90"

    @classmethod
    def poll(cls, context):
        camera_coronal_checklist = [
            obj for obj in bpy.data.objects if "_CORONAL_SLICE_CAM" in obj.name]
        slices_pointer_checklist = [
            obj for obj in bpy.data.objects if "_SLICES_POINTER" in obj.name]
        return camera_coronal_checklist and slices_pointer_checklist

    def execute(self, context):
        obj = context.object
        selected_objects = context.selected_objects
        camera_coronal = [
            obj for obj in bpy.data.objects if "_CORONAL_SLICE_CAM" in obj.name][0]
        slices_pointer = [
            obj for obj in bpy.data.objects if "_SLICES_POINTER" in obj.name][0]
        bpy.ops.object.select_all(action='DESELECT')
        camera_coronal.select_set(True)
        context.view_layer.objects.active = camera_coronal
        for c in camera_coronal.constraints:
            bpy.ops.constraint.apply(constraint=c.name)

        rotate_local(
            target=slices_pointer,
            obj=camera_coronal,
            axis="Y",
            angle=-90
        )
        child_of = camera_coronal.constraints.new(type='CHILD_OF')
        child_of.target = slices_pointer
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_objects:
            obj.select_set(True)

        context.view_layer.objects.active = obj
        return {"FINISHED"}


class BDENTAL_OT_FilpCameraCoronal90Minus(bpy.types.Operator):
    bl_idname = "wm.bdental_flip_camera_coronal_90_minus"
    bl_label = "Coronal 90"

    @classmethod
    def poll(cls, context):
        camera_coronal_checklist = [
            obj for obj in bpy.data.objects if "_CORONAL_SLICE_CAM" in obj.name]
        slices_pointer_checklist = [
            obj for obj in bpy.data.objects if "_SLICES_POINTER" in obj.name]
        return camera_coronal_checklist and slices_pointer_checklist

    def execute(self, context):
        obj = context.object
        selected_objects = context.selected_objects
        camera_coronal = [
            obj for obj in bpy.data.objects if "_CORONAL_SLICE_CAM" in obj.name][0]
        slices_pointer = [
            obj for obj in bpy.data.objects if "_SLICES_POINTER" in obj.name][0]
        bpy.ops.object.select_all(action='DESELECT')
        camera_coronal.select_set(True)
        context.view_layer.objects.active = camera_coronal
        for c in camera_coronal.constraints:
            bpy.ops.constraint.apply(constraint=c.name)

        rotate_local(
            target=slices_pointer,
            obj=camera_coronal,
            axis="Y",
            angle=90
        )
        child_of = camera_coronal.constraints.new(type='CHILD_OF')
        child_of.target = slices_pointer
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_objects:
            obj.select_set(True)

        context.view_layer.objects.active = obj
        return {"FINISHED"}


class BDENTAL_OT_FilpCameraCoronalUpDown(bpy.types.Operator):
    bl_idname = "wm.bdental_flip_camera_coronal_up_down"
    bl_label = "Coronal Up/Down"

    @classmethod
    def poll(cls, context):
        camera_coronal_checklist = [
            obj for obj in bpy.data.objects if "_CORONAL_SLICE_CAM" in obj.name]
        slices_pointer_checklist = [
            obj for obj in bpy.data.objects if "_SLICES_POINTER" in obj.name]
        return camera_coronal_checklist and slices_pointer_checklist

    def execute(self, context):
        obj = context.object
        selected_objects = context.selected_objects
        camera_coronal = [
            obj for obj in bpy.data.objects if "_CORONAL_SLICE_CAM" in obj.name][0]
        slices_pointer = [
            obj for obj in bpy.data.objects if "_SLICES_POINTER" in obj.name][0]
        bpy.ops.object.select_all(action='DESELECT')
        camera_coronal.select_set(True)
        context.view_layer.objects.active = camera_coronal
        for c in camera_coronal.constraints:
            bpy.ops.constraint.apply(constraint=c.name)

        rotate_local(
            target=slices_pointer,
            obj=camera_coronal,
            axis="X",
            angle=180
        )
        child_of = camera_coronal.constraints.new(type='CHILD_OF')
        child_of.target = slices_pointer
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_objects:
            obj.select_set(True)

        context.view_layer.objects.active = obj
        return {"FINISHED"}


class BDENTAL_OT_FilpCameraCoronalLeftRight(bpy.types.Operator):
    bl_idname = "wm.bdental_flip_camera_coronal_left_right"
    bl_label = "Coronal L/R"

    @classmethod
    def poll(cls, context):
        camera_coronal_checklist = [
            obj for obj in bpy.data.objects if "_CORONAL_SLICE_CAM" in obj.name]
        slices_pointer_checklist = [
            obj for obj in bpy.data.objects if "_SLICES_POINTER" in obj.name]
        return camera_coronal_checklist and slices_pointer_checklist

    def execute(self, context):
        obj = context.object
        selected_objects = context.selected_objects
        camera_coronal = [
            obj for obj in bpy.data.objects if "_CORONAL_SLICE_CAM" in obj.name][0]
        slices_pointer = [
            obj for obj in bpy.data.objects if "_SLICES_POINTER" in obj.name][0]
        bpy.ops.object.select_all(action='DESELECT')
        camera_coronal.select_set(True)
        context.view_layer.objects.active = camera_coronal
        for c in camera_coronal.constraints:
            bpy.ops.constraint.apply(constraint=c.name)

        rotate_local(
            target=slices_pointer,
            obj=camera_coronal,
            axis="Z",
            angle=180
        )
        child_of = camera_coronal.constraints.new(type='CHILD_OF')
        child_of.target = slices_pointer
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_objects:
            obj.select_set(True)

        context.view_layer.objects.active = obj
        return {"FINISHED"}

# sagittal camera flip


class BDENTAL_OT_FilpCameraSagittal90Plus(bpy.types.Operator):
    bl_idname = "wm.bdental_flip_camera_sagittal_90_plus"
    bl_label = "Sagittal 90"

    @classmethod
    def poll(cls, context):
        camera_sagittal_checklist = [
            obj for obj in bpy.data.objects if "_SAGITTAL_SLICE_CAM" in obj.name]
        slices_pointer_checklist = [
            obj for obj in bpy.data.objects if "_SLICES_POINTER" in obj.name]
        return camera_sagittal_checklist and slices_pointer_checklist

    def execute(self, context):
        obj = context.object
        selected_objects = context.selected_objects
        camera_sagittal = [
            obj for obj in bpy.data.objects if "_SAGITTAL_SLICE_CAM" in obj.name][0]
        slices_pointer = [
            obj for obj in bpy.data.objects if "_SLICES_POINTER" in obj.name][0]
        bpy.ops.object.select_all(action='DESELECT')
        camera_sagittal.select_set(True)
        context.view_layer.objects.active = camera_sagittal
        for c in camera_sagittal.constraints:
            bpy.ops.constraint.apply(constraint=c.name)

        rotate_local(
            target=slices_pointer,
            obj=camera_sagittal,
            axis="X",
            angle=-90
        )
        child_of = camera_sagittal.constraints.new(type='CHILD_OF')
        child_of.target = slices_pointer
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_objects:
            obj.select_set(True)

        context.view_layer.objects.active = obj
        return {"FINISHED"}


class BDENTAL_OT_FilpCameraSagittal90Minus(bpy.types.Operator):
    bl_idname = "wm.bdental_flip_camera_sagittal_90_minus"
    bl_label = "Sagittal 90"

    @classmethod
    def poll(cls, context):
        camera_sagittal_checklist = [
            obj for obj in bpy.data.objects if "_SAGITTAL_SLICE_CAM" in obj.name]
        slices_pointer_checklist = [
            obj for obj in bpy.data.objects if "_SLICES_POINTER" in obj.name]
        return camera_sagittal_checklist and slices_pointer_checklist

    def execute(self, context):
        obj = context.object
        selected_objects = context.selected_objects
        camera_sagittal = [
            obj for obj in bpy.data.objects if "_SAGITTAL_SLICE_CAM" in obj.name][0]
        slices_pointer = [
            obj for obj in bpy.data.objects if "_SLICES_POINTER" in obj.name][0]

        bpy.ops.object.select_all(action='DESELECT')
        camera_sagittal.select_set(True)
        context.view_layer.objects.active = camera_sagittal
        for c in camera_sagittal.constraints:
            bpy.ops.constraint.apply(constraint=c.name)

        rotate_local(
            target=slices_pointer,
            obj=camera_sagittal,
            axis="X",
            angle=90
        )
        child_of = camera_sagittal.constraints.new(type='CHILD_OF')
        child_of.target = slices_pointer
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_objects:
            obj.select_set(True)

        context.view_layer.objects.active = obj
        return {"FINISHED"}


class BDENTAL_OT_FilpCameraSagittalUpDown(bpy.types.Operator):
    bl_idname = "wm.bdental_flip_camera_sagittal_up_down"
    bl_label = "Sagittal Up/Down"

    @classmethod
    def poll(cls, context):
        camera_sagittal_checklist = [
            obj for obj in bpy.data.objects if "_SAGITTAL_SLICE_CAM" in obj.name]
        slices_pointer_checklist = [
            obj for obj in bpy.data.objects if "_SLICES_POINTER" in obj.name]
        return camera_sagittal_checklist and slices_pointer_checklist

    def execute(self, context):
        obj = context.object
        selected_objects = context.selected_objects
        camera_sagittal = [
            obj for obj in bpy.data.objects if "_SAGITTAL_SLICE_CAM" in obj.name][0]
        slices_pointer = [
            obj for obj in bpy.data.objects if "_SLICES_POINTER" in obj.name][0]

        bpy.ops.object.select_all(action='DESELECT')
        camera_sagittal.select_set(True)
        context.view_layer.objects.active = camera_sagittal
        for c in camera_sagittal.constraints:
            bpy.ops.constraint.apply(constraint=c.name)

        rotate_local(
            target=slices_pointer,
            obj=camera_sagittal,
            axis="Y",
            angle=180
        )
        child_of = camera_sagittal.constraints.new(type='CHILD_OF')
        child_of.target = slices_pointer
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_objects:
            obj.select_set(True)

        context.view_layer.objects.active = obj
        return {"FINISHED"}


class BDENTAL_OT_FilpCameraSagittalLeftRight(bpy.types.Operator):
    bl_idname = "wm.bdental_flip_camera_sagittal_left_right"
    bl_label = "Sagittal L/R"

    @classmethod
    def poll(cls, context):
        camera_sagittal_checklist = [
            obj for obj in bpy.data.objects if "_SAGITTAL_SLICE_CAM" in obj.name]
        slices_pointer_checklist = [
            obj for obj in bpy.data.objects if "_SLICES_POINTER" in obj.name]
        return camera_sagittal_checklist and slices_pointer_checklist

    def execute(self, context):
        obj = context.object
        selected_objects = context.selected_objects
        camera_sagittal = [
            obj for obj in bpy.data.objects if "_SAGITTAL_SLICE_CAM" in obj.name][0]
        slices_pointer = [
            obj for obj in bpy.data.objects if "_SLICES_POINTER" in obj.name][0]

        bpy.ops.object.select_all(action='DESELECT')
        camera_sagittal.select_set(True)
        context.view_layer.objects.active = camera_sagittal
        for c in camera_sagittal.constraints:
            bpy.ops.constraint.apply(constraint=c.name)
        rotate_local(
            target=slices_pointer,
            obj=camera_sagittal,
            axis="Z",
            angle=180
        )
        child_of = camera_sagittal.constraints.new(type='CHILD_OF')
        child_of.target = slices_pointer
        child_of.use_scale_x = False
        child_of.use_scale_y = False
        child_of.use_scale_z = False

        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_objects:
            obj.select_set(True)

        context.view_layer.objects.active = obj
        return {"FINISHED"}


class BDENTAL_OT_NormalsToggle(bpy.types.Operator):
    """ Mesh check normals """

    bl_idname = "wm.bdental_normals_toggle"
    bl_label = "NORMALS TOGGLE"
    bl_options = {"REGISTER", "UNDO"}
    @classmethod
    
    def poll(cls, context):
        return context.object and len(context.selected_objects) == 1 and context.object.type == 'MESH'

    def execute(self, context):
        bpy.context.space_data.overlay.show_face_orientation = bool(
            int(bpy.context.space_data.overlay.show_face_orientation) - 1)

        return{"FINISHED"}


class BDENTAL_OT_FlipNormals(bpy.types.Operator):
    """ Mesh filp normals """

    bl_idname = "wm.bdental_flip_normals"
    bl_label = "FLIP NORMALS"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.object and len(context.selected_objects) == 1 and context.object.type == 'MESH'

    def execute(self, context):
        obj = context.object
        mode = obj.mode
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.flip_normals()
        bpy.ops.object.mode_set(mode=mode)
        return{"FINISHED"}


class BDENTAL_OT_SlicesPointerSelect(bpy.types.Operator):
    """ select slices pointer """

    bl_idname = "wm.bdental_slices_pointer_select"
    bl_label = "SELECT SLICES POINTER"

    @classmethod
    def poll(cls, context):
        checklist = [
            obj for obj in bpy.data.objects if obj.get("bdental_type")=="slices_pointer"]
        if checklist:
            return True
        return False

    def execute(self, context):
        checklist = [
            obj for obj in bpy.data.objects if obj.get("bdental_type")=="slices_pointer"]
        obj = checklist[0]
        context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        return{"FINISHED"}


class BDENTAL_OT_OverhangsPreview(bpy.types.Operator):
    " Survey the model from view"

    bl_idname = "wm.bdental_overhangs_preview"
    bl_label = "Preview Overhangs"

    overhangs_color = [1, 0.2, 0.2, 1.0]
    angle: bpy.props.FloatProperty(
        name="Angle",
        description="Overhangs Angle",
        default=45,
        min=0,
        max=90
    )

    @classmethod
    def poll(cls, context):
        if not context.object:
            return False
        return context.object.type == "MESH" and context.object.mode == "OBJECT" and context.object.select_get()

    def compute_overhangs(self, context):
        bpy.ops.object.select_all(action='DESELECT')
        self.obj.select_set(True)
        bpy.context.view_layer.objects.active = self.obj

        Override, area3D, space3D = CtxOverride(context)

        view_z_local = self.obj.matrix_world.inverted().to_quaternion() @ Vector((0, 0, 1))

        overhangs_faces_index_list = [
            f.index for f in self.obj.data.polygons if abs(f.normal.dot(view_z_local)) > radians(self.angle)
        ]

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.mesh.select_all(action="DESELECT")

        bpy.ops.object.mode_set(mode="OBJECT")

        for i in overhangs_faces_index_list:
            self.obj.data.polygons[i].select = True

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)

        self.obj.vertex_groups.active_index = self.overhangs_vg.index
        bpy.ops.object.vertex_group_assign()
        bpy.ops.object.material_slot_assign()
        bpy.ops.object.mode_set(mode="OBJECT")

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)

    def modal(self, context, event):
        if event.type in {'ESC'}:
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            # change theme color, silly!
            mtx = self.obj.matrix_world.copy()
            if not mtx == self.matrix_world:
                self.matrix_world = mtx
                self.compute_overhangs(context)

        return {'PASS_THROUGH'}

    def execute(self, context):
        bpy.ops.object.select_all(action='DESELECT')
        self.obj.select_set(True)
        bpy.context.view_layer.objects.active = self.obj

        self.overhangs_vg = self.obj.vertex_groups.get(
            "overhangs_vg"
        ) or self.obj.vertex_groups.new(name="overhangs_vg")

        if not self.obj.material_slots:
            mat_white = bpy.data.materials.get(
                "undercuts_preview_mat_white") or bpy.data.materials.new("undercuts_preview_mat_white")
            mat_white.diffuse_color = (0.8, 0.8, 0.8, 1.0)
            self.obj.active_material = mat_white

        for i, slot in enumerate(self.obj.material_slots):
            if slot.material.name == "overhangs_preview_mat_color":
                self.obj.active_material_index = i
                bpy.ops.object.material_slot_remove()

        self.overhangs_mat_color = bpy.data.materials.get(
            "overhangs_preview_mat_color") or bpy.data.materials.new("overhangs_preview_mat_color")
        self.overhangs_mat_color.diffuse_color = self.overhangs_color
        self.obj.data.materials.append(self.overhangs_mat_color)
        self.obj.active_material_index = len(self.obj.material_slots) - 1

        # bpy.ops.object.mode_set(mode="EDIT")
        # bpy.ops.mesh.select_all(action="SELECT")
        # bpy.ops.mesh.normals_make_consistent(inside=False)
        # bpy.ops.mesh.select_all(action="DESELECT")
        # bpy.ops.object.mode_set(mode="OBJECT")

        # #############################____Surveying____###############################
        self.compute_overhangs(context)

        # add a modal timer
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        self.obj = context.object
        self.matrix_world = self.obj.matrix_world.copy()

        wm = context.window_manager
        return wm.invoke_props_dialog(self, width=300)


class BDENTAL_OT_PathCutter(bpy.types.Operator):
    """New path cutter"""

    bl_idname = "wm.bdental_add_path_cutter"
    bl_label = "Path Cutter"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):

        base_mesh = context.object and context.object.select_get(
        ) and context.object.type == "MESH"
        return base_mesh

    def get_select_last(self, context):
        context.view_layer.objects.active = self.base_mesh
        bpy.ops.object.mode_set(mode="EDIT")
        me = self.base_mesh.data
        bm = bmesh.from_edit_mesh(me)
        bm.verts.ensure_lookup_table()
        if bm.select_history:
            return [elem.index for elem in bm.select_history if isinstance(elem, bmesh.types.BMVert)][-1]
        return None

    def modal(self, context, event):
        if not event.type in ["RET", "ESC", "LEFTMOUSE"]:
            return {"PASS_THROUGH"}

        elif event.type in ["LEFTMOUSE"]:

            if event.value == ("PRESS"):
                return {"PASS_THROUGH"}
            if event.value == ("RELEASE"):
                if self.counter == 0:
                    self.start_vert = self.previous_vert = self.get_select_last(
                        context)
                    if self.previous_vert:
                        self.counter += 1
                elif self.counter != 0:
                    if event.shift:
                        self.previous_vert = self.get_select_last(context)
                        self.counter += 1

                        return {"PASS_THROUGH"}
                    else:
                        self.last_vert = self.get_select_last(context)
                        # print("previous_vert", self.previous_vert)
                        # print("last_vert", self.last_vert)
                        if self.last_vert:

                            path = 0

                            bpy.ops.mesh.select_all(action='DESELECT')
                            me = self.base_mesh.data
                            self.bm = bmesh.from_edit_mesh(me)
                            self.bm.verts.ensure_lookup_table()
                            for id in [self.previous_vert, self.last_vert]:
                                self.bm.verts[id].select = True

                            try:
                                bpy.ops.mesh.vert_connect_path()
                                message = [
                                    f"Geodesic path selected {self.counter}", "Cut : Press ENTER | Cancell : Press ESC"]
                                print(message)
                                update_info(message)
                                path = 1
                            except Exception as e:
                                print(e)
                                pass
                            if path == 0:
                                try:
                                    bpy.ops.mesh.select_all(action='DESELECT')
                                    for id in [self.previous_vert, self.last_vert]:
                                        self.bm.verts[id].select = True
                                    bpy.ops.mesh.shortest_path_select()
                                    message = ["Shortest path selected", "Cut : Press ENTER | Cancell : Press ESC"]
                                    print(message)
                                    update_info(message)
                                    path = 1
                                except Exception as e:
                                    print(e)
                                    pass

                            if path == 0:

                                bpy.ops.mesh.select_all(action='DESELECT')
                                bpy.ops.object.vertex_group_select()
                                bpy.ops.object.mode_set(mode="OBJECT")
                                self.base_mesh.data.vertices[self.previous_vert].select = True
                                bpy.ops.object.mode_set(mode="EDIT")
                                message = ["Invalid selection ! Continue...", "Cut : Press ENTER | Cancell : Press ESC"]
                                update_info(message)

                                return {"RUNNING_MODAL"}

                            if path == 1:
                                bpy.ops.object.vertex_group_select()
                                bpy.ops.object.vertex_group_assign()
                                self.previous_vert = self.last_vert
                                self.counter += 1
                                return {"RUNNING_MODAL"}

                return {"RUNNING_MODAL"}

        elif event.type == "ESC":
            if event.value == ("PRESS"):

                for obj in bpy.data.objects:
                    if not obj in self.start_objects:
                        bpy.data.objects.remove(obj)
                for col in bpy.data.collections:
                    if not col in self.start_collections:
                        bpy.data.collections.remove(col)

                for obj in context.visible_objects:
                    obj.hide_set(True)
                for obj in self.start_visible_objects:
                    try:
                        obj.hide_set(False)
                    except:
                        pass

                Override, area3D, space3D = CtxOverride(context)
                context.view_layer.objects.active = self.base_mesh
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action='DESELECT')
                bpy.ops.object.mode_set(mode="OBJECT")

                message = ["CANCELLED"]
                update_info(message)
                sleep(2)
                update_info()
                return {"CANCELLED"}

        elif event.type == "RET":
            if self.counter >= 2:
                if event.value == ("PRESS"):
                    path = 0
                    context.view_layer.objects.active = self.base_mesh
                    bpy.ops.object.mode_set(mode="EDIT")
                    bpy.ops.mesh.select_all(action='DESELECT')
                    me = self.base_mesh.data
                    self.bm = bmesh.from_edit_mesh(me)
                    self.bm.verts.ensure_lookup_table()
                    for id in [self.previous_vert, self.last_vert]:
                        self.bm.verts[id].select = True
                    try:
                        bpy.ops.mesh.vert_connect_path()
                        message = [f"Connected path selected {self.counter}"]
                        print(message)
                        update_info(message)
                        path = 1
                    except Exception as e:
                        message = ["Error from Connect Path"]
                        print(message)
                        print(e)
                        update_info(message)
                        pass
                    if path == 0:
                        try:
                            bpy.ops.mesh.select_all(action='DESELECT')
                            for id in [self.previous_vert, self.last_vert]:
                                self.bm.verts[id].select = True
                            bpy.ops.mesh.shortest_path_select()
                            message = ["Shortest path selected"]
                            print(message)
                            update_info(message)
                            path = 1
                        except Exception as e:
                            message = ["Error from Shortest Path"]
                            print(message)
                            print(e)
                            pass
                    if path == 0:

                        bpy.ops.mesh.select_all(action='DESELECT')
                        bpy.ops.object.vertex_group_select()
                        self.bm.verts[self.previous_vert].select = True
                        message = ["Can't connect the path"]
                        update_info(message)

                        return {"RUNNING_MODAL"}

                    if path == 1:
                        bpy.ops.object.mode_set(mode="EDIT")
                        bpy.ops.object.vertex_group_select()
                        bpy.ops.object.vertex_group_assign()

                        bpy.ops.mesh.select_mode(type="EDGE")
                        bpy.ops.mesh.edge_split(type='EDGE')
                        # bpy.ops.mesh.loop_to_region()
                        bpy.ops.wm.bdental_separate_objects(
                            "EXEC_DEFAULT", SeparateMode="Loose Parts")

                        bpy.ops.object.mode_set(mode="OBJECT")

                        message = ["FINISHED ./"]
                        update_info(message)
                        sleep(1)
                        update_info()
                        return {"FINISHED"}
            else:
                message = ["Please select at least 2 vertices"]
                update_info(message)
                return {"RUNNING_MODAL"}

        return {"RUNNING_MODAL"}

    # def invoke(self, context, event):
    #     if context.space_data.type == "VIEW_3D":
    #         self.base_mesh = context.object
    #         wm = context.window_manager
    #         return wm.invoke_props_dialog(self, width=500)

    #     else:

    #         message = ["Active space must be a View3d"]
    #         icon="COLORSET_02_VEC"
    #         bpy.ops.bdental.message_box("INVOKE_DEFAULT", message=str(message), icon=icon)
    #         return {"CANCELLED"}

    def execute(self, context):
        self.base_mesh = context.object
        self.scn = context.scene
        self.counter = 0

        self.start_objects = bpy.data.objects[:]
        self.start_collections = bpy.data.collections[:]
        self.start_visible_objects = bpy.context.visible_objects[:]

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.mesh.select_all(action='DESELECT')
        for vg in self.base_mesh.vertex_groups:
            self.base_mesh.vertex_groups.remove(vg)

        self.vg = self.base_mesh.vertex_groups.new(name="cut_loop")

        self.start_vert = None
        self.previous_vert = None
        self.last_vert = None
        self.cut_loop = []

        override, _, _ = CtxOverride(context)
        bpy.ops.wm.tool_set_by_id(override, name="builtin.select")
        context.window_manager.modal_handler_add(self)
        message = ["################## Draw path, #################",
                   "################## When done press ENTER. #################"]
        update_info(message)
        return {"RUNNING_MODAL"}


#################################################################################################
# Registration :
#################################################################################################
classes = [
    BDENTAL_OT_checkUpdate,
    BDENTAL_OT_RibbonCutterAdd,
    BDENTAL_OT_RibbonCutter_Perform_Cut,
    BDENTAL_OT_SetConfig,
    BDENTAL_OT_SupportTelegram,
    BDENTAL_OT_RemoveImplant,
    BDENTAL_OT_CurveCutter1_New,
    BDENTAL_OT_CurveCutter1_New_Perform_Cut,
    BDENTAL_OT_CurveCutter2_Cut_New,
    BDENTAL_OT_AddAppTemplate,
    BDENTAL_OT_MessageBox,
    BDENTAL_OT_OpenManual,
    # BDENTAL_OT_Template,
    BDENTAL_OT_Organize,
    BDENTAL_OT_Volume_Render,
    BDENTAL_OT_ResetCtVolumePosition,
    BDENTAL_OT_AddSlices,
    BDENTAL_OT_MultiTreshSegment,
    # BDENTAL_OT_MultiView,
    BDENTAL_OT_MPR,
    BDENTAL_OT_AddReferencePlanes,
    BDENTAL_OT_CtVolumeOrientation,
    BDENTAL_OT_AddMarkupPoint,
    BDENTAL_OT_AddTeeth,
    BDENTAL_OT_AddSleeve,
    BDENTAL_OT_AddImplant,
    BDENTAL_OT_AlignImplants,
    BDENTAL_OT_AddImplantSleeve,
    BDENTAL_OT_AddSplint,
    BDENTAL_OT_Survey,
    BDENTAL_OT_BlockModel,
    BDENTAL_OT_ModelBase,
    BDENTAL_OT_add_offset,
    BDENTAL_OT_hollow_model,
    BDENTAL_OT_AlignPoints,
    BDENTAL_OT_AlignPointsInfo,
    BDENTAL_OT_AddColor,
    BDENTAL_OT_RemoveColor,
    BDENTAL_OT_JoinObjects,
    BDENTAL_OT_SeparateObjects,
    BDENTAL_OT_Parent,
    BDENTAL_OT_Unparent,
    BDENTAL_OT_align_to_front,
    BDENTAL_OT_to_center,
    BDENTAL_OT_center_cursor,
    BDENTAL_OT_OcclusalPlane,
    BDENTAL_OT_OcclusalPlaneInfo,
    BDENTAL_OT_decimate,
    BDENTAL_OT_clean_mesh,
    BDENTAL_OT_clean_mesh2,
    BDENTAL_OT_fill,
    BDENTAL_OT_retopo_smooth,
    BDENTAL_OT_VoxelRemesh,
    BDENTAL_OT_CurveCutterAdd,
    BDENTAL_OT_CurveCutterAdd2,
    BDENTAL_OT_CurveCutterAdd3,
    BDENTAL_OT_CurveCutterCut,
    BDENTAL_OT_CurveCutterCut3,
    BDENTAL_OT_CurveCutter2_ShortPath,
    BDENTAL_OT_AddSquareCutter,
    BDENTAL_OT_square_cut_confirm,
    BDENTAL_OT_square_cut_exit,
    
    BDENTAL_OT_PaintArea,
    BDENTAL_OT_PaintAreaPlus,
    BDENTAL_OT_PaintAreaMinus,
    BDENTAL_OT_PaintCut,
    BDENTAL_OT_AddTube,
    BDENTAL_OT_Matching2D3D,
    BDENTAL_OT_XrayToggle,
    BDENTAL_OT_align_to_cursor,
    BDENTAL_OT_LockToPointer,
    BDENTAL_OT_UnlockFromPointer,
    BDENTAL_OT_MPR2,
    BDENTAL_OT_ImplantToPointer,
    BDENTAL_OT_PointerToImplant,
    BDENTAL_OT_add_3d_text,
    BDENTAL_OT_LockObjects,
    BDENTAL_OT_UnlockObjects,
    BDENTAL_OT_AlignToActive,

    BDENTAL_OT_SplintGuide,
    BDENTAL_OT_AddGuideCuttersFromSleeves,
    BDENTAL_OT_GuideFinalise,
    BDENTAL_OT_ExportMesh,
    BDENTAL_OT_UnderctsPreview,
    BDENTAL_OT_BlockoutNew,
    BDENTAL_OT_ImportMesh,

    BDENTAL_OT_FilpCameraAxial90Plus,
    BDENTAL_OT_FilpCameraAxial90Minus,
    BDENTAL_OT_FilpCameraAxialUpDown,
    BDENTAL_OT_FilpCameraAxialLeftRight,

    BDENTAL_OT_FilpCameraCoronal90Plus,
    BDENTAL_OT_FilpCameraCoronal90Minus,
    BDENTAL_OT_FilpCameraCoronalUpDown,
    BDENTAL_OT_FilpCameraCoronalLeftRight,

    BDENTAL_OT_FilpCameraSagittal90Plus,
    BDENTAL_OT_FilpCameraSagittal90Minus,
    BDENTAL_OT_FilpCameraSagittalUpDown,
    BDENTAL_OT_FilpCameraSagittalLeftRight,

    BDENTAL_OT_GuideAddComponent,
    BDENTAL_OT_GuideSetComponents,
    BDENTAL_OT_GuideSetCutters,

    BDENTAL_OT_NormalsToggle,
    BDENTAL_OT_FlipNormals,
    BDENTAL_OT_SplintGuideGeom,

    BDENTAL_OT_FlyNext,
    BDENTAL_OT_FlyPrevious,
    BDENTAL_OT_RemoveInfoFooter,
    BDENTAL_OT_guide_3d_text,
    BDENTAL_OT_CleanMeshIterative,
    BDENTAL_OT_AutoAlignIcp,
    BDENTAL_OT_SlicesPointerSelect,
    BDENTAL_OT_OverhangsPreview,
    BDENTAL_OT_PathCutter,

]


def register():

    for cls in classes:
        bpy.utils.register_class(cls)
    depsgraph_update_post_handlers = bpy.app.handlers.depsgraph_update_post
    frame_change_post_handlers = bpy.app.handlers.frame_change_post

    MyPostHandlers = [
        "BDENTAL_TresholdMinUpdate",
        "BDENTAL_TresholdMaxUpdate",
        "BDENTAL_SliceUpdate",
        # "BDENTAL_AxialSliceUpdate",
        # "BDENTAL_CoronalSliceUpdate",
        # "BDENTAL_SagittalSliceUpdate",

    ]

    # Remove old handlers :
    depsgraph_handlers_To_Remove = [
        h for h in depsgraph_update_post_handlers if h.__name__ in MyPostHandlers
    ]
    frame_change_handlers_To_Remove = [
        h for h in frame_change_post_handlers if h.__name__ in MyPostHandlers
    ]
    if depsgraph_handlers_To_Remove:
        for h in depsgraph_handlers_To_Remove:
            bpy.app.handlers.depsgraph_update_post.remove(h)
    if frame_change_handlers_To_Remove:
        for h in frame_change_handlers_To_Remove:
            bpy.app.handlers.frame_change_post.remove(h)
    handlers_To_Add = [
        BDENTAL_TresholdMinUpdate,
        BDENTAL_TresholdMaxUpdate,
        BDENTAL_SliceUpdate,
        # BDENTAL_AxialSliceUpdate,
        # BDENTAL_CoronalSliceUpdate,
        # BDENTAL_SagittalSliceUpdate,

    ]
    for h in handlers_To_Add:
        depsgraph_update_post_handlers.append(h)
    for h in handlers_To_Add:
        frame_change_post_handlers.append(h)

    # post_handlers.append(BDENTAL_TresholdUpdate)
    # post_handlers.append(AxialSliceUpdate)
    # post_handlers.append(CoronalSliceUpdate)
    # post_handlers.append(SagittalSliceUpdate)


def unregister():
    global SLICES_TXT_HANDLER
    for _h in SLICES_TXT_HANDLER:
        bpy.types.SpaceView3D.draw_handler_remove(_h, "WINDOW")
        SLICES_TXT_HANDLER = []

    depsgraph_update_post_handlers = bpy.app.handlers.depsgraph_update_post
    frame_change_post_handlers = bpy.app.handlers.frame_change_post
    MyPostHandlers = [
        "BDENTAL_TresholdMinUpdate",
        "BDENTAL_TresholdMaxUpdate",
        "BDENTAL_SliceUpdate"
        # "BDENTAL_AxialSliceUpdate",
        # "BDENTAL_CoronalSliceUpdate",
        # "BDENTAL_SagittalSliceUpdate",

    ]
    # Remove old handlers :
    depsgraph_handlers_To_Remove = [
        h for h in depsgraph_update_post_handlers if h.__name__ in MyPostHandlers
    ]
    frame_change_handlers_To_Remove = [
        h for h in frame_change_post_handlers if h.__name__ in MyPostHandlers
    ]
    if depsgraph_handlers_To_Remove:
        for h in depsgraph_handlers_To_Remove:
            bpy.app.handlers.depsgraph_update_post.remove(h)
    if frame_change_handlers_To_Remove:
        for h in frame_change_handlers_To_Remove:
            bpy.app.handlers.frame_change_post.remove(h)
    try:
        for cls in reversed(classes):
            bpy.utils.unregister_class(cls)
    except Exception as er:
        print("Addon unregister error : ", er)
