# Python imports :
import sys, os, socket, shutil
from os import listdir, system
from os.path import dirname, join, abspath, exists, expanduser
from subprocess import call
from importlib import import_module
import threading
from time import sleep
import zipfile
#############################################################
import bpy
from bpy.props import (
    StringProperty,
    IntProperty,
    FloatProperty,
    EnumProperty,
    FloatVectorProperty,
    BoolProperty,
)
import gpu
from gpu_extras.batch import batch_for_shader
import blf
from math import pi, cos, sin

red_icon = "COLORSET_01_VEC"
orange_icon = "COLORSET_02_VEC"
green_icon = "COLORSET_03_VEC"
blue_icon = "COLORSET_04_VEC"
violet_icon = "COLORSET_06_VEC"
yellow_icon = "COLORSET_09_VEC"
yellow_point = "KEYTYPE_KEYFRAME_VEC"
blue_point = "KEYTYPE_BREAKDOWN_VEC"
DRAW_HANDLERS = []
addon_dir = dirname(dirname(abspath(__file__)))
bdental_app_template_zip_file = join(addon_dir,"Resources","bdental_app_template.zip")

def Add_bdental_app_template() :
    success = 0
    binary_dir = dirname(bpy.app.binary_path)
    version = bpy.app.version_string
    for d in os.listdir(binary_dir) :
        if d in version :
            app_templates_dir = join(binary_dir, d, 'scripts', 'startup', 'bl_app_templates_system')
            with zipfile.ZipFile(bdental_app_template_zip_file, 'r') as zip_ref:
                zip_ref.extractall(app_templates_dir)
            success = 1
            break



def gpu_info_footer(rect_color, text_list, button=False, btn_txt="",pourcentage=100):
    if pourcentage<=0 : 
        pourcentage = 1
    if pourcentage>100 : 
        pourcentage = 100
    def draw_callback_function():
            
        w = int(bpy.context.area.width * (pourcentage/100))
        for i, txt in enumerate((reversed(text_list))) :
            
            h = 30
            # color = [0.4, 0.4, 0.8, 1.000000]
            # color =[0.9, 0.5, 0.000000, 1.000000] 
            draw_gpu_rect(0, h*i, w , h, rect_color)
            blf.position(0, 10, 10+ (h*i), 0) 
            blf.size(0, 40, 30)
            r,g,b,a = (0.0, 0.0, 0.0, 1.0)
            blf.color(0,r,g,b,a)
            blf.draw(0, txt)

        if button :
            
            h = 30
            color =[0.8, 0.258385, 0.041926, 1.0]
            draw_gpu_rect(w-110, 2, 100 , h-4, color)
            blf.position(0,w-85,10, 0) 
            blf.size(0, 40, 30)
            r,g,b,a = (0.0, 0.0, 0.0, 1.0)
            blf.color(0,r,g,b,a)
            blf.draw(0, btn_txt)
    
    

    info_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_callback_function, (), "WINDOW", "POST_PIXEL"
        )
    #redraw scene
    # bpy.ops.wm.redraw_timer(type = 'DRAW_WIN_SWAP', iterations = 1)
    for area in bpy.context.window.screen.areas:
        if area.type == "VIEW_3D":
            area.tag_redraw()
    
    return info_handler

def get_btn_bb(btn_index = 0, btn_width = 100, btn_height = 26, padding_x = 10, padding_y = 2, safe_area = 5 ):
    area3d = None
    area3d_check = [area for area in bpy.context.screen.areas if area.type == "VIEW_3D"]
    if area3d_check :
        area3d = area3d_check[0]
    if area3d :
        w = area3d.width
        x_min = w - padding_x -(btn_width*(btn_index+1))  - (padding_x*btn_index) - safe_area
        x_max = w - padding_x -(btn_width*(btn_index))  - (padding_x*btn_index) + safe_area
        y_min = 0
        y_max = btn_height+safe_area
        btn_bb = {
            "x_min" : x_min,
            "x_max" : x_max,
            "y_min" : y_min,
            "y_max" : y_max
        }
        return btn_bb
    else : return None


    

def draw_gpu_circle(center_2d, radius, segments, color_rgba):

    x,y = center_2d
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
                        (x + w, y + h), (x + w, y) )
                        
        indices = (
                    (0, 1, 2), (0, 2, 3)
                    )

        gpu.state.blend_set('ALPHA')
        shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'TRIS', {"pos": vertices}, indices=indices)
        shader.bind()
        shader.uniform_float("color", rect_color)
        batch.draw(shader)

def update_info( message=[], remove_handlers = True,button=False, btn_txt="",pourcentage=100, redraw_timer = True, rect_color=[0.4, 0.4, 0.8, 1.000000]):
    global DRAW_HANDLERS
    
    if remove_handlers :
        for _h in DRAW_HANDLERS : 
            bpy.types.SpaceView3D.draw_handler_remove(_h, "WINDOW")
        DRAW_HANDLERS = []
    if message :
        info_handler = gpu_info_footer(text_list=message,button=False, btn_txt="",pourcentage=100, rect_color=rect_color) 
        DRAW_HANDLERS.append(info_handler)
    if redraw_timer :
        bpy.ops.wm.redraw_timer(type = 'DRAW_WIN_SWAP', iterations = 1)


class BDENTAL_OT_MessageBox(bpy.types.Operator):
    """Bdental popup message"""

    bl_idname = "bdental.message_box"
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
#############################################################
def ImportReq(REQ_DICT):
    Pkgs = []
    for mod, pkg in REQ_DICT.items():
        try:
            import_module(mod)
        except ImportError:
            Pkgs.append(pkg)

    return Pkgs  

#############################################################
class BDENTAL_OT_InstallRequirements(bpy.types.Operator):
    """ Requirement installer """

    bl_idname = "bdental.installreq"
    bl_label = "INSTALL BDENTAL MODULES"
    message = []
    REQ_DICT = {
    "SimpleITK": "SimpleITK",
    "cv2": "opencv-contrib-python",
    "vtk": "vtk",
    }
    ADDON_DIR = dirname(dirname(abspath(__file__)))
    bdental_modules_archive = join(ADDON_DIR, "Resources", "bdental_modules_archive")
    bdental_3_modules = join(expanduser("~") , "bdental_3_modules")



    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.alignment = "EXPAND"
        row.label(text="", icon="BLENDER")
        
        for l in self.message :
            row = layout.row()
            row.alignment = "EXPAND"
            row.label(text=l)
    
    def execute(self, context) :
    
        if self.isConnected() :
            
            try :
                message = ["Bdental modules online installation processing..."]
                update_info(message)
                print(message)
                self.ReqInstall_online()
                
            except Exception as er :
                print(er)
                message = ["Bdental modules online installation failed !", "Please contact Bdental support."]
                print(message)
                update_info(message)
                sleep(3)
                update_info()
                
                return {"CANCELLED"} 
            

            message = ["Bdental template installation ..."]
            update_info(message)
            print(message)
            sleep(3)
            success = Add_bdental_app_template()
            print(f"add bdental app-template : Success = {bool(success)}")
            
            bpy.ops.script.reload()
            

            message = [
            "Bdental modules online installation completed successfully."
            ]
            
            
            system("cls")
            update_info(message=message, rect_color=[0,1,0,1])
            sleep(3)
            update_info()
            print(message)

            return {"FINISHED"}
            
        else :
           
            message = [
                "Operation cancelled ! ",
                "Please check internet connection, and retry.",
            ]
            print(message)
            update_info(message=message, rect_color=[1,0,0,1])
            sleep(3)
            update_info()
            return {"CANCELLED"} 
               
        
    def ReqInstall_online(self):
        # modules = [m for m in self.REQ_DICT.values()]
        modules = ImportReq(self.REQ_DICT)
        if modules :
            if sys.platform == "darwin" :
                PythonPath = sys.executable
                
                try :
                    command = f'"{PythonPath}" -m pip install -U pip'
                    call(command, shell=True)
                except Exception as er:
                    print(er)

                for module in modules:
                    command = f' "{PythonPath}" -m pip install -U {module} --target "{self.bdental_3_modules}" '
                    call(command, shell=True)

            if sys.platform == "win32" :

                PythonBin = dirname(sys.executable)
                self.dst = join(dirname(dirname(sys.executable)), "lib", "site-packages")
                try :
                    message = [
                        "Updating pip module...",
                    ]
                    print(message)
                    update_info(message)
                    command = f'cd "{PythonBin}" && ".\\python.exe" -m pip install -U pip -t "{self.bdental_3_modules}" '
                    call(
                        command,
                        shell=True,
                    )
                except Exception as er:
                    print(er)

                for module in modules:
                    message = [
                        f"Installing {module} module...",
                    ]
                    print(message)
                    update_info(message)
                    command = f'cd "{PythonBin}" && ".\\python.exe" -m pip install -U "{module}" -t "{self.bdental_3_modules}" '
                    call(command, shell=True)

    def ReqInstall_offline(self):

        zip_files = [join(self.bdental_modules_archive, f) for f in listdir(self.bdental_modules_archive) if f.endswith('.zip')]
        if zip_files :
            if sys.platform == 'win32':
                zip_files = [f for f in zip_files if 'win' in f]
                for z_file in zip_files:
                    shutil.unpack_archive(z_file, self.bdental_3_modules)

            if sys.platform == 'darwin':
                zip_files = [f for f in zip_files if 'mac' in f]
                for z_file in zip_files:
                    shutil.unpack_archive(z_file, self.bdental_3_modules)

            if sys.platform == 'linux':
                zip_files = [f for f in zip_files if 'linux' in f]
                for z_file in zip_files:
                    shutil.unpack_archive(z_file, self.bdental_3_modules)

        whl_files = [join(self.bdental_modules_archive, f) for f in listdir(self.bdental_modules_archive) if f.endswith('.whl')]
        if whl_files :
            if sys.platform == "win32" :
                whl_files = [f for f in whl_files if 'win' in f]
                for w_file in whl_files :
                    PythonBin = dirname(sys.executable)
                    command = f'cd "{PythonBin}" && ".\\python.exe" -m pip install -U "{w_file}" -t "{self.bdental_3_modules}" '
                    call(command, shell=True)
            if sys.platform == "darwin" :
                whl_files = [f for f in whl_files if 'mac' in f]
                for w_file in whl_files :
                    PythonPath = sys.executable
                    command = f' "{PythonPath}" -m pip install -U {w_file} -t "{self.bdental_3_modules}" '
                    call(command, shell=True)
            if sys.platform == "linux" :
                whl_files = [f for f in whl_files if 'linux' in f]
                for w_file in whl_files :
                    PythonPath = sys.executable
                    command = f' "{PythonPath}" -m pip install -U {w_file} -t "{self.bdental_3_modules}" '
                    call(command, shell=True)
    def isConnected(self):
        try:
            sock = socket.create_connection(("www.google.com", 80))
            if sock is not None:
                print("Clossing socket")
                sock.close
            return True
        except OSError:
            pass
            return False

class BDENTAL_PT_InstallReqPanel(bpy.types.Panel):
    """ Install Req Panel"""

    bl_idname = "BDENTAL_PT_InstallReqPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI" 
    bl_category = "BDENTAL"
    bl_label = "BDENTAL"

    def draw(self, context):
        message = [
            "Please ensure internet connexion is OK",
            "Bdental modules will be installed online",
            ]

        layout = self.layout
        
        box = layout.box()
        for l in message :
            row = box.row(align=True)
            row.alert = True
            row.alignment = "EXPAND"
            row.label(text=l)


        box = layout.box()
        row = box.row()
        row.operator("bdental.installreq")


classes = (
BDENTAL_OT_MessageBox,
BDENTAL_OT_InstallRequirements,
BDENTAL_PT_InstallReqPanel,
)
register, unregister = bpy.utils.register_classes_factory(classes)
