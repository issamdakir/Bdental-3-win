import bpy
from os.path import dirname, join,abspath
ADDON_DIR = dirname(dirname(abspath(__file__)))
RESOURCES = join(ADDON_DIR, "Resources")
BDENTAL_LIBRARY_PATH = join(RESOURCES, "Bdental_Library")

user_lib = bpy.context.preferences.filepaths.asset_libraries.get('Bdental Library')
if user_lib :
    user_lib.path = BDENTAL_LIBRARY_PATH