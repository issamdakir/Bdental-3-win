import bpy
from os.path import abspath, basename, exists, join

from bpy.props import (
    StringProperty,
    IntProperty,
    FloatProperty,
    EnumProperty,
    FloatVectorProperty,
    BoolProperty,
)
Wmin, Wmax = -400, 3000

from .Operators.BDENTAL_Utils import AbsPath, set_enum_items
def TresholdMinUpdateFunction(self, context):
    BDENTAL_Props = context.scene.BDENTAL_Props
    GpShader = BDENTAL_Props.GroupNodeName
    TresholdMin = BDENTAL_Props.TresholdMin

    CtVolumeList = [
        obj
        for obj in bpy.context.scene.objects
        if ("BD" in obj.name and "_CTVolume" in obj.name)
    ]
    if context.object in CtVolumeList:
        Vol = context.object
        Preffix = Vol.name[:6]
        GpNode = bpy.data.node_groups.get(f"{Preffix}_{GpShader}")
        Low_Treshold = GpNode.nodes["Low_Treshold"].outputs[0]
        Low_Treshold.default_value = TresholdMin

def TresholdMaxUpdateFunction(self, context):
    BDENTAL_Props = context.scene.BDENTAL_Props
    GpShader = BDENTAL_Props.GroupNodeName
    TresholdMax = BDENTAL_Props.TresholdMax

    CtVolumeList = [
        obj
        for obj in bpy.context.scene.objects
        if ("BD" in obj.name and "_CTVolume" in obj.name)
    ]
    if context.object in CtVolumeList:
        Vol = context.object
        Preffix = Vol.name[:6]
        GpNode = bpy.data.node_groups.get(f"{Preffix}_{GpShader}")
        High_Treshold = GpNode.nodes["High_Treshold"].outputs[0]
        High_Treshold.default_value = TresholdMax
        

def OrganizeSeriesEnumProp_callback(self, context):
    EnumItems = [("EMPTY","EMPTY",str(""))]
    BDENTAL_Props = bpy.context.scene.BDENTAL_Props
    UserDcmDir = AbsPath(BDENTAL_Props.UserDcmDir)
    DcmOrganizeprop = BDENTAL_Props.DcmOrganize
    DcmOrganizeDict = eval(DcmOrganizeprop)
    if UserDcmDir in DcmOrganizeDict.keys() :
        OrganizeReport = DcmOrganizeDict[UserDcmDir]
        EnumItems = []
        for i, (serie, info) in enumerate(OrganizeReport.items()):
            EnumItems.append((serie,serie,str("")))
    return EnumItems
def update_text(self, context):
    if context.object :
        if context.object.type == "FONT" :
            if context.object.get("bdental_type") == "bdental_text" :
                context.object.data.body = self.text
    return None
def brightness_update(self, context):
    GpNodes = [gn for gn in bpy.data.node_groups if "Slices" in gn.name]
    if GpNodes :
        GpNode = GpNodes[0]
        bright_contrast_node = GpNode.nodes["Bright/Contrast"]
        bright_contrast_node.inputs[1].default_value = self.slices_brightness

def contrast_update(self, context):
    GpNodes = [gn for gn in bpy.data.node_groups if "Slices" in gn.name]
    if GpNodes :
        GpNode = GpNodes[0]
        bright_contrast_node = GpNode.nodes["Bright/Contrast"]
        bright_contrast_node.inputs[2].default_value = self.slices_contrast

def update_user_project_dir(self, context):
    if self.UserProjectDir and exists(self.UserProjectDir) :
        self.ProjectNameProp = basename(abspath(self.UserProjectDir))
        filepath = join(self.UserProjectDir, self.ProjectNameProp+".blend")
        bpy.ops.wm.save_as_mainfile(filepath=filepath)
    
     
class BDENTAL_Props(bpy.types.PropertyGroup):

    #####################
    #############################################################################################
    # CT_Scan props :
    #############################################################################################
    #####################
    ProjectNameProp: StringProperty(
        name="Project Name",
        default='Test Project',
        description="Project Name",
    )
    UserProjectDir: StringProperty(
        name="UserProjectDir",
        default="",
        description="User project directory",
        subtype="DIR_PATH",
        update=update_user_project_dir,
    )

    UserDcmDir: StringProperty(
        name="DICOM Path",
        default="",
        description="DICOM Directory Path",
        subtype="DIR_PATH",
    )

    UserImageFile: StringProperty(
        name="User 3D Image File Path",
        default="",
        description="User Image File Path",
        subtype="FILE_PATH",
    )
    Dicom_Series_mode: EnumProperty(items = set_enum_items(["Simple Mode", "Advanced Mode"]), description="Dicom Read Mode", default="Advanced Mode")

    Dicom_Series: EnumProperty(items = OrganizeSeriesEnumProp_callback, description="Dicom series")
    OrganizeInfoProp: StringProperty(
        name="OrganizeInfo",
        default='{}',
        description="Organize Information",
    )
    scan_resolution: FloatProperty(
        name="Scan Resolution",
        description="Scan spacing parameter",
        default=0.0,
        min=0.0,
        max=10.0,
        soft_min=0.0,
        soft_max=10.0,
        step=1,
        precision=3,
    )
    SlicesDir: StringProperty(
        name="SlicesDir",
        default="",
        description="Temporary Slices directory",
        subtype="DIR_PATH",
    )


    #####################

    Data_Types = ["DICOM Series", "3D Image File", ""]
    items = []
    for i in range(len(Data_Types)):
        item = (str(Data_Types[i]), str(Data_Types[i]), str(""), int(i))
        items.append(item)

    DataType: EnumProperty(items=items, description="Data type", default="DICOM Series")

    #######################
    DcmOrganize: StringProperty(
        name="(str) Organize Dicom",
        default="{'Deffault': None}",
        description="Dicom series files list",
    )

    DcmInfo: StringProperty(
        name="(str) DicomInfo",
        default="dict()",
        description="Dicom series files list",
    )
    #######################

    PngDir: StringProperty(
        name="Png Directory",
        default="",
        description=" PNG files Sequence Directory Path",
    )
    #######################

    SlicesDir: StringProperty(
        name="Slices Directory",
        default="",
        description="Slices PNG files Directory Path",
    )
    #######################

    Nrrd255Path: StringProperty(
        name="Nrrd255Path",
        default="",
        description="Nrrd image3D file Path",
    )

    #######################
    IcpVidDict: StringProperty(
            name="IcpVidDict",
            default="None",
            description="ICP Vertices Pairs str(Dict)",
        )
    #######################
    Wmin: IntProperty()
    Wmax: IntProperty()

    #######################
    # SoftTissueMode = BoolProperty(description="SoftTissue Mode ", default=False)

    GroupNodeName: StringProperty(
        name="Group shader Name",
        default="",
        description="Group shader Name",
    )

    #######################

    TresholdMin: IntProperty(
        name="Treshold Min",
        description="Volume Treshold",
        default=600,
        min=Wmin,
        max=Wmax,
        soft_min=Wmin,
        soft_max=Wmax,
        step=1,
        update=TresholdMinUpdateFunction,
    )
    TresholdMax: IntProperty(
        name="Treshold Max",
        description="Volume Treshold",
        default=Wmax,
        min=Wmin,
        max=Wmax,
        soft_min=Wmin,
        soft_max=Wmax,
        step=1,
        update=TresholdMaxUpdateFunction,
    )
    Progress_Bar: FloatProperty(
        name="Progress_Bar",
        description="Progress_Bar",
        subtype="PERCENTAGE",
        default=0.0,
        min=0.0,
        max=100.0,
        soft_min=0.0,
        soft_max=100.0,
        step=1,
        precision=1,
    )
    SoftTreshold: IntProperty(
        name="SOFT TISSU",
        description="Soft Tissu Treshold",
        default=-300,
        min=Wmin,
        max=Wmax,
        soft_min=Wmin,
        soft_max=Wmax,
        step=1,
    )
    BoneTreshold: IntProperty(
        name="BONE",
        description="Bone Treshold",
        default=600,
        min=Wmin,
        max=Wmax,
        soft_min=Wmin,
        soft_max=Wmax,
        step=1,
    )
    TeethTreshold: IntProperty(
        name="Teeth Treshold",
        description="Teeth Treshold",
        default=1400,
        min=Wmin,
        max=Wmax,
        soft_min=Wmin,
        soft_max=Wmax,
        step=1,
    )
    SoftBool: BoolProperty(description="", default=False)
    BoneBool: BoolProperty(description="", default=False)
    TeethBool: BoolProperty(description="", default=False)

    SoftSegmentColor: FloatVectorProperty(
        name="Soft Segmentation Color",
        description="Soft Color",
        default=[0.8, 0.5, 0.38, 1.000000],  # [0.8, 0.46, 0.4, 1.0],[0.63, 0.37, 0.30, 1.0]
        soft_min=0.0,
        soft_max=1.0,
        size=4,
        subtype="COLOR",
    )
    BoneSegmentColor: FloatVectorProperty(
        name="Bone Segmentation Color",
        description="Bone Color",
        default=[0.44, 0.4, 0.5, 1.0],  # (0.8, 0.46, 0.4, 1.0),
        soft_min=0.0,
        soft_max=1.0,
        size=4,
        subtype="COLOR",
    )
    TeethSegmentColor: FloatVectorProperty(
        name="Teeth Segmentation Color",
        description="Teeth Color",
        default=[0.55, 0.645, 0.67, 1.000000],  # (0.8, 0.46, 0.4, 1.0),
        soft_min=0.0,
        soft_max=1.0,
        size=4,
        subtype="COLOR",
    )

    #######################

    CT_Loaded: BoolProperty(description="CT loaded ", default=False)
    CT_Rendered: BoolProperty(description="CT Rendered ", default=False)
    sceneUpdate: BoolProperty(description="scene update ", default=True)
    AlignModalState: BoolProperty(description="Align Modal state ", default=False)

    #######################
    ActiveOperator: StringProperty(
        name="Active Operator",
        default="None",
        description="Active_Operator",
    )
    #######################
    # Guide Components :

    TeethLibList = ["Christian Brenes Teeth Library"]
    items = []
    for i in range(len(TeethLibList)):
        item = (str(TeethLibList[i]), str(TeethLibList[i]), str(""), int(i))
        items.append(item)

    TeethLibrary: EnumProperty(
        items=items,
        description="Teeth Library",
        default="Christian Brenes Teeth Library",
    )

    ImplantLibList = ["NEOBIOTECH_IS_II_ACTIVE"]
    items = []
    for i in range(len(ImplantLibList)):
        item = (str(ImplantLibList[i]), str(ImplantLibList[i]), str(""), int(i))
        items.append(item)

    ImplantLibrary: EnumProperty(
        items=items,
        description="Implant Library",
        default="NEOBIOTECH_IS_II_ACTIVE",
    )
    #######################
    SleeveDiameter: FloatProperty(
        name="Sleeve Diameter",
        description="Sleeve Diameter",
        default=5.0,
        min=0.0,
        max=100.0,
        soft_min=0.0,
        soft_max=100.0,
        step=1,
        precision=1,
    )
    #######################
    SleeveHeight: FloatProperty(
        name="Sleeve Height",
        description="Sleeve Height",
        default=5.0,
        min=0.0,
        max=100.0,
        soft_min=0.0,
        soft_max=100.0,
        step=1,
        precision=1,
    )
    #######################
    HoleDiameter: FloatProperty(
        name="Hole Diameter",
        description="Hole Diameter",
        default=2.0,
        min=0.0,
        max=100.0,
        soft_min=0.0,
        soft_max=100.0,
        step=1,
        precision=1,
    )
    #######################
    HoleOffset: FloatProperty(
        name="Hole Offset",
        description="Sleeve Offset",
        default=0.1,
        min=0.0,
        max=100.0,
        soft_min=0.0,
        soft_max=100.0,
        step=1,
        precision=1,
    )

    #########################################################################################
    # Mesh Tools Props :
    #########################################################################################

    # Decimate ratio prop :
    #######################
    no_material_prop: StringProperty(
        name="No Material",
        default="No Color",
        description="No active material found for active object",
    )
    decimate_ratio: FloatProperty(
        description="Enter decimate ratio ", default=0.5, step=1, precision=2
    )
    #########################################################################################

    CurveCutterNameProp: StringProperty(
        name="Cutter Name",
        default="",
        description="Current Cutter Object Name",
    )

    #####################

    CuttingTargetNameProp: StringProperty(
        name="Cutting Target Name",
        default="",
        description="Current Cutting Target Object Name",
    )

    #####################
    items = [
        
        "Path Split",
        "Ribbon Split",
        "Ribbon Cutter",
        "Frame Cutter",
        # "Curve Cutter 3",
        
        # "Paint Cutter",
        # "Path Cutter",
        
    ]

    Cutting_Tools_Types_Prop: EnumProperty(
        items=set_enum_items(items), description="Select a cutting tool", default="Path Split"
    )

    CurveCutCloseModeList = ["Open Curve", "Close Curve"]
    items = []
    for i in range(len(CurveCutCloseModeList)):
        item = (
            str(CurveCutCloseModeList[i]),
            str(CurveCutCloseModeList[i]),
            str(""),
            int(i),
        )
        items.append(item)
    CurveCutCloseMode: EnumProperty(items=items, description="", default="Close Curve")

    cutting_mode_list = ["Cut inner", "Keep inner"]
    items = []
    for i in range(len(cutting_mode_list)):
        item = (str(cutting_mode_list[i]), str(cutting_mode_list[i]), str(""), int(i))
        items.append(item)

    cutting_mode: EnumProperty(items=items, description="", default="Cut inner")

    TubeWidth: FloatProperty(description="Tube Width ", default=2, step=1, precision=2)

    TubeCloseModeList = ["Open Tube", "Close Tube"]
    items = []
    for i in range(len(TubeCloseModeList)):
        item = (str(TubeCloseModeList[i]), str(TubeCloseModeList[i]), str(""), int(i))
        items.append(item)
    TubeCloseMode: EnumProperty(items=items, description="", default="Close Tube")

    BaseHeight: FloatProperty(
        description="Base Height ", default=10, step=1, precision=2
    )
    SurveyInfo: StringProperty(
        name="Models Survey Local Z",
        default="{}",
        description="Models Survey Local Z",
    )

    #############################################################################################
    # BDENTAL Align Properties :
    #############################################################################################
    IcpVidDict: StringProperty(
        name="IcpVidDict",
        default="None",
        description="ICP Vertices Pairs str(Dict)",
    )

    #######################
    AlignModalState: BoolProperty(description="Align Modal state ", default=False)
    text : StringProperty( name = "3D text", default = "Bdental", update=update_text)
    slices_brightness: FloatProperty(
        description="Slices Brightness ", default=0.0, step=1, precision=3, update=brightness_update
    )
    slices_contrast: FloatProperty(
        description="Slices Contrast ", default=0.2, step=1, precision=3, update=contrast_update
    )

    # axial_slice_flip_vertical : BoolProperty(name="Axial Flip Vertical", default=True, description="Axial Flip Vertical")
    # coronal_slice_flip_vertical : BoolProperty(name="Coronal Flip Vertical", default=True, description="Coronal Flip Vertical")
    # sagital_slice_flip_vertical : BoolProperty(name="Sagital Flip Vertical", default=True, description="Sagital Flip Vertical")
    # axial_slice_flip_horizontal : BoolProperty(name="Axial Flip Horizontal", default=False, description="Axial Flip Horizontal")
    # coronal_slice_flip_horizontal : BoolProperty(name="Coronal Flip Horizontal", default=False, description="Coronal Flip Horizontal")
    # sagital_slice_flip_horizontal : BoolProperty(name="Sagital Flip Horizontal", default=False, description="Sagital Flip Horizontal")
#################################################################################################
# Registration :
#################################################################################################

classes = [
    BDENTAL_Props,
]


def register():

    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.BDENTAL_Props = bpy.props.PointerProperty(type=BDENTAL_Props)


def unregister():

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.BDENTAL_Props


