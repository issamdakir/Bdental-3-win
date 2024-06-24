from typing import overload, Any, Callable, TypeVar, Union
from typing import Tuple, List, Sequence, MutableSequence

Callback = Union[Callable[..., None], None]
Buffer = TypeVar('Buffer')
Pointer = TypeVar('Pointer')
Template = TypeVar('Template')

import vtkmodules.vtkCommonCore
import vtkmodules.vtkRenderingCore
import vtkmodules.vtkRenderingOpenGL2

class vtkClientServerCompositePass(vtkmodules.vtkRenderingCore.vtkRenderPass):
    def GetController(self) -> 'vtkMultiProcessController': ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetPostProcessingRenderPass(self) -> 'vtkRenderPass': ...
    def GetProcessIsServer(self) -> bool: ...
    def GetRenderPass(self) -> 'vtkRenderPass': ...
    def GetServerSideRendering(self) -> bool: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkClientServerCompositePass': ...
    def ProcessIsServerOff(self) -> None: ...
    def ProcessIsServerOn(self) -> None: ...
    def ReleaseGraphicsResources(self, w:'vtkWindow') -> None: ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkClientServerCompositePass': ...
    def ServerSideRenderingOff(self) -> None: ...
    def ServerSideRenderingOn(self) -> None: ...
    def SetController(self, controller:'vtkMultiProcessController') -> None: ...
    def SetPostProcessingRenderPass(self, __a:'vtkRenderPass') -> None: ...
    def SetProcessIsServer(self, _arg:bool) -> None: ...
    def SetRenderPass(self, __a:'vtkRenderPass') -> None: ...
    def SetServerSideRendering(self, _arg:bool) -> None: ...

class vtkSynchronizedRenderers(vtkmodules.vtkCommonCore.vtkObject):
    COMPUTE_BOUNDS_TAG:int
    RESET_CAMERA_TAG:int
    SYNC_RENDERER_TAG:int
    def AutomaticEventHandlingOff(self) -> None: ...
    def AutomaticEventHandlingOn(self) -> None: ...
    def CollectiveExpandForVisiblePropBounds(self, bounds:MutableSequence[float]) -> None: ...
    def FixBackgroundOff(self) -> None: ...
    def FixBackgroundOn(self) -> None: ...
    def GetAutomaticEventHandling(self) -> bool: ...
    def GetCaptureDelegate(self) -> 'vtkSynchronizedRenderers': ...
    def GetFixBackground(self) -> bool: ...
    def GetImageReductionFactor(self) -> int: ...
    def GetImageReductionFactorMaxValue(self) -> int: ...
    def GetImageReductionFactorMinValue(self) -> int: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetParallelController(self) -> 'vtkMultiProcessController': ...
    def GetParallelRendering(self) -> bool: ...
    def GetRenderer(self) -> 'vtkRenderer': ...
    def GetRootProcessId(self) -> int: ...
    def GetWriteBackImages(self) -> bool: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkSynchronizedRenderers': ...
    def ParallelRenderingOff(self) -> None: ...
    def ParallelRenderingOn(self) -> None: ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkSynchronizedRenderers': ...
    def SetAutomaticEventHandling(self, _arg:bool) -> None: ...
    def SetCaptureDelegate(self, __a:'vtkSynchronizedRenderers') -> None: ...
    def SetFixBackground(self, _arg:bool) -> None: ...
    def SetImageReductionFactor(self, _arg:int) -> None: ...
    def SetParallelController(self, __a:'vtkMultiProcessController') -> None: ...
    def SetParallelRendering(self, _arg:bool) -> None: ...
    def SetRenderer(self, __a:'vtkRenderer') -> None: ...
    def SetRootProcessId(self, _arg:int) -> None: ...
    def SetWriteBackImages(self, _arg:bool) -> None: ...
    def WriteBackImagesOff(self) -> None: ...
    def WriteBackImagesOn(self) -> None: ...

class vtkClientServerSynchronizedRenderers(vtkSynchronizedRenderers):
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkClientServerSynchronizedRenderers': ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkClientServerSynchronizedRenderers': ...

class vtkCompositeRGBAPass(vtkmodules.vtkRenderingCore.vtkRenderPass):
    def GetController(self) -> 'vtkMultiProcessController': ...
    def GetKdtree(self) -> 'vtkPKdTree': ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def IsA(self, type:str) -> int: ...
    def IsSupported(self, context:'vtkOpenGLRenderWindow') -> bool: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkCompositeRGBAPass': ...
    def ReleaseGraphicsResources(self, w:'vtkWindow') -> None: ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkCompositeRGBAPass': ...
    def SetController(self, controller:'vtkMultiProcessController') -> None: ...
    def SetKdtree(self, kdtree:'vtkPKdTree') -> None: ...

class vtkParallelRenderManager(vtkmodules.vtkCommonCore.vtkObject):
    class Tags(int): ...
    BOUNDS_TAG:'Tags'
    COMPUTE_VISIBLE_PROP_BOUNDS_RMI_TAG:'Tags'
    LIGHT_INFO_TAG:'Tags'
    LINEAR:int
    NEAREST:int
    RENDER_RMI_TAG:'Tags'
    REN_ID_TAG:'Tags'
    REN_INFO_TAG:'Tags'
    WIN_INFO_TAG:'Tags'
    def AddRenderer(self, __a:'vtkRenderer') -> None: ...
    def AutoImageReductionFactorOff(self) -> None: ...
    def AutoImageReductionFactorOn(self) -> None: ...
    def CheckForAbortComposite(self) -> int: ...
    def CheckForAbortRender(self) -> None: ...
    def ComputeVisiblePropBounds(self, ren:'vtkRenderer', bounds:MutableSequence[float]) -> None: ...
    def ComputeVisiblePropBoundsRMI(self, renderId:int) -> None: ...
    def EndRender(self) -> None: ...
    def GenericEndRenderCallback(self) -> None: ...
    def GenericStartRenderCallback(self) -> None: ...
    def GetAutoImageReductionFactor(self) -> int: ...
    def GetController(self) -> 'vtkMultiProcessController': ...
    @staticmethod
    def GetDefaultRenderEventPropagation() -> bool: ...
    def GetForceRenderWindowSize(self) -> int: ...
    def GetForcedRenderWindowSize(self) -> Tuple[int, int]: ...
    def GetFullImageSize(self) -> Tuple[int, int]: ...
    def GetImageProcessingTime(self) -> float: ...
    def GetImageReductionFactor(self) -> float: ...
    def GetMagnifyImageMethod(self) -> int: ...
    def GetMagnifyImages(self) -> int: ...
    def GetMaxImageReductionFactor(self) -> float: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetParallelRendering(self) -> int: ...
    @overload
    def GetPixelData(self, data:'vtkUnsignedCharArray') -> None: ...
    @overload
    def GetPixelData(self, x1:int, y1:int, x2:int, y2:int, data:'vtkUnsignedCharArray') -> None: ...
    def GetReducedImageSize(self) -> Tuple[int, int]: ...
    @overload
    def GetReducedPixelData(self, data:'vtkUnsignedCharArray') -> None: ...
    @overload
    def GetReducedPixelData(self, x1:int, y1:int, x2:int, y2:int, data:'vtkUnsignedCharArray') -> None: ...
    def GetRenderEventPropagation(self) -> int: ...
    def GetRenderTime(self) -> float: ...
    def GetRenderWindow(self) -> 'vtkRenderWindow': ...
    def GetSyncRenderWindowRenderers(self) -> int: ...
    def GetSynchronizeTileProperties(self) -> int: ...
    def GetUseBackBuffer(self) -> int: ...
    def GetUseCompositing(self) -> int: ...
    def GetUseRGBA(self) -> int: ...
    def GetWriteBackImages(self) -> int: ...
    def InitializeOffScreen(self) -> None: ...
    def InitializePieces(self) -> None: ...
    def InitializeRMIs(self) -> None: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def MagnifyImage(self, fullImage:'vtkUnsignedCharArray', fullImageSize:Sequence[int], reducedImage:'vtkUnsignedCharArray', reducedImageSize:Sequence[int], fullImageViewport:Sequence[int]=..., reducedImageViewport:Sequence[int]=...) -> None: ...
    @staticmethod
    def MagnifyImageLinear(fullImage:'vtkUnsignedCharArray', fullImageSize:Sequence[int], reducedImage:'vtkUnsignedCharArray', reducedImageSize:Sequence[int], fullImageViewport:Sequence[int]=..., reducedImageViewport:Sequence[int]=...) -> None: ...
    @staticmethod
    def MagnifyImageNearest(fullImage:'vtkUnsignedCharArray', fullImageSize:Sequence[int], reducedImage:'vtkUnsignedCharArray', reducedImageSize:Sequence[int], fullImageViewport:Sequence[int]=..., reducedImageViewport:Sequence[int]=...) -> None: ...
    def MagnifyImagesOff(self) -> None: ...
    def MagnifyImagesOn(self) -> None: ...
    def MakeRenderWindow(self) -> 'vtkRenderWindow': ...
    def MakeRenderer(self) -> 'vtkRenderer': ...
    def NewInstance(self) -> 'vtkParallelRenderManager': ...
    def ParallelRenderingOff(self) -> None: ...
    def ParallelRenderingOn(self) -> None: ...
    def RemoveAllRenderers(self) -> None: ...
    def RemoveRenderer(self, __a:'vtkRenderer') -> None: ...
    def RenderEventPropagationOff(self) -> None: ...
    def RenderEventPropagationOn(self) -> None: ...
    def RenderRMI(self) -> None: ...
    def ResetAllCameras(self) -> None: ...
    def ResetCamera(self, ren:'vtkRenderer') -> None: ...
    def ResetCameraClippingRange(self, ren:'vtkRenderer') -> None: ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkParallelRenderManager': ...
    def SatelliteEndRender(self) -> None: ...
    def SatelliteStartRender(self) -> None: ...
    def SetAutoImageReductionFactor(self, _arg:int) -> None: ...
    def SetController(self, controller:'vtkMultiProcessController') -> None: ...
    @staticmethod
    def SetDefaultRenderEventPropagation(val:bool) -> None: ...
    def SetForceRenderWindowSize(self, _arg:int) -> None: ...
    @overload
    def SetForcedRenderWindowSize(self, _arg1:int, _arg2:int) -> None: ...
    @overload
    def SetForcedRenderWindowSize(self, _arg:Sequence[int]) -> None: ...
    def SetImageReductionFactor(self, factor:float) -> None: ...
    def SetImageReductionFactorForUpdateRate(self, DesiredUpdateRate:float) -> None: ...
    def SetMagnifyImageMethod(self, method:int) -> None: ...
    def SetMagnifyImageMethodToLinear(self) -> None: ...
    def SetMagnifyImageMethodToNearest(self) -> None: ...
    def SetMagnifyImages(self, _arg:int) -> None: ...
    def SetMaxImageReductionFactor(self, _arg:float) -> None: ...
    def SetParallelRendering(self, _arg:int) -> None: ...
    def SetRenderEventPropagation(self, _arg:int) -> None: ...
    def SetRenderWindow(self, renWin:'vtkRenderWindow') -> None: ...
    def SetSyncRenderWindowRenderers(self, _arg:int) -> None: ...
    def SetSynchronizeTileProperties(self, _arg:int) -> None: ...
    def SetUseBackBuffer(self, _arg:int) -> None: ...
    def SetUseCompositing(self, _arg:int) -> None: ...
    def SetUseRGBA(self, _arg:int) -> None: ...
    def SetWriteBackImages(self, _arg:int) -> None: ...
    def StartInteractor(self) -> None: ...
    def StartRender(self) -> None: ...
    def StartServices(self) -> None: ...
    def StopServices(self) -> None: ...
    def SyncRenderWindowRenderersOff(self) -> None: ...
    def SyncRenderWindowRenderersOn(self) -> None: ...
    def SynchronizeTilePropertiesOff(self) -> None: ...
    def SynchronizeTilePropertiesOn(self) -> None: ...
    def TileWindows(self, xsize:int, ysize:int, nColumns:int) -> None: ...
    def UseBackBufferOff(self) -> None: ...
    def UseBackBufferOn(self) -> None: ...
    def UseCompositingOff(self) -> None: ...
    def UseCompositingOn(self) -> None: ...
    def WriteBackImagesOff(self) -> None: ...
    def WriteBackImagesOn(self) -> None: ...

class vtkCompositeRenderManager(vtkParallelRenderManager):
    def GetCompositer(self) -> 'vtkCompositer': ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkCompositeRenderManager': ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkCompositeRenderManager': ...
    def SetCompositer(self, c:'vtkCompositer') -> None: ...

class vtkCompositeZPass(vtkmodules.vtkRenderingCore.vtkRenderPass):
    def GetController(self) -> 'vtkMultiProcessController': ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def IsA(self, type:str) -> int: ...
    def IsSupported(self, context:'vtkOpenGLRenderWindow') -> bool: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkCompositeZPass': ...
    def ReleaseGraphicsResources(self, w:'vtkWindow') -> None: ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkCompositeZPass': ...
    def SetController(self, controller:'vtkMultiProcessController') -> None: ...

class vtkCompositedSynchronizedRenderers(vtkSynchronizedRenderers):
    def GetCompositer(self) -> 'vtkCompositer': ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkCompositedSynchronizedRenderers': ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkCompositedSynchronizedRenderers': ...
    def SetCompositer(self, __a:'vtkCompositer') -> None: ...

class vtkCompositer(vtkmodules.vtkCommonCore.vtkObject):
    def CompositeBuffer(self, pBuf:'vtkDataArray', zBuf:'vtkFloatArray', pTmp:'vtkDataArray', zTmp:'vtkFloatArray') -> None: ...
    @staticmethod
    def DeleteArray(da:'vtkDataArray') -> None: ...
    def GetController(self) -> 'vtkMultiProcessController': ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetNumberOfProcesses(self) -> int: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkCompositer': ...
    @staticmethod
    def ResizeFloatArray(fa:'vtkFloatArray', numComp:int, size:int) -> None: ...
    @staticmethod
    def ResizeUnsignedCharArray(uca:'vtkUnsignedCharArray', numComp:int, size:int) -> None: ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkCompositer': ...
    def SetController(self, __a:'vtkMultiProcessController') -> None: ...
    def SetNumberOfProcesses(self, _arg:int) -> None: ...

class vtkCompressCompositer(vtkCompositer):
    def CompositeBuffer(self, pBuf:'vtkDataArray', zBuf:'vtkFloatArray', pTmp:'vtkDataArray', zTmp:'vtkFloatArray') -> None: ...
    @staticmethod
    def CompositeImagePair(localZ:'vtkFloatArray', localP:'vtkDataArray', remoteZ:'vtkFloatArray', remoteP:'vtkDataArray', outZ:'vtkFloatArray', outP:'vtkDataArray') -> None: ...
    @staticmethod
    def Compress(zIn:'vtkFloatArray', pIn:'vtkDataArray', zOut:'vtkFloatArray', pOut:'vtkDataArray') -> None: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkCompressCompositer': ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkCompressCompositer': ...
    @staticmethod
    def Uncompress(zIn:'vtkFloatArray', pIn:'vtkDataArray', zOut:'vtkFloatArray', pOut:'vtkDataArray', lengthOut:int) -> None: ...

class vtkImageRenderManager(vtkParallelRenderManager):
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkImageRenderManager': ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkImageRenderManager': ...

class vtkPHardwareSelector(vtkmodules.vtkRenderingOpenGL2.vtkOpenGLHardwareSelector):
    def CaptureBuffers(self) -> bool: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetProcessIsRoot(self) -> bool: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkPHardwareSelector': ...
    def ProcessIsRootOff(self) -> None: ...
    def ProcessIsRootOn(self) -> None: ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkPHardwareSelector': ...
    def SetProcessIsRoot(self, _arg:bool) -> None: ...

class vtkSynchronizedRenderWindows(vtkmodules.vtkCommonCore.vtkObject):
    SYNC_RENDER_TAG:int
    def AbortRender(self) -> None: ...
    def GetIdentifier(self) -> int: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetParallelController(self) -> 'vtkMultiProcessController': ...
    def GetParallelRendering(self) -> bool: ...
    def GetRenderEventPropagation(self) -> bool: ...
    def GetRenderWindow(self) -> 'vtkRenderWindow': ...
    def GetRootProcessId(self) -> int: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkSynchronizedRenderWindows': ...
    def ParallelRenderingOff(self) -> None: ...
    def ParallelRenderingOn(self) -> None: ...
    def RenderEventPropagationOff(self) -> None: ...
    def RenderEventPropagationOn(self) -> None: ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkSynchronizedRenderWindows': ...
    def SetIdentifier(self, id:int) -> None: ...
    def SetParallelController(self, __a:'vtkMultiProcessController') -> None: ...
    def SetParallelRendering(self, _arg:bool) -> None: ...
    def SetRenderEventPropagation(self, _arg:bool) -> None: ...
    def SetRenderWindow(self, __a:'vtkRenderWindow') -> None: ...
    def SetRootProcessId(self, _arg:int) -> None: ...

class vtkTreeCompositer(vtkCompositer):
    def CompositeBuffer(self, pBuf:'vtkDataArray', zBuf:'vtkFloatArray', pTmp:'vtkDataArray', zTmp:'vtkFloatArray') -> None: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkTreeCompositer': ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkTreeCompositer': ...

