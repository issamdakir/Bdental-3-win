from typing import overload, Any, Callable, TypeVar, Union
from typing import Tuple, List, Sequence, MutableSequence

Callback = Union[Callable[..., None], None]
Buffer = TypeVar('Buffer')
Pointer = TypeVar('Pointer')
Template = TypeVar('Template')

import vtkmodules.vtkCommonCore
import vtkmodules.vtkCommonExecutionModel

class vtkHDFReader(vtkmodules.vtkCommonExecutionModel.vtkDataObjectAlgorithm):
    def CanReadFile(self, name:str) -> int: ...
    def GetCellArrayName(self, index:int) -> str: ...
    def GetCellDataArraySelection(self) -> 'vtkDataArraySelection': ...
    def GetFieldDataArraySelection(self) -> 'vtkDataArraySelection': ...
    def GetFileName(self) -> str: ...
    def GetHasTransientData(self) -> bool: ...
    def GetMaximumLevelsToReadByDefaultForAMR(self) -> int: ...
    def GetNumberOfCellArrays(self) -> int: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetNumberOfPointArrays(self) -> int: ...
    def GetNumberOfSteps(self) -> int: ...
    @overload
    def GetOutputAsDataSet(self) -> 'vtkDataSet': ...
    @overload
    def GetOutputAsDataSet(self, index:int) -> 'vtkDataSet': ...
    def GetPointArrayName(self, index:int) -> str: ...
    def GetPointDataArraySelection(self) -> 'vtkDataArraySelection': ...
    def GetStep(self) -> int: ...
    def GetTimeValue(self) -> float: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkHDFReader': ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkHDFReader': ...
    def SetFileName(self, _arg:str) -> None: ...
    def SetMaximumLevelsToReadByDefaultForAMR(self, _arg:int) -> None: ...
    def SetStep(self, _arg:int) -> None: ...

