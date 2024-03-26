__all__: list[str] = []

import cv2
import cv2.typing
import typing as _typing


from cv2.ximgproc import segmentation as segmentation


# Enumerations
THINNING_ZHANGSUEN: int
THINNING_GUOHALL: int
ThinningTypes = int
"""One of [THINNING_ZHANGSUEN, THINNING_GUOHALL]"""

BINARIZATION_NIBLACK: int
BINARIZATION_SAUVOLA: int
BINARIZATION_WOLF: int
BINARIZATION_NICK: int
LocalBinarizationMethods = int
"""One of [BINARIZATION_NIBLACK, BINARIZATION_SAUVOLA, BINARIZATION_WOLF, BINARIZATION_NICK]"""

DTF_NC: int
DTF_IC: int
DTF_RF: int
GUIDED_FILTER: int
AM_FILTER: int
EdgeAwareFiltersList = int
"""One of [DTF_NC, DTF_IC, DTF_RF, GUIDED_FILTER, AM_FILTER]"""

ARO_0_45: int
ARO_45_90: int
ARO_90_135: int
ARO_315_0: int
ARO_315_45: int
ARO_45_135: int
ARO_315_135: int
ARO_CTR_HOR: int
ARO_CTR_VER: int
AngleRangeOption = int
"""One of [ARO_0_45, ARO_45_90, ARO_90_135, ARO_315_0, ARO_315_45, ARO_45_135, ARO_315_135, ARO_CTR_HOR, ARO_CTR_VER]"""

FHT_MIN: int
FHT_MAX: int
FHT_ADD: int
FHT_AVE: int
HoughOp = int
"""One of [FHT_MIN, FHT_MAX, FHT_ADD, FHT_AVE]"""

HDO_RAW: int
HDO_DESKEW: int
HoughDeskewOption = int
"""One of [HDO_RAW, HDO_DESKEW]"""

SLIC: int
SLICO: int
MSLIC: int
SLICType = int
"""One of [SLIC, SLICO, MSLIC]"""

WMF_EXP: int
WMF_IV1: int
WMF_IV2: int
WMF_COS: int
WMF_JAC: int
WMF_OFF: int
WMFWeightType = int
"""One of [WMF_EXP, WMF_IV1, WMF_IV2, WMF_COS, WMF_JAC, WMF_OFF]"""


EdgeDrawing_PREWITT: int
EDGE_DRAWING_PREWITT: int
EdgeDrawing_SOBEL: int
EDGE_DRAWING_SOBEL: int
EdgeDrawing_SCHARR: int
EDGE_DRAWING_SCHARR: int
EdgeDrawing_LSD: int
EDGE_DRAWING_LSD: int
EdgeDrawing_GradientOperator = int
"""One of [EdgeDrawing_PREWITT, EDGE_DRAWING_PREWITT, EdgeDrawing_SOBEL, EDGE_DRAWING_SOBEL, EdgeDrawing_SCHARR, EDGE_DRAWING_SCHARR, EdgeDrawing_LSD, EDGE_DRAWING_LSD]"""


# Classes
class DisparityFilter(cv2.Algorithm):
    # Functions
    @_typing.overload
    def filter(self, disparity_map_left: cv2.typing.MatLike, left_view: cv2.typing.MatLike, filtered_disparity_map: cv2.typing.MatLike | None = ..., disparity_map_right: cv2.typing.MatLike | None = ..., ROI: cv2.typing.Rect = ..., right_view: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
    @_typing.overload
    def filter(self, disparity_map_left: cv2.UMat, left_view: cv2.UMat, filtered_disparity_map: cv2.UMat | None = ..., disparity_map_right: cv2.UMat | None = ..., ROI: cv2.typing.Rect = ..., right_view: cv2.UMat | None = ...) -> cv2.UMat: ...


class DisparityWLSFilter(DisparityFilter):
    # Functions
    def getLambda(self) -> float: ...

    def setLambda(self, _lambda: float) -> None: ...

    def getSigmaColor(self) -> float: ...

    def setSigmaColor(self, _sigma_color: float) -> None: ...

    def getLRCthresh(self) -> int: ...

    def setLRCthresh(self, _LRC_thresh: int) -> None: ...

    def getDepthDiscontinuityRadius(self) -> int: ...

    def setDepthDiscontinuityRadius(self, _disc_radius: int) -> None: ...

    def getConfidenceMap(self) -> cv2.typing.MatLike: ...

    def getROI(self) -> cv2.typing.Rect: ...


class EdgeDrawing(cv2.Algorithm):
    # Classes
    class Params:
        PFmode: bool
        EdgeDetectionOperator: int
        GradientThresholdValue: int
        AnchorThresholdValue: int
        ScanInterval: int
        MinPathLength: int
        Sigma: float
        SumFlag: bool
        NFAValidation: bool
        MinLineLength: int
        MaxDistanceBetweenTwoLines: float
        LineFitErrorThreshold: float
        MaxErrorThreshold: float

        # Functions
        def __init__(self) -> None: ...



    # Functions
    @_typing.overload
    def detectEdges(self, src: cv2.typing.MatLike) -> None: ...
    @_typing.overload
    def detectEdges(self, src: cv2.UMat) -> None: ...

    @_typing.overload
    def getEdgeImage(self, dst: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
    @_typing.overload
    def getEdgeImage(self, dst: cv2.UMat | None = ...) -> cv2.UMat: ...

    @_typing.overload
    def getGradientImage(self, dst: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
    @_typing.overload
    def getGradientImage(self, dst: cv2.UMat | None = ...) -> cv2.UMat: ...

    def getSegments(self) -> _typing.Sequence[_typing.Sequence[cv2.typing.Point]]: ...

    def getSegmentIndicesOfLines(self) -> _typing.Sequence[int]: ...

    @_typing.overload
    def detectLines(self, lines: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
    @_typing.overload
    def detectLines(self, lines: cv2.UMat | None = ...) -> cv2.UMat: ...

    @_typing.overload
    def detectEllipses(self, ellipses: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
    @_typing.overload
    def detectEllipses(self, ellipses: cv2.UMat | None = ...) -> cv2.UMat: ...

    def setParams(self, parameters: EdgeDrawing.Params) -> None: ...


class DTFilter(cv2.Algorithm):
    # Functions
    @_typing.overload
    def filter(self, src: cv2.typing.MatLike, dst: cv2.typing.MatLike | None = ..., dDepth: int = ...) -> cv2.typing.MatLike: ...
    @_typing.overload
    def filter(self, src: cv2.UMat, dst: cv2.UMat | None = ..., dDepth: int = ...) -> cv2.UMat: ...


class GuidedFilter(cv2.Algorithm):
    # Functions
    @_typing.overload
    def filter(self, src: cv2.typing.MatLike, dst: cv2.typing.MatLike | None = ..., dDepth: int = ...) -> cv2.typing.MatLike: ...
    @_typing.overload
    def filter(self, src: cv2.UMat, dst: cv2.UMat | None = ..., dDepth: int = ...) -> cv2.UMat: ...


class AdaptiveManifoldFilter(cv2.Algorithm):
    # Functions
    @_typing.overload
    def filter(self, src: cv2.typing.MatLike, dst: cv2.typing.MatLike | None = ..., joint: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
    @_typing.overload
    def filter(self, src: cv2.UMat, dst: cv2.UMat | None = ..., joint: cv2.UMat | None = ...) -> cv2.UMat: ...

    def collectGarbage(self) -> None: ...

    @classmethod
    def create(cls) -> AdaptiveManifoldFilter: ...


class FastBilateralSolverFilter(cv2.Algorithm):
    # Functions
    @_typing.overload
    def filter(self, src: cv2.typing.MatLike, confidence: cv2.typing.MatLike, dst: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
    @_typing.overload
    def filter(self, src: cv2.UMat, confidence: cv2.UMat, dst: cv2.UMat | None = ...) -> cv2.UMat: ...


class FastGlobalSmootherFilter(cv2.Algorithm):
    # Functions
    @_typing.overload
    def filter(self, src: cv2.typing.MatLike, dst: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
    @_typing.overload
    def filter(self, src: cv2.UMat, dst: cv2.UMat | None = ...) -> cv2.UMat: ...


class EdgeBoxes(cv2.Algorithm):
    # Functions
    @_typing.overload
    def getBoundingBoxes(self, edge_map: cv2.typing.MatLike, orientation_map: cv2.typing.MatLike, scores: cv2.typing.MatLike | None = ...) -> tuple[_typing.Sequence[cv2.typing.Rect], cv2.typing.MatLike]: ...
    @_typing.overload
    def getBoundingBoxes(self, edge_map: cv2.UMat, orientation_map: cv2.UMat, scores: cv2.UMat | None = ...) -> tuple[_typing.Sequence[cv2.typing.Rect], cv2.UMat]: ...

    def getAlpha(self) -> float: ...

    def setAlpha(self, value: float) -> None: ...

    def getBeta(self) -> float: ...

    def setBeta(self, value: float) -> None: ...

    def getEta(self) -> float: ...

    def setEta(self, value: float) -> None: ...

    def getMinScore(self) -> float: ...

    def setMinScore(self, value: float) -> None: ...

    def getMaxBoxes(self) -> int: ...

    def setMaxBoxes(self, value: int) -> None: ...

    def getEdgeMinMag(self) -> float: ...

    def setEdgeMinMag(self, value: float) -> None: ...

    def getEdgeMergeThr(self) -> float: ...

    def setEdgeMergeThr(self, value: float) -> None: ...

    def getClusterMinMag(self) -> float: ...

    def setClusterMinMag(self, value: float) -> None: ...

    def getMaxAspectRatio(self) -> float: ...

    def setMaxAspectRatio(self, value: float) -> None: ...

    def getMinBoxArea(self) -> float: ...

    def setMinBoxArea(self, value: float) -> None: ...

    def getGamma(self) -> float: ...

    def setGamma(self, value: float) -> None: ...

    def getKappa(self) -> float: ...

    def setKappa(self, value: float) -> None: ...


class FastLineDetector(cv2.Algorithm):
    # Functions
    @_typing.overload
    def detect(self, image: cv2.typing.MatLike, lines: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
    @_typing.overload
    def detect(self, image: cv2.UMat, lines: cv2.UMat | None = ...) -> cv2.UMat: ...

    @_typing.overload
    def drawSegments(self, image: cv2.typing.MatLike, lines: cv2.typing.MatLike, draw_arrow: bool = ..., linecolor: cv2.typing.Scalar = ..., linethickness: int = ...) -> cv2.typing.MatLike: ...
    @_typing.overload
    def drawSegments(self, image: cv2.UMat, lines: cv2.UMat, draw_arrow: bool = ..., linecolor: cv2.typing.Scalar = ..., linethickness: int = ...) -> cv2.UMat: ...


class ContourFitting(cv2.Algorithm):
    # Functions
    @_typing.overload
    def estimateTransformation(self, src: cv2.typing.MatLike, dst: cv2.typing.MatLike, alphaPhiST: cv2.typing.MatLike | None = ..., fdContour: bool = ...) -> tuple[cv2.typing.MatLike, float]: ...
    @_typing.overload
    def estimateTransformation(self, src: cv2.UMat, dst: cv2.UMat, alphaPhiST: cv2.UMat | None = ..., fdContour: bool = ...) -> tuple[cv2.UMat, float]: ...

    def setCtrSize(self, n: int) -> None: ...

    def setFDSize(self, n: int) -> None: ...

    def getCtrSize(self) -> int: ...

    def getFDSize(self) -> int: ...


class SuperpixelLSC(cv2.Algorithm):
    # Functions
    def getNumberOfSuperpixels(self) -> int: ...

    def iterate(self, num_iterations: int = ...) -> None: ...

    @_typing.overload
    def getLabels(self, labels_out: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
    @_typing.overload
    def getLabels(self, labels_out: cv2.UMat | None = ...) -> cv2.UMat: ...

    @_typing.overload
    def getLabelContourMask(self, image: cv2.typing.MatLike | None = ..., thick_line: bool = ...) -> cv2.typing.MatLike: ...
    @_typing.overload
    def getLabelContourMask(self, image: cv2.UMat | None = ..., thick_line: bool = ...) -> cv2.UMat: ...

    def enforceLabelConnectivity(self, min_element_size: int = ...) -> None: ...


class RidgeDetectionFilter(cv2.Algorithm):
    # Functions
    @classmethod
    def create(cls, ddepth: int = ..., dx: int = ..., dy: int = ..., ksize: int = ..., out_dtype: int = ..., scale: float = ..., delta: float = ..., borderType: int = ...) -> RidgeDetectionFilter: ...

    @_typing.overload
    def getRidgeFilteredImage(self, _img: cv2.typing.MatLike, out: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
    @_typing.overload
    def getRidgeFilteredImage(self, _img: cv2.UMat, out: cv2.UMat | None = ...) -> cv2.UMat: ...


class ScanSegment(cv2.Algorithm):
    # Functions
    def getNumberOfSuperpixels(self) -> int: ...

    @_typing.overload
    def iterate(self, img: cv2.typing.MatLike) -> None: ...
    @_typing.overload
    def iterate(self, img: cv2.UMat) -> None: ...

    @_typing.overload
    def getLabels(self, labels_out: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
    @_typing.overload
    def getLabels(self, labels_out: cv2.UMat | None = ...) -> cv2.UMat: ...

    @_typing.overload
    def getLabelContourMask(self, image: cv2.typing.MatLike | None = ..., thick_line: bool = ...) -> cv2.typing.MatLike: ...
    @_typing.overload
    def getLabelContourMask(self, image: cv2.UMat | None = ..., thick_line: bool = ...) -> cv2.UMat: ...


class SuperpixelSEEDS(cv2.Algorithm):
    # Functions
    def getNumberOfSuperpixels(self) -> int: ...

    @_typing.overload
    def iterate(self, img: cv2.typing.MatLike, num_iterations: int = ...) -> None: ...
    @_typing.overload
    def iterate(self, img: cv2.UMat, num_iterations: int = ...) -> None: ...

    @_typing.overload
    def getLabels(self, labels_out: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
    @_typing.overload
    def getLabels(self, labels_out: cv2.UMat | None = ...) -> cv2.UMat: ...

    @_typing.overload
    def getLabelContourMask(self, image: cv2.typing.MatLike | None = ..., thick_line: bool = ...) -> cv2.typing.MatLike: ...
    @_typing.overload
    def getLabelContourMask(self, image: cv2.UMat | None = ..., thick_line: bool = ...) -> cv2.UMat: ...


class SuperpixelSLIC(cv2.Algorithm):
    # Functions
    def getNumberOfSuperpixels(self) -> int: ...

    def iterate(self, num_iterations: int = ...) -> None: ...

    @_typing.overload
    def getLabels(self, labels_out: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
    @_typing.overload
    def getLabels(self, labels_out: cv2.UMat | None = ...) -> cv2.UMat: ...

    @_typing.overload
    def getLabelContourMask(self, image: cv2.typing.MatLike | None = ..., thick_line: bool = ...) -> cv2.typing.MatLike: ...
    @_typing.overload
    def getLabelContourMask(self, image: cv2.UMat | None = ..., thick_line: bool = ...) -> cv2.UMat: ...

    def enforceLabelConnectivity(self, min_element_size: int = ...) -> None: ...


class SparseMatchInterpolator(cv2.Algorithm):
    # Functions
    @_typing.overload
    def interpolate(self, from_image: cv2.typing.MatLike, from_points: cv2.typing.MatLike, to_image: cv2.typing.MatLike, to_points: cv2.typing.MatLike, dense_flow: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
    @_typing.overload
    def interpolate(self, from_image: cv2.UMat, from_points: cv2.UMat, to_image: cv2.UMat, to_points: cv2.UMat, dense_flow: cv2.UMat | None = ...) -> cv2.UMat: ...


class EdgeAwareInterpolator(SparseMatchInterpolator):
    # Functions
    def setCostMap(self, _costMap: cv2.typing.MatLike) -> None: ...

    def setK(self, _k: int) -> None: ...

    def getK(self) -> int: ...

    def setSigma(self, _sigma: float) -> None: ...

    def getSigma(self) -> float: ...

    def setLambda(self, _lambda: float) -> None: ...

    def getLambda(self) -> float: ...

    def setUsePostProcessing(self, _use_post_proc: bool) -> None: ...

    def getUsePostProcessing(self) -> bool: ...

    def setFGSLambda(self, _lambda: float) -> None: ...

    def getFGSLambda(self) -> float: ...

    def setFGSSigma(self, _sigma: float) -> None: ...

    def getFGSSigma(self) -> float: ...


class RICInterpolator(SparseMatchInterpolator):
    # Functions
    def setK(self, k: int = ...) -> None: ...

    def getK(self) -> int: ...

    def setCostMap(self, costMap: cv2.typing.MatLike) -> None: ...

    def setSuperpixelSize(self, spSize: int = ...) -> None: ...

    def getSuperpixelSize(self) -> int: ...

    def setSuperpixelNNCnt(self, spNN: int = ...) -> None: ...

    def getSuperpixelNNCnt(self) -> int: ...

    def setSuperpixelRuler(self, ruler: float = ...) -> None: ...

    def getSuperpixelRuler(self) -> float: ...

    def setSuperpixelMode(self, mode: int = ...) -> None: ...

    def getSuperpixelMode(self) -> int: ...

    def setAlpha(self, alpha: float = ...) -> None: ...

    def getAlpha(self) -> float: ...

    def setModelIter(self, modelIter: int = ...) -> None: ...

    def getModelIter(self) -> int: ...

    def setRefineModels(self, refineModles: bool = ...) -> None: ...

    def getRefineModels(self) -> bool: ...

    def setMaxFlow(self, maxFlow: float = ...) -> None: ...

    def getMaxFlow(self) -> float: ...

    def setUseVariationalRefinement(self, use_variational_refinement: bool = ...) -> None: ...

    def getUseVariationalRefinement(self) -> bool: ...

    def setUseGlobalSmootherFilter(self, use_FGS: bool = ...) -> None: ...

    def getUseGlobalSmootherFilter(self) -> bool: ...

    def setFGSLambda(self, lambda_: float = ...) -> None: ...

    def getFGSLambda(self) -> float: ...

    def setFGSSigma(self, sigma: float = ...) -> None: ...

    def getFGSSigma(self) -> float: ...


class RFFeatureGetter(cv2.Algorithm):
    # Functions
    def getFeatures(self, src: cv2.typing.MatLike, features: cv2.typing.MatLike, gnrmRad: int, gsmthRad: int, shrink: int, outNum: int, gradNum: int) -> None: ...


class StructuredEdgeDetection(cv2.Algorithm):
    # Functions
    @_typing.overload
    def detectEdges(self, src: cv2.typing.MatLike, dst: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
    @_typing.overload
    def detectEdges(self, src: cv2.UMat, dst: cv2.UMat | None = ...) -> cv2.UMat: ...

    @_typing.overload
    def computeOrientation(self, src: cv2.typing.MatLike, dst: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
    @_typing.overload
    def computeOrientation(self, src: cv2.UMat, dst: cv2.UMat | None = ...) -> cv2.UMat: ...

    @_typing.overload
    def edgesNms(self, edge_image: cv2.typing.MatLike, orientation_image: cv2.typing.MatLike, dst: cv2.typing.MatLike | None = ..., r: int = ..., s: int = ..., m: float = ..., isParallel: bool = ...) -> cv2.typing.MatLike: ...
    @_typing.overload
    def edgesNms(self, edge_image: cv2.UMat, orientation_image: cv2.UMat, dst: cv2.UMat | None = ..., r: int = ..., s: int = ..., m: float = ..., isParallel: bool = ...) -> cv2.UMat: ...



# Functions
@_typing.overload
def FastHoughTransform(src: cv2.typing.MatLike, dstMatDepth: int, dst: cv2.typing.MatLike | None = ..., angleRange: int = ..., op: int = ..., makeSkew: int = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def FastHoughTransform(src: cv2.UMat, dstMatDepth: int, dst: cv2.UMat | None = ..., angleRange: int = ..., op: int = ..., makeSkew: int = ...) -> cv2.UMat: ...

@_typing.overload
def GradientDericheX(op: cv2.typing.MatLike, alpha: float, omega: float, dst: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def GradientDericheX(op: cv2.UMat, alpha: float, omega: float, dst: cv2.UMat | None = ...) -> cv2.UMat: ...

@_typing.overload
def GradientDericheY(op: cv2.typing.MatLike, alpha: float, omega: float, dst: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def GradientDericheY(op: cv2.UMat, alpha: float, omega: float, dst: cv2.UMat | None = ...) -> cv2.UMat: ...

@_typing.overload
def HoughPoint2Line(houghPoint: cv2.typing.Point, srcImgInfo: cv2.typing.MatLike, angleRange: int = ..., makeSkew: int = ..., rules: int = ...) -> cv2.typing.Vec4i: ...
@_typing.overload
def HoughPoint2Line(houghPoint: cv2.typing.Point, srcImgInfo: cv2.UMat, angleRange: int = ..., makeSkew: int = ..., rules: int = ...) -> cv2.typing.Vec4i: ...

@_typing.overload
def PeiLinNormalization(I: cv2.typing.MatLike, T: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def PeiLinNormalization(I: cv2.UMat, T: cv2.UMat | None = ...) -> cv2.UMat: ...

@_typing.overload
def RadonTransform(src: cv2.typing.MatLike, dst: cv2.typing.MatLike | None = ..., theta: float = ..., start_angle: float = ..., end_angle: float = ..., crop: bool = ..., norm: bool = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def RadonTransform(src: cv2.UMat, dst: cv2.UMat | None = ..., theta: float = ..., start_angle: float = ..., end_angle: float = ..., crop: bool = ..., norm: bool = ...) -> cv2.UMat: ...

@_typing.overload
def amFilter(joint: cv2.typing.MatLike, src: cv2.typing.MatLike, sigma_s: float, sigma_r: float, dst: cv2.typing.MatLike | None = ..., adjust_outliers: bool = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def amFilter(joint: cv2.UMat, src: cv2.UMat, sigma_s: float, sigma_r: float, dst: cv2.UMat | None = ..., adjust_outliers: bool = ...) -> cv2.UMat: ...

@_typing.overload
def anisotropicDiffusion(src: cv2.typing.MatLike, alpha: float, K: float, niters: int, dst: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def anisotropicDiffusion(src: cv2.UMat, alpha: float, K: float, niters: int, dst: cv2.UMat | None = ...) -> cv2.UMat: ...

@_typing.overload
def bilateralTextureFilter(src: cv2.typing.MatLike, dst: cv2.typing.MatLike | None = ..., fr: int = ..., numIter: int = ..., sigmaAlpha: float = ..., sigmaAvg: float = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def bilateralTextureFilter(src: cv2.UMat, dst: cv2.UMat | None = ..., fr: int = ..., numIter: int = ..., sigmaAlpha: float = ..., sigmaAvg: float = ...) -> cv2.UMat: ...

@_typing.overload
def colorMatchTemplate(img: cv2.typing.MatLike, templ: cv2.typing.MatLike, result: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def colorMatchTemplate(img: cv2.UMat, templ: cv2.UMat, result: cv2.UMat | None = ...) -> cv2.UMat: ...

@_typing.overload
def computeBadPixelPercent(GT: cv2.typing.MatLike, src: cv2.typing.MatLike, ROI: cv2.typing.Rect, thresh: int = ...) -> float: ...
@_typing.overload
def computeBadPixelPercent(GT: cv2.UMat, src: cv2.UMat, ROI: cv2.typing.Rect, thresh: int = ...) -> float: ...

@_typing.overload
def computeMSE(GT: cv2.typing.MatLike, src: cv2.typing.MatLike, ROI: cv2.typing.Rect) -> float: ...
@_typing.overload
def computeMSE(GT: cv2.UMat, src: cv2.UMat, ROI: cv2.typing.Rect) -> float: ...

@_typing.overload
def contourSampling(src: cv2.typing.MatLike, nbElt: int, out: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def contourSampling(src: cv2.UMat, nbElt: int, out: cv2.UMat | None = ...) -> cv2.UMat: ...

@_typing.overload
def covarianceEstimation(src: cv2.typing.MatLike, windowRows: int, windowCols: int, dst: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def covarianceEstimation(src: cv2.UMat, windowRows: int, windowCols: int, dst: cv2.UMat | None = ...) -> cv2.UMat: ...

def createAMFilter(sigma_s: float, sigma_r: float, adjust_outliers: bool = ...) -> AdaptiveManifoldFilter: ...

def createContourFitting(ctr: int = ..., fd: int = ...) -> ContourFitting: ...

@_typing.overload
def createDTFilter(guide: cv2.typing.MatLike, sigmaSpatial: float, sigmaColor: float, mode: int = ..., numIters: int = ...) -> DTFilter: ...
@_typing.overload
def createDTFilter(guide: cv2.UMat, sigmaSpatial: float, sigmaColor: float, mode: int = ..., numIters: int = ...) -> DTFilter: ...

def createDisparityWLSFilter(matcher_left: cv2.StereoMatcher) -> DisparityWLSFilter: ...

def createDisparityWLSFilterGeneric(use_confidence: bool) -> DisparityWLSFilter: ...

def createEdgeAwareInterpolator() -> EdgeAwareInterpolator: ...

def createEdgeBoxes(alpha: float = ..., beta: float = ..., eta: float = ..., minScore: float = ..., maxBoxes: int = ..., edgeMinMag: float = ..., edgeMergeThr: float = ..., clusterMinMag: float = ..., maxAspectRatio: float = ..., minBoxArea: float = ..., gamma: float = ..., kappa: float = ...) -> EdgeBoxes: ...

def createEdgeDrawing() -> EdgeDrawing: ...

@_typing.overload
def createFastBilateralSolverFilter(guide: cv2.typing.MatLike, sigma_spatial: float, sigma_luma: float, sigma_chroma: float, lambda_: float = ..., num_iter: int = ..., max_tol: float = ...) -> FastBilateralSolverFilter: ...
@_typing.overload
def createFastBilateralSolverFilter(guide: cv2.UMat, sigma_spatial: float, sigma_luma: float, sigma_chroma: float, lambda_: float = ..., num_iter: int = ..., max_tol: float = ...) -> FastBilateralSolverFilter: ...

@_typing.overload
def createFastGlobalSmootherFilter(guide: cv2.typing.MatLike, lambda_: float, sigma_color: float, lambda_attenuation: float = ..., num_iter: int = ...) -> FastGlobalSmootherFilter: ...
@_typing.overload
def createFastGlobalSmootherFilter(guide: cv2.UMat, lambda_: float, sigma_color: float, lambda_attenuation: float = ..., num_iter: int = ...) -> FastGlobalSmootherFilter: ...

def createFastLineDetector(length_threshold: int = ..., distance_threshold: float = ..., canny_th1: float = ..., canny_th2: float = ..., canny_aperture_size: int = ..., do_merge: bool = ...) -> FastLineDetector: ...

@_typing.overload
def createGuidedFilter(guide: cv2.typing.MatLike, radius: int, eps: float) -> GuidedFilter: ...
@_typing.overload
def createGuidedFilter(guide: cv2.UMat, radius: int, eps: float) -> GuidedFilter: ...

@_typing.overload
def createQuaternionImage(img: cv2.typing.MatLike, qimg: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def createQuaternionImage(img: cv2.UMat, qimg: cv2.UMat | None = ...) -> cv2.UMat: ...

def createRFFeatureGetter() -> RFFeatureGetter: ...

def createRICInterpolator() -> RICInterpolator: ...

def createRightMatcher(matcher_left: cv2.StereoMatcher) -> cv2.StereoMatcher: ...

def createScanSegment(image_width: int, image_height: int, num_superpixels: int, slices: int = ..., merge_small: bool = ...) -> ScanSegment: ...

def createStructuredEdgeDetection(model: str, howToGetFeatures: RFFeatureGetter = ...) -> StructuredEdgeDetection: ...

@_typing.overload
def createSuperpixelLSC(image: cv2.typing.MatLike, region_size: int = ..., ratio: float = ...) -> SuperpixelLSC: ...
@_typing.overload
def createSuperpixelLSC(image: cv2.UMat, region_size: int = ..., ratio: float = ...) -> SuperpixelLSC: ...

def createSuperpixelSEEDS(image_width: int, image_height: int, image_channels: int, num_superpixels: int, num_levels: int, prior: int = ..., histogram_bins: int = ..., double_step: bool = ...) -> SuperpixelSEEDS: ...

@_typing.overload
def createSuperpixelSLIC(image: cv2.typing.MatLike, algorithm: int = ..., region_size: int = ..., ruler: float = ...) -> SuperpixelSLIC: ...
@_typing.overload
def createSuperpixelSLIC(image: cv2.UMat, algorithm: int = ..., region_size: int = ..., ruler: float = ...) -> SuperpixelSLIC: ...

@_typing.overload
def dtFilter(guide: cv2.typing.MatLike, src: cv2.typing.MatLike, sigmaSpatial: float, sigmaColor: float, dst: cv2.typing.MatLike | None = ..., mode: int = ..., numIters: int = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def dtFilter(guide: cv2.UMat, src: cv2.UMat, sigmaSpatial: float, sigmaColor: float, dst: cv2.UMat | None = ..., mode: int = ..., numIters: int = ...) -> cv2.UMat: ...

@_typing.overload
def edgePreservingFilter(src: cv2.typing.MatLike, d: int, threshold: float, dst: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def edgePreservingFilter(src: cv2.UMat, d: int, threshold: float, dst: cv2.UMat | None = ...) -> cv2.UMat: ...

@_typing.overload
def fastBilateralSolverFilter(guide: cv2.typing.MatLike, src: cv2.typing.MatLike, confidence: cv2.typing.MatLike, dst: cv2.typing.MatLike | None = ..., sigma_spatial: float = ..., sigma_luma: float = ..., sigma_chroma: float = ..., lambda_: float = ..., num_iter: int = ..., max_tol: float = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def fastBilateralSolverFilter(guide: cv2.UMat, src: cv2.UMat, confidence: cv2.UMat, dst: cv2.UMat | None = ..., sigma_spatial: float = ..., sigma_luma: float = ..., sigma_chroma: float = ..., lambda_: float = ..., num_iter: int = ..., max_tol: float = ...) -> cv2.UMat: ...

@_typing.overload
def fastGlobalSmootherFilter(guide: cv2.typing.MatLike, src: cv2.typing.MatLike, lambda_: float, sigma_color: float, dst: cv2.typing.MatLike | None = ..., lambda_attenuation: float = ..., num_iter: int = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def fastGlobalSmootherFilter(guide: cv2.UMat, src: cv2.UMat, lambda_: float, sigma_color: float, dst: cv2.UMat | None = ..., lambda_attenuation: float = ..., num_iter: int = ...) -> cv2.UMat: ...

@_typing.overload
def findEllipses(image: cv2.typing.MatLike, ellipses: cv2.typing.MatLike | None = ..., scoreThreshold: float = ..., reliabilityThreshold: float = ..., centerDistanceThreshold: float = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def findEllipses(image: cv2.UMat, ellipses: cv2.UMat | None = ..., scoreThreshold: float = ..., reliabilityThreshold: float = ..., centerDistanceThreshold: float = ...) -> cv2.UMat: ...

@_typing.overload
def fourierDescriptor(src: cv2.typing.MatLike, dst: cv2.typing.MatLike | None = ..., nbElt: int = ..., nbFD: int = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def fourierDescriptor(src: cv2.UMat, dst: cv2.UMat | None = ..., nbElt: int = ..., nbFD: int = ...) -> cv2.UMat: ...

@_typing.overload
def getDisparityVis(src: cv2.typing.MatLike, dst: cv2.typing.MatLike | None = ..., scale: float = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def getDisparityVis(src: cv2.UMat, dst: cv2.UMat | None = ..., scale: float = ...) -> cv2.UMat: ...

@_typing.overload
def guidedFilter(guide: cv2.typing.MatLike, src: cv2.typing.MatLike, radius: int, eps: float, dst: cv2.typing.MatLike | None = ..., dDepth: int = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def guidedFilter(guide: cv2.UMat, src: cv2.UMat, radius: int, eps: float, dst: cv2.UMat | None = ..., dDepth: int = ...) -> cv2.UMat: ...

@_typing.overload
def jointBilateralFilter(joint: cv2.typing.MatLike, src: cv2.typing.MatLike, d: int, sigmaColor: float, sigmaSpace: float, dst: cv2.typing.MatLike | None = ..., borderType: int = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def jointBilateralFilter(joint: cv2.UMat, src: cv2.UMat, d: int, sigmaColor: float, sigmaSpace: float, dst: cv2.UMat | None = ..., borderType: int = ...) -> cv2.UMat: ...

@_typing.overload
def l0Smooth(src: cv2.typing.MatLike, dst: cv2.typing.MatLike | None = ..., lambda_: float = ..., kappa: float = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def l0Smooth(src: cv2.UMat, dst: cv2.UMat | None = ..., lambda_: float = ..., kappa: float = ...) -> cv2.UMat: ...

@_typing.overload
def niBlackThreshold(_src: cv2.typing.MatLike, maxValue: float, type: int, blockSize: int, k: float, _dst: cv2.typing.MatLike | None = ..., binarizationMethod: int = ..., r: float = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def niBlackThreshold(_src: cv2.UMat, maxValue: float, type: int, blockSize: int, k: float, _dst: cv2.UMat | None = ..., binarizationMethod: int = ..., r: float = ...) -> cv2.UMat: ...

@_typing.overload
def qconj(qimg: cv2.typing.MatLike, qcimg: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def qconj(qimg: cv2.UMat, qcimg: cv2.UMat | None = ...) -> cv2.UMat: ...

@_typing.overload
def qdft(img: cv2.typing.MatLike, flags: int, sideLeft: bool, qimg: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def qdft(img: cv2.UMat, flags: int, sideLeft: bool, qimg: cv2.UMat | None = ...) -> cv2.UMat: ...

@_typing.overload
def qmultiply(src1: cv2.typing.MatLike, src2: cv2.typing.MatLike, dst: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def qmultiply(src1: cv2.UMat, src2: cv2.UMat, dst: cv2.UMat | None = ...) -> cv2.UMat: ...

@_typing.overload
def qunitary(qimg: cv2.typing.MatLike, qnimg: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def qunitary(qimg: cv2.UMat, qnimg: cv2.UMat | None = ...) -> cv2.UMat: ...

@_typing.overload
def readGT(src_path: str, dst: cv2.typing.MatLike | None = ...) -> tuple[int, cv2.typing.MatLike]: ...
@_typing.overload
def readGT(src_path: str, dst: cv2.UMat | None = ...) -> tuple[int, cv2.UMat]: ...

@_typing.overload
def rollingGuidanceFilter(src: cv2.typing.MatLike, dst: cv2.typing.MatLike | None = ..., d: int = ..., sigmaColor: float = ..., sigmaSpace: float = ..., numOfIter: int = ..., borderType: int = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def rollingGuidanceFilter(src: cv2.UMat, dst: cv2.UMat | None = ..., d: int = ..., sigmaColor: float = ..., sigmaSpace: float = ..., numOfIter: int = ..., borderType: int = ...) -> cv2.UMat: ...

@_typing.overload
def thinning(src: cv2.typing.MatLike, dst: cv2.typing.MatLike | None = ..., thinningType: int = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def thinning(src: cv2.UMat, dst: cv2.UMat | None = ..., thinningType: int = ...) -> cv2.UMat: ...

@_typing.overload
def transformFD(src: cv2.typing.MatLike, t: cv2.typing.MatLike, dst: cv2.typing.MatLike | None = ..., fdContour: bool = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def transformFD(src: cv2.UMat, t: cv2.UMat, dst: cv2.UMat | None = ..., fdContour: bool = ...) -> cv2.UMat: ...

@_typing.overload
def weightedMedianFilter(joint: cv2.typing.MatLike, src: cv2.typing.MatLike, r: int, dst: cv2.typing.MatLike | None = ..., sigma: float = ..., weightType: int = ..., mask: cv2.typing.MatLike | None = ...) -> cv2.typing.MatLike: ...
@_typing.overload
def weightedMedianFilter(joint: cv2.UMat, src: cv2.UMat, r: int, dst: cv2.UMat | None = ..., sigma: float = ..., weightType: int = ..., mask: cv2.UMat | None = ...) -> cv2.UMat: ...


