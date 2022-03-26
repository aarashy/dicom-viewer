# NOTE: This code is expected to run in a virtual Python environment.

from abc import ABC, abstractmethod, staticmethod
import os
import unittest
import logging
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import pydicom
import numpy as np
import csv
import sys
import traceback
import highdicom
from highdicom.seg import Segmentation
from pydicom.sr.codedict import codes

from typing import Any, List, Union
DicomFile = Union[pydicom.dataset.FileDataset, pydicom.dicomdir.DicomDir]


#
# ExportRoi
#
logging.info("Hi1")
class ExportRoi(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent) -> None:
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Aarash's Export ROI module"
    self.parent.categories = ["annotation"]
    self.parent.dependencies = []
    self.parent.contributors = ["Aarash Heydari"]
    self.parent.helpText = """
    Performs the selected I/O actions against a loaded DICOM file.
    """
    self.parent.acknowledgementText = "Thank you!"
    self.moduleName = self.__class__.__name__

logging.info("Hi2")
#
# Register sample data sets in Sample Data module
#


#
# ExportRoiWidget
#

class ExportRoiWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation
    self.logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/ExportRoi.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = ExportRoiLogic()

    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
    # (in the selected parameter node).
    self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.actionSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.csvPathSelector.connect("valueChanged(string)", self.updateParameterNodeFromGUI)
# DELETED    # self.ui.invertOutputCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
# self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
#     DELETED# self.ui.invertedOutputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

    # Buttons
    self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

    # Make sure parameter node is initialized (needed for module reload)
    self.initializeParameterNode()

  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    self.removeObservers()

  def enter(self):
    """
    Called each time the user opens this module.
    """
    # Make sure parameter node exists and observed
    self.initializeParameterNode()

  def exit(self):
    """
    Called each time the user opens a different module.
    """
    # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
    self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

  def onSceneStartClose(self, caller, event):
    """
    Called just before the scene is closed.
    """
    # Parameter node will be reset, do not use it anymore
    self.setParameterNode(None)

  def onSceneEndClose(self, caller, event):
    """
    Called just after the scene is closed.
    """
    # If this module is shown while the scene is closed then recreate a new parameter node immediately
    if self.parent.isEntered:
      self.initializeParameterNode()

  def initializeParameterNode(self):
    """
    Ensure parameter node exists and observed.
    """
    # Parameter node stores all user choices in parameter values, node selections, etc.
    # so that when the scene is saved and reloaded, these settings are restored.

    self.setParameterNode(self.logic.getParameterNode())

    # Select default input nodes if nothing is selected yet to save a few clicks for the user
    if not self._parameterNode.GetNodeReference("InputVolume"):
      firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
      if firstVolumeNode:
        self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

  def setParameterNode(self, inputParameterNode):
    """
    Set and observe parameter node.
    Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
    """

    if inputParameterNode:
      self.logic.setDefaultParameters(inputParameterNode)

    # Unobserve previously selected parameter node and add an observer to the newly selected.
    # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
    # those are reflected immediately in the GUI.
    if self._parameterNode is not None:
      self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    if self._parameterNode is not None:
      self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    # Initial GUI update
    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
    self._updatingGUIFromParameterNode = True

    # Update node selectors and sliders
    self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
    # self.ui.csvPathSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolume"))
    # self.ui.invertedOutputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolumeInverse"))
    # self.ui.imageThresholdSliderWidget.value = float(self._parameterNode.GetParameter("Threshold"))
    # self.ui.invertOutputCheckBox.checked = (self._parameterNode.GetParameter("Invert") == "true")

    # Update buttons states and tooltips
     # AARASH: Consider adding this back if I need the input volume to also be a parameter
    if self._parameterNode.GetNodeReference("InputVolume"):#and self._parameterNode.GetNodeReference("OutputVolume"):
      self.ui.applyButton.toolTip = "Apply the selection actions"
      self.ui.applyButton.enabled = True
    else:
      self.ui.applyButton.toolTip = "Select input volume nodes"
      self.ui.applyButton.enabled = False

    # All the GUI updates are done
    self._updatingGUIFromParameterNode = False

  def updateParameterNodeFromGUI(self, caller=None, event=None):
    """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch



    self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)

# NOTE!!!! self.ui.actionSelector.currentItem() could be None or dirQListWidgetItem, which has a _.text() property corresponding to the text in ExportRoi.ui
    if self.ui.actionSelector.currentItem():
        self._parameterNode.SetParameter("Action", self.ui.actionSelector.currentItem().text())
    else: 
        self._parameterNode.SetParameter("Action", None)

    self._parameterNode.SetParameter("CsvFilePath", self.ui.csvPathSelector.plainText)
    # self._parameterNode.SetParameter("Threshold", str(self.ui.imageThresholdSliderWidget.value))
    # self._parameterNode.SetParameter("Invert", "true" if self.ui.invertOutputCheckBox.checked else "false")
    # self._parameterNode.SetNodeReferenceID("OutputVolumeInverse", self.ui.invertedOutputSelector.currentNodeID)

    self._parameterNode.EndModify(wasModified)

  def onApplyButton(self):
    """
    Run processing when user clicks "Apply" button.
    """
    try:
        logging.info("****Attempted Apply******\n\n")
        if self.ui.actionSelector.currentItem() is None:
            raise Exception("Select an action.")
      # # Compute output
      # self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
      #   self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)
        self.logic.process(self.ui.inputSelector.currentNode(), self.ui.actionSelector.currentItem().text(), self.ui.csvPathSelector.plainText)


    except Exception as e:
      slicer.util.errorDisplay("Failed to compute results: "+str(e))
      import traceback
      traceback.print_exc()


#
# ExportRoiLogic
#

#######AARASH CHANGE START HERE#
# An abstraction around ROIs with the necessary APIs to write-back DICOM annotations.
class AbstractRoi(ABC):
    #########################
    ### Interface methods ###
    #########################
    @abstractmethod
    def get_bitmask(self) -> np.ndarray:
        pass
    @abstractmethod
    def get_centroid(self) -> List[float]:
        pass
    @abstractmethod
    def get_area_millimeters(self) -> float:
        pass
    @abstractmethod
    def get_spanned_dicom_files(self) -> List[DicomFile]:
        pass

    #########################
    ## Static util methods ##
    #########################
    
    # Note: The relation between world coordinates and voxel 
    # coordinates is as follows: 
    # World coordinate (0, 0) ~= voxel coordinate (256, 256)
    # in the middle of the photo
    # Positive X coordinates are the patient's right, which is 
    # is the viewer's left side of an image, which trends toward
    # voxel-X-coordinate 0
    # Contrarywise, as world-X -> negative values :: voxel-X -> 512
    # Similarly for Y coordinates, 
    # world-Y -> positive (anterior) :: voxel-Y -> 0
    # world-Y -> negative (posterior) :: voxel-Y -> 512
    # 
    # Because of this inverse relationship, both voxel_to_world 
    # and world_to_voxel negate the coordinate input.
    @staticmethod
    def voxel_to_world(vX: int, vY: int, spacing: List[float], origin: List[float]):
        wX = -(vX * spacing[0]) + origin[0]
        wY = -(vY * spacing[1]) + origin[1]
        return wX, wY

    @staticmethod
    def world_to_voxel(wX: float, wY: float, spacing: List[float], origin: List[float]):
        vX = -(wX - origin[0]) / spacing[0]
        vY = -(wY - origin[1]) / spacing[1]
        return round(vX), round(vY)

    @staticmethod
    # Source: https://stackoverflow.com/questions/44865023/circular-masking-an-image-in-python-using-numpy-arrays
    def create_circular_mask(h, w, centerX, centerY, radius):
        # Grid of indices
        Y, X = np.ogrid[:h, :w]
        # Grid of distances
        dist_from_center = np.sqrt((X - centerX)**2 + (Y-centerY)**2)
        # Grid of 1s and 0s for distances <= radius
        mask = np.array(
            [[1 if dist <= radius else 0 for dist in dist_row] 
                for dist_row in dist_from_center]
            )
        return mask

class DicomViewerState:
    dicom_files: List[DicomFile]
    rois: List[AbstractRoi]

    def __init__(
      self, 
      dicom_files: List[DicomFile], 
      rois: List[AbstractRoi]
    ):
        self.dicom_files = dicom_files
        self.rois = rois

# An abstraction around a DICOM viewer for producing annotated DICOM files.
class AbstractDicomViewerBackend(ABC):
   @abstractmethod
   def list_rois(self) -> List[AbstractRoi]:
       pass

# An abstraction around any action that this tool ought to perform.
# This includes I/O actions such as embedding an annotation into a DICOM overlay, 
# importing annotations from CSV, etc.
class AbstractAction(ABC):
    @abstractmethod
    def apply(self, annotated_dicom: AnnotatedDicom) -> None:
        pass

    def apply_to_all(self, annotated_dicom_list: List[AnnotatedDicom]) -> None:
        print(f"Applying {self} to {len(annotated_dicom_list)} DICOM files.")
        for dcm in annotated_dicom_list:
            try:
                self.apply(dcm)
            except Exception as error:
                print(f"Failed to apply {self} for file {dcm.source_file_path}\nError: {error}")
                traceback.print_tb(sys.exc_info()[2])
                print("Continuing...\n\n")

class SlicerBackend(AbstractDicomViewerBackend):
    # Maintain a handle to the volume node being operated upon. This encompasses all of the DICOM instances of a series.
    def __init__(self, vtkMRMLScalarVolumeNode):
        self.volume_node = vtkMRMLScalarVolumeNode

    def list_rois(self) -> List[AbstractRoi]:
        # Collect all dicom files from the volume node.
        logging.info("Listing annotated dicom...")
        dicom_files: List[DicomFile] = []
        inst_uids = self.volume_node.GetAttribute("DICOM.instanceUIDs").split()
        for inst_uid in inst_uids: 
            source_file_path = slicer.dicomDatabase.fileForInstance(inst_uid)
            dicom_file = pydicom.read_file(source_file_path)
            dicom_files.append(dicom_file)

        # iterate over ROIs. Filter out ROIs that don't reference this volume node.
        roi_nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLAnnotationROINode")
        roi_nodes = [n for n in roi_nodes if n.GetAttribute('AssociatedNodeID') == self.volume_node.GetID()]

        def contains(roi_node, dicom_file):
            z_coordinate = float(dicom_file.ImagePositionPatient[2])
            corners = [0, 0, 0, 0, 0, 0]
            roi_node.GetBounds(corners)
            z_min, z_max = corners[4], corners[5]
            return z_min < z_coordinate and z_max > z_coordinate

        # Append the ROI to all relevant slices
        rois = []
        for roi_node in roi_nodes:
            spanned_dicom_files = [dicom_file for dicom_file in dicom_files if contains(roi_node, dicom_file)]
            logging.info("Found %d relevant dicom for %s" % (len(spanned_dicom_files), roi_node.GetName()))
            roi = SlicerRoi(roi_node, self.volume_node, spanned_dicom_files)
            for annotated_dicom in spanned_dicom_files:
                rois.append(roi)

        logging.info("Done listing ROIs... %d" % len(rois))
        return rois

class SlicerRoi(AbstractRoi):
    def __init__(self, vtkMRMLAnnotationROINode, vtkMRMLScalarVolumeNode, spanned_dicom_files):
        self.roi = vtkMRMLAnnotationROINode
        self.volume_node = vtkMRMLScalarVolumeNode
        self.spanned_dicom_files = spanned_dicom_files

    # Returns the center of ROI in 3D world coordinates.
    def get_centroid(self) -> List[float]:
        # The _.GetXYZ function mutates an array which is passed in.
        xyz: List[float] = [0., 0., 0.]
        self.roi.GetXYZ(xyz)
        return xyz

    # Returns the area of the cross-section of the ROI around the centroid.
    # In this Slicer implementation, we assume the ROI is a rectangle.
    def get_area_millimeters(self):
        radius = [0, 0, 0]
        self.roi.GetRadiusXYZ(radius)
        area = radius[0]*radius[1]*2
        return area

    def get_bitmask(self):
        # For rectangles, we build a rectangle using the corners.
        corners = [0, 0, 0, 0, 0, 0]
        self.roi.GetBounds(corners)
        min_x, min_y = AbstractRoi.world_to_voxel(corners[1], corners[3], self.volume_node.GetSpacing(), self.volume_node.GetOrigin())
        max_x, max_y = AbstractRoi.world_to_voxel(corners[0], corners[2], self.volume_node.GetSpacing(), self.volume_node.GetOrigin())

        mask = np.zeros((512, 512)).astype(int)
        mask[min_x:max_x, min_y:max_y] = 1
        return np.swapaxes(mask, 0, 1)
    
    @abstractmethod
    def get_spanned_dicom_files(self) -> List[DicomFile]:
        return self.spanned_dicom_files

class WriteToDicomOverlay(AbstractAction):
    def __init__(self): 
        # Each DICOM file may store up to 16 overlays. 
        # This map from Series UID to index tracks how many 
        # overlays have already been applied for each DICOM file. 
        self.roi_index_map = {}

    def apply(self, annotated_dicom):
        dcm = annotated_dicom.dicom_file
        for roi_idx, roi in enumerate(annotated_dicom.roi_list):
            mask = roi.get_bitmask()
            if roi_idx > 16:
                # cannot support more than 16 overlays for the same image
                raise Exception("Cannot support more than 16 overlays")
            # packbits converts array of integer 1s and 0s to array of numbers.
            # It interprets a run of 8 bits as a number.
            # i.e. [1, 0, 0, 0, 0, 0, 0, 0] -> [128]
            # The reshaping and flattening is to accomodate DICOM Header's expected format.
            reshaped_mask = np.packbits(mask.reshape(-1,8)[:,::-1].flatten('C'))
            dcm.add_new(0x60000010 + roi_idx*0x20000 , 'US', 512)
            dcm.add_new(0x60000011 + roi_idx*0x20000 , 'US', 512)
            dcm.add_new(0x60000022 + roi_idx*0x20000 , 'LO', "DICOM Overlay annotation added by python script")
            dcm.add_new(0x60000040 + roi_idx*0x20000 , 'CS', 'R')
            dcm.add_new(0x60000050 + roi_idx*0x20000 , 'SS', [1,1])
            dcm.add_new(0x60000100 + roi_idx*0x20000 , 'US', 1)
            dcm.add_new(0x60000102 + roi_idx*0x20000 , 'US', 0)
            dcm.add_new(0x60003000 + roi_idx*0x20000 , 'OW', reshaped_mask)
        print('Saved to:', annotated_dicom.source_file_path)
        dcm.save_as(annotated_dicom.source_file_path)

    def __repr__(self):
        return "WriteToDicomOverlay"

class AppendToCsv(AbstractAction):
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path

    def apply(self, annotated_dicom):
        # Prepare columns values
        series_uid = annotated_dicom.dicom_file.SeriesInstanceUID
        for roi in annotated_dicom.roi_list:
            centroid = roi.get_centroid()
            diameter_mm = 2*np.sqrt(roi.get_area_millimeters() / np.pi)

            worldX, worldY = AbstractRoi.voxel_to_world(
                centroid[0], 
                centroid[1], 
                annotated_dicom.dicom_file.PixelSpacing,
                annotated_dicom.dicom_file.ImagePositionPatient
            )

            # columns = ["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"]
            with open(self.csv_file_path, 'a') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    series_uid,
                    worldX,
                    worldY,
                    str(annotated_dicom.dicom_file.ImagePositionPatient[2]).strip("\'"),
                    diameter_mm
                ])


    def __repr__(self):
        return "AppendToCsv"

# AARASH TODO FINISH AND TEST
class ExportToDicomSeg(AbstractAction): 
  def __init__(self):
    pass

    # Describe the algorithm that created the segmentation
    algorithm_identification = highdicom.AlgorithmIdentificationSequence(
        name='aarash_dicom_plugin',
        version='v1.0',
        family=codes.cid7162.ArtificialIntelligence
    )

    # Describe the segment
    description_segment_1 = highdicom.seg.SegmentDescription(
        segment_number=1,
        segment_label='first segment',
        segmented_property_category=codes.cid7150.Tissue,
        segmented_property_type=codes.cid7166.ConnectiveTissue,
        algorithm_type=highdicom.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=algorithm_identification,
        tracking_uid=highdicom.UID(),
        tracking_id='test segmentation of slide microscopy image'
    )

    # Create the Segmentation instance
    seg_dataset = Segmentation(
        source_images=[image_dataset],
        pixel_array=mask,
        segmentation_type=highdicom.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[description_segment_1],
        series_instance_uid=highdicom.UID(),
        series_number=2,
        sop_instance_uid=highdicom.UID(),
        instance_number=1,
        manufacturer='Manufacturer',
        manufacturer_model_name='Model',
        software_versions='v1',
        device_serial_number='Device XYZ'
    )

# AARASH TODO FINISH AND TEST
class ImportFromCsv(AbstractAction):
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path

    def apply(self, annotated_dicom):
        spacing, origin = annotated_dicom.dicom_file.PixelSpacing, annotated_dicom.dicom_file.ImagePositionPatient
        with open(self.csv_file_path) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            columns = next(csvreader) # Read the first row, which is names of columns rather than data.

            # This dict records number of overlays already in a slice, where key = slice Z position
            NextOverlayIndex = {} 
            for i, row in enumerate(csvreader):
                try:
                     s = row[0].split(",")
                     csvseriesuid, coordX, coordY, coordZ, diameter_mm = s[0], s[1], s[2], s[3].strip("\'"), float(s[4])
                except:
                     print("Failed with len row: ", len(row[0].split(",")), "at i = %d" % i)
                     print(row[0])
                     continue
                if dcmseriesuid == csvseriesuid:
                     print("equality at csvrow = %d" % i)
                     vX, vY = world_to_voxel(float(coordX), float(coordY), spacing, origin)
                     print("ROI at voxel coordinates", coordX, coordY, vX, vY, coordZ)
                     Zlist = [z for z in scans.keys()]
                     scanZ = sorted(Zlist, key=lambda x: np.abs(float(x) - np.round(float(coordZ.rstrip('0')))))[0]
                     print(scanZ)
                     ds = scans[scanZ]
                     overlayIndex = NextOverlayIndex.get(scanZ, 0)
                     mask = create_circular_mask(512, 512, vX, vY, radius=diameter_mm/2*(1/spacing[0]))
                     embed(mask, ds, overlayIndex, OVERLAY_CSV_NAME_SUFFIX)
                     roiNew = osirix.ROI(itype='tPlain',buffer=mask.T.astype(np.bool),name="ROI read from CSV row %d" % i,ipixelSpacing=spacing, iimageOrigin=origin[:2])
                     vc.setROI(roiNew,position=instanceToPixPosition[ds.InstanceNumber])
                     NextOverlayIndex[scanZ] = overlayIndex + 1
                     ds.save_as(pixs[instanceToPixPosition[ds.InstanceNumber]].sourceFile)
                     print("saved file in %s for instance %f with overlayIndex = %d, radius=%f\n\n\n" % (pixs[instanceToPixPosition[ds.InstanceNumber]].sourceFile, ds.InstanceNumber, overlayIndex, diameter_mm))
    def __repr__(self):
        return "ImportFromCsv"

#########AARASH CHANGE END HERE
class ExportRoiLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):
    """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
    ScriptedLoadableModuleLogic.__init__(self)

  def setDefaultParameters(self, parameterNode):
    """
    Initialize parameter node with default settings.
    """
    if not parameterNode.GetParameter("Threshold"):
      parameterNode.SetParameter("Threshold", "100.0")
    if not parameterNode.GetParameter("Invert"):
      parameterNode.SetParameter("Invert", "false")

  def process(self, inputVolume, action=None, csvFilePath=""):
    """
    TODO: Change comments
    Run the processing algorithm.
    Can be used without GUI widget.
    :param inputVolume: volume to be thresholded
    :param outputVolume: thresholding result
    :param imageThreshold: values above/below this threshold will be set to 0
    :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
    """


    logging.info('Processing started')
    
    if action == "CSV":
        action = AppendToCsv(csvFilePath)
    elif action == "DICOM header":
        action = WriteToDicomOverlay()
    elif action == "Import from CSV":
        action = ImportFromCsv(csvFilePath)
    else:
        raise Exception("Invalid action")

    if not inputVolume:
      raise ValueError("Input volume is invalid")

    import time
    startTime = time.time()
    annotated_dicom_list = SlicerBackend(inputVolume).list_annotated_dicom()
    action.apply_to_all(annotated_dicom_list)

    stopTime = time.time()
    logging.info('Processing completed in {0:.2f} seconds'.format(stopTime-startTime))

#
# ExportRoiTest
#

class ExportRoiTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear()

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_ExportRoi1()

  def test_ExportRoi1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")

    # Get/create input data

    import SampleData
    registerSampleData()
    inputVolume = SampleData.downloadSample('ExportRoi1')
    self.delayDisplay('Loaded test data set')

    inputScalarRange = inputVolume.GetImageData().GetScalarRange()
    self.assertEqual(inputScalarRange[0], 0)
    self.assertEqual(inputScalarRange[1], 695)

    outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    threshold = 100

    # Test the module logic

    logic = ExportRoiLogic()

    # Test algorithm with non-inverted threshold
    logic.process(inputVolume, outputVolume, threshold, True)
    outputScalarRange = outputVolume.GetImageData().GetScalarRange()
    self.assertEqual(outputScalarRange[0], inputScalarRange[0])
    self.assertEqual(outputScalarRange[1], threshold)

    # Test algorithm with inverted threshold
    logic.process(inputVolume, outputVolume, threshold, False)
    outputScalarRange = outputVolume.GetImageData().GetScalarRange()
    self.assertEqual(outputScalarRange[0], inputScalarRange[0])
    self.assertEqual(outputScalarRange[1], inputScalarRange[1])

    self.delayDisplay('Test passed')

