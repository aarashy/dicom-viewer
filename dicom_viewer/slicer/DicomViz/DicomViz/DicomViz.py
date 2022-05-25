# NOTE: This code is expected to run in a virtual Python environment.

from abc import ABC, abstractmethod
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
from pydicom.sr.codedict import codes

from typing import Any, List, Union, Dict, Callable

from highdicom import AlgorithmIdentificationSequence, UID
from highdicom.seg import Segmentation, SegmentDescription, SegmentAlgorithmTypeValues, SegmentationTypeValues

#
# DicomViz
#
class DicomViz(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent) -> None:
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Interoperable DICOM Annotation Workflows"
    self.parent.categories = ["Annotation"]
    self.parent.dependencies = []
    self.parent.contributors = ["Aarash Heydari"]
    self.parent.helpText = """
    Performs the selected I/O actions against a loaded DICOM file.
    """
    self.parent.acknowledgementText = "Thank you!"
    self.moduleName = self.__class__.__name__

logging.info("Debug: Successfully loaded module")
#
# Register sample data sets in Sample Data module
#


#
# DicomVizWidget
#

class DicomVizWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/DicomViz.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = DicomVizLogic()

    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
    # (in the selected parameter node).
    self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.actionSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.filePathSelector.connect("valueChanged(string)", self.updateParameterNodeFromGUI)
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
    # self.ui.filePathSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolume"))
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

# NOTE!!!! self.ui.actionSelector.currentItem() could be None or dirQListWidgetItem, which has a _.text() property corresponding to the text in DicomViz.ui
    if self.ui.actionSelector.currentItem():
        self._parameterNode.SetParameter("Action", self.ui.actionSelector.currentItem().text())
    else: 
        self._parameterNode.SetParameter("Action", "")

    self._parameterNode.SetParameter("FilePath", self.ui.filePathSelector.plainText)
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
        self.logic.process(self.ui.inputSelector.currentNode(), self.ui.actionSelector.currentItem().text(), self.ui.filePathSelector.plainText)


    except Exception as e:
      slicer.util.errorDisplay("Failed to compute results: "+str(e))
      import traceback
      traceback.print_exc()


#
# DicomVizLogic
#

#######AARASH CHANGE START HERE#
class DicomFile:
  inner: Union[pydicom.dataset.FileDataset, pydicom.dicomdir.DicomDir]
  file_path: str

  def __init__(
      self, 
      dicom_file: Union[pydicom.dataset.FileDataset, pydicom.dicomdir.DicomDir], 
      file_path: str
    ):
        self.inner = dicom_file
        self.file_path = file_path
  


# An abstraction around ROIs with the necessary APIs to write-back DICOM annotations.
class AbstractRoi(ABC):
    #########################
    ### Interface methods ###
    #########################
    '''
      Returns an array of segmentation pixel data of boolean data type 
      representing a mask image. The array may be a 2D or 3D numpy array.

      If it is a 2D numpy array, it represents the segmentation of a
      single frame image, such as a planar x-ray or single instance from
      a CT or MR series. In this case, `get_spanned_dicom_files` should 
      return a list of size 1. 

      If it is a 3D array, it represents the segmentation of either a
      series of source images (such as a series of CT or MR images) a
      single 3D multi-frame image (such as a multi-frame CT/MR image), or
      a single 2D tiled image (such as a slide microscopy image).

      If ``pixel_array`` represents the segmentation of a 3D image, the
      first dimension represents individual 2D planes. Unless the
      ``plane_positions`` parameter is provided, the frame in
      ``pixel_array[i, ...]`` should correspond to either
      ``source_images[i]`` (if ``source_images`` is a list of single
      frame instances) or source_images[0].pixel_array[i, ...] if
      ``source_images`` is a single multiframe instance.
    '''
    @abstractmethod
    def get_pixel_mask(self) -> np.ndarray:
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
    # Renders a new ROI within the DICOM viewer.
    # Implementation is particularly dependent on the internal data structures
    # of the DICOM viewer software being used.
    @abstractmethod
    def render_new(**kwargs: Dict[str, Any]) -> None:
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

    # dicom_files should be sorted by Z axis increasing
    def __init__(
      self, 
      dicom_files: List[DicomFile], 
      rois: List[AbstractRoi]
    ):
        self.dicom_files = dicom_files
        self.rois = rois
    
    def get_series_instance_uid(self):
      assert(len(self.dicom_files)) > 0
      return self.dicom_files[0].inner.SeriesInstanceUID

# An abstraction around a DICOM viewer for producing annotated DICOM files.
class AbstractDicomViewerBackend(ABC):
   @abstractmethod
   def get_state(self) -> DicomViewerState:
       pass

# An abstraction around any action that this tool ought to perform.
# This includes I/O actions such as embedding an annotation into a DICOM overlay, 
# importing annotations from CSV, etc.
class AbstractAction(ABC):
    @abstractmethod
    def apply(self, state: DicomViewerState) -> None:
        pass

class SlicerBackend(AbstractDicomViewerBackend):
    # Maintain a handle to the volume node being operated upon. This encompasses all of the DICOM instances of a series.
    def __init__(self, vtkMRMLScalarVolumeNode):
        self.volume_node = vtkMRMLScalarVolumeNode

    # Loads the currently rendered DICOM objects and ROIs.
    def get_state(self) -> DicomViewerState:
        logging.info("Listing annotated dicom...")
        dicom_files: List[DicomFile] = []
        inst_uids = self.volume_node.GetAttribute("DICOM.instanceUIDs").split()

        # Load each file by its instance uid, collecting into `dicom_files`
        for inst_uid in inst_uids: 
            file_path = slicer.dicomDatabase.fileForInstance(inst_uid)
            dicom_file = pydicom.read_file(file_path)
            dicom_files.append(DicomFile(dicom_file, file_path))
        dicom_files = sorted(dicom_files, key=lambda x: x.inner.InstanceNumber)

        state = DicomViewerState(dicom_files, [])

        # Collect the ROIs that reference the current volume into the DICOM viewer state
        roi_nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLAnnotationROINode")
        roi_nodes = [n for n in roi_nodes if n.GetAttribute('AssociatedNodeID') == self.volume_node.GetID()]

        
        for roi_node in roi_nodes:
            roi = SlicerRoi(roi_node, self.volume_node, state)
            state.rois.append(roi)

        logging.info("Finished loading %d DICOM files and %d ROIs" % (len(state.dicom_files), len(state.rois)))
        return state

class SlicerRoi(AbstractRoi):
    roi_node: Any
    volume_node: Any
    state: DicomViewerState

    def __init__(self, vtkMRMLAnnotationROINode, vtkMRMLScalarVolumeNode, state: DicomViewerState):
        self.roi_node = vtkMRMLAnnotationROINode
        self.volume_node = vtkMRMLScalarVolumeNode
        self.state = state

    # Returns the center of ROI in 3D world coordinates.
    def get_centroid(self) -> List[float]:
        # The _.GetXYZ function mutates an array which is passed in.
        xyz: List[float] = [0., 0., 0.]
        self.roi_node.GetXYZ(xyz)
        return xyz

    # Returns the area of the cross-section of the ROI around the centroid.
    # In this Slicer implementation, we assume the ROI is a rectangle.
    def get_area_millimeters(self):
        radius = [0, 0, 0]
        self.roi_node.GetRadiusXYZ(radius)
        area = radius[0]*radius[1]*2
        return area

    def get_pixel_mask(self):
        # For rectangles, we build a rectangle using the corners.
        corners = [0, 0, 0, 0, 0, 0]
        self.roi_node.GetBounds(corners)
        min_x, min_y = AbstractRoi.world_to_voxel(corners[1], corners[3], self.volume_node.GetSpacing(), self.volume_node.GetOrigin())
        max_x, max_y = AbstractRoi.world_to_voxel(corners[0], corners[2], self.volume_node.GetSpacing(), self.volume_node.GetOrigin())

        mask = np.zeros((len(self.state.dicom_files), 512, 512)).astype(np.uint8)
        spanned_files = self.get_spanned_dicom_files()
        low_index = min([round(float(dcm.inner.ImagePositionPatient[2])) for dcm in spanned_files])
        high_index = max([round(float(dcm.inner.ImagePositionPatient[2])) for dcm in spanned_files])
        mask[low_index:high_index, min_x:max_x, min_y:max_y] = 1
        return mask
    
    def get_spanned_dicom_files(self) -> List[DicomFile]:
        spanned_dicom_files = [dicom_file for dicom_file in self.state.dicom_files if SlicerRoi.contains(self.roi_node, dicom_file)]
        return spanned_dicom_files

    def render_new(**kwargs: Dict[str, Any]) -> None:
      pass

    def contains(roi_node: Any, dicom_file: DicomFile):
        z_coordinate = float(dicom_file.inner.ImagePositionPatient[2])
        corners = [0, 0, 0, 0, 0, 0]
        roi_node.GetBounds(corners)
        z_min, z_max = corners[4], corners[5]
        return z_min < z_coordinate and z_max > z_coordinate

class WriteToDicomOverlay(AbstractAction):
    def apply(self, state: DicomViewerState):
        # Each DICOM file may store up to 16 overlays. 
        # This map from Series UID to index tracks how many 
        # overlays have already been applied for each DICOM file. 
        roi_idx_map: Dict[str, int] = {}

        for roi in state.rois:
          for dcm in roi.get_spanned_dicom_files():
            mask: np.ndarray = roi.get_pixel_mask()
            roi_idx = roi_idx_map.get(dcm.inner.ImagePositionPatient[2], 0)
            roi_idx_map[dcm.inner.ImagePositionPatient[2]] = roi_idx + 1

            if roi_idx > 16:
                # cannot support more than 16 overlays for the same image
                raise Exception("Cannot support more than 16 overlays")
            # packbits converts array of integer 1s and 0s to array of numbers.
            # It interprets a run of 8 bits as a number.
            # i.e. [1, 0, 0, 0, 0, 0, 0, 0] -> [128]
            # The reshaping and flattening is to accomodate filePathSelector's expected format.
            reshaped_mask = np.packbits(mask.reshape(-1,8)[:,::-1].flatten('C'))
            dcm.inner.add_new(0x60000010 + roi_idx*0x20000 , 'US', 512)
            dcm.inner.add_new(0x60000011 + roi_idx*0x20000 , 'US', 512)
            dcm.inner.add_new(0x60000022 + roi_idx*0x20000 , 'LO', "DICOM Overlay annotation added by python script")
            dcm.inner.add_new(0x60000040 + roi_idx*0x20000 , 'CS', 'R')
            dcm.inner.add_new(0x60000050 + roi_idx*0x20000 , 'SS', [1,1])
            dcm.inner.add_new(0x60000100 + roi_idx*0x20000 , 'US', 1)
            dcm.inner.add_new(0x60000102 + roi_idx*0x20000 , 'US', 0)
            dcm.inner.add_new(0x60003000 + roi_idx*0x20000 , 'OW', reshaped_mask)
          print('Saved to:', dcm.file_path)
          dcm.inner.save_as(dcm.file_path)

    def __repr__(self):
        return "WriteToDicomOverlay"

class AppendToCsv(AbstractAction):
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path

    def apply(self, state: DicomViewerState):
      for roi in state.rois:
          spanned_dcm = roi.get_spanned_dicom_files()
          assert(len(spanned_dcm) > 0)
          center_dcm: DicomFile = spanned_dcm[len(spanned_dcm) / 2]
          # Prepare columns values
          series_uid = center_dcm.inner.SeriesInstanceUID
          centroid = roi.get_centroid()
          diameter_mm = 2*np.sqrt(roi.get_area_millimeters() / np.pi)

          worldX, worldY = AbstractRoi.voxel_to_world(
              centroid[0], 
              centroid[1], 
              center_dcm.inner.PixelSpacing,
              center_dcm.inner.ImagePositionPatient
          )

          # columns = ["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"]
          with open(self.csv_file_path, 'a') as csv_file:
              writer = csv.writer(csv_file)
              writer.writerow([
                  series_uid,
                  worldX,
                  worldY,
                  str(center_dcm.dicom_file.ImagePositionPatient[2]).strip("\'"),
                  diameter_mm
              ])


    def __repr__(self):
        return "AppendToCsv"

# AARASH TODO FINISH AND TEST
class ExportToDicomSeg(AbstractAction): 
  def __init__(self, out_dir: str):
    self.out_dir = out_dir

  def apply(self, state: DicomViewerState):
    for roi in state.rois:
      # Describe the algorithm that created the segmentation
      algorithm_identification = AlgorithmIdentificationSequence(
          name='aarash_dicom_plugin',
          version='v1.0',
          family=codes.cid7162.ArtificialIntelligence
      )

      # Describe the segment
      description_segment_1 = SegmentDescription(
          segment_number=1,
          segment_label='first segment',
          segmented_property_category=codes.cid7150.Tissue,
          segmented_property_type=codes.cid7166.ConnectiveTissue,
          algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC,
          algorithm_identification=algorithm_identification,
          tracking_uid=UID(),
          tracking_id='test segmentation of slide microscopy image'
      )

      logging.info(np.shape(roi.get_pixel_mask()))
      # Create the Segmentation instance
      seg_dataset = Segmentation(
          source_images=[dcm.inner for dcm in state.dicom_files],
          pixel_array=roi.get_pixel_mask(),
          segmentation_type=SegmentationTypeValues.BINARY,
          segment_descriptions=[description_segment_1],
          series_instance_uid=UID(),
          series_number=2,
          sop_instance_uid=UID(),
          instance_number=1,
          manufacturer='Manufacturer',
          manufacturer_model_name='Model',
          software_versions='v1',
          device_serial_number='Device XYZ'
      )

      seg_dataset.save_as(self.out_dir + "seg.dcm")
      print("Saved!")

# AARASH TODO FINISH AND TEST
RoiGeneratorFunc = Callable[[str, float, float, int, float], None]
class ImportFromCsv(AbstractAction):
    def __init__(self, csv_file_path: str, roi_generator_function: RoiGeneratorFunc):
        self.csv_file_path = csv_file_path
        self.roi_generator_function = roi_generator_function

    def apply(self, state: DicomViewerState):
        spacing, origin = state.dicom_files[0].inner.PixelSpacing, state.dicom_files[0].inner.ImagePositionPatient
        series_uid = state.get_series_instance_uid()
        with open(self.csv_file_path) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            columns = next(csvreader) # Read the first row, which is names of columns rather than data.

            # Find a row where the SeriesUID matches the current series
            for i, row in enumerate(csvreader):
                try:
                     s = row[0].split(",")
                     csv_series_uid = s[0]
                     coordX, coordY = float(s[1]), float(s[2])
                     # Remove formatting from Z world coordinate and round it to the nearest int
                     coordZ: int = np.round(float(s[3].strip("\'").rstrip('0')))
                     diameter_mm = float(s[4])
                except:
                     print("Encountered error at row at idx = %d" % i)
                     print(row[0])
                     continue
                if series_uid == csv_series_uid:
                     print("equality at csvrow = %d" % i)
                     vX, vY = AbstractRoi.world_to_voxel(float(coordX), float(coordY), spacing, origin)
                     print(f"ROI at coordinates {coordX}, {coordY}, {vX}, {vY}, {coordZ}")
                     self.roi_generator_function(csv_series_uid, coordX, coordY, coordZ, diameter_mm)
    def __repr__(self):
        return "ImportFromCsv"

#########AARASH CHANGE END HERE
class DicomVizLogic(ScriptedLoadableModuleLogic):
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

  def process(self, inputVolume, action=None, filePath=""):
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
    
    if action == "Append to CSV":
        action = AppendToCsv(filePath)
    elif action == "Embed in DICOM Overlay":
        action = WriteToDicomOverlay()
    elif action == "Import from CSV":
        action = ImportFromCsv(filePath)
    elif action == "Export to DICOM Segmentation Object":
        action = ExportToDicomSeg(filePath)
    else:
        raise Exception("Invalid action")

    if not inputVolume:
      raise ValueError("Input volume is invalid")

    import time
    startTime = time.time()
    state: DicomViewerState = SlicerBackend(inputVolume).get_state()
    action.apply(state)

    stopTime = time.time()
    logging.info('Processing completed in {0:.2f} seconds'.format(stopTime-startTime))

#
# DicomVizTest
#

class DicomVizTest(ScriptedLoadableModuleTest):
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
    self.test_DicomViz1()

  def test_DicomViz1(self):
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
    inputVolume = SampleData.downloadSample('DicomViz1')
    self.delayDisplay('Loaded test data set')

    inputScalarRange = inputVolume.GetImageData().GetScalarRange()
    self.assertEqual(inputScalarRange[0], 0)
    self.assertEqual(inputScalarRange[1], 695)

    outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    threshold = 100

    # Test the module logic

    logic = DicomVizLogic()

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

