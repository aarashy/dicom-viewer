import osirix
import dicom
import numpy as np
import csv
import sys
import traceback

class IoAction:
    def apply_to_all(self, annotated_dicom_list):
        print("Applying %s to %d dicom files." % (self, len(annotated_dicom_list)))
        for annotated_dicom in annotated_dicom_list:
            try:
                self.apply(annotated_dicom)
            except Exception as error:
                print("Failed to apply %s for file %s\nError: %s" 
                    % (self, annotated_dicom.source_file_path, error))
                traceback.print_tb(sys.exc_info()[2])
                print("Continuing...\n\n")

    def apply(self, annotated_dicom):
        pass

class WriteToDicomOverlay(IoAction):
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
        dcm.save_as(annotated_dicom.source_file_path)

    def __repr__(self):
        return "WriteToDicomOverlay"

class AppendToCsv(IoAction):
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path

    def apply(self, annotated_dicom):
        # Prepare columns values
        series_uid = annotated_dicom.dicom_file.SeriesInstanceUID
        for roi in annotated_dicom.roi_list:
            centroid = roi.get_centroid()
            worldX, worldY = Roi.voxel_to_world(
                centroid[0], 
                centroid[1], 
                annotated_dicom.dicom_file.PixelSpacing,
                annotated_dicom.dicom_file.ImagePositionPatient
            )
            diameter_mm = 2*np.sqrt(roi.get_area_millimeters() / np.pi)

            # columns = ["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"]
            with open(self.csv_file_path, 'ab') as csv_file:
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


class DicomViewerBackendInterface:
    def list_annotated_dicom(self):
        pass

class OsirixBackend(DicomViewerBackendInterface):
    # Maintain a handle to the OsiriX backend
    def __init__(self, osirix):
        self.osirix = osirix

    def list_annotated_dicom(self):
        annotated_dicom_list = []

        # The DCMPix objects each correspond to 1 DICOM file.
        # For 3D cross-sectional images, this is a 2D image slice.
        # The roi_tuples elements each correspond to the list of ROIs for a given DCMPix.
        viewer_controller   = self.osirix.frontmostViewer()
        pixs = viewer_controller.pixList(movieIdx=0) 
        roi_tuples = viewer_controller.roiList(movieIdx=0)
        # Collect annotated slices
        for slice_idx, roi_list in enumerate(roi_tuples):
            pix = pixs[slice_idx]
            if len(roi_list) > 0:
                dicom_file = dicom.read_file(pix.sourceFile)
                roi_list = [OsirixRoi(roi, pix, dicom_file) for roi in roi_list]
                annotated_dicom_list.append(AnnotatedDicom(dicom_file, pix.sourceFile, roi_list))

        return annotated_dicom_list

class AnnotatedDicom:
    def __init__(self, dicom_file, source_file_path, roi_list):
        self.dicom_file = dicom_file
        self.source_file_path = source_file_path
        self.roi_list = roi_list

# An abstraction around ROIs with the necessary APIs to write-back DICOM annotations.
class Roi:
    #########################
    ### Interface methods ###
    #########################
    def get_bitmask(self):
        pass
    def get_centroid(self):
        pass
    def get_area_millimeters(self):
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
    # world-X -> negative values (patient's left) :: voxel-X -> 512
    # Similarly for Y coordinates, 
    # world-Y -> positive (anterior) :: voxel-Y -> 0
    # world-Y -> negative (posterior) :: voxel-Y -> 512
    # 
    # Because of this inverse relationship, both voxel_to_world 
    # and world_to_voxel negate the coordinate input.
    @staticmethod
    def voxel_to_world(vX, vY, spacing, origin):
        wX = -(vX * spacing[0]) + origin[0]
        wY = -(vY * spacing[1]) + origin[1]
        return wX, wY
    
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

class OsirixRoi(Roi):
    DEFAULT_CIRCLE_DIAMETER_MM   = 6 # 6 millimeter
    DEFAULT_CIRCLE_AREA_MM = np.sqrt(6/np.pi)
    DEFAULT_COLOR = (252, 15, 41) # C
    DEFAULT_OPACITY = 10.691605567932129
    DEFAULT_THICKNESS = 0.5

    def __init__(self, roi, pix, dicom_file):
        self.roi = roi
        self.pix = pix
        self.dicom_file = dicom_file

    def get_centroid(self):
        return self.roi.centroid()

    def get_area_millimeters(self):
        # roi.roiArea() is units of cm^2. We want units of mm^2.
        # Thus we mutiply area by the square of mm per cm, which is 10^2 = 100.
        area = self.roi.roiArea()*100
        if area == 0:
            # Default for points, which have roiArea == 0
            area = OsirixRoi.DEFAULT_CIRCLE_AREA_MM
        return area

    def get_bitmask(self):
        # Change the rendered color of the ROI to help indicate it was operated on.
        self.beautify() 
        roi = self.roi
        # Some OsiriX ROI types have built-in support for bitmask-ification.
        # Others need to be manually interpreted.
        if roi.type in ['tPlain', 'tPencil', 'tCPolygon', 'tOPolygon']:
            bitmask = self.pix.getMapFromROI(roi).astype(np.uint8)
            return np.swapaxes(bitmask, 0, 1)

        # For ovals we embed a circle based on the average of the major and minor axis. 
        # For points we infer a 6mm diameter. 
        if len(roi.points) == 1 or roi.type == 't2DPoint' or roi.type == 'tOval':
            roiDiam = 2*np.sqrt(self.get_area_millimeters() / np.pi)
            if roiDiam == 0:
                roiDiam = OsirixRoi.DEFAULT_CIRCLE_DIAMETER_MM
            centX, centY = roi.centroid()[0], roi.centroid()[1]
            return Roi.create_circular_mask(
                512,
                512,
                roi.centroid()[0],
                roi.centroid()[1],
                radius=(roiDiam/2)*(1/self.dicom_file.PixelSpacing[0]))
        
        # For rectangles, we build a rectangle using the corners.
        if roi.type == 'tROI':
            mask = np.zeros((512, 512)).astype(int)
            points = roi.points
            xs = points[:, 0]
            ys = points[:, 1]
            min_x, max_x = int(np.round(min(xs))), int(np.round(max(xs)))            
            min_y, max_y = int(np.round(min(ys))), int(np.round(max(ys)))
            mask[min_x:max_x, min_y:max_y] = 1
            return np.swapaxes(mask, 0, 1)

        print("Unknown ROI type provided.")
        raise Exception("Unknown ROI type provided.")

    # Renders the OsiriX ROI in a translucent red.
    def beautify(self):
        self.roi.color = OsirixRoi.DEFAULT_COLOR
        self.roi.thickness = OsirixRoi.DEFAULT_THICKNESS
        self.roi.opacity = OsirixRoi.DEFAULT_OPACITY

CSV_FILE_PATH  = "./Users/aarash/annotations.csv"

print("wtf")
def embed_rois():
    osirix_backend = OsirixBackend(osirix)
    actions = [WriteToDicomOverlay(), AppendToCsv(CSV_FILE_PATH)]
    dicom_files = osirix_backend.list_annotated_dicom()
    for action in actions:
        print("wtf")
        action.apply_to_all(dicom_files)

if __name__ == "__main__":
    embed_rois()
    print("wtf2")
    # Refresh OsiriX 
    osirix.frontmostViewer().needsDisplayUpdate()
