#HeaderInfo
#type=ROITool
#name=Export ROI into DICOM Overlay and CSV
#version=2021.08.05
#author=aheyd@berkeley.edu
#EndHeaderInfo

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