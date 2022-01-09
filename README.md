# dicom-viz
Python modules and scripts that interoperate between data formats and DICOM viewer software

### Export ROI to DICOM overlay and CSV
This script extracts an ROI drawn using native annotation tools and embeds the area as a segmentation mask bitmap into the DICOM Overlay header. The DICOM file, which now includes both the image pixel data and the annotation overlay segmentation mask, can be transported for interoperable viewing within other DICOM viewers. The annotation bitmap can be parsed into a `numpy::nd.array` for use within standard machine learning workflows. In addition to modifying the DICOM file in place, the script accepts the path to a study-wide CSV file, appending the coordinates and diameter of the ROI as a new row in the table. 

To execute the procedure, first set the path to the CSV file within the `export_roi.py` script and save it. Then, open a DICOM series in OsiriX/Horos. Draw one or more ROIs using OsiriX / Horos’ annotation tools. Run the script via pyOsiriX and observe that the DICOM Overlay Pixel Data has been populated. Additionally, the CSV file has one new row appended to it for each drawn ROI.

### Import from CSV
This script takes as input the path to a CSV file containing the format displayed in Figure 4. A circular ROI bitmap is embedded into the DICOM header for each lung nodule associated with the loaded image series. 

To execute the procedure, first set the path to the CSV file within the `import_from_csv.py` script and save it. Then, open a DICOM series in OsiriX/Horos. Run the script via pyOsiriX and observe that the DICOM Overlay Pixel Data has been populated.

### Clear DICOM header
This script can be used to undo the effects of the former two scripts. It erases the Overlay content of all image files within the currently-loaded DICOM series. To execute the procedure, open a DICOM series in OsiriX/Horos and run the `clear_dicom_overlays.py` script via pyOsiriX.
