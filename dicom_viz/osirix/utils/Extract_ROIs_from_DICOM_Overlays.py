#HeaderInfo
#type=ROITool
#name=DICOM_Overlays->ROIs
#version=2018.01.29
#author=Matthew.Lewis@UTSouthwestern.edu
#EndHeaderInfo

import osirix
import dicom
import numpy as np
import shutil
import os

vc = osirix.frontmostViewer()
#print(vc)

pixs = vc.pixList()
#rois=vc.roiList(movieIdx=0)
#print(pixs)
p=0
wait = vc.startWaitProgressWindow('Extracting ROIs from DICOM Overlays…', len(pixs))
for pix in pixs:
    ds = dicom.read_file(pix.sourceFile)

    for j in range(16):
        try:
            mask = np.fromstring(ds[0x60003000+0x20000*j].value,dtype=np.uint8)
            mask = np.packbits(np.unpackbits(mask).reshape(-1,8)[:,::-1])
            mask = np.unpackbits(mask).reshape(ds[0x60000011+0x20000*j].value,ds[0x60000010+0x20000*j].value,order='F')
            roi_stuff = ds[0x60000022+0x20000*j].value.split('/')  
            roiNew = osirix.ROI(itype='tPlain',buffer=mask.astype(np.bool),name=roi_stuff[0],DCMPix=pix)
            # roiNew.color = eval(roi_stuff[1])
            vc.setROI(roiNew,position=p)
            vc.needsDisplayUpdate()                   
        except Exception as e:
            print("Error: ")
            print(e)
            print("——-")
            print(pix)
            print(p)
            print("====")

    p=p+1
    print(p)
    wait.incrementBy(1.0)

vc.endWaitWindow(wait)