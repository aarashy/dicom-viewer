#HeaderInfo
#type=ROITool
#name=ROIs->DICOM_Overlays
#version=2018.01.29
#author=Matthew.Lewis@UTSouthwestern.edu
#EndHeaderInfo

#itype (Optional[str]): The type of ROI to create.  Currently can only be none of
# |             tPlain, tCPolygon (default), tOPolygon, tPencil.
#help(roi)
#raise Warning('W')

import osirix
import dicom
import numpy as np
import shutil
import os

tempdir = '/tmp/SBI'
shutil.rmtree(tempdir,ignore_errors=True)
os.mkdir(tempdir)

vc = osirix.frontmostViewer()
pixs = vc.pixList()
rois=vc.roiList(movieIdx=0)


wait = vc.startWaitProgressWindow('Embedding OsiriX ROIs in DICOM Overlays…', len(rois))
for i in range(len(rois)):
    if len(rois[i]):
        ds = dicom.read_file(pixs[i].sourceFile)
        
        supported_rois = ()
        for j in range(len(rois[i])):
            if (rois[i][j].type == 'tPlain') or (rois[i][j].type == 'tPencil') or (rois[i][j].type == 'tCPolygon') or (rois[i][j].type == 'tOPolygon'):
	            supported_rois += rois[i][j],
        
        for j in range(len(supported_rois)):
            #roi = rois[i][j]
            roi = supported_rois[j]
            print(roi.type)
            test = roi.name+'/'+repr(roi.color)
            wtf=test.split('/')
            roi.color=eval(wtf[1])
            print(roi.color)		    

            mask = pixs[i].getMapFromROI(roi).astype(np.uint8)
            w = np.packbits(mask.flatten('F'))
            w = np.packbits(np.unpackbits(w).reshape(-1,8)[:,::-1])

            ds.add_new(0x60000010 + j*0x20000 , 'US', pixs[i].shape[1])
            ds.add_new(0x60000011 + j*0x20000 , 'US', pixs[i].shape[0])
            ds.add_new(0x60000022 + j*0x20000 , 'LO', roi.name+'/'+repr(roi.color) )
            ds.add_new(0x60000040 + j*0x20000 , 'CS', 'R')
            ds.add_new(0x60000050 + j*0x20000 , 'SS', [1,1])
            ds.add_new(0x60000100 + j*0x20000 , 'US', 1)
            ds.add_new(0x60000102 + j*0x20000 , 'US', 0)
            ds.add_new(0x60003000 + j*0x20000 , 'OW', w)

        ds.save_as(tempdir + '/' + str(i) + str(j) +'.dcm')

    wait.incrementBy(1.0)

vc.endWaitWindow(wait)		

try:
    files = []
    for fl in os.listdir(tempdir):
        files.append(tempdir + '/' + fl)

    #Get the databaseBroser
    bc = osirix.currentBrowser()
    bc.copyFilesIntoDatabaseIfNeeded(files)

    del(files)
except Exception:
    pass
