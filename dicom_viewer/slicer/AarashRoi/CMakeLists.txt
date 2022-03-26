cmake_minimum_required(VERSION 3.13.4)

project(ExportRoi Export ROI module)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/AarashRoi")
set(EXTENSION_CATEGORY "Annotation")
set(EXTENSION_CONTRIBUTORS "Aarash Heydari (Fulton and Haste)")
set(EXTENSION_DESCRIPTION "Does what I tell it to do")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/AarashRoi.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/AarashRoi/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(ExportRoi)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
