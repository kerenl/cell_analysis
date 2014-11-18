CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:11710

LoadImages:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:11|show_window:False|notes:\x5B\x5D]
    File type to be loaded:individual images
    File selection method:Text-Exact match
    Number of images in each group?:3
    Type the text that the excluded images have in common:Do not use
    Analyze all subfolders within the selected folder?:None
    Input image file location:Default Input Folder\x7C.
    Check image sets for missing or duplicate files?:No
    Group images by metadata?:No
    Exclude certain files?:No
    Specify metadata fields to group by:
    Select subfolders to analyze:
    Image count:2
    Text that these images have in common (case-sensitive):AP323-3_w1Phase
    Position of this image in each group:_w2Brightfield
    Extract metadata from where?:None
    Regular expression that finds metadata in the file name:None
    Type the regular expression that finds metadata in the subfolder path:None
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:2
    Load the input as images or objects?:Images
    Name this loaded image:bf
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:NucleiOutlines
    Channel number:1
    Rescale intensities?:Yes
    Text that these images have in common (case-sensitive):AP323-3_w2YFPCube
    Position of this image in each group:_w1GFP
    Extract metadata from where?:None
    Regular expression that finds metadata in the file name:None
    Type the regular expression that finds metadata in the subfolder path:None
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:2
    Load the input as images or objects?:Images
    Name this loaded image:yfp
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:NucleiOutlines
    Channel number:1
    Rescale intensities?:Yes

ImageMath:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:3|show_window:False|notes:\x5B\x5D]
    Operation:Invert
    Raise the power of the result by:1.0
    Multiply the result by:1.0
    Add to result:0
    Set values less than 0 equal to 0?:Yes
    Set values greater than 1 equal to 1?:Yes
    Ignore the image masks?:No
    Name the output image:inverted
    Image or measurement?:Image
    Select the first image:bf
    Multiply the first image by:1.0
    Measurement:
    Image or measurement?:Image
    Select the second image:
    Multiply the second image by:1
    Measurement:

EnhanceOrSuppressFeatures:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\x5D]
    Select the input image:inverted
    Name the output image:CorrInv
    Select the operation:Enhance
    Feature size:10
    Feature type:Speckles
    Range of hole sizes:1,10

IdentifyPrimaryObjects:[module_num:4|svn_version:\'Unknown\'|variable_revision_number:8|show_window:False|notes:\x5B\x5D]
    Select the input image:CorrInv
    Name the primary objects to be identified:cells
    Typical diameter of objects, in pixel units (Min,Max):10,100
    Discard objects outside the diameter range?:Yes
    Try to merge too small objects with nearby larger objects?:No
    Discard objects touching the border of the image?:Yes
    Select the thresholding method:RobustBackground Adaptive
    Threshold correction factor:1
    Lower and upper bounds on threshold:0,1
    Approximate fraction of image covered by objects?:0.01
    Method to distinguish clumped objects:None
    Method to draw dividing lines between clumped objects:None
    Size of smoothing filter:10
    Suppress local maxima that are closer than this minimum allowed distance:5
    Speed up by using lower-resolution image to find local maxima?:Yes
    Name the outline image:None
    Fill holes in identified objects?:Yes
    Automatically calculate size of smoothing filter?:Yes
    Automatically calculate minimum allowed distance between local maxima?:Yes
    Manual threshold:0.0
    Select binary image:RobustBackground Adaptive
    Retain outlines of the identified objects?:No
    Automatically calculate the threshold using the Otsu method?:Yes
    Enter Laplacian of Gaussian threshold:.5
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Automatically calculate the size of objects for the Laplacian of Gaussian filter?:Yes
    Enter LoG filter diameter:5
    Handling of objects if excessive number of objects identified:Continue
    Maximum number of objects:500
    Select the measurement to threshold with:None

EnhanceOrSuppressFeatures:[module_num:5|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\x5D]
    Select the input image:yfp
    Name the output image:CorrCfp
    Select the operation:Enhance
    Feature size:3
    Feature type:Circles
    Range of hole sizes:1,10

Crop:[module_num:6|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\x5D]
    Select the input image:CorrCfp
    Name the output image:CropCFP
    Select the cropping shape:Objects
    Select the cropping method:Coordinates
    Apply which cycle\'s cropping pattern?:Every
    Left and right rectangle positions:1,100
    Top and bottom rectangle positions:1,100
    Coordinates of ellipse center:500,500
    Ellipse radius, X direction:400
    Ellipse radius, Y direction:200
    Use Plate Fix?:No
    Remove empty rows and columns?:No
    Select the masking image:Nuclei
    Select the image with a cropping mask:None
    Select the objects:cells

MeasureObjectIntensity:[module_num:7|svn_version:\'Unknown\'|variable_revision_number:3|show_window:False|notes:\x5B\x5D]
    Hidden:1
    Select an image to measure:CropCFP
    Select objects to measure:cells

IdentifyPrimaryObjects:[module_num:8|svn_version:\'Unknown\'|variable_revision_number:8|show_window:False|notes:\x5B\x5D]
    Select the input image:CropCFP
    Name the primary objects to be identified:foci
    Typical diameter of objects, in pixel units (Min,Max):1,40
    Discard objects outside the diameter range?:Yes
    Try to merge too small objects with nearby larger objects?:No
    Discard objects touching the border of the image?:No
    Select the thresholding method:RobustBackground PerObject
    Threshold correction factor:1
    Lower and upper bounds on threshold:0,1
    Approximate fraction of image covered by objects?:0.01
    Method to distinguish clumped objects:Intensity
    Method to draw dividing lines between clumped objects:Intensity
    Size of smoothing filter:10
    Suppress local maxima that are closer than this minimum allowed distance:5
    Speed up by using lower-resolution image to find local maxima?:Yes
    Name the outline image:peaks
    Fill holes in identified objects?:Yes
    Automatically calculate size of smoothing filter?:Yes
    Automatically calculate minimum allowed distance between local maxima?:Yes
    Manual threshold:0.0
    Select binary image:RobustBackground PerObject
    Retain outlines of the identified objects?:Yes
    Automatically calculate the threshold using the Otsu method?:Yes
    Enter Laplacian of Gaussian threshold:.5
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Automatically calculate the size of objects for the Laplacian of Gaussian filter?:Yes
    Enter LoG filter diameter:5
    Handling of objects if excessive number of objects identified:Continue
    Maximum number of objects:500
    Select the measurement to threshold with:None

OverlayOutlines:[module_num:9|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Display outlines on a blank image?:No
    Select image on which to display outlines:inverted
    Name the output image:OrigOverlay
    Select outline display mode:Color
    Select method to determine brightness of outlines:Max of image
    Width of outlines:1
    Select outlines to display:peaks
    Select outline color:Red

RelateObjects:[module_num:10|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\x5D]
    Select the input child objects:foci
    Select the input parent objects:cells
    Calculate distances?:None
    Calculate per-parent means for all child measurements?:Yes
    Calculate distances to other parents?:No
    Parent name:Do not use

DisplayDataOnImage:[module_num:11|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\x5D]
    Display object or image measurements?:Object
    Select the input objects:foci
    Measurement to display:Location_Center_X
    Select the image on which to display the measurements:inverted
    Text color:red
    Name the output image that has the measurements displayed:DisplayImage
    Font size (points):10
    Number of decimals:0
    Image elements to save:Image

MeasureObjectIntensity:[module_num:12|svn_version:\'Unknown\'|variable_revision_number:3|show_window:False|notes:\x5B\x5D]
    Hidden:1
    Select an image to measure:CropCFP
    Select objects to measure:foci

MeasureObjectSizeShape:[module_num:13|svn_version:\'1\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D]
    Select objects to measure:cells
    Calculate the Zernike features?:Yes

ExportToSpreadsheet:[module_num:14|svn_version:\'Unknown\'|variable_revision_number:7|show_window:False|notes:\x5B\x5D]
    Select or enter the column delimiter:Comma (",")
    Prepend the output file name to the data file names?:Yes
    Add image metadata columns to your object data file?:No
    Limit output to a size that is allowed in Excel?:No
    Select the columns of measurements to export?:No
    Calculate the per-image mean values for object measurements?:No
    Calculate the per-image median values for object measurements?:No
    Calculate the per-image standard deviation values for object measurements?:No
    Output file location:Default Output Folder\x7C.
    Create a GenePattern GCT file?:No
    Select source of sample row name:Metadata
    Select the image to use as the identifier:None
    Select the metadata to use as the identifier:None
    Export all measurements, using default file names?:Yes
    Press button to select measurements to export:foci\x7CIntensity_IntegratedIntensity_CropCFP,foci\x7CNumber_Object_Number,foci\x7CParent_cells,cells\x7CNumber_Object_Number,cells\x7CIntensity_IntegratedIntensity_CropCFP,cells\x7CAreaShape_Area,cells\x7CAreaShape_MinorAxisLength,cells\x7CAreaShape_MajorAxisLength,cells\x7CChildren_foci_Count
    Data to export:Image
    Combine these object measurements with those of the previous object?:No
    File name:Image.csv
    Use the object name for the file name?:No
    Data to export:cells
    Combine these object measurements with those of the previous object?:No
    File name:Nuclei.csv
    Use the object name for the file name?:No
    Data to export:foci
    Combine these object measurements with those of the previous object?:Yes
    File name:DATA.csv
    Use the object name for the file name?:Yes

DisplayDataOnImage:[module_num:15|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\x5D]
    Display object or image measurements?:Object
    Select the input objects:foci
    Measurement to display:Number_Object_Number
    Select the image on which to display the measurements:inverted
    Text color:red
    Name the output image that has the measurements displayed:DisplayImage
    Font size (points):1
    Number of decimals:1
    Image elements to save:Image
