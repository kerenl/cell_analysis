CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:11710

LoadImages:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:11|show_window:True|notes:\x5B\x5D]
    File type to be loaded:individual images
    File selection method:Text-Regular expressions
    Number of images in each group?:3
    Type the text that the excluded images have in common:Do not use
    Analyze all subfolders within the selected folder?:None
    Input image file location:Default Input Folder\x7CDropbox/optogenetics/microscopy/scope_data/2013-08-02-X1/X1-new-YFP-settings
    Check image sets for missing or duplicate files?:No
    Group images by metadata?:No
    Exclude certain files?:No
    Specify metadata fields to group by:region,promotor,image,TP
    Select subfolders to analyze:P3,ls101
    Image count:2
    Text that these images have in common (case-sensitive):w1Phase
    Position of this image in each group:_w2Brightfield
    Extract metadata from where?:File name
    Regular expression that finds metadata in the file name:^(?P<Strain>\x5BA-Z\x5D\x5B0-9\x5D).(?P<TP>.*)-(?P<ImageNum>.*)_w1Phase.TIF
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
    Text that these images have in common (case-sensitive):w3YFPCube
    Position of this image in each group:2
    Extract metadata from where?:File name
    Regular expression that finds metadata in the file name:^(?P<Strain>\x5BA-Z\x5D\x5B0-9\x5D).(?P<TP>.*)-(?P<ImageNum>.*)_w3YFPCube.TIF
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:3
    Load the input as images or objects?:Images
    Name this loaded image:yfp
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedImageOutlines
    Channel number:1
    Rescale intensities?:No

ImageMath:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D]
    Operation:Invert
    Raise the power of the result by:1
    Multiply the result by:1
    Add to result:0
    Set values less than 0 equal to 0?:Yes
    Set values greater than 1 equal to 1?:Yes
    Ignore the image masks?:No
    Name the output image:bf_inverted
    Image or measurement?:Image
    Select the first image:bf
    Multiply the first image by:1
    Measurement:
    Image or measurement?:Image
    Select the second image:
    Multiply the second image by:1
    Measurement:

SaveImages:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:7|show_window:True|notes:\x5B\x5D]
    Select the type of image to save:Image
    Select the image to save:bf_inverted
    Select the objects to save:None
    Select the module display window to save:None
    Select method for constructing file names:From image filename
    Select image name for file prefix:bf
    Enter single file name:OrigBlue
    Do you want to add a suffix to the image file name?:Yes
    Text to append to the image name:_inverted
    Select file format to use:tiff
    Output file location:Default Output Folder sub-folder\x7Cinverted
    Image bit depth:16
    Overwrite existing files without warning?:Yes
    Select how often to save:Every cycle
    Rescale the images? :No
    Save as grayscale or color image?:Grayscale
    Select colormap:gray
    Store file and path information to the saved image?:Yes
    Create subfolders in the output folder?:No

EnhanceOrSuppressFeatures:[module_num:4|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Select the input image:bf_inverted
    Name the output image:CorrInv
    Select the operation:Enhance
    Feature size:10
    Feature type:Speckles
    Range of hole sizes:1,10

IdentifyPrimaryObjects:[module_num:5|svn_version:\'Unknown\'|variable_revision_number:8|show_window:True|notes:\x5B\x5D]
    Select the input image:CorrInv
    Name the primary objects to be identified:cells
    Typical diameter of objects, in pixel units (Min,Max):7,40
    Discard objects outside the diameter range?:Yes
    Try to merge too small objects with nearby larger objects?:No
    Discard objects touching the border of the image?:Yes
    Select the thresholding method:RobustBackground Adaptive
    Threshold correction factor:1
    Lower and upper bounds on threshold:0.0011,1.0
    Approximate fraction of image covered by objects?:0.01
    Method to distinguish clumped objects:Shape
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

MeasureObjectSizeShape:[module_num:6|svn_version:\'1\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Select objects to measure:cells
    Calculate the Zernike features?:Yes

MeasureObjectNeighbors:[module_num:7|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Select objects to measure:cells
    Method to determine neighbors:Within a specified distance
    Neighbor distance:5
    Retain the image of objects colored by numbers of neighbors for use later in the pipeline (for example, in SaveImages)?:No
    Name the output image:ObjectNeighborCount
    Select colormap:Default
    Retain the image of objects colored by percent of touching pixels for use later in the pipeline (for example, in SaveImages)?:No
    Name the output image:PercentTouching
    Select a colormap:Default

MeasureObjectIntensity:[module_num:8|svn_version:\'Unknown\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D]
    Hidden:1
    Select an image to measure:yfp
    Select objects to measure:cells

DisplayDataOnImage:[module_num:9|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Display object or image measurements?:Object
    Select the input objects:cells
    Measurement to display:Number_Object_Number
    Select the image on which to display the measurements:bf_inverted
    Text color:red
    Name the output image that has the measurements displayed:DisplayImage
    Font size (points):10
    Number of decimals:2
    Image elements to save:Image

ExportToSpreadsheet:[module_num:10|svn_version:\'Unknown\'|variable_revision_number:7|show_window:True|notes:\x5B\x5D]
    Select or enter the column delimiter:Comma (",")
    Prepend the output file name to the data file names?:Yes
    Add image metadata columns to your object data file?:Yes
    Limit output to a size that is allowed in Excel?:No
    Select the columns of measurements to export?:No
    Calculate the per-image mean values for object measurements?:No
    Calculate the per-image median values for object measurements?:No
    Calculate the per-image standard deviation values for object measurements?:No
    Output file location:Default Output Folder\x7CDocuments/Bio Research/Image Analysis/X1
    Create a GenePattern GCT file?:No
    Select source of sample row name:Metadata
    Select the image to use as the identifier:None
    Select the metadata to use as the identifier:None
    Export all measurements, using default file names?:Yes
    Press button to select measurements to export:
    Data to export:Filtered1
    Combine these object measurements with those of the previous object?:No
    File name:DATA.csv
    Use the object name for the file name?:Yes

ExportToDatabase:[module_num:11|svn_version:\'Unknown\'|variable_revision_number:20|show_window:True|notes:\x5B\x5D]
    Database type:SQLite
    Database name:DefaultDB
    Add a prefix to table names?:No
    Table prefix:Expt_
    SQL file prefix:SQL_
    Output file location:Default Output Folder\x7CDocuments/Bio Research/Image Analysis/X1
    Create a CellProfiler Analyst properties file?:Yes
    Database host:
    Username:
    Password:
    Name the SQLite database file:segmentation_output.db
    Calculate the per-image mean values of object measurements?:No
    Calculate the per-image median values of object measurements?:No
    Calculate the per-image standard deviation values of object measurements?:No
    Calculate the per-well mean values of object measurements?:No
    Calculate the per-well median values of object measurements?:No
    Calculate the per-well standard deviation values of object measurements?:No
    Export measurements for all objects to the database?:Select...
    Select the objects:cells
    Maximum # of characters in a column name:64
    Create one table per object or a single object table?:Single object table
    Enter an image url prepend if you plan to access your files via http:
    Write image thumbnails directly to the database?:No
    Select the images you want to save thumbnails of:
    Auto-scale thumbnail pixel intensities?:Yes
    Select the plate type:None
    Select the plate metadata:None
    Select the well metadata:None
    Include information for all images, using default values?:No
    Hidden:1
    Hidden:1
    Hidden:0
    Select an image to include:bf_inverted
    Use the image name for the display?:Yes
    Image name:Channel1
    Channel color:gray
    Do you want to add group fields?:No
    Enter the name of the group:
    Enter the per-image columns which define the group, separated by commas:ImageNumber, Image_Metadata_TP, Image_Metadata_image, Image_Metadata_strain 
    Do you want to add filter fields?:No
    Automatically create a filter for each plate?:No
