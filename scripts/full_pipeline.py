import matplotlib.patches
import matplotlib.pyplot as plt
import shapely.geometry
import skimage.io, skimage.color
import os,sys,random,math
import numpy as np
import utils2
from collections import namedtuple


######## TODO - make the shape bigger
#        coords = coords2.buffer(2)
#http://toblerity.org/shapely/manual.html#object.buffer
######



ProcessedCell=namedtuple('ProcessedCell',['coords','outline','skeleton','mesh','regions'])

def save_outlines_as_tif(outlines,data,fn):
    fig = plt.figure()
    #plt.imshow(np.fliplr(phase_data))
    plt.imshow(data,cmap='gray')
    
    #plt.show()
    #sys.exit()
    #ax = fig.add_subplot(111)

    for i,outline in enumerate(outlines):
        print "imaging outine",i,"from",len(outlines)
        for e in outline:
            print e
            plt.plot(e[1],e[0],'ro')
        plt.annotate(
            "cell_%d"%(i), xy = (outline[0,1], outline[0,0]), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        #ax.invert_yaxis()
    #plt.show()

def mask_signal(image,cell):    
    yfp_profile=[]
    print "number of regions",len(cell.regions)
    for region in cell.regions:
        g1 = shapely.geometry.Polygon(region)
        data_mask=np.zeros(image.shape)
        signal=0
        counter=0
        for i in range(int(math.floor(g1.bounds[0])),int(math.ceil(g1.bounds[2]))+1):
            for j in range(int(math.floor(g1.bounds[1])),int(math.ceil(g1.bounds[3]))+1):
                if g1.contains(shapely.geometry.Point(i,j)):
                    signal+=image[i][j]
                    counter+=1
                    #data_mask[j,i]=1   #why does it need to be j,i ???
                    data_mask[i,j]=1   #why does it need to be j,i ???
                    #ax0.plot(i,j,'ro')
        
        print "new region",region,"contains",np.sum(data_mask),"points"
        signal1=image*data_mask
        #ax1.imshow(data_mask,origin="lower",cmap = matplotlib.cm.Greys_r)
        #yfp_profile.append(np.sum(signal1)/np.sum(data_mask))#signal/counter)#np.sum(signal)/np.sum(data_mask))
        if counter==0:
            yfp_profile.append(0)
        else:
            yfp_profile.append(signal/counter)
    return yfp_profile
    #plt.plot(yfp_profile)
    #plt.show()

if __name__=="__main__":

    fig = plt.figure()
    ax = fig.add_subplot(111)

    processed_cells=[]
    #filename_phase = os.path.join('/Users/kerenlasker/projects/opto/microscopy/scripts/data/few/set1','KL5010-1_phase_set1_mask.tif')
    
    #filename_phase = os.path.join('/Users/kerenlasker/projects/opto/microscopy/scripts/data/few/set1','KL5010-1_phase_set1_one_cell2.tif')

    #filename_yfp = os.path.join('/Users/kerenlasker/projects/opto/microscopy/scripts/data/few/set1','KL5010-1_YFP_set1.tif')
    #filename_yfp = os.path.join('/Users/kerenlasker/projects/opto/microscopy/scripts/data/few/set1','KL5010-1_YFP_set1_one_cell2.tif')

    filename_phase = os.path.join('/Users/kerenlasker/projects/opto/microscopy/scripts/data/few/set2','KL5010-1_phase.tif')
    filename_yfp = os.path.join('/Users/kerenlasker/projects/opto/microscopy/scripts/data/few/set2','KL5010-1_yfp.tif')
    
    #transform the data
    data1 = skimage.io.imread(filename_phase)
    if len(data1.shape)==3:
        data1=skimage.color.rgb2gray(data1)
    phase_data=1.-data1
    yfp_data1 = skimage.io.imread(filename_yfp)
    if len(yfp_data1.shape)==3:
        yfp_data1=skimage.color.rgb2gray(yfp_data1)
    yfp_data=yfp_data1
    polygons = utils2.get_cells(yfp_data)
    outlines=[]
    yfp_profiles=[]
    for i,coords in enumerate(polygons[:10]):
        print "-------polygon",i
        outline = utils2.get_outline_from_polygon_coordinates(coords)
        plt.imshow(phase_data)
        for e in outline:
            plt.plot(e[1],e[0],'yo')
        #plt.show()
        print "outline size",len(outline)
        outlines.append(outline)
        skel = utils2.get_skeleton_from_outline_with_extension(outline)
        #ax.plot(skel[:,0],skel[:,1],'bo')
        [lines,regions,rects]=utils2.create_mesh2(skel,outline)
        #plt.annotate("cell_%d"%(i), xy = (skel[0,0], skel[0,1]), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5), arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        processed_cells.append(ProcessedCell(coords,outline,skel,lines,regions))
        #plt.show()
        yfp_profile=mask_signal(yfp_data,processed_cells[-1])
        yfp_profiles.append(yfp_profile)
        print "profile",i,yfp_profile
    plt.show()

    #save polygons to tif
    save_outlines_as_tif(outlines,yfp_data,"a.tif")

    #show the yfp profiles
    #plt.figure()
    fig, ((ax1, ax2,ax3,ax4,ax5), (ax6, ax7,ax8,ax9,ax10),(ax11, ax12,ax13,ax14,ax15), (ax16, ax17,ax18,ax19,ax20),) = plt.subplots(nrows=4, ncols=5)
    parts=[ax1,ax2,ax3,ax4,ax5,ax6, ax7,ax8,ax9,ax10,ax11, ax12,ax13,ax14,ax15,ax16, ax17,ax18,ax19,ax20]
    for i,ee in enumerate(yfp_profiles):
        print "prof",i
        parts[i].plot(ee,label="cell_%d"%(i+1))
        parts[i].legend()
    plt.show()
