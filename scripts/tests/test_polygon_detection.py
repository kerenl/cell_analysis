import random
import unittest
import os,sys
import skimage,skimage.io
import skimage.segmentation
import skimage.color
import numpy as np
sys.path.append("../")
import utils2
from skimage.filter import threshold_otsu
import matplotlib.pyplot as pp
import matplotlib.patches
import matplotlib.pyplot as plt
import shapely.geometry


class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        filename_phase = os.path.join('/Users/kerenlasker/projects/opto/microscopy/scripts/data/few/set1','KL5010-1_phase_set1_mask.tif')
        #'KL5010-1_phase_set1_one_cell2.tif')
        filename_yfp = os.path.join('/Users/kerenlasker/projects/opto/microscopy/scripts/data/few/set1','KL5010-1_YFP_set1_one_cell2.tif')
        data1 = skimage.io.imread(filename_phase)
        print "before gray min,max:",np.amin(data1),np.amax(data1)
        if len(data1.shape)==3:
            data1=skimage.color.rgb2gray(data1)
        self.phase_data = data1
        print "after gray min,max:",np.amin(self.phase_data),np.amax(self.phase_data)
        #self.phase_data = self.phase_data[::-1]
        self.phase_data=1.-self.phase_data #invert the image
        yfp_data1 = skimage.io.imread(filename_yfp)
        if len(yfp_data1.shape)==3:
            yfp_data1=skimage.color.rgb2gray(yfp_data1)
        self.yfp_data = yfp_data1
        #self.yfp_data = yfp_data1[::-1]
  

    def _test_binary_image(self): #binary does not rotate axis.
            # apply threshold
            img = self.phase_data.copy()
            print "min,max:",np.amin(img),np.amax(img)
            thresh = threshold_otsu(img)
            print "threshold",thresh
            binary = self.phase_data > thresh
            print "min,max:",np.amin(binary),np.amax(binary)
            bw=binary


            # Remove connected to image border
            cleared = bw.copy()
            skimage.segmentation.clear_border(cleared)
            #up to now nothing is rotated.

            # label image regions
            label_image = skimage.measure.label(cleared)
            skimage.io.imsave("phase_binary.tif",label_image*255)

            #find_contours
            borders = np.logical_xor(bw, cleared)
            label_image[borders] = -1
            image_label_overlay = skimage.color.label2rgb(label_image, image=img)
            skimage.io.imsave("phase_binary2.tif",image_label_overlay*255)
                        

    def _test_polygon_detection(self):
            #transform the data
            polygons = utils2.get_cells(self.phase_data)
            print "number of polygons",len(polygons)
            self.assertTrue(len(polygons)==23)
            img=self.phase_data.copy()
            colors=[50,100,150,200,250]
            print "number of polygons",len(polygons)
            for i,c in enumerate(colors):
                for [xi,yi] in polygons[i]:
                    img[xi,yi]= c
            skimage.io.imsave("phase_polygon.tif",img)
            skimage.io.imsave("phase.tif",self.phase_data)

    def _test_polygon_line_intersection(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        polygon = utils2.get_cells(self.phase_data)[0]
        for pp in polygon:
            ax.plot(pp[0],pp[1],'yo')

        outline = utils2.get_outline_from_polygon_coordinates(polygon)
        skel = utils2.get_skeleton_from_outline(outline)
        lines = utils2.find_intersecting_lines_from_outline(skel,outline)

        for line in lines:
            ax.plot([line[0][0],line[1][0]],[line[0][1],line[1][1]],'r-')
        plt.show()
            

    def test_build_mesh(self):
        print "============="
        fig = plt.figure()
        ax = fig.add_subplot(111)
        polygons = utils2.get_cells(self.phase_data)
        polygon = polygons[0]
        outline = utils2.get_outline_from_polygon_coordinates(polygon)
        for pp in polygon:
            ax.plot(pp[0],pp[1],'yo')
        skel = utils2.get_skeleton_from_outline_with_extension(outline)
        #find voronoi
        [edges_x1,edges_y1] = utils2.get_voroni_inside_polygon(outline)
        #detect branch, the result is already sorted
        [edges_x3,edges_y3]=utils2.remove_branches(edges_x1,edges_y1)
        #for e in edges_x3:
        #    ax.plot(e[0],e[1],'ro')
        skel=edges_x3
        # for e in edges_y3:
        #     ax.plot(e[0],e[1],'bo')

        #extend to get to the ends of the polygon
        poly = shapely.geometry.Polygon(outline)
        ### add the first point
        d=[skel[0][0]-skel[-1][0],skel[0][1]-skel[-1][1]]
        dp=[d[1],-d[0]]
        line = shapely.geometry.LineString([skel[0],[skel[0][0]+d[0]*20,skel[0][1]+d[1]*20]])
        #ax.plot([skel[0][0],skel[0][0]+d[0]*3],[skel[0][1],skel[0][1]+d[1]*3],'b-')
        #ax.plot([skel[0][0],skel[0][0]+dp[0]*3],[skel[0][1],skel[0][1]+dp[1]*3],'g-')
        ip_first = utils2.get_polygon_line_intersection(poly,line)
        #check if there is another point
        #1. get the line going through ip_first
        l1 = shapely.geometry.LineString([ip_first,[ip_first[0]+dp[0]*4,ip_first[1]+dp[1]*4]])
        a1=utils2.get_polygon_line_intersection(poly,l1)
        l2 =  shapely.geometry.LineString([ip_first,[ip_first[0]-dp[0]*4,ip_first[1]-dp[1]*4]])
        a2=utils2.get_polygon_line_intersection(poly,l2)
        #ax.plot([ip_first[0],ip_first[0]+dp[0]*3],[ip_first[1],ip_first[1]+dp[1]*3],'b-')
        #ax.plot([ip_first[0],ip_first[0]-dp[0]*3],[ip_first[1],ip_first[1]-dp[1]*3],'g-')
        print "a1",a1,"a2",a2
        mid_a = [a1[0]+(a2[0]-a1[0])/2,a1[1]+(a2[1]-a1[1])/2]
        ax.plot(mid_a[0],mid_a[1],'bo')
        print "mid_a",mid_a
        line = shapely.geometry.LineString([mid_a,[mid_a[0]+d[0]*20,mid_a[1]+d[1]*20]])
        ip_first = utils2.get_polygon_line_intersection(poly,line)
        ax.plot(ip_first[0],ip_first[1],'go')
            
        # last_line = [[14.0, 12.875], [17.35084833795014, 10.0]]
        # #ax.plot([last_line[0][0],last_line[1][0]],[last_line[0][1],last_line[1][1]],'r-')
        # dp=[ 16.409090909090907,19.125]
        # p1=[(last_line[0][0]+last_line[1][0])/2,(last_line[0][1]+last_line[1][1])/2] # last_line_mid_poin
        # p1_cont=[p1[0]-dp[0]*3,p1[1]-dp[1]*3]
        # ax.plot([p1[0],p1_cont[0]],[p1[1],p1_cont[1]],'r-')
        plt.show()
        #now make a line from the middle of the last line to the end of the polygon
        
    def _test_polygon_skeleton(self):
        #transform the data
        polygons = utils2.get_cells(self.phase_data)
        outline = utils2.get_outline_from_polygon_coordinates(polygons[0])
        img=self.phase_data.copy()
        for [xi,yi] in outline:
            img[xi,yi]= 255
        skimage.io.imsave("phase_outline.tif",img)
        skel = utils2.get_skeleton_from_outline_with_extension(outline)
        img=self.phase_data.copy()
        for [xi,yi] in skel:
            img[xi,yi]= 255
        skimage.io.imsave("phase_skeleton.tif",img)
        [lines,regions,rects]=utils2.create_mesh2(skel,outline)
        img=self.phase_data.copy()
        # pp.figure()
        # pp.subplot(1,3,1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #plt.imshow(self.phase_data,origin='upper')
        colors=['ro','bo','go']
        for ii in outline:
            ax.plot(ii[0],ii[1],'ro')
        for i,line in enumerate(lines):
            ax.plot([line[0][0],line[1][0]],[line[0][1],line[1][1]],'g-')
        for rect in rects:
            ax.add_patch(rect)
            #print regions[i]
            #pp.plot([line[0][0],line[1][0]],[line[0][1],line[1][1]],'b-',alpha=0.1)
        #pp.show()  
        plt.show()          
        sys.exit()
            
            
    def _test_draw_outline(self):
            polygons = utils2.get_cells(self.phase_data)
            outline = utils2.get_outline_from_polygon_coordinates(polygons[0])
            img = self.phase_data.copy()
            print img.shape
            for [xi,yi] in outline:
                img[xi,yi]= 255
            skimage.io.imsave("phase_outline.tif",self.phase_data)
        
if __name__ == '__main__':
    unittest.main()
