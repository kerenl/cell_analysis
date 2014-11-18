import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from skimage import io
import os,math
from skimage.filter import threshold_otsu
from skimage.segmentation import clear_border
#from skimage.morphology import label, closing, square
from skimage.color import label2rgb
from skimage.measure import regionprops
import skimage
from skimage.morphology import medial_axis
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from skimage import draw
import networkx as nx
import skeleton_to_mesh
import shapely.geometry
import matplotlib
import scipy.ndimage
SKELETON_RESOLUTION=20

def get_outline_from_polygon_coordinates(coords):
    #coords - polygon coordinates
    aax=[]
    aay=[]
    points = []
    for i,p1 in enumerate(coords):
        num=0
        for j,p2 in enumerate(coords):
            if i==j:
                continue
            if math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))<1.001:
                num=num+1
        if num<4:
            aax.append(coords[i,0])
            aay.append(coords[i,1])
            points.append(shapely.geometry.Point([coords[i,0],coords[i,1]]))
    outline=np.zeros([len(aax),2])
    #sort points
    outline[0,:]=[points[0].x,points[0].y]
    f_ind=0
    used_inds=[]
    for i in range(0,len(points)):
        dist=9999
        closest_ind=-1
        for j in range(0,len(points)):
            if j in used_inds:
                continue
            if points[f_ind].distance(points[j])<dist:
                dist = points[f_ind].distance(points[j])
                closest_ind=j
        if closest_ind==-1:
            print "did not find a point for",i
            sys.exit(-1)
        outline[i,:]=[points[f_ind].x,points[f_ind].y]
        f_ind=closest_ind
        used_inds.append(closest_ind)
    
    return outline


def find_intersecting_lines_from_outline(skel,outline):
    #for each point on the skel find the closest point on the outline and draw a line
    #make a polygon out of the outline
    lines=[]
    poly = shapely.geometry.Polygon(outline)
    if not poly.is_valid:
        print "ERROR: The polygon is not valid. Exiting"
        l1 = shapely.geometry.LineString(poly.exterior.coords)
        if not l1.is_simple:
            print "ERROR: The polydon exterioir is not simple"
        return
    #find intersection points for each point of the outline
    print range(1,len(skel)-1),len(skel)
    dp=[skel[-1][1]-skel[0][1],-(skel[-1][0]-skel[0][0])]
    for i in range(0,len(skel)):
        #try:
            p1=skel[i]
            line = shapely.geometry.LineString([p1,[p1[0]+dp[0]*200, p1[1]+dp[1]*200]])
            xx = poly.intersection(line)
            if isinstance(xx,shapely.geometry.MultiLineString):
                for linei in xx:
                    if shapely.geometry.Point(p1).distance(shapely.geometry.Point(linei.coords[0])) <0.01:
                        ip1=linei.coords[1]
                    else:
                        continue 
            else:
                if len(xx.coords)==1: #the point touches the polygon
                    ip1=xx.coords[0]
                else: #the point is in the polygon
                    ip1=xx.coords[1]
      
            line = shapely.geometry.LineString([p1,[p1[0]-dp[0]*20, p1[1]-dp[1]*20]])
            xx = poly.intersection(line)
            if isinstance(xx,shapely.geometry.MultiLineString):
                for linei in xx:
                    if shapely.geometry.Point(p1).distance(shapely.geometry.Point(linei.coords[0])) <0.01:
                        ip2=linei.coords[1]
                    else:
                        continue 
            else:
                if len(xx.coords)==1: #the point touches the polygon
                    ip2=xx.coords[0]
                else: #the point is in the polygon
                    ip2=xx.coords[1]
            lines.append([ip1,ip2])
            #except:
            #print "did not draw a line for",i
    #add an extra line for the last patch
    p1=lines[-1][0]
    p2=lines[-1][1]
    p3=[poly.bounds[2],poly.bounds[3]]
    box=poly.bounds
    #sys.exit()
    d=[p2[0]-p1[0],p2[1]-p1[1]]
    step=5
    eline = [[p3[0]-d[0]*step,p3[1]-d[1]*step],[p3[0]+d[0]*step,p3[1]+d[1]*step]]
    print eline
    lines.append(eline)
    return lines

def create_mesh2(skel,outline):
    #for each point on the skel find the closest point on the outline and draw a line
    #make a polygon out of the outline
    lines=[]
    poly = shapely.geometry.Polygon(outline)
    if not poly.is_valid:
        print "ERROR: The polygon is not valid. Exiting"
        l1 = shapely.geometry.LineString(poly.exterior.coords)
        if not l1.is_simple:
            print "ERROR: The polydon exterioir is not simple"
        return
    #find intersection points for each point of the outline
    print range(1,len(skel)-1),len(skel)
    dp=[skel[-1][1]-skel[0][1],-(skel[-1][0]-skel[0][0])]
    d=[skel[-1][0]-skel[0][0],-(skel[-1][1]-skel[0][1])]
    for i in range(0,len(skel)):
        #try:
            p1=skel[i]
            line = shapely.geometry.LineString([p1,[p1[0]+dp[0]*200, p1[1]+dp[1]*200]])
            ip1=get_polygon_line_intersection(poly,line)
            line = shapely.geometry.LineString([p1,[p1[0]-dp[0]*20, p1[1]-dp[1]*20]])
            ip2=get_polygon_line_intersection(poly,line)
            lines.append([ip1,ip2])
            #except:
            #print "did not draw a line for",i

    #add an extra line for the last patch
    print "dp:",dp
    dpp=[dp[1],-dp[0]]
    p1=[(lines[-1][0][0]+lines[-1][1][0])/2,(lines[-1][0][1]+lines[-1][1][1])/2] # last_line_mid_poin
    p1_cont=[p1[0]-dpp[0]*5,p1[1]-dpp[1]*5]
   
    line = shapely.geometry.LineString([p1,p1_cont])
    print "before intersection"
    xx = poly.intersection(line)
    if isinstance(xx,shapely.geometry.MultiLineString):
        for linei in xx:
            if shapely.geometry.Point(p1).distance(shapely.geometry.Point(linei.coords[0])) <0.01:
                ip1=linei.coords[1]
            else:
                continue 
    else:
        if len(xx.coords)==1: #the point touches the polygon
            ip1=xx.coords[0]
        else: #the point is in the polygon
            ip1=xx.coords[1]
    print "LAST POINT",p1,ip1
    # #now do in d and find the two intersection with the polygon
    # d=[lines[-1][1][0]-lines[-1][0][0],lines[-1][1][1]-lines[-1][0][1]]
    # print "start point",ip1
    # print "minus point",[ip1[0]-3*d[0],ip1[1]-3*d[1]]
    # print "after point",[ip1[0]+3*d[0],ip1[1]+3*d[1]]
    # eline = [[ip1[0]-3*d[0],ip1[1]-3*d[1]],[ip1[0]+3*d[0],ip1[1]+3*d[1]]]
    # lines.append(eline)

    # #add an extra line for the first patch
    # p1=[(lines[0][0][0]+lines[0][1][0])/2,(lines[0][0][1]+lines[0][1][1])/2] # last_line_mid_poin
    # p1_cont=[p1[0]+dpp[0]*5,p1[1]+dpp[1]*5]
   
    # line = shapely.geometry.LineString([p1,p1_cont])
    # xx = poly.intersection(line)
    # if isinstance(xx,shapely.geometry.MultiLineString):
    #     for linei in xx:
    #         if shapely.geometry.Point(p1).distance(shapely.geometry.Point(linei.coords[0])) <0.01:
    #             ip1=linei.coords[1]
    #         else:
    #             continue 
    # else:
    #     if len(xx.coords)==1: #the point touches the polygon
    #         ip1=xx.coords[0]
    #     else: #the point is in the polygon
    #         ip1=xx.coords[1]
    # print "FIRST POINT",p1,ip1
    # #now do in d and find the two intersection with the polygon
    # eline = [[ip1[0]-3*d[0],ip1[1]-3*d[1]],[ip1[0]+3*d[0],ip1[1]+3*d[1]]]
    # lines = [eline] + lines

    #print lines[-1],eline
    #draw the mesh
    regions=[]
    rects=[]
    for i in range(len(lines)-1):
        region = get_region(lines[i],lines[i+1],outline)
        if region==None:
            continue
        #print list(region.exterior.coords)
        rect = matplotlib.patches.Polygon(list(region.exterior.coords),color=np.random.rand(3,1))
        #ax.add_patch(rect)
        regions.append(list(region.exterior.coords))
        rects.append(rect)
    return [lines,regions,rects]

def remove_branches(edges_x1,edges_y1):
    nodes_dic={}
    for e1 in edges_x1:
        nodes_dic[str(e1)]=e1
    for e1 in edges_y1:
        nodes_dic[str(e1)]=e1
    print edges_x1[0],edges_x1[1]
    G=nx.Graph()
    for i in range(len(edges_x1)):
        G.add_edge(str(edges_x1[i]),str(edges_y1[i]))
    #find all nodes with rank 1
    sources=[]
    for n in nx.nodes(G):
        if len(G.neighbors(n))==1:
            sources.append(n)
    #find the longest shortest path between the sources
    longest_path=[]
    for i,s1 in enumerate(sources):
        for s2 in sources[i+1:]:
            sp=nx.shortest_path(G,source=s1,target=s2)
            if len(sp)>len(longest_path):
                longest_path=sp
    #get the skeleton with no branches
    edges_x1_no_sk=[]
    edges_y1_no_sk=[]
    for i in range(len(longest_path)-1):
        edges_x1_no_sk=np.append(edges_x1_no_sk,nodes_dic[longest_path[i]])
        edges_y1_no_sk=np.append(edges_y1_no_sk,nodes_dic[longest_path[i+1]])
    return[edges_x1_no_sk.reshape(len(longest_path)-1,2),edges_y1_no_sk.reshape(len(longest_path)-1,2)]

def adjust_bounds(ax, points):
    ptp_bound = points.ptp(axis=0)
    ax.set_xlim(points[:,0].min() - 0.1*ptp_bound[0],
                points[:,0].max() + 0.1*ptp_bound[0])
    ax.set_ylim(points[:,1].min() - 0.1*ptp_bound[1],
                points[:,1].max() + 0.1*ptp_bound[1])


def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments



def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
        
    z = np.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    
    ax = plt.gca()
    ax.add_collection(lc)
    
    return lc


def is_point_inside_polygon(x,y,poly):
    pp=shapely.geometry.Polygon(poly)
    return shapely.geometry.Point([x,y]).within(shapely.geometry.Polygon(poly))
'''
    #http://www.ariel.com.au/a/python-point-int-poly.html
    n = len(poly)
    inside =False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside
'''
def get_region(l1,l2,outline):
    print l1,l2
    #create a polygon from the two lines
    #extend l1
    d1=[l1[1][0]-l1[0][0],l1[1][1]-l1[0][1]]
    p1=[l1[0][0]-d1[0]*10,l1[0][1]-d1[1]*10]
    p2=[l1[1][0]+d1[0]*10,l1[1][1]+d1[1]*10]
    d2=[l2[1][0]-l2[0][0],l2[1][1]-l2[0][1]]
    p3=[l2[0][0]-d2[0]*10,l2[0][1]-d2[1]*10]
    p4=[l2[1][0]+d2[0]*10,l2[1][1]+d2[1]*10]
    points_order=[p1,p2,p3,p4]
    if shapely.geometry.Point(p2).distance(shapely.geometry.Point(p3))>shapely.geometry.Point(p2).distance(shapely.geometry.Point(p4)):
        points_order=[p1,p2,p4,p3]
    large_region=shapely.geometry.Polygon(points_order)
    g1 = shapely.geometry.Polygon(outline)
    #print large_region
    try:
        region=g1.intersection(large_region)
    except:
        print "can not intersect, returning NULL"
        return None
    if not isinstance(region,shapely.geometry.polygon.Polygon):
        return None
    rect = matplotlib.patches.Polygon(list(region.exterior.coords),color='g')
    #plt.add_patch(rect)
    return region


def get_skeleton_from_outline(outline):
    vor = Voronoi(outline)
    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi diagram is not 2-D")

    points=outline
    #ax.scatter(points[:,0],points[:,1],c='g')
    edges_x11=[]
    edges_y11=[]
    #plt.scatter(vor.vertices[:,0],vor.vertices[:,1],c='r')
    for vpair in vor.ridge_vertices:
        if -1 in vpair:
            continue
        v0=vor.vertices[vpair[0]]
        v1=vor.vertices[vpair[1]]
        if not is_point_inside_polygon(v0[0],v0[1],outline):
            continue
        if not is_point_inside_polygon(v1[0],v1[1],outline):
            continue
        #plt.plot([v0[0],v1[0]],[v0[1],v1[1]])
        edges_x11.append(v0)
        edges_y11.append(v1)

    edges_x1=np.array(edges_x11)
    edges_y1=np.array(edges_y11)

    #detect branch, the result is already sorted
    [edges_x3,edges_y3]=remove_branches(edges_x1,edges_y1)

    #define the skeleton
    skel=np.zeros([len(edges_x3)+3,2])
    for i in range(len(edges_x3)-1):
        skel[i+1,:]=edges_x3[i+1]
    skel[-3,:]=edges_x3[-1]
    skel[-2,:]=edges_y3[-1]

    #extend to get to the ends of the polygon
    poly = shapely.geometry.Polygon(outline)
    ### add the first point
    pivot_ind=10
    if len(skel)<7:
        pivot_ind=len(skel)-2
    d=[skel[pivot_ind][0]-skel[1][0],skel[pivot_ind][1]-skel[1][1]]
    line = shapely.geometry.LineString([skel[pivot_ind],[skel[pivot_ind][0]-d[0]*200,skel[pivot_ind][1]-d[1]*200]])
    ip_first=[]
    ip_last=[]
    xx = poly.intersection(line)
    if isinstance(xx,shapely.geometry.MultiLineString):
        for linei in xx:
            if shapely.geometry.Point(p1).distance(shapely.geometry.Point(linei.coords[0])) <0.01:
                ip_first=linei.coords[1]
            else:
                continue
    else:
         print "First not multi"
         print xx
         ip_first=xx.coords[1]
    print "First point coordinates",ip_first
    ### add the last point
    pivot_ind=len(skel)-5
    if pivot_ind<1:
        pivot_ind=1
    d=[skel[-2][0]-skel[pivot_ind][0],skel[-2][1]-skel[pivot_ind][1]]
    line = shapely.geometry.LineString([skel[pivot_ind],[skel[pivot_ind][0]+d[0]*200,skel[pivot_ind][1]+d[1]*200]])
    xx = poly.intersection(line)
    if isinstance(xx,shapely.geometry.MultiLineString):
        for linei in xx:
            if shapely.geometry.Point(p1).distance(shapely.geometry.Point(linei.coords[0])) <0.01:
                ip_last=linei.coords[1]
            else:
                continue
    else:
         print "First not multi"
         ip_last=xx.coords[1]
    print "Last point coordinates",ip_last
    skel[0,:]=ip_first
    skel[-1,:]=ip_last
    skel_line = shapely.geometry.LineString(skel)
    L=SKELETON_RESOLUTION
    #interpolate, sample L points
    interpolate_line = np.zeros([L+1,2])
    linel=skel_line.length
    for i in range(L+1):
        pp=skel_line.interpolate(linel/L*i)
        interpolate_line[i,:]=[pp.coords[0][0],pp.coords[0][1]]
    return interpolate_line


def get_voroni_inside_polygon(outline):
        #voronoi analysis
    vor = Voronoi(outline)
    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi diagram is not 2-D")

    points=outline
    edges_x11=[]
    edges_y11=[]
    for vpair in vor.ridge_vertices:
        if -1 in vpair:
            continue
        v0=vor.vertices[vpair[0]]
        v1=vor.vertices[vpair[1]]
        if not is_point_inside_polygon(v0[0],v0[1],outline):
            continue
        if not is_point_inside_polygon(v1[0],v1[1],outline):
            continue
        edges_x11.append(v0)
        edges_y11.append(v1)

    edges_x1=np.array(edges_x11)
    edges_y1=np.array(edges_y11)
    return [edges_x1,edges_y1]

def get_polygon_line_intersection(poly,line):
    xx = poly.intersection(line)
    ip=None
    if isinstance(xx,shapely.geometry.Point):
        ip=xx.coords[0]
    elif isinstance(xx,shapely.geometry.LineString):
        for i in range(2):
            #TODO - hack, change
            p1=shapely.geometry.Point(line.coords[0])
            p2=shapely.geometry.Point(xx.coords[i])
            if p1.distance(p2)>0.01:
                ip=p2.coords[0]
    else:#multiline
        for linei in xx:
            if shapely.geometry.Point(p1).distance(shapely.geometry.Point(linei.coords[0])) <0.01:
                ip=linei.coords[1]
            else:
                continue
    return ip


def get_skeleton_from_outline_with_extension(outline):
    #find voronoi
    [edges_x1,edges_y1] = get_voroni_inside_polygon(outline)
    #detect branch, the result is already sorted
    [edges_x3,edges_y3]=remove_branches(edges_x1,edges_y1)
    
  
    skel=edges_x3

    #extend to get to the start of the polygon
    ip_first=-1
    num_extra=0
    poly = shapely.geometry.Polygon(outline)
    d=[skel[0][0]-skel[-1][0],skel[0][1]-skel[-1][1]]
    dp=[d[1],-d[0]]
    try:
        line = shapely.geometry.LineString([skel[0],[skel[0][0]+d[0]*20,skel[0][1]+d[1]*20]])
        ip_first = get_polygon_line_intersection(poly,line)
        l1 = shapely.geometry.LineString([ip_first,[ip_first[0]+dp[0]*4,ip_first[1]+dp[1]*4]])
        a1=get_polygon_line_intersection(poly,l1)
        l2 =  shapely.geometry.LineString([ip_first,[ip_first[0]-dp[0]*4,ip_first[1]-dp[1]*4]])
        a2=get_polygon_line_intersection(poly,l2)
        mid_a = [a1[0]+(a2[0]-a1[0])/2,a1[1]+(a2[1]-a1[1])/2]
        line = shapely.geometry.LineString([mid_a,[mid_a[0]+d[0]*20,mid_a[1]+d[1]*20]])
        ip_first = get_polygon_line_intersection(poly,line)
        num_extra=num_extra+1
    except:
        print "can not extend the polygon beyond the first point"
    print "ip_first",ip_first

    #extend to get to the end of the polygon
    ip_last=-1
    try:
        line = shapely.geometry.LineString([skel[-1],[skel[-1][0]-d[0]*20,skel[-1][1]-d[1]*20]])
        ip_last = get_polygon_line_intersection(poly,line)
        l1 = shapely.geometry.LineString([ip_last,[ip_last[0]+dp[0]*4,ip_last[1]+dp[1]*4]])
        a1=get_polygon_line_intersection(poly,l1)
        l2 =  shapely.geometry.LineString([ip_last,[ip_last[0]-dp[0]*4,ip_last[1]-dp[1]*4]])
        a2=get_polygon_line_intersection(poly,l2)
        mid_a = [a1[0]+(a2[0]-a1[0])/2,a1[1]+(a2[1]-a1[1])/2]
        line = shapely.geometry.LineString([mid_a,[mid_a[0]-d[0]*20,mid_a[1]-d[1]*20]])
        ip_last = get_polygon_line_intersection(poly,line)
        print "ip_last",ip_last
        num_extra=num_extra+1
    except:
        print "can not extend the polygon beyond the last point"
    skel=np.zeros([len(edges_x3)+num_extra,2])
    if ip_first != -1:
        skel[0,:]=ip_first    
    for i in range(len(edges_x3)):
        skel[i+1,:]=edges_x3[i]
    if ip_last != -1:
        skel[-1,:]=ip_last
    print "before"
    skel_line = shapely.geometry.LineString(skel)
    print "after"
    L=SKELETON_RESOLUTION
    #interpolate, sample L points
    interpolate_line = np.zeros([L+1,2])
    linel=skel_line.length
    for i in range(L+1):
        pp=skel_line.interpolate(linel/L*i)
        interpolate_line[i,:]=[pp.coords[0][0],pp.coords[0][1]]
    print "after inter"
    return interpolate_line


def get_cells(image):
    '''
    Get cellls from the polygon.
    '''
    # apply threshold    
    thresh = threshold_otsu(image)
    binary = image > thresh
    bw=binary

    # Remove connected to image border
    cleared = bw.copy()
    clear_border(cleared)

    # label image regions
    label_image = skimage.measure.label(cleared)
    #find_contours
    borders = np.logical_xor(bw, cleared)
    label_image[borders] = -1
    image_label_overlay = label2rgb(label_image, image=image)

    #extract the regions and get a polygon per region
    polygons=[]
    for i,region in enumerate(regionprops(label_image)):
        # skip small images
        if region.area < 100:
            continue
        a=np.zeros([len(region.coords),2])
        #a=np.zeros(
        for i in range(len(region.coords)):
            a[i,:]=[region.coords[i][0],region.coords[i][1]]
        
        polygons.append(a)
    return polygons







'''
#peform dilation
        di = np.zeros(bw.shape)
        for c in region.coords:
            di[c[0]][c[1]]=1
        dii=scipy.ndimage.binary_dilation(di)
        print type(dii)
        aa=np.zeros([np.sum(dii),2])
        counter=0
        for i in range(dii.shape[0]):
            for j in range(dii.shape[1]):
                if dii[i][j]==1:
                    aa[counter,:]=[i,j]
                    counter=counter+1
'''


def get_skeletons(image):
    # Compute the medial axis (skeleton) and the distance transform
    skel, distance = medial_axis(data, return_distance=True)
    dist_on_skel = distance * skel
    return dist_on_skel
    

if __name__=="__main__":
    #### load a tif image of a cell
    filename = os.path.join('/Users/kerenlasker/projects/opto/microscopy/scripts/data/few/', 'phase_inverted.tif')
    #filename = os.path.join('/Users/kerenlasker/projects/opto/microscopy/scripts/data', 'cell.tif')
    data = io.imread(filename)
    
    #label_image_regions(data)
    polygons = get_cells(data)
    print "number of polygons",len(polygons)
    print polygons[0]
    aax=[]
    aay=[]
    for i,p1 in enumerate(polygons[0]):
        num=0
        for j,p2 in enumerate(polygons[0]):
            if i==j:
                continue
            if math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))<1.001:
                num=num+1
        print p1[0],p1[1],num
        if num<4:
            aax.append(polygons[0][i,0])
            aay.append(polygons[0][i,1])
    outline=np.zeros([len(aax),2])
    outline[:,0]=aax
    outline[:,1]=aay
    skel =  get_skeleton_from_outline(outline)
    print skel


    #[pintx,pinty,q]=skeleton_to_mesh.skeleton_to_mesh(skel,points,prevpoint,lng,stp)


    
    #skeleton_to_mesh.skeleton_to_mesh(sk,outline,prevpoint,lng,stp)
            
    # fig, ax = plt.subplots()
    # ax.plot(polygons[0][:,0],polygons[0][:,1] , 'r.')
    # ax.plot([212,190],[30,50],'y.')
    # ax.plot(aax, aay, 'g.')
    # plt.show()
    # sys.exit()
    #dist_on_skel = get_skeletons(data)
    #print np.where(dist_on_skel>0)
    #coords = get_skeleton_coords(dist_on_skel)
    #sort_coords(coords)
    #mat = skeleton_coords_to_distance_matrix(coords)
    #order=get_ordered_skeleton(mat,coords)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    # ax1.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
    # ax1.scatter(polygons[0][:,0],polygons[0][:,1])
    # ax1.axis('off')
    # ax2.imshow(dist_on_skel, cmap=plt.cm.spectral, interpolation='nearest')
    # ax2.scatter(coords[0], coords[1], color='white')
    # ax2.axis('off')
    # fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
    # plt.show()
    # sys.exit()


