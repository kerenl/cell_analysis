import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from skimage import io
import os,math
from skimage.filter import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import label, closing, square
from skimage.color import label2rgb
from skimage.measure import regionprops
import skimage
from skimage.morphology import medial_axis
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from skimage import draw


def remove_branches(edges_x1,edges_y1):
    delta=0.0001
    edges_x2=[];
    edges_y2=[];
    while True:
        for i in range(len(edges_x1)):
            p1=np.sum((abs(edges_x1-edges_x1[i][0])<delta) & (abs(edges_y1-edges_y1[i][0])<delta))>1
            p2=np.sum((abs(edges_x1-edges_x1[i][1])<delta) & (abs(edges_y1-edges_y1[i][1])<delta))>1
            if p1 and p2:
                #print i,"adding"
                edges_x2.append([edges_x1[i][0],edges_x1[i][1]])
                edges_y2.append([edges_y1[i][0],edges_y1[i][1]])
                #else:
                #print i,"not adding"
        if (len(edges_x1)-len(edges_x2))<=2:
            #print "here"
            edges_x3=np.array(edges_x2)
            edges_y3=np.array(edges_y2)
            break
        else:
            #print "here1"
            edges_x1 = np.array(edges_x2)
            edges_y1 = np.array(edges_y2)
            edges_x2=[];
            edges_y2=[];
    print "edges X3"
    return [edges_x1,edges_y1]

# def intxy2(ax,ay,bx,by):
#     # modified intxy, finds the first point along line a where an intersection 
#     # rewritten to accept unsorted sets, 2xN of N segments
#     # finds all intersections, only gives the row numbers in the first set
#     if len(ax)==0:
#         print "ax is of size 0"
#         return []
#     res = np.zeros([len(ax),1])
#     for i in range(len(ax)):
#         #vector to next vertex in a
#         u=[ax[i][1]-ax[i][0], ay[i][1]-ay[i][0]]
#         #go through each vertex in b
#         for j in range(len(bx)):
#             #check for intersections at the vertices
#             #if (ax[0,i]==bx[0,j] and ay[0,i]==by[0,j]) or (ax[1,i]==bx[0,j] and ay[1,i]==by[0,j]) or (ax[0,i]==bx[1,j] and ay[0,i]==by[1,j]) or (ax[1,i]==bx[1,j] and ay[1,i]==by[1,j]):
#             print ax[i],bx[j]
#             if (ax[i][0]==bx[j] and ay[i][0]==by[j]) or (ax[i][1]==bx[j] and ay[i][1]==by[j]) or (ax[i][0]==bx[j] and ay[i][0]==by[j]) or (ax[i][1]==bx[j] and ay[i][1]==by[j]):            
#                 res[i] = 1;
#                 continue
#             #vector from ith vertex in a to jth vertex in b
#             v=[bx[j][1]-ax[i][0], by[j][1]-ay[i][0]];
#             #vector from ith+1 vertex in a to jth vertex in b
#             w=[bx[j][0]-ax[i][1], by[j][0]-ay[i][1]];
#             print v,w
#             sys.exit()
#             '''
#             %vector from ith vertex of a to jth-1 vertex of b
#             vv=[bx(1,j)-ax(1,i) by(1,j)-ay(1,i)];
#             %vector from jth-1 vertex of b to jth vertex of b
#             z=[bx(2,j)-bx(1,j) by(2,j)-by(1,j)];
#             %cross product of u and v
#             cpuv=u(1)*v(2)-u(2)*v(1);
#             %cross product of u and vv
#             cpuvv=u(1)*vv(2)-u(2)*vv(1);
#             %cross product of v and z
#             cpvz=v(1)*z(2)-v(2)*z(1);
#             %cross product of w and z
#             cpwz=w(1)*z(2)-w(2)*z(1);
#             % check if there is an intersection
#             if c
#             '''


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


def get_skeleton_from_outline(outline):
    fig, ax = plt.subplots()
    vor = Voronoi(outline)
    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi diagram is not 2-D")

    points=outline
    plt.scatter(points[:,0],points[:,1],c='g')
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

    #detect branch
    [edges_x3,edges_y3]=remove_branches(edges_x1,edges_y1)
    print "edgesx3"
    print edges_x3
    for i in range(len(edges_x3)):
        plt.plot([edges_x3[i][0],edges_y3[i][0]],[edges_x3[i][1],edges_y3[i][1]])
    plt.show()
    
    sys.exit()
    
    voronoi_plot_2d(vor)
    plt.show()
    sys.exit()
    edges_x=[]
    edges_y=[]
    for vpair in vor.ridge_vertices:
        if vpair[0] >= 0  and vpair[1] >= 0:
            v0=vor.vertices[vpair[0]]
            v1=vor.vertices[vpair[1]]
            edges_x.append([v0[0],v1[0]])
            edges_y.append([v1[0],v1[1]])
            #ax.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k', linewidth=2)
    #ax.scatter(outline[:,0],outline[:,1])
    #ax.scatter(edges_x,edges_y)
    #plt.show()
    #sys.exit()
    #print "number of edges",len(edges_x)
    #print "First edge:",edges_x[0],edges_y[0]
    #sys.exit()
    # remove vertices crossing the boundary
    print "len edges_x",len(edges_x)
        
    ptp_bound = vor.points.ptp(axis=0)
    center = vor.points.mean(axis=0)
    print "center",center
    sys.exit()
    #https://docs.scipy.org/doc/scipy-0.12.0/reference/tutorial/spatial.html
    ind=-1
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        ind=ind+1
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[i] + direction * ptp_bound.max()
            edges_x.append([vor.vertices[i,0], far_point[0]])
            edges_y.append([vor.vertices[i,1], far_point[1]])
            
    edges_x1=[]
    edges_y1=[]
    for i in range(len(edges_x)):
        if point_inside_polygon(edges_x[i][0],edges_y[i][0],outline) and point_inside_polygon(edges_x[i][1],edges_y[i][1],outline):
            edges_x1.append(edges_x[i])
            edges_y1.append(edges_y[i])
    edges_x1=np.array(edges_x1)
    edges_y1=np.array(edges_y1)

    #detect branch
    delta=0.0001
    edges_x2=[];
    edges_y2=[];
    while True:
        for i in range(len(edges_x1)):
            p1=np.sum((abs(edges_x1-edges_x1[i][0])<delta) & (abs(edges_y1-edges_y1[i][0])<delta))>1
            p2=np.sum((abs(edges_x1-edges_x1[i][1])<delta) & (abs(edges_y1-edges_y1[i][1])<delta))>1
            if p1 and p2:
                edges_x2.append([edges_x1[i][0],edges_x1[i][1]])
                edges_y2.append([edges_y1[i][0],edges_y1[i][1]])
        if (len(edges_x1)-len(edges_x2))<=2:
            edges_x3=np.array(edges_x2)
            edges_y3=np.array(edges_y2)
            break
        else:
            edges_x1 = np.array(edges_x2)
            edges_y1 = np.array(edges_y2)
            edges_x2=[];
            edges_y2=[];
    edges_x3=np.array(edges_x1)
    edges_y3=np.array(edges_y1)


    #TODO - missing removing cycles
    edges_x=edges_x3
    edges_y=edges_y3

    #sort the points
    vx2=[]
    vy2=[]


    colorline(edges_x,edges_y)
    adjust_bounds(ax, outline)
    plt.show()

    for i in range(len(edges_x2)): #find the first point
        if np.sum(np.sum((abs(edges_x-edges_x[i][0])<delta) & (abs(edges_y-edges_y[i][0])<delta)))==1:
            vx2=edges_x[i]
            vy2=edges_y[i]
            break
        elif np.sum(np.sum((abs(edges_x-edges_x[i][1])<delta) & (abs(edges_y-edges_y[i][1])<delta)))==1:
            vx2=np.fliplr(edges_x[i])
            vy2=np.fliplr(edges_x[i])
            break
    k=1


    
    '''
    FIND THE ERROR HERE:File "utils.py", line 119, in get_skeleton_from_outline
    f1=np.where((abs(edges_x[:,0]-vx2[k])<delta) & (abs(edges_y[:,0]-vy2[k])<delta) & ((abs(edges_x[:,1]-vx2[k-1])>=delta) | (abs(edges_y[:,1]-vy2[k-1])>    =delta)))
    IndexError: too many indices for array
    compare to test.py

    Things left:
    I know how to segment
    get the polygon
    find the outline
    find the skeleton
    find the mesh
    now i need to find how to get a slice of the mesh and sum all pixels there
    '''



    
    
   
    while True: # in this cycle sort all points after the first one
        f1=np.where((abs(edges_x[:,0]-vx2[k])<delta) & (abs(edges_y[:,0]-vy2[k])<delta) & ((abs(edges_x[:,1]-vx2[k-1])>=delta) | (abs(edges_y[:,1]-vy2[k-1])>=delta)))
        f2=np.where((abs(edges_x[:,1]-vx2[k])<delta) & (abs(edges_y[:,1]-vy2[k])<delta) & ((abs(edges_x[:,0]-vx2[k-1])>=delta) | (abs(edges_y[:,0]-vy2[k-1])>=delta)))
        if len(f1[0])>0:
            vx2=np.append(vx2,edges_x[f1[0],1])
            vy2=np.append(vy2,edges_y[f1[0],1])
        elif len(f2[0])>0:
            vx2=np.append(vx2,edges_x[f2[0],0])
            vy2=np.append(vy2,edges_y[f2[0],0])
        else:
            break
        k=k+1

    stp=1
    skel=np.zeros([len(vx2),2])
    skel_diff=np.zeros([len(vx2)-1,2])
    skel[0,0]=vx2[0]
    skel[0,1]=vy2[0]
    for i in range(1,len(vx2)):
        skel[i,0]=vx2[i]
        skel[i,1]=vy2[i]
        skel_diff[i-1,0]=vx2[i]-vx2[i-1]
        skel_diff[i-1,1]=vy2[i]-vy2[i-1]
    #interpolate skeleton to equal step, extend outside of the cell and smooth
    l=np.cumsum(np.append(0,np.sum(skel_diff*skel_diff,1)))

    skelx_interp = np.interp(np.linspace(0,l[-1],l[-1]/stp),l,vx2)
    skely_interp = np.interp(np.linspace(0,l[-1],l[-1]/stp),l,vy2)

    skel_interp=np.zeros([len(skelx_interp),2])
    
    for i in range(len(skelx_interp)):
        skel_interp[i,0]=skelx_interp[i]
        skel_interp[i,1]=skely_interp[i]

    colorline(vx2,vy2)
    colorline(skelx_interp,skely_interp)
    adjust_bounds(ax, points)
    plt.show()
   
    lng0 = l[-1]
    lng=15
    sz = lng0/stp
    coord = np.append(points,points[0])
    coord=coord.reshape(len(points)+1,2)
    skel2 = np.append(np.append(skel_interp[0]*(lng/stp+1) - skel_interp[1]*lng/stp,skel_interp),skel_interp[-1]*(lng/stp+1) - skel_interp[-2]*lng/stp)
    skel2=skel2.reshape(2+len(skel_interp),2)

    skel2_diff=np.zeros([len(skel2[:,0])-1,2])
    for i in range(1,len(skel2[:,0])):
        skel2_diff[i-1,0]=skel2[i,0]-skel2[i-1,0]
        skel2_diff[i-1,1]=skel2[i,1]-skel2[i-1,1]
    l=np.cumsum(np.append(0,np.sum(skel2_diff*skel2_diff,1)))
    l,i = np.unique(l,return_index=True);
    skel2 = skel2[i,:];

    L=80
    [tmp1,tmp2,indS,indC]=skeleton_to_mesh.intxyMulti(skel2[2:,0],skel2[2:,1],coord[:,0],coord[:,1])
    aa=np.array([min(skeleton_to_mesh.modL(indC,1)),min(skeleton_to_mesh.modL(indC,L/2+1))])
    tmp1=aa.argmin()
    prevpoint=aa[tmp1]
    if prevpoint==1:
        prevpoint=L/2
    lng=15
    return skel
'''
    #skel3=signal.savgol_filter(skel2,5,2)
    #mesh_points_x=[]
    #mesh_points_y=[]
    #for i in range(len(pintx)):
    #mesh_points_x.append([pintx[i,0],pinty[i,0]])
    #mesh_points_y.append([pintx[i,1],pinty[i,1]])
    #ax.plot(mesh_points_x,mesh_points_y)
    
    #adjust_bounds(ax, points)

'''



def get_cells(image):
    '''
    Get cellls from the polygon.
    '''
    new_image=np.ones([3,image.shape[0],image.shape[1]],dtype=float)
    # apply threshold
    thresh = threshold_otsu(image)
    binary = image > thresh
    bw=image


    # Remove connected to image border
    cleared = bw.copy()
    clear_border(cleared)


    #cleared = bw
    
    # label image regions
    label_image = label(cleared)
    #skimage.measure.label
    #find_contours
    borders = np.logical_xor(bw, cleared)
    label_image[borders] = -1
    image_label_overlay = label2rgb(label_image, image=image)

    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 2))
    # print "====="
    # ax.imshow(label_image, cmap=plt.cm.gray)
    # plt.show()


    #extract the regions and get a polygon per region
    polygons=[]
    for i,region in enumerate(regionprops(label_image)):
        # skip small images
        if region.area < 100:
            continue
        #polygons.append(matplotlib.path.Path(region.coords))
        a=np.zeros(region.coords.shape)
        a[:,0]=region.coords[:,1]
        a[:,1]=region.coords[:,0]
        polygons.append(a)   
    return polygons

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


