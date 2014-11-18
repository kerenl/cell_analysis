import numpy as np
from scipy import ndimage
from skimage.morphology import medial_axis
import matplotlib.pyplot as plt
from skimage import io
import os,sys,random,math
from skimage.color import rgb2gray
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.filter import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import label, closing, square
from skimage.measure import regionprops
from skimage.color import label2rgb
import matplotlib.patches as mpatches
import matplotlib.mlab as mlab
from sklearn import preprocessing
import matplotlib.path
#http://www.tp.umu.se/~nylen/pylect/advanced/scikit-learn/index.html#id9
#http://scikit-learn.org/stable/auto_examples/cluster/plot_segmentation_toy.html
#http://scikit-image.org/docs/dev/auto_examples/plot_blob.html
#http://peekaboo-vision.blogspot.com/2012/02/simplistic-minimum-spanning-tree-in.html

def extend_skeleton(ordered_skeleton,polygon):
    vv=polygon#.vertices
    coords=ordered_skeleton
    #extent the line between the first two points in both direction and see when you get outside of the polygon
    #get line from two points
    fig, ax = plt.subplots()
    ax.scatter(vv[:,0], vv[:,1],c='green')
    ax.set_aspect('equal')
    a0=coords[:,0]
    a1=coords[:,1]
    m=(a1[1]-a0[1])/(a1[0]-a0[0])
    b=a1[1]-m*a1[0]
    #annotate(ax, '', coords[:,0],  coords[:,1])
    annotate(ax, '', coords[:,0],  coords[:,1],'red')
    annotate(ax, '', coords[:,0],  [coords[:,0][0],b+m],'orange')
    annotate(ax, '', coords[:,0],  [coords[:,0][0],b+2*m],'blue')
    annotate(ax, '', coords[:,0],  [coords[:,0][0],b+3*m],'green')

    # print coords[:,0],  coords[:,1],coords[:,0]-coords[:,1]
    # annotate(ax, '', coords[:,coords.shape[1]-2],  coords[:,coords.shape[1]-3])
    # #print  coords[:,coords.shape[1]-2],  coords[:,len(coords)-1], coords[:,len(coords)-2]-coords[:,len(coords)-1]
    plt.show()

def get_cells(image):
    '''
    Get cellls from the polygon.
    '''
    new_image=np.ones([3,image.shape[0],image.shape[1]],dtype=float)
    # apply threshold
    thresh = threshold_otsu(image)
    bw=image

    # remove artifacts connected to image border
    cleared = bw.copy()
    clear_border(cleared)

    # label image regions
    label_image = label(cleared)
    #skimage.measure.label
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
        #polygons.append(matplotlib.path.Path(region.coords))
        print (region.coords.shape)
        a=np.zeros(region.coords.shape)
        a[:,0]=region.coords[:,1]
        a[:,1]=region.coords[:,0]
        polygons.append(a)   
    return polygons

def pca_analysis(region):
    datar=region.coords.astype(np.float)
    data = preprocessing.scale(datar) #Scaled data has zero mean and unit variance:
    print data.shape
    mu = [0,0]
    print mu
    eigenvectors, eigenvalues, V  = np.linalg.svd(data.T, full_matrices=False)
    projected_data = np.dot(data, eigenvectors)
    sigma = projected_data.std(axis=0).mean()
    print sigma
    fig, ax = plt.subplots()
    ax.scatter(data[:,0], data[:,1])
    ax.set_aspect('equal')
    for axis in eigenvectors:
        print "====",mu,mu + sigma * axis
        annotate(ax, '', mu, mu + sigma * axis)
    plt.show()
    

def annotate(ax, name, start, end,color='red'):
    print start,end
    arrow = ax.annotate(name,
                        xy=end, xycoords='data',
                        xytext=start, textcoords='data',
                        arrowprops=dict(facecolor=color, width=2.0))
    return arrow


def get_ordered_skeleton(dist_matrix,coords,copy_matrix=True):
#def get_minimum_spanning_tree(dist_matrix,coords,copy_matrix=True):
    """dist_matrix are edge weights of fully connected graph"""
    X = dist_matrix
    if copy_matrix:
        X = X.copy()
 
    if X.shape[0] != X.shape[1]:
        raise ValueError("X needs to be square matrix of edge weights")
    n_vertices = X.shape[0]
    spanning_edges = []
     
    # initialize with node 0:                                                                                         
    visited_vertices = [0]                                                                                            
    num_visited = 1
    # exclude self connections:
    diag_indices = np.arange(n_vertices)
    X[diag_indices, diag_indices] = np.inf
     
    while num_visited != n_vertices:
        new_edge = np.argmin(X[visited_vertices], axis=None)
        # 2d encoding of new_edge from flat, get correct indices                                                      
        new_edge = divmod(new_edge, n_vertices)
        new_edge = [visited_vertices[new_edge[0]], new_edge[1]]                                                       
        # add edge to tree
        spanning_edges.append(new_edge)
        visited_vertices.append(new_edge[1])
        # remove all edges inside current tree
        X[visited_vertices, new_edge[1]] = np.inf
        X[new_edge[1], visited_vertices] = np.inf                                                                     
        num_visited += 1
    order =  np.vstack(spanning_edges)
    ordered_corrds_x=[]
    ordered_corrds_y=[]
    for ind in order:
        ordered_corrds_x.append(coords[0,ind[0]])
        ordered_corrds_y.append(coords[1,ind[0]])
    ordered_corrds=np.zeros([2,len(ordered_corrds_x)])
    ordered_corrds[0,:]=ordered_corrds_x
    ordered_corrds[1,:]=ordered_corrds_y    
    return ordered_corrds

def skeleton_coords_to_distance_matrix(coords):
    mat = np.zeros([coords.shape[1],coords.shape[1]])
    for i in range(coords.shape[1]):
        for j in range(coords.shape[1]):
            mat[i,j] = math.sqrt((coords[0,i]-coords[0,j])*(coords[0,i]-coords[0,j])+(coords[1,i]-coords[1,j])*(coords[1,i]-coords[1,j]))
    return mat

def get_skeleton_coords(dist_on_skel):
    non_zero_points = np.nonzero(dist_on_skel)
    coords = np.zeros([2,len(non_zero_points[0])])
    coords[0,:]=non_zero_points[1]
    coords[1,:]=non_zero_points[0]
    return coords #coords[:,0] is the first point

def sort_coords(coords):
    num_points = coords.shape[1]
    pp=[]
    for i in range(num_points):
        pp.append(coords[:,i])
    #compute centroid
    cent = [(sum(coords[0,:])/num_points),(sum(coords[1,:])/num_points)]
    # sort by polar angle
    pp.sort(key=lambda p: math.atan2(p[1]-cent[1],p[0]-cent[0]))
    print pp

def is_point_within_polygon(point, polygonVertexCoords):
     path = matplotlib.path.Path( polygonVertexCoords)
     return path.contains_point(point[0], point[1]) 

def get_random_color(pastel_factor = 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]
 
def color_distance(c1,c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])
 
def generate_new_color(existing_colors,pastel_factor = 0.5):
    max_distance = None
    best_color = None
    for i in range(0,100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color,c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color
    #return [int(255*best_color[0]),int(255*best_color[1]),int(255*best_color[2])]

def get_random_colors(num_colors):
    colors=[]
    for i in range(0,num_colors):
      colors.append(generate_new_color(colors,pastel_factor = 0.9))
    return colors

def label_image_regions(image):
    #test--
    #new_image=np.zeros([3,image.shape[0],image.shape[1]],dtype=float)
    new_image=np.ones([3,image.shape[0],image.shape[1]],dtype=float)
    # apply threshold
    thresh = threshold_otsu(image)
    #close small holes with binary closing (skip)
    #bw = closing(image > thresh, square(3))
    bw=image

    # remove artifacts connected to image border
    cleared = bw.copy()
    clear_border(cleared)

    # label image regions
    label_image = label(cleared)
    borders = np.logical_xor(bw, cleared)
    label_image[borders] = -1
    image_label_overlay = label2rgb(label_image, image=image)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6)) ##
    ax.imshow(image_label_overlay) ##
    
    rand_colors = get_random_colors(len(regionprops(label_image)))
    for i,region in enumerate(regionprops(label_image)):
        #pca_analysis(region)
        rand_color=rand_colors[i]
        # skip small images
        if region.area < 100:
            continue
        for coord in region.coords:
            new_image[:,coord[0],coord[1]]=rand_color

        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect) ##
        ax.scatter(region.coords[:,1],region.coords[:,0]) ##
    plt.show()
    sys.exit()
    fig, ax = plt.subplots()
    rolled_image = np.rollaxis(new_image, 0, 3) 
    im = ax.imshow(rolled_image)
    plt.show()

def detect_cells(image):
    image_gray = rgb2gray(image)

    blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

    blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)

    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    colors = ['yellow', 'lime', 'red']
    titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
          'Determinant of Hessian']
    sequence = zip(blobs_list, colors, titles)

    for blobs, color, title in sequence:
        fig, ax = plt.subplots(1, 1)
        ax.set_title(title)
        ax.imshow(image, interpolation='nearest')
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax.add_patch(c)

    plt.show()




def microstructure(l=256):
    """
    Synthetic binary data: binary microstructure with blobs.

    Parameters
    ----------

    l: int, optional
        linear size of the returned image

    """
    n = 5
    x, y = np.ogrid[0:l, 0:l]
    mask_outer = (x - l/2)**2 + (y - l/2)**2 < (l/2)**2
    mask = np.zeros((l, l))
    generator = np.random.RandomState(1)
    points = l * generator.rand(2, n**2)
    mask[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    mask = ndimage.gaussian_filter(mask, sigma=l/(4.*n))
    return mask > mask.mean()


def get_skeletons(image):
    # Compute the medial axis (skeleton) and the distance transform
    skel, distance = medial_axis(data, return_distance=True)
    dist_on_skel = distance * skel
    return dist_on_skel

#filename = os.path.join('/Users/kerenlasker/projects/opto/microscopy/scripts/data', 'pPOS70-1_w1Phase_mask.tif')

if __name__=="__main__":

    #### load a tif image of a cell
    filename = os.path.join('/Users/kerenlasker/projects/opto/microscopy/scripts/data', 'cell.tif')
    data = io.imread(filename)
    
    #label_image_regions(data)
    polygons = get_cells(data)
    dist_on_skel = get_skeletons(data)
    #print np.where(dist_on_skel>0)
    coords = get_skeleton_coords(dist_on_skel)
    sort_coords(coords)
    mat = skeleton_coords_to_distance_matrix(coords)
    order=get_ordered_skeleton(mat,coords)
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

    print "========= BEFORE EXTEND SEKELTON"
    extend_skeleton(order,polygons[0])
    print "========= AFTER EXTEND SEKELTON"

    sys.exit()
    #print np.nonzero(dist_on_skel)
    # Distance to the background for pixels of the skeleton

