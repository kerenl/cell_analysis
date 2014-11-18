
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import numpy as np
import operator,math
from scipy import interpolate,signal

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

import skeleton_to_mesh





# Data manipulation:

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments


# Interface to LineCollection:


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


def point_inside_polygon(x,y,poly):
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




def adjust_bounds(ax, points):
    ptp_bound = points.ptp(axis=0)
    ax.set_xlim(points[:,0].min() - 0.1*ptp_bound[0],
                points[:,0].max() + 0.1*ptp_bound[0])
    ax.set_ylim(points[:,1].min() - 0.1*ptp_bound[1],
                points[:,1].max() + 0.1*ptp_bound[1])




ff=open("/Users/kerenlasker/projects/opto/microscopy/scripts/data/cell_coord.txt")
lines = ff.readlines()
ff.close()
points_x=[]
points_y=[]
for line in lines:
    s=line.split()
    points_x.append(float(s[0]))
    points_y.append(float(s[1])) 
points = np.zeros([len(points_x),2])
n_points=len(points_x)
for i in range(len(points_x)):
        points[i,0]=points_x[i]
        points[i,1]=points_y[i]


edges_x=[]
edges_y=[]


#points = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])
vor = Voronoi(points)
#Plot it:
fig, ax = plt.subplots()

if vor.points.shape[1] != 2:
    raise ValueError("Voronoi diagram is not 2-D")

#ax.plot(vor.points[:,0], vor.points[:,1], '.')
#ax.plot(vor.vertices[:,0], vor.vertices[:,1], 'o')


for simplex in vor.ridge_vertices:
    simplex = np.asarray(simplex)
    if np.all(simplex >= 0):
        #ax.plot(vor.vertices[simplex,0], vor.vertices[simplex,1], 'k-')
        edges_x.append(vor.vertices[simplex,0])
        edges_y.append(vor.vertices[simplex,1])

ptp_bound = vor.points.ptp(axis=0)

center = vor.points.mean(axis=0)
colorl=['g--','b--','r--','b--','k--','y--','g--','b--','r--','b--','k--','y--','g--','b--','r--','b--','k--','y--']
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
        print [vor.vertices[i,0], far_point[0]],[vor.vertices[i,1], far_point[1]]
        #ax.plot([vor.vertices[i,0], far_point[0]],
        #        [vor.vertices[i,1], far_point[1]], colorl[0])
        edges_x.append([vor.vertices[i,0], far_point[0]])
        edges_y.append([vor.vertices[i,1], far_point[1]])

edges_x1=[]
edges_y1=[]
for i in range(len(edges_x)):
    print edges_x[i], edges_y[i]
    if point_inside_polygon(edges_x[i][0],edges_y[i][0],points) and point_inside_polygon(edges_x[i][1],edges_y[i][1],points):
        #ax.plot(edges_x[i],edges_y[i],colorl[0])
        edges_x1.append(edges_x[i])
        edges_y1.append(edges_y[i])
print type(edges_x1)
edges_x1=np.array(edges_x1)
print type(edges_x1)
edges_y1=np.array(edges_y1)


# #remove vertices outside
# q = logical(inpolygon(vx(1,:),vy(1,:),coord(:,1),coord(:,2))...
#             .* inpolygon(vx(2,:),vy(2,:),coord(:,1),coord(:,2)));
# vx = reshape(vx([q;q]),2,[]);
# vy = reshape(vy([q;q]),2,[]);
    
# #remove isolated points
# if isempty(vx), res=0;return; end
# t = ~((abs(vx(1,:)-vx(2,:))<delta)&(abs(vy(1,:)-vy(2,:))<delta));
# vx = reshape(vx([t;t]),2,[]);
# vy = reshape(vy([t;t]),2,[]);

#make a graph and mark branch points
# Import graphviz
import sys
sys.path.append('..')
sys.path.append('/usr/lib/graphviz/python/')
sys.path.append('/usr/lib64/graphviz/python/')
import gv

# Import pygraph
import pygraph
from pygraph.classes.graph import graph
from pygraph.classes.digraph import digraph
from pygraph.algorithms.searching import breadth_first_search
from pygraph.readwrite.dot import write

# Graph creation
gr = graph()


nodes1={}
for i in range(len(edges_x1)):
    key1=str(edges_x1[i][0])+"_"+str(edges_y1[i][0])
    key2=str(edges_x1[i][1])+"_"+str(edges_y1[i][1])
    if key1 not in nodes1:
        nodes1[key1]=[edges_x1[i][0],edges_y1[i][0]]
    if key2 not in nodes1:
        nodes1[key2]=[edges_x1[i][1],edges_y1[i][1]]
for key in nodes1.keys():
    gr.add_nodes([key])
        
for i in range(len(edges_x1)):
    key1=str(edges_x1[i][0])+"_"+str(edges_y1[i][0])
    key2=str(edges_x1[i][1])+"_"+str(edges_y1[i][1])
    gr.add_edge((key1,key2))


#detect branch
for node in gr.nodes():
    if gr.node_order(node) > 2:
        print "BIG NODE",node
        st,pre,post= pygraph.algorithms.searching.depth_first_search(gr,root=node)
        nd={}
        print gr.neighbors(node)
        for neighbor in gr.neighbors(node):
            nd[neighbor]=post.index(neighbor)
        print nd
        sorted_nd=sorted(nd.items(),key=operator.itemgetter(1))
        path_size={}
        last_val=0
        for k,v in sorted_nd:
            path_size[k]=v-last_val
            last_val=v+1
        print path_size
#sys.exit()        
# Draw as PNG
dot = write(gr)
gvv = gv.readstring(dot)
gv.layout(gvv,'dot')
gv.render(gvv,'png','europe.png')


delta=0.0001
edges_x2=[];
edges_y2=[];
while True:
    print edges_x1
    for i in range(len(edges_x1)):
        p1=np.sum((abs(edges_x1-edges_x1[i][0])<delta) & (abs(edges_y1-edges_y1[i][0])<delta))>1
        print "========================="
        p2=np.sum((abs(edges_x1-edges_x1[i][1])<delta) & (abs(edges_y1-edges_y1[i][1])<delta))>1
        print p1,p2
        if p1 and p2:
            edges_x2.append([edges_x1[i][0],edges_x1[i][1]])
            edges_y2.append([edges_y1[i][0],edges_y1[i][1]])
    print edges_x2
    print edges_y2
    if (len(edges_x1)-len(edges_x2))<=2:
        edges_x3=np.array(edges_x2)
        edges_y3=np.array(edges_y2)
        break
    else:
        edges_x1 = np.array(edges_x2)
        edges_y1 = np.array(edges_y2)
        edges_x2=[];
        edges_y2=[];
    # vx3 = vx;
    # vy3 = vy;
edges_x3=np.array(edges_x1)
edges_y3=np.array(edges_y1)


#TODO - missing removing cycles
edges_x=edges_x3
edges_y=edges_y3

#sort the points
vx2=[]
vy2=[]
for i in range(len(edges_x2)): #find the first point
    if np.sum(np.sum((abs(edges_x-edges_x[i][0])<delta) & (abs(edges_y-edges_y[i][0])<delta)))==1:
        vx2=edges_x[i]
        vy2=edges_y[i]
        break
    elif np.sum(np.sum((abs(edges_x-edges_x[i][1])<delta) & (abs(edges_y-edges_y[i][1])<delta)))==1:
        vx2=np.fliplr(edges_x[i])
        vy2=np.fliplr(edges_x[i])
        break
print "The first point is",vx2,vy2
k=1

while True: # in this cycle sort all points after the first one
    print k,vx2
    print vx2[k]
    print edges_x[:,0]
    f1=np.where((abs(edges_x[:,0]-vx2[k])<delta) & (abs(edges_y[:,0]-vy2[k])<delta) & ((abs(edges_x[:,1]-vx2[k-1])>=delta) | (abs(edges_y[:,1]-vy2[k-1])>=delta)))
    f2=np.where((abs(edges_x[:,1]-vx2[k])<delta) & (abs(edges_y[:,1]-vy2[k])<delta) & ((abs(edges_x[:,0]-vx2[k-1])>=delta) | (abs(edges_y[:,0]-vy2[k-1])>=delta)))
    print f1,f2
    print len(f1[0]),len(f2[0])
    if len(f1[0])>0:
        print "here f1"
        vx2=np.append(vx2,edges_x[f1[0],1])
        vy2=np.append(vy2,edges_y[f1[0],1])
    elif len(f2[0])>0:
        print "heref2"
        vx2=np.append(vx2,edges_x[f2[0],0])
        vy2=np.append(vy2,edges_y[f2[0],0])
    else:
        break
    k=k+1

print vx2
#colorline(vx2,vy2)
    #sys.exit()

'''    
skel=[vx2' vy2'];
    if size(vx2,2)<=1
        res = 0;
        return;
    end
'''
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
print "SKEL"
print skel
print "DIFF"
print skel_diff
l=np.cumsum(np.append(0,np.sum(skel_diff*skel_diff,1)))

# l - the X points
#vx2 - the F(X) values
#0:stp:l(end) - the query points
skelx_interp = np.interp(np.linspace(0,l[-1],l[-1]/stp),l,vx2)
skely_interp = np.interp(np.linspace(0,l[-1],l[-1]/stp),l,vy2)
# fx = interpolate.interp1d(l,vx2)
# fy = interpolate.interp1d(l,vy2)
# skelx_interp=[]
# skely_interp=[]
# xvals = np.linspace(0,l[-1],l[-1]/stp)
# for i in range(len(xvals)):
#     skelx_interp.append(fx(xvals[i]))
#     skely_interp.append(fy(xvals[i]))

print "============|"
print skelx_interp
print "============|"
print skely_interp
skel_interp=np.zeros([len(skelx_interp),2])
for i in range(len(skelx_interp)):
    skel_interp[i,0]=skelx_interp[i]
    skel_interp[i,1]=skely_interp[i]

#colorline(vx2,vy2)
#colorline(skelx_interp,skely_interp)

lng0 = l[-1]
lng=15
sz = lng0/stp
print points
coord = np.append(points,points[0])
coord=coord.reshape(len(points)+1,2)
#extend the skeleton
print skel_interp[0]*(lng/stp+1) - skel_interp[1]*lng/stp
#print np.append(skel_interp[0]*(lng/stp+1) - skel_interp[1]*lng/stp,skel_interp)
skel2 = np.append(np.append(skel_interp[0]*(lng/stp+1) - skel_interp[1]*lng/stp,skel_interp),skel_interp[-1]*(lng/stp+1) - skel_interp[-2]*lng/stp)
skel2=skel2.reshape(2+len(skel_interp),2)
#print skel2

skel2_diff=np.zeros([len(skel2[:,0])-1,2])
for i in range(1,len(skel2[:,0])):
    skel2_diff[i-1,0]=skel2[i,0]-skel2[i-1,0]
    skel2_diff[i-1,1]=skel2[i,1]-skel2[i-1,1]
l=np.cumsum(np.append(0,np.sum(skel2_diff*skel2_diff,1)))
l,i = np.unique(l,return_index=True);
skel2 = skel2[i,:];


print type(skel2)
print coord
L=80
[tmp1,tmp2,indS,indC]=skeleton_to_mesh.intxyMulti(skel2[2:,0],skel2[2:,1],coord[:,0],coord[:,1])
print skeleton_to_mesh.modL(indC,1)
aa=np.array([min(skeleton_to_mesh.modL(indC,1)),min(skeleton_to_mesh.modL(indC,L/2+1))])
tmp1=aa.argmin()
prevpoint=aa[tmp1]
if prevpoint==1:
    prevpoint=L/2
print skel2
lng=15
[pintx,pinty,q]=skeleton_to_mesh.skeleton_to_mesh(skel,points,prevpoint,lng,stp)

#skel3=signal.savgol_filter(skel2,5,2)
mesh_points_x=[]
mesh_points_y=[]
for i in range(len(pintx)):
    mesh_points_x.append([pintx[i,0],pinty[i,0]])
    mesh_points_y.append([pintx[i,1],pinty[i,1]])
ax.plot(mesh_points_x,mesh_points_y)
    

adjust_bounds(ax, points)

plt.show()
sys.exit()


#_adjust_bounds(ax, vor.points)

    # return ax.figure


'''
    9-8-7-6-5-1-2-3
              |
              -1





1 2 3 -1 5 6 7 8 9
3 2 -1 9 8 7 6 5 1
'''
