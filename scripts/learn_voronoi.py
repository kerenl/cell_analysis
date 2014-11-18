import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt


points=np.zeros([13,2])
points[0,:]=[0,0]
points[1,:]=[1,0]
points[2,:]=[2,0]
points[3,:]=[3,0]
points[4,:]=[4,0]
points[5,:]=[4,1]
points[6,:]=[4,2]
points[7,:]=[3,4]
points[8,:]=[2,4]
points[9,:]=[1,4]
points[10,:]=[0,4]
points[11,:]=[0,1]
points[12,:]=[0,0]

vor = Voronoi(points)
#voronoi_plot_2d(vor)
plt.scatter(points[:,0],points[:,1],c='g')
plt.scatter(vor.vertices[:,0],vor.vertices[:,1],c='r')
for vpair in vor.ridge_vertices:
    if -1 in vpair:
        continue
    print vor.vertices[vpair[0]],vor.vertices[vpair[1]]
    v0=vor.vertices[vpair[0]]
    v1=vor.vertices[vpair[1]]   
    plt.plot([v0[0],v1[0]],[v0[1],v1[1]])
#plt.show()



#learn shapely

import shapely.geometry

points2=[]
for p in points:
    points2.append([p[0],p[1]])
print points2
poly = shapely.geometry.Polygon(points)#[(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 0.0), (1.0, 2.0), (2.0, 2.0), (3.0, 2.0), (4.0, 2.0)])
#points2)
print poly
print "is valid",poly.is_valid
l1 = shapely.geometry.LineString(poly.exterior.coords)
print "is simple",l1.is_simple
p1=np.array([1,1])
p2=np.array([-1,1])
d1=p1-p2
dp=np.array([-d1[1],d1[0]])
print [p1, p1+dp*100]
line = shapely.geometry.LineString([p1, p1+dp*100])
print "before intersection"
xx = poly.intersection(line)
print "line",line
print "xx",dir(xx)
for c in xx.coords:
    print "coord",c
print "intersection",line.intersects(poly)
sys.exit()




# plot
#voronoi_plot_2d(vor)
#http://stackoverflow.com/questions/20515554/colorize-voronoi-diagram
# # colorize
# print vor.regions
# for region in vor.regions:
#     if not -1 in region:
#         polygon = [vor.vertices[i] for i in region]
#         print polygon
#         plt.fill(*zip(*polygon))

# plt.show()
