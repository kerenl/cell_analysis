import matplotlib.patches
import matplotlib.pyplot as plt
import shapely.geometry
from skimage import io
import os,sys,random,math
import numpy as np


#load figure and see the coordinates
#### load a tif image of a cell
filename = os.path.join('/Users/kerenlasker/projects/opto/microscopy/scripts/data', 'cell.tif')
data = io.imread(filename)
print dir(data)
print "size:",data.size
print "shape:",data.shape
#define an empty figure the size of data and set 1 based on the polygon shape
data_mask=np.zeros(data.shape)
g1 = shapely.geometry.Polygon([[1, 1], [1, 20], [10, 20],[10,1]])
print g1.bounds
for i in range(int(math.floor(g1.bounds[0])),int(math.ceil(g1.bounds[2]))+1):
    for j in range(int(math.floor(g1.bounds[1])),int(math.ceil(g1.bounds[3]))+1):
        data_mask[i][j]=1
print data[:,0]
print data*data_mask
plt.imshow(data*data_mask)
plt.show()
        
print dir(g1)
print "is simple",g1.is_simple,g1.is_valid
print list(g1.exterior.coords)
for coord in list(g1.coords):
    print coord
    data_mask[coord]=1
plt.imshow(data*data_mask)
plt.show()


sys.exit()



fig = plt.figure()
ax = fig.add_subplot(111)
g1 = shapely.geometry.Polygon([[2, 1], [8, 1], [8, 4]])
g2 = shapely.geometry.Polygon([[4, 0.5], [6, 0.5], [6, 7],[4,7]])
g12=g1.intersection(g2)
rect1 = matplotlib.patches.Polygon(list(g1.exterior.coords),color='y')
rect2 = matplotlib.patches.Polygon(list(g2.exterior.coords),color='r')
rect3 = matplotlib.patches.Polygon(list(g12.exterior.coords),color='g')
# rect2 = matplotlib.patches.Rectangle((0,150), 300, 20, color='red')
# rect3 = matplotlib.patches.Rectangle((-300,-50), 40, 200, color='#0099FF')
# circle1 = matplotlib.patches.Circle((-200,-250), radius=90, color='#EB70AA')
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
# ax.add_patch(rect3)
# ax.add_patch(circle1)
plt.xlim([-4, 10])
plt.ylim([-4, 10])
plt.show()


