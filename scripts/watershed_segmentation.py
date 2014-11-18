import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os
from skimage import io
import matplotlib.pyplot as plt
from sklearn import cluster
import time,sys

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
#http://scikit-learn.org/stable/auto_examples/cluster/plot_segmentation_toy.html

from skimage.morphology import watershed
from skimage.feature import peak_local_max

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)
plt.figure(figsize=(17, 9.5))
plt.subplots_adjust(left=.001, right=.999, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

# Generate an initial image with two overlapping circles
filename = os.path.join('/Users/kerenlasker/projects/opto/microscopy/scripts/data', 'pPOS70-1_w1Phase_mask.tif')
image = io.imread(filename)

# estimate bandwidth for mean shift
bandwidth = cluster.estimate_bandwidth(image, quantile=0.3)
print bandwidth
ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)

 # predict cluster memberships
t0 = time.time()
ms.fit(image)
t1 = time.time()
if hasattr(ms, 'labels_'):
    y_pred = ms.labels_.astype(np.int)
else:
    y_pred = ms.predict(image)
print y_pred
sys.exit()
# plot
plt.subplot(4, 1, 1)
plt.title('cluster', size=18)
plt.scatter(image[:, 0], image[:, 1], color=colors[y_pred].tolist(), s=10)

if hasattr(ms, 'cluster_centers_'):
            centers = ms.cluster_centers_
            center_colors = colors[:len(centers)]
            plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xticks(())
plt.yticks(())
plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
plt.show()
sys.exit()

# Convert the image into a graph with the value of the gradient on the
# edges.
graph = image.img_to_graph(img, mask=mask)

# Take a decreasing function of the gradient: we take it weakly
# dependent from the gradient the segmentation is close to a voronoi
graph.data = np.exp(-graph.data / graph.data.std())

# Force the solver to be arpack, since amg is numerically
# unstable on this example
labels = spectral_clustering(graph, n_clusters=100, eigen_solver='arpack')
label_im = -np.ones(mask.shape)
label_im[mask] = labels

plt.matshow(img)
plt.matshow(label_im)

sys.exit()

# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance to the background
distance = ndimage.distance_transform_edt(image)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                            labels=image)
markers = ndimage.label(local_maxi)[0]
labels = watershed(-distance, markers, mask=image)

fig, axes = plt.subplots(ncols=3, figsize=(8, 2.7))
ax0, ax1, ax2 = axes

ax0.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax0.set_title('Overlapping objects')
ax1.imshow(-distance, cmap=plt.cm.jet, interpolation='nearest')
ax1.set_title('Distances')
ax2.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
ax2.set_title('Separated objects')

for ax in axes:
    ax.axis('off')

fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
                    right=1)
plt.show()
