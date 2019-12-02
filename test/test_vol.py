"""
Testing volume calculations from a set of 3d points
"""
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import numpy as np
import time

t0 = time.time()
points = np.random.rand(30000, 2)   # 30 random points in 2-D
t1 = time.time()
print("Generate random pts: " + str(t1-t0))

t0 = time.time()
hull = ConvexHull(points)
t1 = time.time()
print("Convex Hull: " + str(t1-t0))

plt.plot(points[:,0], points[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

t0 = time.time()
area = hull.area
t1 = time.time()
print("Area Calculation: " + str(t1-t0))

plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
plt.show()

# compute the edge orientation (with arctan),
# rotate the convex hull using this orientation in order to compute easily the bounding rectangle area with min/max of x/y of the rotated convex hull,
# Store the orientation corresponding to the minimum area found,
# Return the rectangle corresponding to the minimum area found
# https://gis.stackexchange.com/questions/22895/finding-minimum-area-rectangle-for-given-points
