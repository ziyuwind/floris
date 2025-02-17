import numpy as np
import matplotlib.pyplot as plt
from shapely import Polygon,MultiPolygon,Point

D =126.
boundaries=[(0.0, 0.0), (0.0, 50*D), (50*D, 50*D), (50*D, 0.0), (0.0, 0.0)]
boundary_polygon = MultiPolygon([Polygon(boundaries)])
boundary_line = boundary_polygon.boundary

# contains: False if point is located at the boundary
print(boundary_polygon.contains(Point([0.,0.])))
print(boundary_polygon.contains(Point([0.01,0.])))
print(boundary_polygon.contains(Point([0.01,0.01])))

# touch
print(boundary_polygon.touches(Point([0.0,0.0])))
print(boundary_polygon.touches(Point([0.01,0.0])))

# contains or touch
print(boundary_polygon.contains(Point([0.0,0.0])) and boundary_polygon.touches(Point([0.0,0.0])))
# plt.show()
