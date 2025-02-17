import numpy as np
import matplotlib.pyplot as plt
from shapely import Polygon,MultiPolygon
folder_name = '/home/ziyu/solvers/floris/cases_ziyu/Mosetti_CaseC/'
wt50_spi120_objective_candidate_log = np.load(folder_name+'Mosetti_CaseC_GRS_wt50_spi120_output/Mosetti_CaseC_wt50_spi120_objective_candidate_log.npy')
wt50_spi120_num_objective_calls_log = np.load(folder_name+'Mosetti_CaseC_GRS_wt50_spi120_output/Mosetti_CaseC_wt50_spi120_num_objective_calls_log.npy')
wt50_spi120_initial_layout = np.load(folder_name+'Mosetti_CaseC_GRS_wt50_spi120_output/Mosetti_CaseC_wt50_spi120_initial_layout.npy')
wt50_spi120_final_layout = np.load(folder_name+'Mosetti_CaseC_GRS_wt50_spi120_output/Mosetti_CaseC_wt50_spi120_final_layout.npy')

# Handle default initial location plotting
default_initial_locs_plotting_dict = {
    "marker":"o",
    "color":"b",
    "linestyle":"None",
    "label":"Initial locations",
}

# Handle default final location plotting
default_final_locs_plotting_dict = {
    "marker":"o",
    "color":"r",
    "linestyle":"None",
    "label":"New locations",
}

default_plot_boundary_dict = {
    "color":"k",
    "alpha":0.1,
    "edgecolor":None
}
D =126.
boundaries=[(0.0, 0.0), (0.0, 50*D), (50*D, 50*D), (50*D, 0.0), (0.0, 0.0)]
boundary_polygon = MultiPolygon([Polygon(boundaries)])
boundary_line = boundary_polygon.boundary

fig,ax = plt.subplots(1,1,figsize=(9,6))
ax.set_aspect("equal")

# boundaries
for line in boundary_line.geoms:
    xy = np.array(line.coords)
    ax.fill(xy[:,0], xy[:,1], **default_plot_boundary_dict)

ax.plot(wt50_spi120_initial_layout[0,:],wt50_spi120_initial_layout[1,:],**default_initial_locs_plotting_dict)
ax.plot(wt50_spi120_final_layout[0,:],wt50_spi120_final_layout[1,:],**default_final_locs_plotting_dict)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
plt.show()
