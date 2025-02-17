import numpy as np
import matplotlib.pyplot as plt

from floris import (
    FlorisModel,
    WindRose,
)
from floris.optimization.layout_optimization.layout_optimization_gridded import (
    LayoutOptimizationGridded,
)

# Load the Floris model
fmodel = FlorisModel('./cases_ziyu/inputs/gch.yaml')

# Load the wind rose from csv
wind_rose = WindRose.read_csv_long(
    "./cases_ziyu/Mosetti_CaseC/Mosetti_CaseC_wind_rose.csv", wd_col="wd", ws_col="ws", freq_col="freq_val", ti_col_or_value=0.075
)

D = 125.88

# Set the boundaries
# The boundaries for the turbines, specified as vertices
boundaries = [(0.0, 0.0), (0.0, 50*D), (50*D, 50*D), (50*D, 0.0), (0.0, 0.0)]

# Set up the optimization object with 5D spacing
layout_opt = LayoutOptimizationGridded(
    fmodel,
    boundaries,
    min_dist_D=5., # results in spacing of 5*125.88 = 629.4 m
    min_dist=None, # Alternatively, can specify spacing directly in meters
)

layout_opt.optimize()

# Note that the "initial" layout that is provided with the fmodel is
# not used by the layout optimization.
layout_opt.plot_layout_opt_results()

plt.show()
