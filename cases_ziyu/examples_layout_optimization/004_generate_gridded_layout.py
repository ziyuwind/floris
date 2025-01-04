"""Example: Gridded layout design
This example shows a layout optimization that places as many turbines as
possible into a given boundary using a gridded layout pattern.
"""

import matplotlib.pyplot as plt
import numpy as np

from floris import FlorisModel, WindRose
from floris.optimization.layout_optimization.layout_optimization_gridded import (
    LayoutOptimizationGridded,
)


if __name__ == '__main__':
    # Load the Floris model
    fmodel = FlorisModel('cases_ziyu/inputs/gch.yaml')

    # Set the boundaries
    # The boundaries for the turbines, specified as vertices
    boundaries = [(0.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0), (1000.0, 0.0), (0.0, 0.0)]
    D = 126
    H = 60*D
    W = 30*D
    boundaries = [(0.0, 0.0), (0.0, H), (W, H), (W, H - 1./3.*H), (W - 1./2.*W, H - 1./3.*H), (W - 1./2.*W, H - 1./3.*H - 1/3.*H), (W, H - 1./3.*H - 1./3.*H), (0, 0)]
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
