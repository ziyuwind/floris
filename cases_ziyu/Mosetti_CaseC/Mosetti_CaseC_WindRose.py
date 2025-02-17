import numpy as np
import matplotlib.pyplot as plt

from floris import (
    FlorisModel,
    WindRose,
)
from floris.optimization.layout_optimization.layout_optimization_gridded import (
    LayoutOptimizationGridded,
)
from floris.optimization.layout_optimization.layout_optimization_random_search import (
    LayoutOptimizationRandomSearch,
)

# Load the wind rose from csv
wind_rose = WindRose.read_csv_long(
    "./cases_ziyu/Mosetti_CaseC/Mosetti_CaseC_wind_rose.csv", wd_col="wd", ws_col="ws", freq_col="freq_val", ti_col_or_value=0.075
)

# Load the Floris model
wind_rose.plot()

plt.show()
