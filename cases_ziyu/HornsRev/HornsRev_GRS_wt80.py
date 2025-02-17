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
import pandas as pd
import os
from floris.utilities import cosd, sind

# 使用 ~ 表示 $HOME 目录
home_path = os.path.expanduser("~")

# 读取CSV文件
df = pd.read_csv(home_path+'/solvers/floris/cases_ziyu/HornsRev/HornsRev_windrose.csv')

# 访问特定的列
ws = df['ws']  # 获取ws列
wd = df['wd']  # 获取wd列
freq_val = df['freq_val']  # 获取freq_val列
# Load the wind rose from csv
wind_speeds = np.unique(ws.values)
wind_directions = np.unique(wd.values)
freq_table = freq_val.values
freq_table = freq_table / freq_table.sum()
freq_table = freq_table.reshape((len(wind_speeds),len(wind_directions))).T
ti_table  = 0.075

wind_rose = WindRose(wind_directions=wind_directions,
                     wind_speeds=wind_speeds,
                     ti_table=ti_table,
                     freq_table=freq_table)
# WindRose.read_csv_long最好不要用
# wind_rose = WindRose.read_csv_long(
#     os.path.join(home_path, "solvers/floris/cases_ziyu/HornsRev/HornsRev_windrose.csv"), wd_col="wd", ws_col="ws", freq_col="freq_val", ti_col_or_value=0.075
# )

# Load the Floris model
fmodel = FlorisModel(os.path.join(home_path, "solvers/floris/cases_ziyu/inputs/HornsRev.yaml"))
D = 80.

wt_num = 50
wt_interval = 5*D
num_x = 10
num_y = int(wt_num/num_x)
layout_x = np.linspace(0,(num_x-1)*wt_interval,num_x,endpoint=True)
layout_y = np.linspace(0,(num_y-1)*wt_interval,num_y,endpoint=True)
layout_x, layout_y = np.meshgrid(layout_x, layout_y)
layout_x = layout_x.flatten()
layout_y = layout_y.flatten()

# # # Define a 10x8 turbine farm
D = 80.
layout_x = np.zeros([8,10])
layout_y = np.zeros([8,10])
for i in range(8):
    for j in range(10):
        layout_x[i,j] = j*7*D + -i*sind(7)*7*D
        layout_y[i,j] = 0. + i*cosd(7)*7*D
layout_x = layout_x.flatten()
layout_y = layout_y.flatten()


fmodel.set(layout_x=layout_x, layout_y=layout_y,wind_shear=0.11,wind_data=wind_rose)

# Parameters of layout optimization
n_individuals=4
relegation_number=2
max_workers=n_individuals+2 #多增加2个核
seconds_per_iteration=1.
total_optimization_seconds=2.
grid_step_size=0.5*D
use_dist_based_init=True

boundaries=[(0.0, 0.0),
            (0*7*D + -7*sind(7)*7*D, 0. + 7*cosd(7)*7*D), 
            (9*7*D + -7*sind(7)*7*D, 0. + 7*cosd(7)*7*D), 
            (9*7*D + -0*sind(7)*7*D, 0. + 0*cosd(7)*7*D), 
            (0.0, 0.0)]
min_dist=None
min_dist_D=5.
distance_pmf=None
interface="multiprocessing"  # Options are 'multiprocessing', 'mpi4py', None
enable_geometric_yaw=False
random_seed=1
use_value=False

# Perform the optimization
d = np.round(np.linspace(15*D,0,400,endpoint=False)[::-1],4)
p = np.linspace(1,1,400,endpoint=False)
p = p/sum(p)
distance_pmf = {"d":d, "p":p}

# Other options that users can try
# 1.
# distance_pmf = {"d": [100, 1000], "p": [0.8, 0.2]}
# 2.
# p = gamma.pdf(np.linspace(0, 900, 91), 15, scale=20); p = p/p.sum()
# distance_pmf = {"d": np.linspace(100, 1000, 91), "p": p}

layout_opt = LayoutOptimizationRandomSearch(
    fmodel=fmodel,
    boundaries=boundaries,
    min_dist=min_dist,
    min_dist_D=min_dist_D,
    distance_pmf=distance_pmf,
    n_individuals=n_individuals,
    seconds_per_iteration=seconds_per_iteration,
    total_optimization_seconds=total_optimization_seconds,
    interface=interface,  # Options are 'multiprocessing', 'mpi4py', None
    max_workers=max_workers,
    grid_step_size=grid_step_size,
    relegation_number=relegation_number,
    enable_geometric_yaw=enable_geometric_yaw,
    use_dist_based_init=use_dist_based_init,
    random_seed=random_seed,
    use_value=use_value,
)


layout_opt.describe()

layout_opt.optimize()




file_name = os.path.join(home_path, f"solvers/floris/cases_ziyu/Mosetti_CaseC/Mosetti_CaseC_wt{wt_num}_")
# np.save(file_name+'initial_layout.npy',np.array([layout_opt.x_initial,layout_opt.y_initial]))
# np.save(file_name+'final_layout.npy',np.array([layout_opt.x_opt,layout_opt.y_opt]))
# np.save(file_name+'objective_candidate_log.npy',layout_opt.objective_candidate_log)
# np.save(file_name+'num_objective_calls_log.npy',layout_opt.num_objective_calls_log)

layout_opt.plot_distance_pmf()
layout_opt.plot_layout_opt_results()
layout_opt.plot_progress()
plt.show()
