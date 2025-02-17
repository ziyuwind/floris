"""Example 1: Opening FLORIS and Computing Power

This example illustrates several of the key concepts in FLORIS. It demonstrates:

  1) Initializing a FLORIS model
  2) Changing the wind farm layout
  3) Changing the incoming wind speed, wind direction and turbulence intensity
  4) Running the FLORIS simulation
  5) Getting the power output of the turbines

Main concept is introduce FLORIS and illustrate essential structure of most-used FLORIS calls
"""


import numpy as np
import matplotlib.pyplot as plt
from floris import FlorisModel
from floris.utilities import sind, cosd

import sys
import os
# 使用 ~ 表示 $HOME 目录
home_path = os.path.expanduser("~")
sys.path.append(home_path+'/solvers/floris/cases_ziyu')

import matplotlib_params

LES_173_353 = np.loadtxt(r'/mnt/d/OneDrive/paper/Zhang2024analytical/Exp/HRWF/LES_173_353.txt',skiprows=1)
LES_direction=LES_173_353[:,0]
LES_direction=np.round(LES_direction,0)
# LES_direction[12]=222
# LES_direction[54]=312
LES_power=LES_173_353[:,1]
# 12 221 54
LES_direction[11]+=1
LES_direction[12]+=1
LES_direction[53]+=1
LES_direction[54]+=1
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
s = 100


# dashed line
full_wd = np.array([173, 192, 201, 222, 242, 251, 270, 288, 295, 312, 328, 335, 353])
full_wd_false = np.array([173, 192, 201, 222, 242, 251, 270, 288, 295, 312, 328, 335, 353])
power_full_wd = np.array([0.611,0.883, 0.821, 0.705, 0.804, 0.852, 0.611, 0.847, 0.813, 0.718, 0.826, 0.875, 0.613])
Sx = np.array([7,21.3,15.0,9.3,15.0, 21.3, 7, 22.9, 16.3, 10.5, 16.3, 22.9, 7.0])
wd_label = "\mathrm{wind}"
for i in range(len(full_wd)):
    axes[0].plot([full_wd[i],full_wd[i]],[0,power_full_wd[i]],linestyle='--',linewidth=1.0,color='k')
    axes[0].text(full_wd[i]-0.5, 0.425, s=f'$\\theta_{wd_label}={full_wd_false[i]}^\circ$',ha='right',va='bottom', rotation=90,fontsize='x-small')  
    axes[0].text(full_wd[i]+0.5, 0.425, s=f'$S_x/D={Sx[i]}$',ha='left',va='bottom', rotation=90,fontsize='x-small')  


# The FlorisModel class is the entry point for most usage.
# Initialize using an input yaml file
fmodel = FlorisModel("./cases_ziyu/inputs/HornsRev.yaml")

# Define a 10x8 turbine farm
D = 80.
layout_x = np.zeros([8,10])
layout_y = np.zeros([8,10])
for i in range(8):
    for j in range(10):
        layout_x[i,j] = j*7*D + -i*sind(7)*7*D
        layout_y[i,j] = 0. + i*cosd(7)*7*D
layout_x = layout_x.reshape(1,80)
layout_y = layout_y.reshape(1,80)
layout_x = layout_x.tolist()[0]
layout_y = layout_y.tolist()[0]
wd_array = np.arange(173,353+1,1)

fmodel.set(
    layout_x=layout_x,
    layout_y=layout_y,
    wind_directions=wd_array,
    wind_speeds=np.ones(len(wd_array))*8.,
    turbulence_intensities=np.ones(len(wd_array))*0.077,
)

# After the set method, the run method is called to perform the simulation
fmodel.run()

# There are functions to get either the power of each turbine, or the farm power
# turbine_powers = fmodel.get_turbine_powers() / 1000.0
farm_power = fmodel.get_farm_power() / 1000.0

# max 709.686592605545
farm_power_efficiency = farm_power/(80*709.686592605545)

s=50
axes[0].scatter(LES_direction, LES_power,facecolors='none',edgecolor='k',s=s,edgecolors='k')
axes[0].plot(wd_array, farm_power_efficiency,color='k')

fmt='png'
dpi=256
fig.savefig(home_path+'/solvers/floris/cases_ziyu/fig/HornsRev_power.'+fmt,dpi=dpi)

