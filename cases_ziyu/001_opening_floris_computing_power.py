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

from floris import FlorisModel


# The FlorisModel class is the entry point for most usage.
# Initialize using an input yaml file
fmodel = FlorisModel("./cases_ziyu/inputs/gch.yaml")

# Changing the wind farm layout uses FLORIS' set method to a two-turbine layout
fmodel.set(layout_x=[0, 500.0], layout_y=[0.0, 0.0])

fmodel.set(
    wind_directions=np.array([270.0, 280.0, 290.0, 300.0]),
    wind_speeds=[8.0, 8.0, 10.0, 10.0],
    turbulence_intensities=np.array([0.06, 0.06, 0.06, 0.06]),
)
fmodel.set(yaw_angles=np.array([[10.,5.]]))
# After the set method, the run method is called to perform the simulation
fmodel.run()

# There are functions to get either the power of each turbine, or the farm power
turbine_powers = fmodel.get_turbine_powers() / 1000.0
farm_power = fmodel.get_farm_power() / 1000.0

print("The turbine power matrix should be of dimensions 4 (n_findex) X 2 (n_turbines)")
print(turbine_powers)
print("Shape: ", turbine_powers.shape)

print("The farm power should be a 1D array of length 4 (n_findex)")
print(farm_power)
print("Shape: ", farm_power.shape)
