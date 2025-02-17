import matplotlib
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap
def plot_vertical_rotor(ax,location,D,gamma):
    yc = location[0]
    zc = location[1]
    theta = np.linspace(0,2.*np.pi,100)
    y = yc + D/2*np.cos(gamma)*np.cos(theta)
    z = zc + D/2*np.sin(theta)
    ax.plot(y,z,color='k',linewidth=1,linestyle='--')


def plot_horizontal_rotor(ax,location,D,gamma,nacelle,color):
    xc = location[0]
    yc = location[1]
    xr = np.linspace(xc-D/2*np.sin(gamma),xc+D/2*np.sin(gamma),num=2)
    yr = np.linspace(yc+D/2*np.cos(gamma),yc-D/2*np.cos(gamma),num=2)
    xn = np.linspace(xc+0.25*nacelle*np.cos(gamma),xc+nacelle*np.cos(gamma),num=2)
    yn = np.linspace(yc+0.25*nacelle*np.sin(gamma),yc+nacelle*np.sin(gamma),num=2)
    ax.plot(xr,yr,color=color,linewidth=1.5)
    ax.plot(xn,yn,color=color,linewidth=2)

params = {

        'font.family':'Times New Roman', # Computer Modern Roman
        'mathtext.fontset':'cm',# stix
        'font.size':16,
        # 'text.usetex':True,
        # axes  
          'xtick.labelsize':14,
          'ytick.labelsize':14,
          
          'xtick.direction':'in',
          'ytick.direction':'in',
          'xtick.top':'True',
          'ytick.right':'True',
        #   'xtick.major.width':1,
        #   'xtick.major.size':4,
        #   'ytick.major.size':4,
          'axes.labelsize':16,
          'axes.linewidth':1.5,
          'lines.linewidth':2,

        # text
        # legend  
          'legend.fontsize':16,
        }

matplotlib.rcParams.update(params)

newcolors = cm.get_cmap('turbo',256)
newcolors = newcolors(np.linspace(0.05,1,256))
cmap = ListedColormap(newcolors)
resolution = 256