
#import matplotlib


import matplotlib as mpl
mpl.use('qt5agg')

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.image import AxesImage
from numpy.random import rand

import scipy.interpolate as intp
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter



import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import numpy as np
from tkinter import *



import sys
sys.path.insert(1,'scripts/')
import fluence as fluence
import process_sim as sim
import process_func as prf


# prepare sim information
datadir='/vol/astro7/lofar/sim/pipeline/events/272886233/1/coreas/proton/'
fileno='000091'
Binc=1.1837 # lofar



###### prep info from sim
sim_info=sim.ProcessData(datadir,fileno,30,80)

antUVW=prf.GetUVW(sim_info['antenna_position'], 0, 0, 0, sim_info['zenith'], sim_info['azimuth'],Binc)
rbf = intp.Rbf(antUVW.T[0], antUVW.T[1], sim_info['fluence'],smooth =0,function='quintic')

dist_scale=600
ti = np.linspace(-dist_scale, dist_scale, 150)
XI, YI = np.meshgrid(ti, ti)
ZI = rbf(XI, YI)


print('energy: {0:.2e}'.format(sim_info['energy']*1e9))
print('zenith: {0:.0f}'.format(sim_info['zenith']*180/np.pi))
print('azimuth: {0:.0f}'.format((int(sim_info['azimuth']*180/np.pi)%360)))
###########################################





# Make the main tkinter window
root = tk.Tk()


# setting the title and
root.title('Plotting in Tkinter')

# setting the dimensions of
# the main window
root.geometry("1000x500")


tabControl = ttk.Notebook(root)
tab1 = ttk.Frame(tabControl)
tab2 = ttk.Frame(tabControl)


tabControl.add(tab1, text ='Shower View')
tabControl.add(tab2, text ='Event View')

tabControl.pack(expand = 1, fill ="both")

gframe = ttk.Frame(master=root,borderwidth=1)
gframe.pack(side=tk.TOP,fill=tk.BOTH, expand=1)
fig = Figure(figsize = (8, 5),dpi = 100)

cframe1 = tk.Frame(master=tab1, borderwidth=10)
canvas1 = FigureCanvasTkAgg(fig, master=cframe1)
canvas1.draw()
canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
toolbar = NavigationToolbar2Tk(canvas1, cframe1)
toolbar.update()
canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
cframe1.pack(side=tk.LEFT,fill=tk.BOTH, expand=1)

# make figure for frame 1

#x, y, c, s = rand(4, 100)

#fig=plt.figure()
#
def onpick3(event):
    ind = event.ind
    print('onpick3 scatter:', ind)


    ax2.clear()
    ax3.clear()
    ax4.clear()

    # Raw electric field
    ax2.plot(sim_info['e_time']*1e9,sim_info['xyz_trace'][ind[0]][0],color='red')
    ax2.plot(sim_info['e_time']*1e9,sim_info['xyz_trace'][ind[0]][1],color='blue')
    ax2.plot(sim_info['e_time']*1e9,sim_info['xyz_trace'][ind[0]][2],color='green')
    ax2.set_xlabel('time (ns)', fontsize=8)
    ax2.set_ylabel('xyz (V/m)', fontsize=8)

    # On sky, filtered
    ax3.plot(sim_info['e_time']*1e9,sim_info['poldata'][ind[0]][0],color='red')
    ax3.plot(sim_info['e_time']*1e9,sim_info['poldata'][ind[0]][1],color='blue')
    ax3.set_xlabel('time (ns)', fontsize=8)
    ax3.set_ylabel('on sky (V/m)', fontsize=8)

    # Voltage (+ antenna model)
    ax4.plot(sim_info['e_time']*1e9,sim_info['voltage'][ind[0]][0],color='red')
    ax4.plot(sim_info['e_time']*1e9,sim_info['voltage'][ind[0]][1],color='blue')
    ax4.set_xlabel('time (ns)', fontsize=8)
    ax4.set_ylabel('voltage (V)', fontsize=8)
    mid=np.argmax(np.abs(sim_info['poldata'][ind[0]][0]))


    ax2.set_xlim(100,300)
    ax3.set_xlim(100,300)

    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax3.tick_params(axis='both', which='major', labelsize=8)
    ax4.tick_params(axis='both', which='major', labelsize=8)



    fig.canvas.flush_events()
    fig.tight_layout()

    fig.canvas.draw()

# adding the subplot
gs = GridSpec(3, 2, width_ratios=[1, 1],
                       height_ratios=[1,1, 1], figure=fig)

#ax1 = fig.add_subplot(231,aspect=1)
ax1 = fig.add_subplot(gs[:-1,0],aspect=1)
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2, 1])
ax5 = fig.add_subplot(gs[2, 0])

ax1.tick_params(axis='both', which='major', labelsize=8)
ax2.tick_params(axis='both', which='major', labelsize=8)
ax3.tick_params(axis='both', which='major', labelsize=8)
ax4.tick_params(axis='both', which='major', labelsize=8)
ax5.tick_params(axis='both', which='major', labelsize=8)

#draw radio footprint
ax1.pcolor(XI, YI, ZI,vmax=np.max(sim_info['fluence']), vmin=0,cmap=cm.jet)
ax1.scatter(antUVW.T[0],antUVW.T[1],20,c=sim_info['fluence'],vmax=np.max(sim_info['fluence']), vmin=0,edgecolors='white',cmap=cm.jet,picker=True)

ax1.set_xlim([-200,200])
ax1.set_ylim([-200,200])
ax1.set_xlabel('vxB (m)', fontsize=8)
ax1.set_ylabel('vxvxB (m)', fontsize=8)

#draw shower hillas curve

xhillas=np.arange(10,1600,5)
yhillas=sim_info['hillas'][0]*((xhillas-sim_info['hillas'][1])/(sim_info['hillas'][2]-sim_info['hillas'][1]))**((sim_info['hillas'][2]-sim_info['hillas'][1])/(sim_info['hillas'][3]+sim_info['hillas'][4]*xhillas+sim_info['hillas'][5]*xhillas**2)) * np.exp((sim_info['hillas'][2]-xhillas)/(sim_info['hillas'][3]+sim_info['hillas'][4]*xhillas+sim_info['hillas'][5]*xhillas**2))


ax5.plot(xhillas,yhillas,color='black',linestyle=':')
ax5.set_xlabel('g/cm2', fontsize=8)
ax5.set_ylabel('N(particles)', fontsize=8)
ax5.set_xlim([100,1600])

ax2.set_xlabel('time (ns)', fontsize=8)
ax2.set_ylabel('electric field (V/m)', fontsize=8)
ax3.set_xlabel('time (ns)', fontsize=8)
ax3.set_ylabel('filtered (V/m)', fontsize=8)
ax4.set_xlabel('time (ns)', fontsize=8)
ax4.set_ylabel('voltage (V)', fontsize=8)


fig.canvas.mpl_connect('pick_event', onpick3)

#fig.tight_layout()









'''
cframe1 = tk.Frame(master=tab1, borderwidth=10)
canvas1 = FigureCanvasTkAgg(fig, master=cframe1)
canvas1.draw()
canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
toolbar = NavigationToolbar2Tk(canvas1, cframe1)
toolbar.update()
canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
cframe1.pack(side=tk.LEFT,fill=tk.BOTH, expand=1)
'''




def update_event():
    global ev,tStep,freqMax,alabels,text_box

    print('hi')

update_event()

tk.mainloop()
