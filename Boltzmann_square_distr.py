# -*- coding: utf-8 -*-
"""
Created on Mon May 22 17:01:43 2023

@author: cleme
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from itertools import combinations
from time import perf_counter
import pickle

t1 = perf_counter()

dpi_value = 100
plt.rcParams['figure.dpi'] = dpi_value

x_length = 1    
y_length = 1

np.random.seed(1)

N = 1*100
radius = 0.02
tmax = 50
h = 0.01
nb_steps_tot = int(np.round(tmax/h, 10))

nb_steps_bis = nb_steps_tot
# nb_steps_bis = 30

name_file = 'data_squareN%.0ftmax%.1fradius%.3fh%.4f.pickle'%(N, tmax, radius, h)
with open(name_file, 'rb') as handle:
    R_ans, V_ans = pickle.load(handle)


#ANIMATION
# xarr = np.zeros((N, nb_steps_bis+1))
# yarr = np.zeros((N, nb_steps_bis+1))

# for ii in range(nb_steps_bis+1):
#     xarr[:, ii], yarr[:, ii] = R_ans[ii]

# t_anim_ini = perf_counter()

# fig = plt.figure()
# fig.set_dpi(dpi_value)
# fig.set_size_inches(7*x_length, 6.5*y_length)

# styles = {'facecolor': 'red', 'edgecolor':'black', 'linewidth': 0.3}

# ax = plt.axes(xlim=(0, x_length), ylim=(0, y_length))
# ax.yaxis.set_ticks([])
# ax.xaxis.set_ticks([])
# circles = []
# for i in range(N):
#     circles.append(plt.Circle((xarr[i, 0], yarr[i, 0]), radius, **styles))

# def init():
#     for circle in circles:
#         ax.add_patch(circle)
#     return circles

# def animate(i):
#     for j, circle in enumerate(circles):
#         x, y = xarr[j, i], yarr[j, i]
#         circle.center = (x, y)
#     return circles

# anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nb_steps_bis+1)

# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=40, bitrate=3000)
# anim.save('squareN%.0ftmax%.1fradius%.3fh%.4f.mp4'%(N, tmax, radius, h), writer=writer)
# t_anim_fin = perf_counter()
# print('Time to animate:', t_anim_fin - t_anim_ini)



#DISTRIBUTION
fig2, axs2 = plt.subplots(1,1,figsize=(7, 5))

nb_max_collisions = [key for key in V_ans][-1]

vx1, vy1 = V_ans[0]
norm_v = np.sqrt(vx1**2 + vy1**2)

#The temperature is fixed by the (conserved) energy
energy_per_particle = np.sum(0.5 * norm_v**2)/N
T = energy_per_particle
#Alternative view, via equipartition
# T = np.mean(norm_v**2) / 2

most_probable_speed = np.sqrt(T)
mean_speed = np.sqrt(np.pi/2)*most_probable_speed
axs2.axvline(x = most_probable_speed, c='red')
axs2.axvline(x = mean_speed, c='green')

v_arr = np.linspace(0, 6*most_probable_speed, 100)
maxwell_boltzmann = (v_arr/T) * np.exp(-v_arr**2/(2*T))
axs2.plot(v_arr, maxwell_boltzmann)



nb_bins = 100
bins = list(np.linspace(0, 6*most_probable_speed, nb_bins)) + [100*most_probable_speed]

counts_tot = np.zeros(nb_bins)

for uu in range(1000,40000):
# for uu in range(10):
    vx1, vy1 = V_ans[uu]
    norm_v = np.sqrt(vx1**2 + vy1**2)
    counts = np.histogram(norm_v, bins)[0]
    counts_tot += counts


# axs2.stairs(counts_tot, bins, density=True, histtype='step')

bins_middle = [(bins[i+1] + bins[i])/2 for i in range(len(bins)-1)]
width_bins = np.array([(bins[i+1] - bins[i]) for i in range(len(bins)-1)])

int_distr = np.sum(width_bins*counts_tot)

distribution = counts_tot / int_distr


axs2.step(bins_middle, distribution)
axs2.set_xlim(0, 5*most_probable_speed)
# axs2.hist(vitesses, bins, density=True, histtype='step') 



t2 = perf_counter()
print('Total time of execution:', t2 - t1)
