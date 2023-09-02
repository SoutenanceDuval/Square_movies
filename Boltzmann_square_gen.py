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

x_length = 1    
y_length = 1

# np.random.seed(1)

N = 1*100
radius = 0.02
tmax = 50
h = 0.01

def overlaps(coord_particle1, coord_particle2):   
    return np.hypot(*(coord_particle1 - coord_particle2)) <= 2*radius

r0 = np.zeros((2, N))
ta = perf_counter()
for uu in range(N):
    while 1:
        Ax = radius + (x_length - 2*radius) * np.random.random()
        Ay = radius + (y_length - 2*radius) * np.random.random()
        coord_particleA = np.array([Ax, Ay])
        overlap_loc = False
        for index in range(N):
            coord_particleB = r0[:, index]
            if overlaps(coord_particleA, coord_particleB):
                overlap_loc = True
                break
        if not overlap_loc:
            r0[:, uu] = coord_particleA
            break
tb = perf_counter()

print('Time for initial config:', tb - ta)

v0 = np.zeros((2,N))
speed_amplitude = 3.5
v0x = (np.random.random(N) - 0.5)*speed_amplitude
v0y = (np.random.random(N) - 0.5)*speed_amplitude
v0[0][:], v0[1][:] = v0x, v0y

def next_collision_pair(pos1, pos2, vit1, vit2, t0):
    Delta_pos = pos1 - pos2
    Delta_vit = vit1 - vit2
    Discriminant = (np.dot(Delta_pos, Delta_vit))**2 - np.dot(Delta_vit, Delta_vit)*(np.dot(Delta_pos, Delta_pos) - 4*radius**2)
    if np.dot(Delta_pos, Delta_vit) < 0:
        approaching = True
    else:
        approaching = False
    if approaching and Discriminant>0:
        t_pair = t0 - (np.dot(Delta_pos, Delta_vit) + np.sqrt(Discriminant))/np.dot(Delta_vit, Delta_vit)
    else:
        t_pair = 10**100
    return t_pair

def next_collision_wall(pos1, vit1, t0):
    posx, posy = pos1
    vitx, vity = vit1
    tposs = []
    if vitx != 0.0:
        tposs.append(np.abs((np.sign(vitx) + 1)*x_length / 2 - posx - np.sign(vitx)*radius)/np.abs(vitx))
    if vity != 0.0:
        tposs.append(np.abs((np.sign(vity) + 1)*y_length / 2 - posy - np.sign(vity)*radius)/np.abs(vity))
        
    return t0 + np.min(tposs)

def collision_pair(pos1, pos2, vit1, vit2):
    Delta_pos = pos1 - pos2
    Delta_vit = vit1 - vit2
    e_normal = Delta_pos / np.linalg.norm(Delta_pos)
    new_vit1 = vit1 - e_normal * (np.dot(Delta_vit, e_normal))
    new_vit2 = vit2 + e_normal * (np.dot(Delta_vit, e_normal))
    return new_vit1, new_vit2

def collision_wall(pos1, vit1):
    posx, posy = pos1
    vitx, vity = vit1
    digits_thresh = 8
    if np.round(posx - radius, digits_thresh) == 0.0 or np.round(posx + radius, digits_thresh) == x_length:
        vitx *= -1
    elif np.round(posy - radius, digits_thresh) == 0.0 or np.round(posy + radius, digits_thresh) == y_length:
        vity *= -1
    vit1 = np.array([vitx, vity])
    return vit1

def event_disks(r_loc, v_loc, t_loc):
    
    tps_min_pair = 10**10
    for pair in combinations(np.arange(N),2):
        ind1, ind2 = pair
        pos1 = r_loc[:, ind1]
        pos2 = r_loc[:, ind2]
        vit1 = v_loc[:, ind1]
        vit2 = v_loc[:, ind2]
        tps_pair = next_collision_pair(pos1, pos2, vit1, vit2, t_loc)
        if tps_pair < tps_min_pair:
            tps_min_pair = tps_pair
            next_pair = pair
            
    tps_min_wall = 10**10
    for ind1 in range(N):
        pos1 = r_loc[:, ind1]
        vit1 = v_loc[:, ind1]
        tps_wall = next_collision_wall(pos1, vit1, t_loc)
        if tps_wall < tps_min_wall:
            tps_min_wall = tps_wall
            next_ind = ind1
            
    bool_wall, bool_pair = 0, 0
    t_next = min(tps_min_pair, tps_min_wall)
    r_loc += (t_next - t_loc)*v_loc
        
    if tps_min_wall < tps_min_pair:
        # print('Wall. t_loc', t_loc, 't_next', t_next, next_ind)
        pos1 = r_loc[:, next_ind]
        vit1 = v_loc[:, next_ind]
        v_loc[:, next_ind] = collision_wall(pos1, vit1)
        bool_wall = 1
        
    else:
        # print('Binary collision. t_loc', t_loc, 't_next', t_next)
        ind1_next, ind2_next = next_pair
        new_vit1, new_vit2 = collision_pair(r_loc[:, ind1_next], r_loc[:, ind2_next], v_loc[:, ind1_next], v_loc[:, ind2_next])
        v_loc[:, ind1_next] = new_vit1
        v_loc[:, ind2_next] = new_vit2
        bool_pair = 1
        
    return r_loc, v_loc, t_next, bool_wall, bool_pair


nb_steps_tot = int(np.round(tmax/h, 10))

R_ans = np.zeros((nb_steps_tot + 1, r0.shape[0], r0.shape[1]))
V_ans = {}

R_ans[0] = r0.copy()
V_ans[0] = v0

t0 = 0
nb_steps_cumule = 0
next_t = t0 + h
t_loc_check = 0

nb_collisions_walls = 0
nb_collisions_pairs = 0

t_dyn_ini = perf_counter()

while t0 < tmax:
    R, V, T, bool_wall, bool_pair = event_disks(np.copy(r0), np.copy(v0), t0)
    nb_collisions_walls += bool_wall
    nb_collisions_pairs += bool_pair
    if bool_pair == 1:
        V_ans[nb_collisions_pairs] = V
        if nb_collisions_pairs % 10 == 0:
            print(t0, nb_collisions_pairs)
        
    
    if T > next_t:
        
        R_loc = np.copy(r0)
        first_time_after_event = h * (int(np.round(t0/h, 10)) + 1)
        t_loc_check = first_time_after_event

        R_loc += v0 * (first_time_after_event - t0)
        nb_steps_cumule += 1
        next_t += h
        R_ans[nb_steps_cumule] = R_loc.copy()
        
        i = 1
        while next_t < T and nb_steps_cumule < nb_steps_tot:
            R_loc += v0 * h
            nb_steps_cumule += 1
            next_t += h
            t_loc_check = first_time_after_event + i*h
            i += 1
            R_ans[nb_steps_cumule] = R_loc.copy()
    
    r0, v0, t0 = np.copy(R), np.copy(V), T

t_dyn_fin = perf_counter()

n_events = nb_collisions_pairs + nb_collisions_walls

print('Time for dynamique moléculaire:', t_dyn_fin - ta)
print('Nb events: ', n_events)
print('Nb wall collisions: ', nb_collisions_walls)
print('Nb pair collisions: ', nb_collisions_pairs, '\t', 'Nb collisions par particules: ', nb_collisions_pairs/N)

data = [R_ans, V_ans]
name_file = 'data_squareN%.0ftmax%.1fradius%.3fh%.4fspeed_amplitude%.3f.pickle'%(N, tmax, radius, h, speed_amplitude)
with open(name_file, 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)





t2 = perf_counter()
print('Total time of execution:', t2 - t1)
