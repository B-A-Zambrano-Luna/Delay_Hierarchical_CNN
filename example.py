# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:48:36 2023

@author: LENOVO
"""

import time
import Q_p
import test_functions as test
from Q_p_delay_CNN import p_adic_delay_CNN
import joblib
import math
import numpy as np


delta_t = 0.05
t = 20
p = 2
K = 12  # max 12 for 1 min
delay = -2
Z_k = Q_p.Z_N_group(p, K)
L = 1
# Parameters


# Nonlinealities


# Kernels

def A(x):
    return -4*math.sin(math.pi*(1-Q_p.norm_p(x, p)))


# A = test.char_function(0, -1, p)


J = 0


def B(x):
    return math.cos((math.pi)*(1-Q_p.norm_p(x, p)))


# Stimuli

def U(x):
    return math.sin(math.pi*(1-Q_p.norm_p(x, p)))

# U = 2*test.char_function(0, -4, p)
# threhold


Z = (-0.15)*test.char_function(0, -K, p)

# Initial state


# X_0 = 0

def X_0_aux(x):
    if x == 0:
        return 1
    else:
        return 0


def X_0(x, t):
    return 0


solver = p_adic_delay_CNN()

start = time.time()

solver.solution(J, A, B, U, Z,
                X_0, t, delta_t, Z_k,
                delay=delay, L=L)

end = time.time()
print("Time ejecution solver", end-start)

# Heat map
start = time.time()
heat_map = solver.plot(function="state",
                       xlabels=False,
                       ylabels=False,
                       title=" ",
                       with_tree=True)
end = time.time()
print("Time ejecution plot", end-start)

# Save image
imag_heat_map = heat_map.get_figure()
imag_heat_map.savefig('simulation_1_state_r-2.png',
                      dpi="figure",
                      bbox_inches="tight")

# Fractal map
# start = time.time()
# solver.plot_fract(function="E", m=0, s=0.5, screen_shot=True,
#                   size_points=2)
# end = time.time()
# print("Time ejecution plot", end-start)

# load a model
#solver_1 = joblib.load('Amari_solver.pkl')
