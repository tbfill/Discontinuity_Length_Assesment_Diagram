import numpy as np
import matplotlib.pyplot as plt
from BS7910 import BS7910
from DLAD import DLAD

sigma_Y = 48.05
sigma_u = 70.8
E = 29000.
Kmat = 77.287
P_sum = 30
W = 12.
c = 1.0 #np.linspace(0.01,0.8,48)
B = 1.
P_ratio = 0.0
HAZ = False
flaw_type = 'through_thickness_flaw' # 'edge_flaw' 'through_thickness_flaw' 'surface_flaw'
P_sum =  np.flip(np.linspace(1.,48,48)) # np.array([45.,40.])
P_ratio = np.linspace(0,1,6) # np.array([1.0]) #

DLAD_eval = DLAD(flaw_type,HAZ,E,sigma_Y,sigma_u,Kmat,c,W,B,P_ratio,P_sum)

DLAD_eval.find_a_for_load()

