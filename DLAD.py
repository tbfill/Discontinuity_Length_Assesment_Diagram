from BS7910 import BS7910
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import sys
class DLAD:
    def __init__(self,flaw_type,HAZ,E,sigma_Y,sigma_u,Kmat,c,W,B,P_ratio,P_sum):
        self.flaw_type = flaw_type
        self.HAZ = HAZ
        self.E = E
        self.sigma_Y = sigma_Y
        self.sigma_u = sigma_u
        self.Kmat = Kmat
        
        self.c = c
        self.W = W
        self.B = B
        
        self.P_ratio = P_ratio
        self.P_sum = P_sum
        
        self.BS7910_vals = BS7910(self.flaw_type,self.E,self.sigma_Y,self.sigma_u,self.Kmat)
        if self.flaw_type == 'through_thickness_flaw':
            self.BS7910_vals.set_through_thickness_params(self.W,-1.)
        elif self.flaw_type == 'edge_flaw':
            self.BS7910_vals.set_edge_params(self.W,-1.)
        elif self.flaw_type == 'surface_flaw':
            self.BS7910_vals.set_surface_params(self.c,self.W,-1.,self.B)
    
    def find_a_for_load(self):
            
        fig = plt.figure()
        max_val = 0.
        for ind,val in enumerate(self.P_ratio):
            Pm = np.zeros(self.P_sum.shape[0])
            Pb  = np.zeros(self.P_sum.shape[0])
            if val > 0:
                Pm = self.P_sum * val
                Pb = self.P_sum - Pm
            elif val == 0:
                Pm = np.zeros(self.P_sum.shape[0])
                Pb  = self.P_sum
            Qm = np.zeros(self.P_sum.shape[0])
            if self.HAZ == True:
                Qm = self.sigma_Y - self.P_sum
            Qb =np.zeros(self.P_sum.shape[0])
            param_res = np.zeros(self.P_sum.shape[0])
            guess_Lr = 0.5
            guess = 0.1
            if self.flaw_type == 'surface_flaw':
                guess=self.BS7910_vals.B/4
            for ind2,val2 in enumerate(self.P_sum):
                #res = minimize(self.find_Lr_1,guess_Lr,args=[self.BS7910_vals],method='Nelder-Mead')
                #guess_Lr = res.x[0]
                self.BS7910_vals.set_loads(Pm[ind2],Pb[ind2],Qm[ind2],Qb[ind2])
                res = minimize(self.cost_find_a,guess,method='Nelder-Mead')
                param_res[ind2] = res.x
                guess = res.x
            
            max_val = np.max([max_val,np.max(param_res)])
            plt.plot(self.P_sum,param_res,label='$P_m/(P_m+P_b)=$'+str(np.round(val,decimals=2)))
            #print(self.P_sum)
            #print(param_res)
        
        plt.xlabel('$P_m+P_b(ksi)$')
        plt.ylabel('$a(in)$')
        plt.legend()
        plt.ylim(0,np.min([max_val+0.25,self.W+0.25]))
        plt.grid()
        plt.show()
        
    def cost_find_a(self,a):
        #print('Pm='+str(self.BS7910_vals.Pm))
        #print('a='+str(a))
        self.BS7910_vals.a = a
        Lr_a =  self.BS7910_vals.point_Lr()
        #print('Lr='+str(Lr_a))
        #print(Lr)
        if self.flaw_type == 'surface_flaw' and a/self.BS7910_vals.B > 0.99:
            self.BS7910_vals.a = self.BS7910_vals.B*0.99
            Kr_a = self.BS7910_vals.point_Kr()
            cost = a/self.BS7910_vals.B * self.cost_function(Lr_a,Kr_a)
        Kr_a = self.BS7910_vals.point_Kr()
        #print('Kr='+str(Kr_a))
        cost = self.cost_function(Lr_a,Kr_a)
        #print('cost='+str(cost))
        return cost
        
    def cost_function(self,Lr_a,Kr_a):
        f_1 = self.BS7910_vals.FAD_disc_yield(1.)
        f_Lr = self.BS7910_vals.FAD_disc_yield(Lr_a)
        Lr_max = (self.BS7910_vals.sigma_Y+self.BS7910_vals.sigma_u)/2*self.BS7910_vals.sigma_Y # equation 25
        f_Lrmax = self.BS7910_vals.FAD_disc_yield(Lr_max)
        cost = 0.
        if Lr_a <= 1 and Kr_a >= (1.5)**(-0.5):
            cost = np.abs(Kr_a-f_Lr)
        elif Lr_a > 1 and Kr_a >= (1.5)**(-0.5):
            cost = ((Kr_a-(1.5)**(-0.5))**2+(Lr_a-1)**2)**(0.5)
        elif Lr_a <= 1 and f_1 <= Kr_a and Kr_a < 1.5**(-0.5):
            cost = np.min([np.abs(Lr_a-1),np.abs(Kr_a-f_Lr)])
        elif Lr_a > 1 and f_1 <= Kr_a and Kr_a < 1.5**(-0.5):
            cost = np.abs(Lr_a-1)
        elif Lr_a <= 1. and Kr_a < f_1:
            cost = ((Kr_a-f_1)**2 + (Lr_a-1)**2)**0.5
        elif Lr_a > 1. and Lr_a < Lr_max and f_Lrmax <= Kr_a and Kr_a < f_1:
            cost = np.abs(Kr_a-f_Lr)
        elif Lr_max < Lr_a and f_Lrmax <= Kr_a and Kr_a <= f_1:
            cost = ((Kr_a - f_Lrmax)**2 + (Lr_a-Lr_max)**2)**0.5
        elif Lr_a > 1. and Kr_a < f_Lrmax:
            cost = np.min([np.abs(Lr_a-Lr_max),np.abs(Kr_a-f_Lr)])
        return cost
        
    def find_Lr_1(self,a,data):
        BS7910_vals = data[0]
        BS7910_vals.a = a
        Lr_a =  BS7910_vals.point_Lr()
        cost = np.abs(1-Lr_a)
        
        return cost
        