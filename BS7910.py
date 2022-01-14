import numpy as np
import sys

class BS7910:
    def __init__(self,flaw_type,E,sigma_Y,sigma_u,Kmat):
        self.flaw_type = flaw_type
        self.E = E
        self.sigma_Y = sigma_Y
        self.sigma_u = sigma_u
        self.Kmat = Kmat
        
        self.c = -1.
        self.W = -1.
        self.a = -1.
        self.B = -1.
        self.Pm = -1.
        self.Pb = -1.
        self.Qm = -1.
        self.Qb = -1.
        self.Lr = -1.
        self.Kr = -1.
    
    def set_loads(self,Pm,Pb,Qm,Qb):
        self.Pm = Pm
        self.Pb = Pb
        self.Qm = Qm
        self.Qb = Qb
        
    def set_through_thickness_params(self,W,a):
        self.W = W
        self.a = a
        
    def set_edge_params(self,W,a):
        self.W = W
        self.a = a
        
    def set_surface_params(self,c,W,a,B):
        self.c = c
        self.W = W
        self.a = a
        self.B = B
        
        
    def FAD_disc_yield(self,Lr):
        Lr_max = (self.sigma_Y+self.sigma_u)/2*self.sigma_Y # equation 25
        N = 0.3*(1-self.sigma_Y/self.sigma_u)          # equation 28
        Delta_eps = 0.0375 * (1-0.001*self.sigma_Y) # equation 8
        lambda1 = 1+self.E*Delta_eps/self.sigma_Y
        f_Lr = 0.
        if type(Lr) is np.ndarray:
            f_Lr = np.zeros(Lr.shape[0])
        f_1 = (lambda1+1/(2*lambda1))**(-1/2)
        f_Lr = np.where(Lr < 1., (1+1/2*Lr**2)**(-1/2),
                        np.where(Lr < Lr_max,f_1*Lr**((N-1)/(2*N)),0.) )
        
        return f_Lr
    
    def point_Lr(self):
        sigma_ref = 0.
        if self.flaw_type == 'through_thickness_flaw':
            sigma_ref = self.through_thickness_sigma_ref()
        elif self.flaw_type == 'edge_flaw':
            sigma_ref = self.edge_sigma_ref() 
        elif self.flaw_type == 'surface_flaw':
            sigma_ref = self.surface_sigma_ref()
        self.Lr = sigma_ref/self.sigma_Y
        return self.Lr
        
    def through_thickness_sigma_ref(self):
        # BS7910 P.5.1
        # Step to prevent optimizer from finding crack lengths longer than physically possible
        sigma_ref = np.where((2*self.a)<self.W,
                    (self.Pb+(self.Pb**2+9*self.Pm**2)**0.5)/(3*(1-(2*self.a/self.W))),
                    5000.+(2*self.a+1)**2)
        return sigma_ref
        
    def edge_sigma_ref(self):
        # Step to prevent optimizer from finding crack lengths longer than physically possible
        sigma_ref = np.where((self.a)<self.W,
                    (self.Pb+(self.Pb**2+9*self.Pm**2)**0.5)/(3*(1-(self.a/self.W))),
                    5000.+(self.a+1)**2)
        return sigma_ref
        
    def surface_sigma_ref(self):
        # BS7910 P.6.1
        alpha = np.where(self.W>=2*(self.c+self.B),(2*self.a/self.B)/(1+self.B/self.c),
                        (self.a/self.B)*(2*self.c/self.W))
        # Step to prevent optimizer from finding crack lengths longer than physically possible
        sigma_ref = np.where(alpha < 1,
                            (self.Pb+(self.Pb**2+9*self.Pm**2*(1-alpha)**2)**0.5)/(3*(1-alpha))**2,
                            np.inf)
        return sigma_ref

###################################################################################################
        
    def point_Kr(self):
        M = 0.
        fw = 0.
        Mm = 0.
        Mb = 0.
        if self.flaw_type == 'through_thickness_flaw':
            M,fw,Mm,Mb = self.through_thickness_fw()
        elif self.flaw_type == 'edge_flaw':
            M,fw,Mm,Mb = self.edge_fw()
        elif self.flaw_type == 'surface_flaw':
            M,fw,Mm,Mb = self.surface_fw()
        #Section M.1
        K1p = M*fw*(Mm*self.Pm+Mb*self.Pb)*(np.pi*self.a)**(0.5)
        K1s = (Mm*self.Qm+Mb*self.Qb)*(np.pi*self.a)**(0.5)
        K1s,V = self.V_calc(K1p,K1s)
        self.Kr = (K1p+V*K1s)/(self.Kmat)
        return self.Kr
        
    def through_thickness_fw(self):
        # BS7910 M.3.1
        M = np.ones(self.a.shape[0])
        fw = (1/np.cos(np.pi*self.a/self.W))**(0.5)
        Mm = np.ones(self.a.shape[0])
        Mb = np.ones(self.a.shape[0])
        return M,fw,Mm,Mb

    def edge_fw(self):
        # BS7910 M.3.2              
        M = np.ones(self.a.shape[0]) #np.where((self.a)<self.W,np.ones(self.a.shape[0]),5000.+(self.a+1)**2)
        fw = np.ones(self.a.shape[0])
        Mm = 1.12-0.23*(self.a/self.W) + 10.6*(self.a/self.W)**2 - 21.7*(self.a/self.W)**3+30.4*(self.a/self.W)**4
        Mb = np.ones(self.a.shape[0])
        return M,fw,Mm,Mb
    
    def surface_fw(self):
        # BS7910 M.3.3
        theta = np.pi
        if (np.max(self.a)/(2*self.c)) > 1.:
            sys.exit("a/(2*c)="+str(self.a/(2*self.c))+'. It must be less than 1')
        if theta > np.pi:
            sys.exit("theta="+str(theta)+'. It must be less than pi')
        if (np.max(self.a)/self.B) > 1.:
            print(self.a)
            sys.exit("a/B="+str(self.a/self.B)+'. Crack goes through plate thickness')
        M = 1.
        if type(self.a) is np.ndarray:
            M = np.ones(self.a.shape[0])
        fw = (1/np.cos((np.pi*self.c/self.W)*(self.a/self.B)**0.5))**(0.5)
        
        # M.4.1.2
        M1 = np.where((self.a/(2*self.c))<=0.5,1.13-0.09*(self.a/self.c),((self.c/self.a)**0.5)*(1+0.04*self.c/self.a))
        M2 = np.where((self.a/(2*self.c))<=0.5,0.89/(0.2+self.a/self.c)-0.54,0.2*(self.c/self.a)**4)
        M3 = np.where((self.a/(2*self.c))<=0.5,0.5-1/(0.65+self.a/self.c)+14*(1-self.a/self.c)**24,-0.11*(self.c/self.a)**4)
        g = np.where((self.a/(2*self.c))<=0.5,1+(0.1+0.35*(self.a/self.B)**2)*(1-np.sin(theta))**2 , 
                     1+(0.1+0.35*(self.c/self.a)*(self.a/self.B)**2)*(1-np.sin(theta))**2)
        f_theta = np.where((self.a/(2*self.c))<=0.5,((self.a/self.c)**2*np.cos(theta)**2+np.sin(theta)**2)**0.25,
                           ((self.c/self.a)**2*np.sin(theta)**2+np.cos(theta)**2)**0.25)
        Phi = np.where((self.a/(2*self.c))<=0.5, (1+1.464*(self.a/self.c)**1.65)**0.5, 
                       (1+1.464*(self.c/self.a)**1.65)**0.5)
        Mm = (M1+M2*(self.a/self.B)**2+M3*(self.a/self.B)**4)*(g*f_theta/Phi)
        
        # M.4.1.3
        q =  np.where((self.a/(2*self.c))<=0.5,0.2+self.a/self.c+0.6*self.a/self.B,
                                        0.2*self.c/self.a+0.6*self.a/self.B)
        H1 = np.where((self.a/(2*self.c))<=0.5,1-0.34*self.a/self.B-0.11*(self.a/self.c)*(self.a/self.B),
                      1-(0.04+0.41*(self.c/self.a))*(self.a/self.B)+(0.55-1.93*(self.c/self.a)**0.75+1.38*(self.c/self.a)**1.5)*(self.a/self.B)**2)
        G1 = np.where((self.a/(2*self.c))<=0.5,-1.22-0.12*self.a/self.c,
                                         -2.11+0.77*self.c/self.a)
        G2 =np.where((self.a/(2*self.c))<=0.5,0.55-1.05*(self.a/self.c)**0.75+0.47*(self.a/self.c)**1.5,
                                        0.55-0.72*(self.c/self.a)**0.75+0.14*(self.c/self.a)**1.5)
        H2 = 1 + G1*(self.a/self.B)+G2*(self.a/self.B)**2
        H = 1+(H2-H1)*np.sin(theta)**q
        Mb = H*Mm
        
        return M,fw,Mm,Mb
        
    def V_calc(self,K1p,K1s):
        K1s_0 = K1s
        K1s = np.where(K1s<0,0,K1s)
        V = 1.
        if type(K1p) is np.ndarray:
            V = np.ones(K1p.shape[0])
        beta = 1 # assume plane stress
        h = np.where(self.Lr<1.085,
                    (1+0.15*self.Lr**7)**(-1)*(0.2+0.8*np.exp(-0.5*self.Lr**7))*np.exp(1.3*self.Lr**6-0.8*self.Lr**12),
                    (1+0.15*self.Lr**7)**(-1)*(0.2+0.8*np.exp(-0.5*self.Lr**7)) )
        aeff = self.a+(1/(2*np.pi*beta))*(K1s/self.sigma_Y)**2
        
        Kjs =  (aeff/self.a)**0.5*K1s
        Kjs = np.where(Kjs<K1s,K1s,Kjs)
        V = np.where(K1s/(K1p/self.Lr) <= 4., np.where(self.Lr<1.05,np.minimum(1+0.2*self.Lr+0.02*K1s*(self.Lr/K1p)*(1+2*self.Lr),3.1-2*self.Lr),1.),
                 Kjs/K1s*h)
        V = np.where(K1s_0<=0,1,V)
        return K1s,V