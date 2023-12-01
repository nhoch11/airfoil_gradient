# import necessary packages
import json
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from VPM_primal import Primal
from adjoint_calcs import adjoint_calcs
import time

# set precision to 15 digits of precision
np.set_printoptions(precision=15)

# this class will calculate and display the gradient attribute
class Adjoint:
	#__init__ called when an instance of a class is created
    def __init__(self, input_json): # file is a json
        # self.input_file stores the name of the json file. ex: "2412_200.json"
        
        # call and run the adjoint_calcs class/functions
        calcs= adjoint_calcs(input_json).run()
        
        # assign the returned data to object attributes here
        self.A = calcs[0]
        self.B = calcs[1]
        self.airfoil_points = calcs[2]
        self.n = calcs[3]
        self.l_k = calcs[4]
        self.v_inf = calcs[5]
        self.gamma = calcs[6]
        self.dA = calcs[7]
        self.dB = calcs[8]
        self.partial_CL = calcs[9] 
        

    # calculate g, then g transpose
    def calc_g(self):
        # not sure exactly why this assumption was here, but it does belong somewhere in the code
        # ASSUMPTION: the length of the nth panel is always 0.0 -or close to zero
    
        # initalize g^T vector ( a row vector)
        self.gT = np.zeros((1,self.n))
        
        # this for loop pattern only holds for indexes 1 through and including n-2,
        for i in range(1,self.n-1):
            # the points are technically nondimensionalized as x/c and y/c, so no chord is needed (or chord = 1) in eq 4.36
            # derivative is with respect to gamma values, so we are left with lengths
            # in gT, gamma is treated as independent of x and y design variables, and the derivative is taken WRT gamma
            self.gT[0,i] = (self.l_k[i] + self.l_k[i-1])/self.v_inf
        
        # populate the first and last indices
        self.gT[0,0] = self.l_k[0]/self.v_inf  
        self.gT[0,self.n-1] = self.l_k[self.n-2]/self.v_inf   
        
        # transpose gT to get g
        self.g = np.transpose(self.gT)

   
    # solve for v in the adjont equation A^T v = g
    def calc_v(self):
        
        # transpose A matrix
        self.A_transpose = np.transpose(self.A)

        # solve adjoint equation A^T v = g
        self.v = np.linalg.solve(self.A_transpose,self.g)

        #print("v", "\n", self.v)


    # calculate the f term
    def calc_f(self):
        # f = (d A)/(d X(alpha))*gamma - (d b)/( X(alpha))
        # get the dA /dX_a and dB / dX_a
        
        # initialize empty matrices. dA_gamma combines the dA and gamma terms. dB will then be added to this to get f
        self.dA_gamma = np.zeros((self.n,self.n*2,1))
        self.f = np.zeros((self.n,self.n*2))

        # loop for each design variable
        for i in range(0,self.n*2):
            
            # multiply each dA matrix with gamma vector, notice negative sign
            self.dA_gamma[:,i] = -np.matmul(self.dA[:,:,i],self.gamma)
            
    
        # this gets rid of a meaningless/annoying dimension 1
        self.dA_gamma = np.squeeze(self.dA_gamma)
        
        # add the dB vectors to get f
        self.f = self.dA_gamma + self.dB
        
        #print("f0", self.f[:,0])


    # multiply the vT and f terms
    def calc_vTf(self):

        # transpose v vector
        self.vT = self.v.transpose()

        # initialize empty array of appropriate size
        self.vTf= np.zeros((1,self.n*2))

        # loop for each design variable
        for i in range(0,self.n*2):

            # calculate vTf for f layer. (vT stays the same)
            self.vTf[0,i] = np.matmul(self.vT,self.f[:,i])
        
        #print("vTf[:,0]", self.vTf[:,0])


    # calculate the gradient
    def calc_gradient(self):
        
        # calculate the final gradients
        self.gradient = self.vTf + self.partial_CL

        #print("gradient", "\n",self.gradient.transpose())

    
    # make data presentable
    def make_pretty_table(self):
        
        # initalize empty list
        labels = []
        
        # make a label for each design variable
        for i in range(0, self.n):
            labels.append("x"+ str(i))
        for i in range(0, self.n):
            labels.append("y"+ str(i))

        # tabulate data, headers = "keys" displays the lists in column form
        print("\n", tabulate({"design variable": labels , "CL gradient": self.gradient.transpose()}, floatfmt=".14f",headers = "keys", stralign = "right"), "\n")
        

    def run(self):
        self.calc_g()
        self.calc_v()
        self.calc_f()
        self.calc_vTf()
        self.calc_gradient()
        self.make_pretty_table()

if __name__ == "__main__":
    # take start time
    tstart = time.time()

    # create an Adjoint object of the input airfoil
    gradient1 = Adjoint("airfoil_adjoint/2412_200.json")

    # run the program
    gradient1.run()

    # take end time and display total computation time
    comp_time = (time.time()-tstart)
    print("Computation Time: ", round(comp_time, 4), " seconds")
    print("done")