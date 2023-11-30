# import necessary packages
import json
import numpy as np
import math
import matplotlib.pyplot as plt
from tabulate import tabulate
from cmath import *
import time

# set precision to 15 digits of precision
np.set_printoptions(precision=15)

class Complex_Step:
	#__init__ called when an instance of a class is created
    def __init__(self,input_file, var_index, step): #file is a json
        # self.input_file stores the name of the json file. ex: "2412_200.json"
        self.input_file = input_file
        #index of variable that derivative of A matrix will be taken with respect to
        self.var_index = var_index
        #step size
        self.step = step

#func to get variables from input file
    def get_airfoil_points(self):
        # Read in the json (jayson) file
        json_string = open(self.input_file).read()
        # make a json dictionary
        input_dictionary = json.loads(json_string)
        # NACAfiles gets the txt file associated with 'airfoils' key in the json
        NACAfiles = input_dictionary["airfoils"]
        # the list associated with airfoils key may have multiple files (airfoils)
        for file in NACAfiles:
            # the list associated with airfoils key may have multiple files (airfoils)
            with open(file, "r") as f: # "r" means open for reading
                    # create an empty list
                    info = []
                    # go through each line in the NACAfile airfoil
                    for line in f:
                        # take the text and create numerical values for each chunk
                        info.append([complex(coordinate) for coordinate in line.split()])
                        # close the file
                    f.close()
            # store airfoil points as attribute self.mypoints        
            self.mypoints = np.array(info)
            
        # store number of points as self.n
        self.n = len(self.mypoints)
        #print(self.mypoints)
        # split the airfoil points and make X_a vector of x coords then y coords
        self.x_vals = self.mypoints[:,0]
        self.y_vals = self.mypoints[:,1]
        
        self.X_a = np.concatenate([self.x_vals.reshape(-1,1), self.y_vals.reshape(-1,1)])

        #print("X_a", self.X_a)
        return self.n

    # modify the airfoil points by stepping the correct variable up or down
    def change_points(self):
            self.X_a[self.var_index] = self.X_a[self.var_index] + complex(0.0, self.step)


    # function to get control points 
    def get_Control_Points(self):    
        self.xycp = np.zeros((2*(self.n-1),1),dtype=complex)
        for i in range(0,self.n-1):
            self.xycp[i] = (self.X_a[i+1]+self.X_a[i])/2
            self.xycp[i+self.n-1] = (self.X_a[i+self.n+1]+self.X_a[i+self.n])/2

        #print("xycp",self.xycp)

        #print("xycp shape: ",np.shape(self.xycp))
        #print("X_a shape: ",np.shape(self.X_a))

    def get_alpha_and_Vinf(self):
        # Read in the json file
        json_string = open(self.input_file).read()
        # look for an "alpha[deg]" key in json
        input_dictionary = json.loads(json_string)
        #store array of alpha values
        self.alpha = np.radians(input_dictionary["alpha[deg]"])
        self.Vinf = complex(input_dictionary["freestream_velocity"])	

        #print("alpha", self.alpha)
        #print("Vinf", self.Vinf)


    # calculate and store all l_i/l_j values as l_k. k so we dont get mixed up with i or j.
    def calc_l_k(self):
        # initialize l_k of zeros. the length will be one less than n
        self.l_k = np.zeros((self.n-1,1),dtype=complex)
        # calculate l_k
        for k in range(0,self.n-1):
            self.l_k[k] = np.sqrt((self.X_a[k+1]-self.X_a[k])**2 + (self.X_a[k+1+self.n]-self.X_a[k + self.n])**2)

        #print("l_k shape", np.shape(self.l_k))
        #print("l_k", self.l_k)

    # calculate and store all xi values   NOTE there may be a way to save computation by only calculating the first 2x2 once and using for all the i iterations
    def calc_xi_eta(self):
        # initalize n-1 by n-1 matrix of zeros. indexing will be j,i. jth panel influence on ith control point. this is how the P matrix is indexed
        self.xi = np.zeros((self.n-1,self.n-1),dtype=complex)
        self.eta = np.zeros((self.n-1,self.n-1),dtype=complex)
        # build xi and eta matrix
        for j in range(0,self.n-1):
            for i in range(0,self.n-1): 
                # note the xycp y values are indexed by j + n and j + (n-1)  there are 1 fewer control points than nodes #########################issue might be here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                self.xi[j,i] = ((self.X_a[j+1]-self.X_a[j])*(self.xycp[i]-self.X_a[j])+(self.X_a[j+self.n+1]-self.X_a[j+self.n])*(self.xycp[i+self.n-1]-self.X_a[j+self.n]))/self.l_k[j]
                self.eta[j,i] = (-(self.X_a[j+self.n+1]-self.X_a[j+self.n])*(self.xycp[i]-self.X_a[j])+(self.X_a[j+1]-self.X_a[j])*(self.xycp[i+self.n-1]-self.X_a[j+self.n]))/self.l_k[j]

        #print("xi",self.xi)
        #print("eta", self.eta)
    def c_atan2(self,x,y):
        a=x.real
        b=x.imag
        c=y.real
        d=y.imag
        return complex(math.atan2(a,c),(c*b-a*d)/(a**2+c**2))

    # calculate and store phi values for each ji index
    def calc_phi_psi(self):
        # initalize n-1 by n-1 matrix of zeros. indexing will be j,i. jth panel influence on ith control point. this is how the P matrix is indexed
        self.phi = np.zeros((self.n-1,self.n-1),dtype=complex)
        self.psi = np.zeros((self.n-1,self.n-1),dtype=complex)
        # build phi and psi matrix  indexing will be j,i. jth panel influence on ith control point. this is how the P matrix is indexed
        for j in range(0,self.n-1):
            for i in range(0,self.n-1): 
                self.phi[j,i] = self.c_atan2(self.eta[j,i]*self.l_k[j], self.eta[j,i]**2 + self.xi[j,i]**2 - self.xi[j,i]*self.l_k[j])
                self.psi[j,i] = 0.5*np.log((self.xi[j,i]**2 + self.eta[j,i]**2)/((self.xi[j,i]-self.l_k[j])**2 + self.eta[j,i]**2))
        #print("phi", self.phi)
        #print("psi", self.psi)

    # calculate and store sub P matrices ( matrices in eq 4.26 Handbook)
    def calc_P(self):
        # initialize 4-dimensional matrices ( a matrix of matrices)  first index is row of matrices, second is column (this results in a sub matrix), 3rd is the row of the sub matrix, 4th is the column resulting in a value.
        # subP_xy is the middle left hand 2 by 2 matrix in eq 4.26
        self.subP_xy = np.zeros((self.n-1,self.n-1,2,2),dtype=complex)
        # subP_greek is the right hand 2 by matrix in eq 4.26. greek because its got all the xi, eta, phi, psi, etc terms
        self.subP_greek = np.zeros((self.n-1, self.n-1, 2,2),dtype=complex)

        self.subPxygreek = np.zeros((self.n-1, self.n-1, 2,2),dtype=complex)

        self.P = np.zeros((self.n-1, self.n-1,2,2),dtype=complex)

        # the j loop is for the influence of the jth panel 
        for j in range(0,self.n-1):
            # the i loop is for the influence of the jth panel on every ith control point (there are n-1 control points)
            for i in range(0,self.n-1):
                # build subP xy matrices. The subP xy includes the scalar term (1/(2*np.pi*self.l_k[j]**2)) ep 4.26 handbook
                self.subP_xy[j,i,0,0] = (self.X_a[j+1] - self.X_a[j])
                self.subP_xy[j,i,0,1] = (-(self.X_a[j + self.n+1] - self.X_a[j + self.n]))
                self.subP_xy[j,i,1,0] = (self.X_a[j + self.n+1] - self.X_a[j + self.n])
                self.subP_xy[j,i,1,1] = (self.X_a[j+1] - self.X_a[j])

                # build subP_greek matrices
                self.subP_greek[j,i,0,0] = self.l_k[j]*self.phi[j,i]- self.xi[j,i]*self.phi[j,i] + self.eta[j,i]*self.psi[j,i]
                self.subP_greek[j,i,0,1] =                            self.xi[j,i]*self.phi[j,i] - self.eta[j,i]*self.psi[j,i]
                self.subP_greek[j,i,1,0] =  self.eta[j,i]*self.phi[j,i] - self.l_k[j]*self.psi[j,i]  +self.xi[j,i]*self.psi[j,i] - self.l_k[j]
                self.subP_greek[j,i,1,1] = -self.eta[j,i]*self.phi[j,i]                              -self.xi[j,i]*self.psi[j,i] + self.l_k[j]
        
            # P matrix,   This performs a matrix multiplication (2x2) by (2,2) for each j,i iteration. (it executes equation 4.26 for each j,i combination)
            self.subPxygreek[j] = np.matmul(self.subP_xy[j],self.subP_greek[j])
            self.P[j] = (1/(2*np.pi*(self.l_k[j]**2)))*self.subPxygreek[j]
            #print("subP_greek[0,0]", self.subP_greek[0,0])
        #print("P matrix", "\n", self.P)

    # function to get the a matrix this matches the 2412 example A matrix solution to 14 or 15 sig figs
    def calc_A_matrix(self):
        
        # make an empty n by n matrix
        self.A = np.zeros((self.n,self.n),dtype=complex)
        # i loop is for control points
        for i in range(0, self.n-1):   # python is end exclusive
            # j loop is for airfoil data points
            for j in range(0, self.n-1):  # python is end exclusive
                
                #populate current A matrix location
                self.A[i,j] = self.A[i,j] + ((self.X_a[i+1]-self.X_a[i])/self.l_k[i])*self.P[j,i,1,0] - ((self.X_a[i +self.n+1]- self.X_a[i + self.n])/self.l_k[i])*self.P[j,i,0,0]
                # adjust j+1 A matrix location
                self.A[i,j+1] = self.A[i,j+1] + ((self.X_a[i +1]-self.X_a[i])/self.l_k[i])*self.P[j,i,1,1] - ((self.X_a[i +self.n+1]-self.X_a[i + self.n])/self.l_k[i])*self.P[j,i,0,1]
                
        # put 1 in the first and last columns of the last row	
        self.A[self.n-1,0] = self.A[self.n-1,self.n-1] = 1.0
        
        #print("A matrix", self.A)

    # this function makes a B matrix    
    def calc_B_matrix(self):
        #Initialize B matrix
        self.B = np.zeros((self.n,1),dtype=complex)
        # this loop populates the B matrix
        for i in range(0,self.n-1):
            self.B[i] = self.Vinf*((self.X_a[i+self.n+1] - self.X_a[i+self.n])*complex(np.cos(self.alpha), 0) - (self.X_a[i+1] - self.X_a[i])*complex(np.sin(self.alpha), 0))/self.l_k[i]
            

        # multiply by vinf
        self.B
    
        #print("B matrix", "\n", self.B)

    # this function solves for gammas
    def calc_gamma(self):
        #Solve for gamma
        self.gamma = np.linalg.solve(self.A,self.B)
        #print(" gamma vector ","\n", self.gamma)

    def calc_CL(self):
        # NOTE we are not holding gamma constant here, so the complex step should give 
        # the total gradient for the particular design variable 
        #(if we did hold gamma constant, we would get the 'partial CL/ partial design variable' term)
        
        self.grad = complex(0,0)
        for i in range(0,self.n-1):
            self.grad = self.grad + (self.l_k[i]*(self.gamma[i] + self.gamma[i+1]))/self.Vinf
        

       

    def run(self):
        
        self.get_airfoil_points()
        self.change_points()
        self.get_Control_Points()
        self.get_alpha_and_Vinf()
        self.calc_l_k()
        self.calc_xi_eta()
        self.calc_phi_psi()
        self.calc_P()
        self.calc_A_matrix()
        self.calc_B_matrix()
        self.calc_gamma()
        self.calc_CL()
       
    

        return self.grad



if __name__ == "__main__":
    
    # set step size
    h = 1.0E-200
    # get number of design points from
    i = 0
    airfoil1 = Complex_Step("2412_200.json", i, h) 
    n = airfoil1.get_airfoil_points() 

    # take start time
    tstart = time.time()

    # iterate through all the design variables
    gradient = []

    for i in range(0, 2*n):
        # calculate the grad (the gradient for a single design variable) and append it to the list
        gradient.append(np.imag(Complex_Step("2412_200.json", i, h).run())/h )
        print("done with grad ", i)
    # initalize empty list
    labels = []
    
    # make a label for each design variable
    for i in range(0, n):
        labels.append("x"+ str(i))
    for i in range(0, n):
        labels.append("y"+ str(i))

    # tabulate data, headers = "keys" displays the lists in column form
    print("\n", tabulate({"design variable": labels , "CL gradient": gradient}, floatfmt=".14f", headers = "keys", stralign = "right"), "\n")

    # take end time and display total computation time
    comp_time = (time.time()-tstart)
    print("Computation Time: ", round(comp_time, 4), " seconds")
    print("done")