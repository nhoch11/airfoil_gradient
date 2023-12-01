# import necessary packages
import json
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# set precision to 15 digits of precision
np.set_printoptions(precision=15)

class Primal:
	#__init__ called when an instance of a class is created
    def __init__(self,input_file): #file is a json
        # self.input_file stores the name of the json file. ex: "2412_200.json"
        self.input_file = input_file
        
   
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
                        info.append([float(coordinate) for coordinate in line.split()])
                        # close the file
                    f.close()
            # store airfoil points as attribute self.mypoints        
            self.mypoints = np.array(info)
            

    # this function is needed for my modified json containing all alphas at which to analyze airfoil
    def get_alpha(self):
        # Read in the json file
        json_string = open(self.input_file).read()
        # look for an "alpha[deg]" key in json
        input_dictionary = json.loads(json_string)
        #store array of alpha values
        self.alpha = np.radians(input_dictionary["alpha[deg]"])	


    # this function gets the value of v infinity from the json. (only done once)
    def get_v_inf(self):
        # Read in the json (jayson) file
        json_string = open(self.input_file).read()
        # make a json dictionary
        input_dictionary = json.loads(json_string)
        self.v_inf = input_dictionary["freestream_velocity"]
        
    # function to get control points 
    def get_Control_Points(self):         
        self.cp = (self.mypoints[:-1] + self.mypoints[1:]) / 2.0 
        
    # function to get a p matrix. This is called within the A matrix function 
    def get_P_matrix(self, jpoint,j1point,cpoint):
        #Calculate length of jth panel (l_j) (4.23)
        #get x and y value of point j
        x_j = jpoint[0] #check this!!!!!!!!!!!!!!!!!!!!!
        y_j = jpoint[1]
        #get x and y value of point j+1
        x_j1 = j1point[0] 
        y_j1 = j1point[1]
        # get x and y value at control point i
        x = cpoint[0]
        y = cpoint[1]

        #calculate l_j at the current j iteration
        l_j = np.sqrt((x_j1 - x_j)**2 + (y_j1 - y_j)**2)
        
        # these matrices make calculating xi and eta a bit easier
        v_mat = np.array([[(x_j1 - x_j),(y_j1 - y_j)],[(-y_j1 + y_j), (x_j1 - x_j)]])
        w_mat = np.array([[(x - x_j)],[(y - y_j)]])
        
        # a column vector containing xi an eta from matrix multiplication
        xieta = (1/l_j)*np.matmul(v_mat,w_mat)
        
        # pull xi and etea from the xieta column vector. make them floats, 1x1 not arrays
        xi = float(xieta[0])
        eta = float(xieta[1])
    
        #calculate phi and psi (4.25 - 4.26)
        phi = float(np.arctan2(eta*l_j, (eta**2 + xi**2 - xi*l_j)))
        psi = float(0.5*np.log((xi**2 + eta**2)/((xi-l_j)**2 + eta**2)))
        
        #calculate our P matrix (4.27)

        # greek matrix helps break up the pmatrix calculation for coding
        greek_mat = np.array([[((phi*(l_j - xi)) + (eta*psi)),((xi*phi) - (eta*psi))],[((eta*phi) - ((l_j - xi)*psi) - l_j),((-eta*phi) - (xi*psi) + l_j)]])
        # noticed pmatrix uses the transpose of a previously used matrix
        transposeVmat= np.transpose(v_mat)
        # matrix multiplication
        AB = np.matmul(transposeVmat,greek_mat)
        # output a p matrix. this changes multiple times for each instance of a class, so I didn't make it a class attribute
        pmatrix = (1/(2*np.pi*l_j*l_j))*AB
        #return pmatrix for use in the current A matrix iteration
        return pmatrix

    # function to get the a matrix
    def get_A_matrix(self):
        # get the number of points of the airfoil data
        n = len(self.mypoints)
        # make an empty n by n matrix
        A = np.zeros((n,n))
        # i loop is for control points
        for i in range(0, n-1):
            # j loop is for airfoil data points
            for j in range(0, n-1):
                #get x and y value of point j
                x_i = self.mypoints[i, 0]
                y_i = self.mypoints[i, 1]
                #get x and y value of point j+1
                x_i1 = self.mypoints[i+1, 0]
                y_i1 = self.mypoints[i+1, 1]
                l_i = np.sqrt((x_i1 - x_i)**2 + (y_i1 - y_i)**2)
                # get a p matrix for the current iteration
                P = self.get_P_matrix(self.mypoints[j],self.mypoints[j+1], self.cp[i] )
                #populate current A matrix location
                A[i,j] = A[i,j] + ((x_i1-x_i)/l_i)*P[1,0] - ((y_i1-y_i)/l_i)*P[0,0]
                # adjust j+1 A matrix location
                A[i,j+1] = A[i,j+1] + ((x_i1-x_i)/l_i)*P[1,1] - ((y_i1-y_i)/l_i)*P[0,1]
                
        #put 1 in the first and last columns of the last row	
        A[n-1,0] = A[n-1,n-1] = 1.0
        # make attribute for Amatrix
        self.Amatrix = A
        print(self.Amatrix)

    # this function makes a B matrix    
    def get_B_matrix(self):
        #Initialize B matrix
        n = len(self.mypoints)
        B = np.zeros((n,1))
        # this loop populates the B matrix
        for i in range(0,n-1):
            x_i = self.mypoints[i,0]
            y_i = self.mypoints[i,1]
            x_i1 = self.mypoints[i+1,0]
            y_i1 = self.mypoints[i+1, 1]
            l_i = np.sqrt((x_i1 - x_i)**2 + (y_i1 - y_i)**2)
            b_input = (((y_i1 - y_i)*np.cos(self.alpha) - (x_i1 - x_i)*np.sin(self.alpha))/l_i)
            B[i] = b_input
        # store B matrix in b matrix attribute
        self.Bmatrix = B	

    # this function solves for gammas
    def get_gamma(self):
        #Solve for gamma
        self.gamma = np.linalg.solve(self.Amatrix,(self.v_inf*self.Bmatrix))
        print(" gamma vector ",self.gamma)

    # Solve for ~CL (4.36)
    def get_CL(self):
        n = len(self.mypoints)
        # initial value for CL

        # the points are technically nondimensionalized as x/c and y/c, so no chord is needed (or chord = 1) in eq 4.36
        myCL = 0
        for i in range(0,n-1):
            x_i = self.mypoints[i,0]
            y_i = self.mypoints[i,1]
            x_i1 = self.mypoints[i+1,0]
            y_i1 = self.mypoints[i+1, 1]
            l_i = np.sqrt((x_i1 - x_i)**2 + (y_i1 - y_i)**2)
            inputCL = (l_i*float((self.gamma[i]) + float(self.gamma[i+1])))/self.v_inf
            myCL = myCL + inputCL
        # store CL value as attribute
        self.CL = myCL


    # the run fucntion takes the name of the json file and create and initiallizes the instance of the class, 
    # and gets CL, Cmle, and Cmc4 values at each alpha value instance.

    #****Note: I modified the json file format so I can evaluate all alpha angles in one instance of a class
    def run(self):
        # get airfoil points (to be used for all alpha angles in this object)
        
        data = np.zeros((1,2))  
        
        self.get_airfoil_points()
        self.get_alpha()
        self.get_v_inf()
        self.get_Control_Points()
        self.get_A_matrix()
        self.get_B_matrix()
        self.get_gamma()
        self.get_CL()
        #populate data array
        data = [np.degrees(self.alpha), self.CL]
    
        # make data presentable
        airfoilLabel = str(self.input_file)
        # get rid of ending junk
        airfoilLabel1 = airfoilLabel.replace("_200.json"," ")
        print()
        print("Data for NACA", airfoilLabel1)
        print()
        print ("alpha(deg)     C_L")
        print(data[0], "          ", data[1])

        return self.Amatrix, self.Bmatrix, self.mypoints, self.v_inf, self.gamma





if __name__ == "__main__":
    airfoil1 = Primal("2412_10.json") 
    airfoil1.run() 