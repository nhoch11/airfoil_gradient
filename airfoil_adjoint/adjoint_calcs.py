# This method calculates and manually differentiates the A matrix

# import necessary packages
import json
import numpy as np
import matplotlib.pyplot as plt


# set precision to 15 digits of precision
np.set_printoptions(precision=15)

class adjoint_calcs:
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
            
        # store number of points as self.n
        self.n = len(self.mypoints)
        print("Airfoil has "+str(self.n)+" points")
        # split the airfoil points and make X_a vector of x coords then y coords
        self.x_vals = self.mypoints[:,0]
        self.y_vals = self.mypoints[:,1]
        # combine x and y values into a design variable vector
        self.X_a = np.concatenate([self.x_vals.reshape(-1,1), self.y_vals.reshape(-1,1)])


    # function to get control points 
    def get_Control_Points(self):    

        # initialize empty array
        self.xycp = np.zeros((2*(self.n-1),1))
        
        # create control points located halfway along each panel. assumes the trailing edge panel is negligibly small 
        for i in range(0,self.n-1):
            self.xycp[i] = (self.X_a[i+1]+self.X_a[i])/2
            self.xycp[i+self.n-1] = (self.X_a[i+self.n+1]+self.X_a[i+self.n])/2


    # read in freestream data
    def get_alpha_and_Vinf(self):
        # Read in the json file
        json_string = open(self.input_file).read()
        # look for an "alpha[deg]" key in json
        input_dictionary = json.loads(json_string)
        #store array of alpha values
        self.alpha = np.radians(input_dictionary["alpha[deg]"])
        self.Vinf = input_dictionary["freestream_velocity"]	


    # calculate and store all l_i/l_j values as l_k. k so we dont get mixed up with i or j.
    def calc_l_k(self):
        # initialize l_k of zeros. the length will be one less than n
        self.l_k = np.zeros((self.n-1,1))
        # calculate l_k
        for k in range(0,self.n-1):
            self.l_k[k] = np.sqrt((self.X_a[k+1]-self.X_a[k])**2 + (self.X_a[k+1+self.n]-self.X_a[k + self.n])**2)

    
    # calculate and store all xi values   NOTE there may be a way to save computation by only calculating the first 2x2 once and using for all the i iterations
    def calc_xi_eta(self):
        # initalize n-1 by n-1 matrix of zeros. indexing will be j,i. jth panel influence on ith control point. this is how the P matrix is indexed
        self.xi = np.zeros((self.n-1,self.n-1))
        self.eta = np.zeros((self.n-1,self.n-1))
        # build xi and eta matrix
        for j in range(0,self.n-1):
            for i in range(0,self.n-1): 
                # note the xycp y values are indexed by j + n and j + (n-1)  there are 1 fewer control points than nodes #########################issue might be here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                self.xi[j,i] = ((self.X_a[j+1]-self.X_a[j])*(self.xycp[i]-self.X_a[j])+(self.X_a[j+self.n+1]-self.X_a[j+self.n])*(self.xycp[i+self.n-1]-self.X_a[j+self.n]))/self.l_k[j]
                self.eta[j,i] = (-(self.X_a[j+self.n+1]-self.X_a[j+self.n])*(self.xycp[i]-self.X_a[j])+(self.X_a[j+1]-self.X_a[j])*(self.xycp[i+self.n-1]-self.X_a[j+self.n]))/self.l_k[j]


    # calculate and store phi values for each ji index
    def calc_phi_psi(self):
        # initalize n-1 by n-1 matrix of zeros. indexing will be j,i. jth panel influence on ith control point. this is how the P matrix is indexed
        self.phi = np.zeros((self.n-1,self.n-1))
        self.psi = np.zeros((self.n-1,self.n-1))
        # build phi and psi matrix  indexing will be j,i. jth panel influence on ith control point. this is how the P matrix is indexed
        for j in range(0,self.n-1):
            for i in range(0,self.n-1): 
                self.phi[j,i] = np.arctan2(self.eta[j,i]*self.l_k[j], (self.eta[j,i]**2 + self.xi[j,i]**2 - self.xi[j,i]*self.l_k[j]))
                self.psi[j,i] = 0.5*np.log((self.xi[j,i]**2 + self.eta[j,i]**2)/((self.xi[j,i]-self.l_k[j])**2 + self.eta[j,i]**2))
        

    # calculate and store sub P matrices ( matrices in eq 4.26 Handbook)
    def calc_P(self):
        # initialize 4-dimensional matrices ( a matrix of matrices)  first index is row of matrices, second is column (this results in a sub matrix), 3rd is the row of the sub matrix, 4th is the column resulting in a value.
        
        # each subP_xy is the middle left hand 2 by 2 matrix in eq 4.26
        self.subP_xy = np.zeros((self.n-1,self.n-1,2,2))
        # each subP_greek is the right hand 2 by 2 matrix in eq 4.26. greek because its got all the xi, eta, phi, psi, etc terms
        self.subP_greek = np.zeros((self.n-1, self.n-1, 2,2))
        # when each subPxy and subPgreek are multiplied, result is a 2 x 2
        self.subPxygreek = np.zeros((self.n-1, self.n-1, 2,2))

        self.P = np.zeros((self.n-1, self.n-1,2,2))

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
                self.subP_greek[j,i,0,0] = (self.l_k[j]- self.xi[j,i])*self.phi[j,i] + self.eta[j,i]*self.psi[j,i]
                self.subP_greek[j,i,0,1] = self.xi[j,i]*self.phi[j,i] - self.eta[j,i]*self.psi[j,i]
                self.subP_greek[j,i,1,0] = self.eta[j,i]*self.phi[j,i] - (self.l_k[j]-self.xi[j,i])*self.psi[j,i] -self.l_k[j]
                self.subP_greek[j,i,1,1] = -self.eta[j,i]*self.phi[j,i] -self.xi[j,i]*self.psi[j,i] + self.l_k[j]
        
            # P matrix,   This performs a matrix multiplication (2x2) by (2,2) for each j,i iteration. (it executes equation 4.26 for each j,i combination)
            self.subPxygreek[j] = np.matmul(self.subP_xy[j],self.subP_greek[j])
            self.P[j] = (1/(2*np.pi*(self.l_k[j]**2)))*self.subPxygreek[j]
    

    # function to get the a matrix this matches the 2412 example A matrix solution to 14 or 15 sig figs
    def calc_A_matrix(self):
        
        # make an empty n by n matrix
        self.A = np.zeros((self.n,self.n))
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
        self.B = np.zeros((self.n,1))
        # this loop populates the B matrix
        for i in range(0,self.n-1):
            self.B[i]  = ((self.X_a[i+self.n+1] - self.X_a[i+self.n])*np.cos(self.alpha) - (self.X_a[i+1] - self.X_a[i])*np.sin(self.alpha))/self.l_k[i]
        # multiply by vinf
        self.B = self.Vinf*self.B
        #print("B matrix", "\n", self.B)


    # this function solves for gammas
    def calc_gamma(self):
        #Solve for gamma
        self.gamma = np.linalg.solve(self.A,self.B)
        #print(" gamma vector ",self.gamma)


    # calculate the coefficient of lift 
    def calc_CL(self):
        
        # initialize value of zero
        self.CL = 0.0

        # sum CL terms
        for i in range(0, self.n-1):
            self.CL = self.CL + self.l_k[i]*(self.gamma[i]+ self.gamma[i+1])/self.Vinf
        
        #print("CL = ", self.CL)


    # make an identity matrix to be indexed for the dx/dX_a and the dy/dX_a
    def calc_dxy(self):

        # the derivative of x0 WRT all design variables corresponds to the 0th row 
        # of this identiy matrix and so on
        self.dxy = np.identity(2*self.n)
   

    # make an vector containting the dxcp / dX_a
    def calc_dxycp(self):
        # dxcp is a 2*(number of cpoints) x 2*(number of nodes)  example: 10 airfoil points, size is 18x20
        self.dxycp = np.zeros((2*(self.n-1),2*self.n))
        # for ease of indexing, dxcp1 will be the row rather than the column.           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Note this!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #row 0    : [dxcp0/dx0, dxcp0/dx1, dxcp0/dx2,... dxcp0/dy0, ... dxcp0/dyn]
        #row 1    : [dxcp1/dx0, dxcp1/dx1, dxcp1/dx2,... dxcp1/dy0, ... dxcp1/dyn]
        # ....
        #row (n-1): [dycp0/dx0, dycp0/dx1, dycp0/dx2,... dycp0/dy0, ... dycp0/dyn]
        #row (n)  : [dycp1/dx0, dycp1/dx1, dycp1/dx2,... dycp1/dy0, ... dycp1/dyn]
        # ....

        # while there are n points, there are n-1 control points
        for i in range(0,self.n-1): # python is end exclusive
            # derivative of control points with respect to x coordinates
            self.dxycp[i] = (self.dxy[i+1] + self.dxy[i])/2
            # the y index is i + n -1 (-1 because there are 1 fewer control points than nodes),  the dxy indices have to jump ahead so we dont compute dycp0 with a dx_n/dX_a
            self.dxycp[i+self.n-1] = (self.dxy[i+self.n+1] + self.dxy[i+self.n])/2


    # calculate and store dl_k/dX_a 
    def calc_dl_k(self):
        # initalize matrix, size should be n-1 rows with 2n columns
        self.dl_k =  np.zeros((self.n-1,2*self.n))
        for k in range(0,self.n-1): # python is end exclusive
            # split the big equation into 2 parts
            junk1 = 0.5*((self.X_a[k+1]-self.X_a[k])**2 + (self.X_a[k+self.n+1]-self.X_a[k+self.n])**2)**(-0.5)
            junk2 = (2*(self.X_a[k+1]-self.X_a[k])*(self.dxy[k+1]-self.dxy[k]) + 2*(self.X_a[k+self.n+1]-self.X_a[k+self.n])*(self.dxy[k+self.n+1]-self.dxy[k+self.n]))
            # combine 2 parts
            self.dl_k[k] = junk1*junk2 


    # calculate and store dxi/dX_a and d_eta/dX_a
    def calc_dxi_and_deta(self):
        # initialize, size of dxi and deta should be 2(n-1)x 2(n-1) x 2n
        self.dxi = np.zeros(((self.n-1),(self.n-1), 2*self.n))
        self.deta = np.zeros(((self.n-1),(self.n-1), 2*self.n))
        # index is j,i,:
        for j in range(0,self.n-1):
            for i in range(0,self.n-1):
                # break up dxi[j,i] into manageable bits
                xijunk1 = ((self.dxy[j+1] -self.dxy[j])*(self.xycp[i]-self.X_a[j]))/self.l_k[j]

                xijunk2 = ((self.X_a[j+1]-self.X_a[j])*(self.dxycp[i]-self.dxy[j]))/self.l_k[j]
 
                xijunk3 = ((self.dxy[j+self.n+1] -self.dxy[j+self.n])*(self.xycp[i+self.n-1]-self.X_a[j+self.n]))/self.l_k[j]

                xijunk4 = ((self.X_a[j+self.n+1]-self.X_a[j+self.n])*(self.dxycp[i+self.n-1]-self.dxy[j+self.n]))/self.l_k[j]

                xijunk5 = -(self.dl_k[j]*(self.X_a[j+1]-self.X_a[j])*(self.xycp[i]-self.X_a[j]))/(self.l_k[j]**2)

                xijunk6 = -(self.dl_k[j]*(self.X_a[j+self.n+1]-self.X_a[j+self.n])*(self.xycp[i+self.n-1]-self.X_a[j+self.n]))/(self.l_k[j]**2)
               
                # combine bits for d xi
                self.dxi[j,i] = xijunk1 + xijunk2 + xijunk3 + xijunk4 + xijunk5 + xijunk6

                # break up dxi[j,i] into manageable bits
                etajunk1 = -((self.dxy[j+self.n+1] -self.dxy[j+self.n])*(self.xycp[i]-self.X_a[j]))/self.l_k[j]

                etajunk2 = -((self.X_a[j+self.n+1]-self.X_a[j+self.n])*(self.dxycp[i]-self.dxy[j]))/self.l_k[j]
 
                etajunk3 = ((self.dxy[j+1] -self.dxy[j])*(self.xycp[i+self.n-1]-self.X_a[j+self.n]))/self.l_k[j]

                etajunk4 = ((self.X_a[j+1]-self.X_a[j])*(self.dxycp[i+self.n-1]-self.dxy[j+self.n]))/self.l_k[j]

                etajunk5 = (self.dl_k[j]*(self.X_a[j+self.n+1]-self.X_a[j+self.n])*(self.xycp[i]-self.X_a[j]))/(self.l_k[j]**2)

                etajunk6 = -(self.dl_k[j]*(self.X_a[j+1]-self.X_a[j])*(self.xycp[i+self.n-1]-self.X_a[j+self.n]))/(self.l_k[j]**2)

                # combine bits for d eta
                self.deta[j,i] =  etajunk1 + etajunk2 + etajunk3 + etajunk4 + etajunk5 + etajunk6
        

    # calculate and store dphi /dX_a and dpsi/dX_a
    def calc_dphi_and_dpsi(self):
        # size should be the same as dxi and deta
        self.dphi = np.zeros(((self.n-1),(self.n-1), 2*self.n))
        self.dpsi = np.zeros(((self.n-1),(self.n-1), 2*self.n))

        # each ji index has its own dphi an dpsi vector
        for j in range(0,self.n-1):
            for i in range(0,self.n-1):
                # break up dphi equation into manageable bits
                dphijunk1 = (self.deta[j,i]*self.l_k[j] + self.eta[j,i]*self.dl_k[j])*((self.eta[j,i]**2 + self.xi[j,i]**2 - self.xi[j,i]*self.l_k[j])/((self.eta[j,i]**2 + self.xi[j,i]**2 - self.xi[j,i]*self.l_k[j])**2 + (self.eta[j,i]*self.l_k[j])**2))
                dphijunk2 = (2*self.eta[j,i]*self.deta[j,i] + self.dxi[j,i]*(2*self.xi[j,i]-self.l_k[j])-self.xi[j,i]*self.dl_k[j])*((-self.eta[j,i]*self.l_k[j])/((self.eta[j,i]**2 + self.xi[j,i]**2 -self.xi[j,i]*self.l_k[j])**2 + (self.eta[j,i]*self.l_k[j])**2))
                # combine the dphi bits
                self.dphi[j,i] = dphijunk1 + dphijunk2

                # break up dpsi equation into manageable bits
                dpsijunk1 = 1/((self.xi[j,i]**2 + self.eta[j,i]**2)*((self.xi[j,i]-self.l_k[j])**2 + self.eta[j,i]**2))
                dpsijunk2 = ((self.xi[j,i]-self.l_k[j])**2 +self.eta[j,i]**2)*(self.xi[j,i]*self.dxi[j,i] +self.eta[j,i]*self.deta[j,i]) - (((self.xi[j,i]-self.l_k[j])*(self.dxi[j,i]-self.dl_k[j]))+self.eta[j,i]*self.deta[j,i])*(self.xi[j,i]**2 + self.eta[j,i]**2)
                # combine the bits
                self.dpsi[j,i] = dpsijunk1*dpsijunk2


    # calculate and store dP
    def calc_dP(self):
        # size is a (n-1) x (n-1) x 2 x 2 x 2n
        self.dP = np.zeros((self.n-1,self.n-1,2,2,2*self.n))
        

        # create empty d subP / dX_a matrices
        dsubP_xy = np.zeros((self.n-1,self.n-1,2,2,2*self.n))
        dsubP_greek = np.zeros((self.n-1,self.n-1,2,2,2*self.n))
        dpjunk1 = np.zeros((self.n-1,self.n-1,2,2,2*self.n))
        dpjunk2 = np.zeros((self.n-1,self.n-1,2,2,2*self.n))
        dpjunk3 = np.zeros((self.n-1,self.n-1,2,2,2*self.n))

        # each ji index will have quantity (2n) 2 by 2 matrices
        for j in range(0,self.n-1):
            for i in range(0,self.n-1):
                # first calculate the dsubP elements
                dsubP_xy[j,i,0,0] = (self.dxy[j+1]-self.dxy[j])
                dsubP_xy[j,i,0,1] = -(self.dxy[j+self.n+1]-self.dxy[j+self.n])
                dsubP_xy[j,i,1,0] = (self.dxy[j+self.n+1]- self.dxy[j+self.n])
                dsubP_xy[j,i,1,1] = (self.dxy[j+1]- self.dxy[j])

                # calculate dsubP_greek elements
                dsubP_greek[j,i,0,0] =  self.dl_k[j]*self.phi[j,i] + self.l_k[j]*self.dphi[j,i] - self.dxi[j,i]*self.phi[j,i] - self.xi[j,i]*self.dphi[j,i] + self.deta[j,i]*self.psi[j,i] + self.eta[j,i]*self.dpsi[j,i]  
                dsubP_greek[j,i,0,1] =                                                            self.dxi[j,i]*self.phi[j,i] + self.xi[j,i]*self.dphi[j,i] - self.deta[j,i]*self.psi[j,i] - self.eta[j,i]*self.dpsi[j,i]
                dsubP_greek[j,i,1,0] =  self.deta[j,i]*self.phi[j,i] + self.eta[j,i]*self.dphi[j,i] - self.dl_k[j]*self.psi[j,i] - self.l_k[j]*self.dpsi[j,i] + self.dxi[j,i]*self.psi[j,i] + self.xi[j,i]*self.dpsi[j,i] - self.dl_k[j]
                dsubP_greek[j,i,1,1] = -self.deta[j,i]*self.phi[j,i] - self.eta[j,i]*self.dphi[j,i]                                                           - self.dxi[j,i]*self.psi[j,i] - self.xi[j,i]*self.dpsi[j,i] + self.dl_k[j]

                # multiply the derivative l_k term by the non-derivative subPxygreek matrices
                # multiply each element of this ji 2x2 by the dl_k vector
                for k in range(0,2):
                    for l in range(0,2):
                        dpjunk1[j,i,k,l] = (-self.dl_k[j]/(np.pi*(self.l_k[j])**3))*self.subPxygreek[j,i,k,l]

                # this junk2 term has regular l_k term, derivative subPxy matrix, and regular subP greek matrix  
                for k in range(0, 2*self.n):
                    dpjunk2[j,i,:,:,k] = (1/(2*np.pi*(self.l_k[j]**2)))*np.matmul(dsubP_xy[j,i,:,:,k], self.subP_greek[j,i])
                    # this is [2 x 2 ][2 x 2 x 2n] = [2 x 2 x 2n]
                    # the junk3 term has regular l_k term, regular subPxy terms and derivative subP greek matrix
                    dpjunk3[j,i,:,:,k] = (1/(2*np.pi*(self.l_k[j]**2)))*np.matmul(self.subP_xy[j,i],dsubP_greek[j,i,:,:,k])
                
                # combine the 3 chain rule terms
                self.dP[j,i] = dpjunk1[j,i] + dpjunk2[j,i] + dpjunk3[j,i]
        

    # calculate thhe dA /dX_A matrix
    def calc_dA(self):

        # make an empty n by n by 2n matrix 
        self.dA = np.zeros((self.n,self.n,2*self.n))
        
        # in the regular A matrix, the ji index switches to ij. this is reflected here.
        for i in range(0, self.n-1):
            for j in range(0, self.n-1):
                # at each ij iteration, we calculate a dAij and dAi(j+1)

                # break dAij into smaller chunks
                dAj_junk1 = ((self.l_k[i]*(self.dxy[i+1]-self.dxy[i])- self.dl_k[i]*(self.X_a[i+1]-self.X_a[i]))/(self.l_k[i]**2))*self.P[j,i,1,0]
                dAj_junk2 = ((self.X_a[i+1]-self.X_a[i])/self.l_k[i])*self.dP[j,i,1,0]
                dAj_junk3 = -((self.l_k[i]*(self.dxy[i+self.n+1]-self.dxy[i+self.n])- self.dl_k[i]*(self.X_a[i+self.n+1]-self.X_a[i+self.n]))/(self.l_k[i]**2))*self.P[j,i,0,0]
                dAj_junk4 = -((self.X_a[i+self.n+1]-self.X_a[i+self.n])/self.l_k[i])*self.dP[j,i,0,0]
                
                # combine the smaller chunks
                self.dA[i,j] = self.dA[i,j] + dAj_junk1 + dAj_junk2 + dAj_junk3 + dAj_junk4
                
                # break dAi(j+1) into smaller chunks
                dAj1_junk1 = ((self.l_k[i]*(self.dxy[i+1]-self.dxy[i])- self.dl_k[i]*(self.X_a[i+1]-self.X_a[i]))/(self.l_k[i]**2))*self.P[j,i,1,1]
                dAj1_junk2 = ((self.X_a[i+1]-self.X_a[i])/self.l_k[i])*self.dP[j,i,1,1]
                dAj1_junk3 = -((self.l_k[i]*(self.dxy[i+self.n+1]-self.dxy[i+self.n])- self.dl_k[i]*(self.X_a[i+self.n+1]-self.X_a[i+self.n]))/(self.l_k[i]**2))*self.P[j,i,0,1]
                dAj1_junk4 = -((self.X_a[i+self.n+1]-self.X_a[i+self.n])/self.l_k[i])*self.dP[j,i,0,1]

                # combine the smaller chunks                
                self.dA[i,j+1] = self.dA[i,j+1] + dAj1_junk1 + dAj1_junk2 + dAj1_junk3 + dAj1_junk4
                    
        #print("dA matrix [:,:, 0]", "\n", self.dA[:,:,0])


    # calculate thhe dB /dX_A matrix
    def calc_dB(self):

        # make an empty n by 2n matrix
        self.dB = np.zeros((self.n,2*self.n))
        # i loop goes through each airfoil point
        for i in range(0, self.n-1):
            # break up the dB term into smaller chunks
            dbjunk1 = (self.Vinf/self.l_k[i]**2)
            dbjunk2 = self.l_k[i]*(self.dxy[i+self.n+1]-self.dxy[i+self.n])*np.cos(self.alpha) 
            dbjunk3 = -self.dl_k[i]*(self.X_a[i+self.n+1]-self.X_a[i+self.n])*np.cos(self.alpha)
            dbjunk4 = -self.l_k[i]*(self.dxy[i+1]-self.dxy[i])*np.sin(self.alpha)
            dbjunk5 = self.dl_k[i]*(self.X_a[i+1]-self.X_a[i])*np.sin(self.alpha)
            
            # combine the smaller chunks with the appropriate operation
            self.dB[i] = dbjunk1*(dbjunk2+dbjunk3+dbjunk4+dbjunk5)
        
        #print("dB", "\n", self.dB[:,0])

    # calculate the partial of CL WRT the design variables 
    def calc_partial_CL(self):

        # initialize partial CL 
        self.partial_CL = np.zeros((1,2*self.n))
        
        # in this partial derivative, we treat gamma as independent of the x and y design variables. 
        # the derivative of x and y variables is accounted for in the dl_k term
        for i in range(0,self.n-1):
            self.partial_CL = self.partial_CL + ((self.dl_k[i])*(self.gamma[i+1]+self.gamma[i]))/self.Vinf

        #print("partial_CL[:,0]", self.partial_CL[:,0])

        
    def run(self):
        print("-"*40)
        print("Start Adjoint Gradient Calculation")
        self.get_airfoil_points()
        self.get_Control_Points()
        self.get_alpha_and_Vinf()
        self.calc_l_k()
        self.calc_xi_eta()
        self.calc_phi_psi()
        self.calc_P()
        self.calc_A_matrix()
        self.calc_B_matrix()
        print("Solving for gamma")
        self.calc_gamma()
        print("gamma found")
        self.calc_CL()
        self.calc_dxy()
        self.calc_dxycp()
        self.calc_dl_k()
        print("Working on the dGreeks...")
        self.calc_dxi_and_deta()
        self.calc_dphi_and_dpsi()
        print("Building dP...")
        self.calc_dP()
        print("Building dA...")
        self.calc_dA()
        print("done with dA")
        self.calc_dB()
        self.calc_partial_CL()
        print("-"*40)
        
        # some of these returned values were used when checking the code against finite difference
        return self.A, self.B, self.mypoints, self.n, self.l_k, self.Vinf, self.gamma, self.dA, self.dB, self.partial_CL, self.CL


if __name__ == "__main__":
    diff_A = adjoint_calcs("2412_10.json") 
    diff_A.run() 
    print("done")
        
       