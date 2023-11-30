# import necessary packages
import json
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from stepSolve import StepSolve
import time
import math
from cmath import *
from complex_step import Complex_Step
from complexGamma import complex_Gamma

# set precision to 15 digits of precision
np.set_printoptions(precision=15)


        

if __name__ == "__main__":
    tstart = time.time()
    

    
    # select a input json, the index of the variable to change, and the step size h
    # x0=0 indicates we want the derivative of l_k with respect to x0
    x0 = 0
    #step size is h  (if smaller than 0.000001, it starts to diverge)
    h = 0.00000002   #.000149 blows up, 0.0002 is stable

    stepUp = StepSolve("2412_10.json", x0, h).run()
    
    # change step size to negative, then run again
    stepDown = StepSolve("2412_10.json", x0, h*-1).run()
    
    dxi = (stepUp[1] - stepDown[1])/(2*h)
    #print("dxi", "\n", dxi)

    deta = (stepUp[2] - stepDown[2])/(2*h)
    #print("deta", "\n", deta)

    dphi = (stepUp[3] - stepDown[3])/(2*h)
    #print("dphi", "\n", dphi)

    dpsi = (stepUp[4] - stepDown[4])/(2*h)
    #print("dpsi", "\n", dpsi)

    dP = (stepUp[5] - stepDown[5])/(2*h)
    #print("dP[0,0]", "\n", dP[0, :])

    dA = (stepUp[6] - stepDown[6])/(2*h)
    #print("dA", "\n", dA)

    dB = (stepUp[7] - stepDown[7])/(2*h)
    #print("dB", "\n", dB)

    dCL = (stepUp[8] - stepDown[8])/(2*h)
    print("dCL", "\n", dCL)

    dsubP = (stepUp[9] - stepDown[9])/(2*h)
    #print("dsubP", "\n", dsubP)
    
    
    
    tstart = time.time()
    x0 = 0
    print("gradient with respect to x"+ str(x0) )
    h = 1.0E-200
    cStep = Complex_Step("2412_10.json", x0, h).run()
    
    dA_complex = np.imag(cStep[6])/h
    print("dA complex step", "\n", dA_complex)

    print("gradient with respect to x"+ str(x0) )
    dB_complex = np.imag(cStep[7])/h
    print("dB complex", "\n", dB_complex)

    dUda1 = np.imag(cStep[8])/h
    #print("dUda1", "\n", dUda1)

    dCL_complex = np.imag(cStep[9])/h
    #print("dCL complex", "\n", dCL_complex)

    gT = np.imag(cStep[10])/h
    #print("gT", "\n", gT)

    gradient = np.matmul(gT, dUda1) + dCL_complex
    #print("gradient", "\n", gradient)

    CL_check =  np.imag(cStep[11])/h
    print("CL_check", "\n", CL_check)

    complex_solve = complex_Gamma("2412_10.json", h).run()
    A = np.real(complex_solve[2])
    #print("A", A)

    gamma = np.real(complex_solve[0])
    #print("gamma", gamma)


    f0 = -np.matmul(np.real(dA_complex), gamma) + np.real(dB_complex)
    #print("f0", f0)

    A_inv_f = np.matmul(np.linalg.inv(A), f0)
    #print("A inv f ", A_inv_f)

    gTu = np.matmul(gT, A_inv_f)
    #print("gTu", gTu)

    grad = gTu + dCL_complex
    print("grad", grad)

    print

    comp_time = (time.time()-tstart)
    print("Computation Time 1 gradient: ", round(comp_time, 4), " seconds")
    print("Computation Time full gradient: ", round(comp_time, 4)*400, " seconds")

    print("done")