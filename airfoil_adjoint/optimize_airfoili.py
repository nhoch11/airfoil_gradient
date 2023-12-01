# import necessary packages
import json
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from VPM_adjoint import Adjoint

# import airfoil points and copy to an X0.txt vector that can be adjusted

# calculate the first gradient of X0
        # call Adjoint
        # store the data

# do a loop while difference between gradients is greater than say .001, or just do a number of iterations

    # calculate the direction P

    # approximate the Hessian
        # perturb X in the P direction by arbitrary alpha_bar step of 0.00001
        # use forward diff
    
    # calc the step size alpha

    # step the geometry 
        # Xnew = X + alpha*P

    # calculate the gradient for this new step and compare to the existing gradient

    # store or print results

    # stop if the gradient difference is close to zero 


# give results