# import necessary packages
import json
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from VPM_adjoint import Adjoint

# import airfoil points and copy to an X0.txt vector that can be adjusted

input_file = "airfoil_adjoint/2412_50.json"
json_string = open(input_file).read()
# make a json dictionary
input_dictionary = json.loads(json_string)
# airfoil_txt gets the txt file associated with 'airfoils' key in the json
airfoil_txt = input_dictionary["airfoils"]
# the list associated with airfoils key may have multiple files (airfoils)
for file in airfoil_txt:
    # the list associated with airfoils key may have multiple files (airfoils)
    with open(file, "r") as f: # "r" means open for reading
            # create an empty list
            info = []
            # go through each line in the airfoil .txt
            for line in f:
                # take the text and create numerical values for each chunk
                info.append([float(coordinate) for coordinate in line.split()])
                # close the file
            f.close()
    # store airfoil points as X0        
    X0 = np.array(info)
    
X0_txt = "airfoil_adjoint/X0.txt"
with open(X0_txt, "w") as output_file:
    for x, y in X0:
        output_file.write(f"{x} {y} \n")


# calculate the first gradient of X0
# call Adjoint
adjoint_X0 = Adjoint("airfoil_adjoint/2412_50_adjoint.json")
adjoint_X0.run()
grad_X0 = adjoint_X0.gradient
# store the data
gradient_txt = "airfoil_adjoint/gradient.txt"
with open(gradient_txt, "w") as output_file:
    for x in grad_X0:
        output_file.write(f"{x}\n")


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