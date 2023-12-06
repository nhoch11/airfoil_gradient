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
grad_table = adjoint_X0.gradient_table
# store the data for gradient of shape X0
gradient_txt = "airfoil_adjoint/gradient.txt"
with open(gradient_txt, "w") as output_file:
    for x in grad_X0:
        output_file.write(grad_table)


# feed in X0 points and gradient into the loop
grad = grad_X0
X_shape = X0


# set up loop stop conditions
max_iter = 10
e = 0.001
iter = 0
difference = 1

# do a loop while difference between gradients is greater than say .001, or just do a number of iterations
while iter <= max_iter or difference >= e:
    # trim the gradient vector,  remove trailing edge and some leading edge gradient values
    rows = grad_table.split('\n')
    data = [row.split() for row in rows]
    table = [list(map(str.strip, row)) for row in data] 
    # change to zeros, dont delete
    rows_to_delete = set([2,3,4, 49,50,51, 52,53,54,55,56, 95,96,97,98,99,100,101])
    table = [element for index, element in enumerate(table) if index not in rows_to_delete]
    grad = np.delete(grad, [0,1,2, 47,48,49, 50,51,52,53,54, 93,94,95,96,97,98,99], 1)
    
    # calculate the direction P
    grad_norm = np.linalg.norm(grad)
    print("norm\n",grad_norm)
    p = grad/grad_norm
    print("p direction\n",p)
    

    # approximate the Hessian
    # perturb X in the P direction by arbitrary alpha_bar step of 0.000001
    for i in range(p/2):
        X_perturbed(0,i) = X_shape + 0.00001*p
    # write to a txt file
    X_perturb_txt = "airfoil_adjoint/X_perturbed.txt"
    with open(X_perturb_txt, "w") as output_file:
        for x, y in X_perturbed:
            output_file.write(f"{x} {y} \n")
    adjoint_h = Adjoint("airfoil_adjoint/X_perturbed.json")
    adjoint_h.run()
    grad_h = adjoint_h.gradient
    grad_h = np.delete(grad_h, [0,1,2, 47,48,49, 50,51,52,53,54, 93,94,95,96,97,98,99], 1)
    # use forward diff
    hessianp = (grad_h - grad)/0.00001
    
    # calc the step size alpha
    alpha = np.dot(grad, hessianp)/(p*np.dot(grad,hessianp))

    # step the geometry 
        # Xnew = X + alpha*P

    # calculate the gradient for this new step and compare to the existing gradient

    # store or print results
    break
    # stop if the gradient difference is close to zero 


# give results