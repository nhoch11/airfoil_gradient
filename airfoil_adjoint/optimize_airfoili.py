# import necessary packages
import json
import os
import time
import shutil
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from VPM_adjoint import Adjoint


#a function that plots airfoils and angle of attack for each HW problem
def plotAirfoils():
      
    fig, ax = plt.subplots(figsize=(8, 6)) 
    fig.subplots_adjust(right = 0.85)
    folder = "airfoil_adjoint/stepped_airfoils"
    
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        with open(file_path, "r") as f: # "r" means open for reading
            # create an empty list
            info = []
        
            # go through each line in the NACAfile airfoil
            for line in f:
                # take the text and create numerical values for each chunk
                info.append([float(coordinate) for coordinate in line.split()])
                # close the file
            f.close()

        # turn the list into an array that we can do math on
        info = np.array(info)

        # assign the zero column as x coordinates
        x = info[:, 0]
        y = info[:, 1]
        # To make a label for the airfoil plots, Get string of file
        airfoilLabel = str(file)
        # get rid of ending junk
        airfoilLabel1=airfoilLabel.replace("airfoil_adjoint/","")
        #get rid of underscores
        airfoilLabel2 = airfoilLabel1.replace(".txt", " ")
        # assign a random color to airfoil
        col = (np.random.random(), np.random.random(), np.random.random())
        # plot the points
        ax.plot(x,y, c=col, label=str(airfoilLabel2))
        ax.scatter(x,y, c=col)
    # make the plot look pretty
    ax.set_aspect(6)
    ax.legend(loc='center right',bbox_to_anchor=(1.2, 0.5))
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    # Display all the plots associated with this json (aifoils and relative wind vector)    
    plt.show() 
     

# import airfoil points and copy to an X0.txt vector that can be adjusted
def run(input_file):
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

    stepped_airfoil_dir = "airfoil_adjoint/stepped_airfoils"

    step0_txt = os.path.join(stepped_airfoil_dir, f"step0.txt")
    with open(step0_txt, "w") as output_file:
        for x, y in X0:
            output_file.write(f"{x} {y} \n")
    



    # calculate the first gradient of X0
    # call Adjoint
    print("Calculating Gradient of initial shape")
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

    X_shape = X0.ravel('F').reshape(-1,1)
    n = np.size(X_shape)

    # set up loop stop conditions
    max_iter = 6
    e = 0.00001
    iter = 0
    difference = 1.0
    


    # do a loop while difference between gradients is greater than say .001, or just do a number of iterations
    while iter <= max_iter or difference >= e:
        print("Begin Step "+str(iter+1))
        # trim the gradient vector,  remove trailing edge and some leading edge gradient values
        rows = grad_table.split('\n')
        data = [row.split() for row in rows]
        table = [list(map(str.strip, row)) for row in data] 
        
        # change constrained points sensitivites to zeros in table, dont delete
        rows_to_zero = [2,3,4, 25,26,27,28, 48,49,50,51, 52,53,54,55, 75,76,77,78, 98,99,100,101]
        for i in rows_to_zero:
            table[i][1] = 0.0

        # change constrained points sensitivites to zeros in the grad vector
        points_to_zero = [0,1,2, 23,24,25,26, 46,47,48,49, 50,51,52,53, 73,74,75,76, 96,97,98,99]
        grad[:,points_to_zero] = 0.0
        

        # calculate the direction P
        grad_norm = np.linalg.norm(grad)
        #print("norm\n",grad_norm)
        p = grad/grad_norm
        #print("p direction\n",p)
        

        # approximate the Hessian
        # perturb X in the P direction by arbitrary alpha_bar step of 0.000001
        alpha_bar = 0.000001
        X_perturbed = X_shape - alpha_bar*np.transpose(p)

        # reshape for txt writing
        X_perturbed_points = X_perturbed.reshape((int(n/2),2), order = 'F')
        # write to a txt file
        X_perturb_txt = "airfoil_adjoint/X_perturbed.txt"
        with open(X_perturb_txt, "w") as output_file:
            for x, y in X_perturbed_points:
                output_file.write(f"{x} {y} \n")
        
        
        # calc the gradient of the slightly perturbed geometry
        adjoint_h = Adjoint("airfoil_adjoint/X_perturbed.json")
        adjoint_h.run()
        grad_h = adjoint_h.gradient

        # change constrained points sensitivites to zeros in the grad vector
        grad_h[:,points_to_zero] = 0.0

        # use forward diff to estimate the hessian
        hessianp = (grad_h - grad)/alpha_bar
        
        # calc the step size alpha
        alpha = np.matmul(grad.flatten(), np.transpose(hessianp.flatten()))/(np.matmul(hessianp.flatten(),np.transpose(hessianp.flatten())))

        # step the geometry Xnew = X + alpha*P
        X_new = X_shape - np.transpose(alpha*p)
        X_new_points = X_new.reshape((int(n/2),2), order = 'F')
        
        # write to a txt file
        X_step_txt = os.path.join(stepped_airfoil_dir, f"step{iter + 1}.txt")
        with open(X_step_txt, "w") as output_file:
            for x, y in X_new_points:
                output_file.write(f"{x} {y} \n")
        step_file = "airfoil_adjoint/step.txt"
        shutil.copyfile(X_step_txt, step_file)

    
        
        # calculate the gradient for this new step and compare to the existing gradient
        adjoint_new = Adjoint("airfoil_adjoint/X_step.json")
        adjoint_new.run()
        grad_new = adjoint_h.gradient
        grad_table_new = adjoint_new.gradient_table
        rows_new = grad_table_new.split('\n')
        data_new = [row.split() for row in rows_new]
        table_new = [list(map(str.strip, row)) for row in data_new] 
        
        # change constrained points sensitivites to zeros in table, dont delete
        for i in rows_to_zero:
            table[i][1] = 0.0

        # stop if the gradient difference is close to zero 
        difference = np.max(grad_new-grad)
        if difference < e:
            print("Different between gradients is close to zero. Ending optimiziation...")
            break
        print("Ended Step "+str(iter+1))
        iter += 1
        grad = grad_new
        table = table_new
        X_shape = X_new

        if iter == max_iter:
            print("Max iterations reached. Ending optimiziation...")
            break
        
        


    # give results

if __name__ == "__main__":
    # take start time
    tstart = time.time()
    
    input_file = "airfoil_adjoint/2412_50.json"
    run(input_file)
    plotAirfoils()

    # take end time and display total computation time
    comp_time = (time.time()-tstart)
    print("Computation Time: ", round(comp_time, 4), " seconds")
    print("done")