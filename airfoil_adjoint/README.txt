Airfoil optimization code by Nathan Hoch

Written for MAE 5370 Final Project Fall 2023

RUN adaptive_step.py or fixed_step.py files

The adaptive_step.py and fixed_step.py will 
    -read in and write out airfoil information
    -call adjoint gradient calculation
    -iterate the optimizer
    -write each step's new geometry
    -plot the airfoil geometries
    -plot the CL value for each iteration

Optimizer settings may be adjusted at points with sandwiched between pound signs like this:

