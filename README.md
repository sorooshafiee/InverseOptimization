# Source for InvOpt paper

## Introduction
This repo contains all source code that reproduce the experiments in our paper: [Data-driven Inverse Optimization with Incomplete Information](http://arxiv.org/abs/1512.05489) paper. 

**We welcome any feedback and suggestions! Note that we put in maximum effort to write high quality codes. However, they may still contain bugs or not be efficient enough.**

## Prerequisites
All optimization problems are implemented in MATLAB. The implementations rely on the following third-party software: [CPLEX Studio](http://www-01.ibm.com/software/commerce/optimization/cplex-cp-optimizer/index.html), [MOSEK](https://www.mosek.com/), and [YALMIP](https://github.com/johanlofberg/YALMIP). t is necessary to install these software and add their respective directories to the MATLAB path before running the codes.

### CPLEX and MEX impelementatios
CPLEX is a solver for mixed-integer linear, quadratic and second-order cone programs. In our implementations, we use CPLEX 12.6 via the ILOG C++ interface. All C++ codes are compiled into MEX binaries, which can be called from MATLAB. In Windows, the MEX binaries are generated using Microsoft Visual Studio (VS)  compiler (see [here](http://ch.mathworks.com/help/matlab/ref/mex.html) for more information) . To generate the MEX binaries, you need to run

> \> make(CPLEX\_root\_directory)

in MATLAB command windows.

### YALMIP
YALMIP is used to interface with MOSEK. To add YALMIP directories to the MATLAB path, you need to run. You can download it from [here](http://users.isy.liu.se/johanl/yalmip/) or its [git repo](https://github.com/johanlofberg/YALMIP). YALMIP is used to interface with MOSEK. To add YALMIP directories to the MATLAB path, you need to run

> \> addpath(genpath(YALMIP\_root\_directory))

### MOSEK

You can download MOSEK from its [website](http://www.mosek.com). Free versions for academia are available. You can add MOSEK directories to your MATLAB path by running

> \> addpath(MOSEK\_root\_directory\MATLAB\_version)

**Note:** You can use the *savepath* command in MATLAB to save the YALMIP and MOSEK paths permanently. Otherwise, you will need to run the above *addpath* commands whenever you restart MATLAB.

## Reproducing the results
First, clone the repo

> $ git clone https://github.com/sorooshafiee/InverseOptimization.git

Then run

> \> make(CPLEX\_root\_directory)

in MATLAB command window. To reproduce the simulation results, you need to run m-file scripts.