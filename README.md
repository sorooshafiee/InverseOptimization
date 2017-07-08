# Source for InvOpt paper

## Introduction
This repo contains all source code required to reproduce the [Data-driven Inverse Optimization with Incomplete Information](http://arxiv.org/abs/1512.05489) paper. 

**Note that we did our best to write high quality codes. However, it may contain bugs or not be efficient enough. Any feedback on codes are more than welcome!**

## Prerequisites
All optimization problems are implemented in MATLAB. Our implementations rely on third-party software including [CPLEX Studio](http://www-01.ibm.com/software/commerce/optimization/cplex-cp-optimizer/index.html), [MOSEK](https://www.mosek.com/), and [YALMIP](https://github.com/johanlofberg/YALMIP). It is necessary to install and add these software to MATLAB path to generate simulation results. 

### CPLEX and MEX impelementatios
CPLEX is known as one of the best commercial optimization software package. We used CPLEX 12.6 via the ILOG C++ interface to solve linear, quadratic, and second order cones programs. Our implementation requires to install MEX in MATLAB. If you are a windows user, you need Microsoft Visual Studio (VS) to provide C++ compiler for MEX files. For more information, see [here](http://ch.mathworks.com/help/matlab/ref/mex.html). To generate compiled MEX files, you need to run

> \> make(CPLEX\_root\_directory)

in MATLAB command windows.

### YALMIP
You need to download YALMIP from [here](http://users.isy.liu.se/johanl/yalmip/) or its [git repo](https://github.com/johanlofberg/YALMIP). Then add YALMIP directories to your MATLAB path by running

> \> addpath(genpath(YALMIP\_root\_directory))

### MOSEK

You need to download MOSEK from its [website](http://www.mosek.com). Free versions for academia are available. You need to add MOSEK directories to your MATLAB path by running

> \> addpath(MOSEK\_root\_directory\MATLAB\_version)

**Note** that you can then use *savepath* command MATLAB in order to save YALMIP, Sedumi, and MOSEK path permanently. Otherwise, you need to run the above *addpath* commands whenever you run MATLAB.

## Reproducing the results
First, clone the repo

> $ git clone https://github.com/sorooshafiee/InvOpt.git

Then run

> \> make(CPLEX\_root\_directory)

in MATLAB command window. To reproduce the simulation results, you need to run m-file scripts.