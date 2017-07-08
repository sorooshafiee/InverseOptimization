# Inverse optimization using open source software 

## Introduction
If you are not familiar with MEX functions or you do not have access to the commercial solvers, we implement an open source, full MATLAB implementation of our work. As we code everything in MATLAB, the codes are not efficient in terms of speed in comparison to C++ implementations. Hence, we just provide a demo for the figures 2, 7, 8 (demo1.m, demo2.m, demo3.m respectively) in our paper.

## Prerequisites
All optimization problems are implemented in MATLAB. Our implementations rely on open source software [YALMIP](https://github.com/johanlofberg/YALMIP) and [Sedumi](https://github.com/sqlp/sedumi). 

### YALMIP
You need to download YALMIP from [here](http://users.isy.liu.se/johanl/yalmip/) or its [git repo](https://github.com/johanlofberg/YALMIP). Then add YALMIP directories to your MATLAB path by running

> \> addpath(genpath(YALMIP\_root\_directory))

### Sedumi
You need to download Sedumi from [git repo](https://github.com/sqlp/sedumi). Then add Sedumi directories to your MATLAB path by running

> \> addpath(genpath(Sedumi\_root\_directory))

### CBC
CBC is an open-source mixed integer programming software which is part of [OPTI Toolbox](http://www.i2c2.aut.ac.nz/Wiki/OPTI).  Add OPTI Toolbox directories to your MATLAB path by running

> \> addpath(genpath(OPTI Toolbox\_root\_directory))


**Note** that a mixed integer programming software is required to solve the inverse optimization problem proposed [here](http://arxiv.org/pdf/1507.03266v3.pdf). 

## Demo
To download YALMIP and sedumi, you can simply run *third_party.m* MATLAB file to download YALMIP and Sedumi. Then, add YALMIP and sedumi to your MATLAB path, and run the scripts demo1.m to demo4.m.