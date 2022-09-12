**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Yilin Li 
* Tested on: Windows 11, i7-12700H @ 2.3GHz 16GB, GTX 3060 (Personal Laptop)

## Introduction
This project implements a flocking simulation based on the Reynolds Boids algorithm, along with two levels of optimization: a uniform grid, and a uniform grid with semi-coherent memory access.

![](images/Naive_40000boids.gif)
*Naive Algorithm with 40000 boids and 128 block size*

![](images/Unifrom_40000boids.gif)
*Uniform Grid Search Algorithm with 40000 boids and 128 block size*

![](images/Coherent_40000boids.gif)
*Coherent Uniform Grid Algorithm with 40000 boids and 128 block size*

![](images/Coherent_100000boids.gif)

*Coherent Uniform Grid Algorithm with 100000 boids and 128 block size*
