**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Yilin Li 
* Tested on: Windows 11, i7-12700H @ 2.3GHz 16GB, GTX 3060 (Personal Laptop)

## Introduction
This project implements a flocking simulation based on the Reynolds Boids algorithm, along with two levels of optimization: a uniform grid, and a uniform grid with semi-coherent memory access.

![](images/Naive_40000boids.gif)
*Naive Algorithm with 40000 boids and 128 block size.*

![](images/Unifrom_40000boids.gif)
*Uniform Grid Search Algorithm with 40000 boids and 128 block size.*

![](images/Coherent_40000boids.gif)
*Coherent Uniform Grid Algorithm with 40000 boids and 128 block size.*

## Performance Analysis 
* For each implementation, how does changing the number of boids affect performance? Why do you think this is?
  * As we can observe from the figure below, increasing the number of boids will decrease the performance badly for the Naive Algorithm. Increasing the number of boids will also decrease the performance of Uniform Search Algorithm but not as bad as Naive Algorithm. Notice that Coherent Algorithm is barely influenced within our tesing range. 
  * The results indicate that 100000 boids is still far away from the limit of Coherent Algorithm. For Naive Algorithm, each thread will search through every other boids in the environment and thus it has the worst performance. For Uniform Algorithm, as number of boids increases, its neighbor candidates increases but the range of searching is 8 grid cells by maximum and thus the performance will not be damaged too bad. The Coherent Algorithm, however, optimize the memeory access which utilizes the power of contiguous memory on blocks. Remeber most of the slowness is caused by memory accessing. 
![](images/p_boids.png)
* For each implementation, how does changing the block count and block size affect performance? Why do you think this is?
  * sdfljkahsdf
