**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* RHUTA JOSHI
  * [LinkedIn](https://www.linkedin.com/in/rcj9719/)
  * [Website](https://sites.google.com/view/rhuta-joshi)

* Tested on: Windows 10 Home, i5-7200U CPU @ 2.50GHz, NVIDIA GTX 940MX 4096 MB (Personal Laptop), RTX not supported
* GPU Compatibility: 5.0

# Boids Assemble! #
## Introduction

Boids are a computer simulation of an animal(eg. fish/bird) that flies in flocks or swarms.
In this assignment I have implemented a flocking simulation based on the Reynolds Boids algorithm, optimized using a uniform grid. Another level of optimization to be implemented is using a uniform grid with semi-coherent memory access.


![](images/50k_default.gif)

## Observations

After 3-4 minutes uniform grid simulation running continuously, we can see that all particles slowly get aligned in one direction
![](images/50k_3min.gif)

Increasing the maximumspeed of each particle by a factor of 2:
![](images/50k_2xSpeed.gif)

## Blooper

I thought this blooper was an interesting visualization, totally wrong of course. I was calculating the updated velocity incorrectly.
![](images/5k_naive_blooper.gif)

# Performance Analysis #

Charts - To be updated

**Q. For each implementation, how does changing the number of boids affect performance? Why do you think this is?**

**A.** As the number of boids increase, the performance drops in both naive and uniform grid methods. This is because the number of neighboring boids increases as each cell gets more densely packed. Since we are running 1 thread per boid for calculating updated velocities, each thread has to take more velocities into account for this calculation, thus affecting the overall performance.

**Q. For each implementation, how does changing the block count and block size affect performance? Why do you think this is?**

**A.** To be answered

**Q. For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?**

**A.** To be answered

**Q. Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!**

**A.** For a large number of boids, reduced cellWidth and comparison with 27 neighboring cells gives better performance. This might be because each thread does not have to determine which closest 8 cells among its neighbors have boids affecting its velocity. Also, even if the number of cells is more, the total number of affecting boids may be less since the cell width has been reduced. The results may also depend on whether the cell width is lesser than the smallest radius of boid influence or is it equal to the largest boid influence distance.


