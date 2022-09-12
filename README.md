**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Dongying Liu
  * [LinkedIn](https://www.linkedin.com/in/dongying-liu/), [personal website](https://vivienliu1998.wixsite.com/portfolio)
* Tested on:  Windows 11, i7-11700 @ 2.50GHz, NVIDIA GeForce RTX 3060


# Project Description
![Flocking Simulation GIF](/images/result.gif)
Uniformed Gird, Coherent Memory Buffer, 500,000 Boids.

This project is about creating a Boids flocking simulation. 
Particles(boids) behave according to three rules:
1. **cohesion** - boids move towards the perceived center of mass of their neighbors
2. **separation** - boids avoid getting to close to their neighbors
3. **alignment** - boids generally try to move with the same direction and speed as their neighbors

These three rules specify a boid's velocity change in a timestep. 
The code is based on [Conard Parker's notes](http://www.vergenet.net/~conrad/boids/pseudocode.html) with slight adaptations.
This project is basically getting familiar with writing simple CUDA kernels and using them.

I've implemented the update function(main simulation step) using three ways:

1. **NAIVE** - Each boid checks every other boid in the simulation to compute its resulting velocity.

2. **UNIFORM GRID** - Each boid is pre-processed into a uniform grid cell. Every boid only need to check its 27 neighbor cells to compute its velocity change. 

3. **COHERENT GRID** - Same with the Uniform Grid, however, the position buffer and velocity buffer are sorted as well according to the grid_cell_idx so we can read from the contiguous memory when doing the simulation.

# Performance Analysis

## Framerate change with increasing count of boids

![How Boids Affect FPS](/images/boid_count_with_visualization.png)
![How Boids Affect FPS](/images/boid_count_without_visualization.png)

As the graph shows, generally, the FPS declines as the number of boids increases. 
For the naive method, the performance is worse than the other two methods, since every boids need to check for all the other boids, and for the other two methods we used the grid cell to narrow down the total boids need to be checked for one boid. 
What's more, we can tell the coherent grid acts a little bit better than uniform grid, since we rearraged the position and velocity buffer so the memory can store contiguously.

![How Boids Affect FPS](/images/visualization_comparison.png)

Since we are doing the visualization, the framerate is not reporting the framerate for the the simulation only. So, we can tell from the graph the perfromance is better without visualization.

## Framerate change with increasing block size
![How Block_Size Affect FPS](/images/block_size.png)
Tested with Coherent, 500,000 boids.

From the graph we can see there are only a mild falloff in FPS when block size increases. I think the reason is changing the block size does not change the number of threads running together at the same time, it only changes the configuration of the threads, which is how many blocks and how many threads in one block.

# Questions
Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!

**Q: For each implementation how does changing the number of boids affect performance? Why do you think this is?**

When the number of boids increases, the framerate dereases significantly. For each boids, we need to check for its 26 neighbor grid cells to change it velocity. When the number of boids increase, the number of boids in each grid cell increases generally. Although we only check for the boids in 27 cells, the total number of boids checked increase however. However, limited the total number of boids checked for each boid indeed increase the framerate, this can be told comparing the naive method with the other two methods.

**Q: For each implementation, how does changing the block count and block size affect performance? Why do you think this is?**

As said in the Performance Analysis. I think the reason is, changing the block count and block size do not change the number of threads running together at the same time, it only changes the configuration of the threads, which is how many blocks and how many threads in one block.

**Q: For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?**

Yes, there is a performance improvements with the coherent uniform grid and it is what I expected. I think the reason is because of the contiguous memory access. It is always slow when access memory that are not contiguous. Althought we spend extra time to create the sorted position and velocity buffers, the time saved for memory access is much more than the cost, it's a good trade off.

**Q: Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!**

I toggled between test checking 27 neighbor with cell_width=neighbor_distance and checking 8 neighbor with cell_width=double_neighbor_distance with uniform grid. When boids_count = 50,000, it turns out checking 8 neighbors(780 FPS) are somehow slightly faster than checking 27 neighbors(690 FPS). When boids_count = 100,000, checking 27 neighbors(523 FPS) are faster than checking 8 neighbors(370 FPS). 
Instinctively thinking, checking 27 neighbors with cell_width=neighbor_distance(assume neighbor_distance=1) is equal to checking 3*3*3 volume of boids. However, checking 8 neighbor with cell_width=double_neighbor_distance is equal to checking 4*4*4 volume of boids. So, it seems checking 8 neighbors will leads to checking more number of boids(assume the number of boids in every 1*1*1 cell are same). So, checking 27 neighbors with cell_width=neighbor_distance might performance better than checking 8 neighbor with cell_width=double_neighbor_distance. 
