**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Nadine Adnane
  * [LinkedIn](https://www.linkedin.com/in/nadnane/)
* Tested on my personal laptop (ASUS ROG Zephyrus M16):
* **OS:** Windows 11
* **Processor:** 12th Gen Intel(R) Core(TM) i9-12900H, 2500 Mhz, 14 Core(s), 20 Logical Processor(s) 
* **GPU:** NVIDIA GeForce RTX 3070 Ti Laptop GPU

Note: I useed a late day for this assignment.

### Results

## Naive Implementation
5000 boids
<img src="images/naive-5000.gif" width="500">

50,000 boids
<img src="images/naive-50000.gif" width="500">

100,000 boids
<img src="images/naive-100000.gif" width="500">

## Uniform Grid
5000 boids
<img src="images/uniform-5000.gif" width="500">

50,000 boids
<img src="images/uniform-50000.gif" width="500">

100,000 boids
<img src="images/uniform-100000.gif" width="500">

## Coherent Grid
5000 boids
<img src="images/coherent-5000.gif" width="500">

50,000 boids
<img src="images/coherent-50000.gif" width="500">

100,000 boids
<img src="images/coherent-100000.gif" width="500">

## Graphs

<img src="images/num_boids_vs_fps.png" width="500">

<img src="images/num_boids_vs_fps_no_vis.png" width="500">

<img src="images/block_size_vs_fps.png" width="500">

## Performance Analysis

# How does changing the number of boids affect performance? Why do you think this is?

Across all of the implementation methods, the performance decreases as the number of boids increases (as one would expect!). This makes sense since for each boid, we have to consider the impact of its neighboring boids on its velocity. More boids in the simulation means more neighbors to consider per boid, and thus more calculations to be done, especially for the naive implementation where we are not culling the pool of potential neighbors! For the naive implementation, there was a significant drop in the frame rate as the number of particles in the simulation increased. The decreased performance also makes sense for the grid-based methods, as even though we are culling the pool of potential neighbors, there are still more neighbors to consider per cell as the total number of boids increases (and thus more calculations to be performed!). 

# How does changing the block count and block size affect performance? Why do you think this is?

Based off of my tests, there appeared to be a slight increase in performance when the block size is set to around 32, but otherwise it was somewhat difficult to tell if the difference in performance was particularly significant, as there wasn't much of a change in FPS during the tests that I ran. From there on, there seemed to be a point of diminishing returns. I think that increasing the block size causes an increase in performance at first because there are more threads available to be used.

# For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?

Yes! There was a significant improvement in performance with the coherent uniform grid as compared to the scattered grid method, which was as I had expected. This makes sense, since the coherent grid method involves sorting the boid data (velocities and positions) to be contiguous in memory (and match the ordering of the grid index), so the data could be accessed more directly without the need for dev_particleArrayIndices.

# Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!

Yes, changing the cell width and checking 27 cells instead of 8 did seem to affect the performance - I actually saw a very slight increase in performance when the cell width was changed from 1X to 2X and we were checking 27 neighboring cells instead of only 8. At first this was surprising - I assumed that checking 27 cells would surely be slower. However, I think the performance increase is due to the fact that even if we are only checking 8 cells, we still have to go through the process of figuring out which of the 8 cells to check. Also, since I only observed a slight increase in performance (at least from the FPS), I think more testing would be needed to actually be sure that the performance increase was significant.