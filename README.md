**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

## YUHAN LIU
  * [LinkedIn](https://www.linkedin.com/in/yuhan-liu-/), [personal website](https://liuyuhan.me/)
  * Tested on: Windows 11 Pro, Ultra 7 155H @ 1.40 GHz 32GB, RTX 4060 8192MB (Personal Laptop)

<p>
<img src="https://github.com/yuhanliu-tech/GPU-CUDA-Flocking/blob/main/images/screenshot.png" width="500"/>
<em>Screenshot of Boids simulation with 100,000 particles.</em>
</p>

## Performance Analysis

For each implementation, how does changing the number of boids affect performance? Why do you think this is?
* **Naive**: The complexity of the boid's behavior is O(N*N), where N is the number of boids, because, to update the position of each boid, we must compare its distance to all other boids in order to resolve neighbor rules. According to the plot, the performance of this approach exceeds that of the grid implementations for small boid counts; this is because the naive approach avoids the upfront operations related to grid preprocessing. However, the program becomes exponentially slower with more boids, to the point where the grid optimization is crucial to render large boid counts with satisfactory frame rates. 
* **Scattered Uniform Grid**: The uniform grid implementation drastically reduces the number of neighbors needed to check for each boid. Thus, its performance exceeds that of the naive implementation when the boid count is high enough to necessitate grid preprocessing. However, its performance to boid count has the same decreasing relationship because an increase in boids still leads to an increase in neighbor checks to satisfy the simulation rules. 
* **Coherent Grid**: The coherent grid implementation introduces an optimization on top of the scattered uniform grid. Though the method of updating boid position has not changed, the coherent grid's contiguous memory advantage boosts its performance. It has the same boid count to frame rate relationship as the previous two implementations, but it displays much better performance for the same number of boids. 

<img src="https://github.com/yuhanliu-tech/GPU-CUDA-Flocking/blob/main/images/FPS_boidnum.png" width="1000"/>

For each implementation, how does changing the block count and block size affect performance? Why do you think this is?
* For all three implementations, the performance increases as block size grows to 32. When the block size is below that of a warp (32 threads), the GPU is underutilized. As a result, adding more blocks increases parallelism, thereby enhancing performance. At block sizes larger than 32, performance seems to plateau (with a bit of variance). This may be because there is a tradeoff between increasing GPU utilization and limiting resource availability for each block. That is, the GPU hits its hardware limits in terms of scheduling blocks on SMs.    

<img src="https://github.com/yuhanliu-tech/GPU-CUDA-Flocking/blob/main/images/FPS_blocksize.png" width="500"/>

For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
* Yes, the coherent grid implementation improves upon the performance of the scattered grid implementation for all boid counts. This was expected because the implementation reduces memory latency and indirection by making the velocities and positions of all boids in a cell contiguous in memory. 

Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!
* When tested, checking 8 neighboring cells is generally faster compared to checking 27. However, when the boid count becomes extremely large, the performance gap closes. At 500K boids, checking 27 neighbors results in a frame rate of 90FPS, exceeding that of checking 8, which hovered at around 80 FPS. This is likely because, at high boid densities, the spatial efficiency gained by using smaller cells and checking more neighbors outweighs the benefit of checking fewer cells with larger boid populations. 

<p>
<img src="https://github.com/yuhanliu-tech/GPU-CUDA-Flocking/blob/main/images/gif.gif" width="500"/>
<em>GIF of Boids simulation with 10,000 particles.</em>
</p>
