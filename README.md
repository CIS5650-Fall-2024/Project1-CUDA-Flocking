**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Shixuan Fang
  * [LinkedIn](https://www.linkedin.com/in/shixuan-fang-4aba78222/)
* Tested on: Windows 11, i7-12700k, RTX 3080Ti (Personal)


## Project Description


This project is about creating flocking simulation based on Craig Reynolds algorithm. In this algorithm, boids behave under the following three rules:
1. cohesion - boids move towards the perceived center of mass of their neighbors
2. separation - boids avoid getting to close to their neighbors
3. alignment - boids generally try to move with the same direction and speed as their neighbors

I implemented this algorithm with three methods and also analyzed their performance:
1. Boids with Naive Neighbor Search, where each boid will iterate over all boids.
2. Optimization with uniform spatial grid, where boids are seperated into different unifrom grids and therefore can reduce the number of boids that need to be checked every time.
3. Further optimization with coherent grid, which is similar to method 2 but with coherent memory access


## Screenshots

![scattered](https://user-images.githubusercontent.com/54868517/189559176-99473284-7fb2-48c6-81c2-26b1430d6fe6.gif)
![cohenert](https://user-images.githubusercontent.com/54868517/189559188-a67446ff-d56e-4603-8e9d-a047965b6e07.gif)


## Performance Analysis
![figure 1](https://user-images.githubusercontent.com/54868517/189559288-ccf3618c-51d6-4102-994e-1c7c71d3cff9.jpg)
**figure 1: FPS and number of boids in 3 different methods with visualization**
![figure 2](https://user-images.githubusercontent.com/54868517/189559292-23459485-74e5-4e97-8460-ba114e292c5a.jpg)
**figure 2: FPS and number of boids in 3 different methods without visualization**
![figure3](https://user-images.githubusercontent.com/54868517/189559618-e5090b02-c13c-44ea-949e-30eaf43dae5a.jpg)
**figure 3: FPS and Block Size under Coherent Grid with 100k boids**

## Questions
**1. For each implementation, how does changing the number of boids affect performance? Why do you think this is?**

For all three methods, increasing the number of boids will decrease the performace and average FPS. Increasing the number of boids will increase the number of neighbors for each boid, and although we are doing this in parallel, we still have to check neighbors in each thread. Therefore, increase boid number will increase the time of computation for each thread, thus decrease performance.

**2. For each implementation, how does changing the block count and block size affect performance? Why do you think this is?**

As seen in figure 3, the FPS increases when the blocksize is small, but after a certain point it remains almost same. I think this is because when there are not enough warps in each block, it can't hide the latency caused by memory access and other computation, thus increasing number of threads will increase the performance; however, after there are enough warps, then block size/number of thread doesn't matters too much since each block can compute in parallel and doesn't depend on others.

**3. For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?**

The coherent uniform grid actually cause a huge performace improment. This isn't the outcome as I expected becasue I created two new buffers for coherent position and velocity, and I believe that copy data would also cost some time, which would balence the time accessing global data in GPU. Especially for the visualization off case, the program can runs up to around 3000 FPS with 5000 boids, which almost double the performance compare to scattered grid. I think this is because accessing global memory is very expensive in GPU programming.

**4. Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!**

Yes, this affect performance but not significantly. The actual "space" of checking 27 neighboring cells with half gridWidth is actually smaller that checking 8 neighboring cells with double gridWidth, but will also cause more gridcells, so the performace of these two methods will depends on how boids are distributed in the space. Smaller grid width means less memory access in each grid, but also means more grid to check, so these are trade-offs.


