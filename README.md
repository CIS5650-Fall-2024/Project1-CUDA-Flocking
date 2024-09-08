**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Yifan Lu
  * [LinkedIn](https://www.linkedin.com/in/yifan-lu-495559231/), [personal website](http://portfolio.samielouse.icu/)
* Tested on: Windows 11, AMD Ryzen 7 5800H 3.20 GHz, Nvidia GeForce RTX 3060 Laptop GPU (Personal Laptop)

### Flocking Final Result


### Performance Analysis
#### Benchmark for 3 Methods
If there is no simulation running, the frame rate is around 2600FPS.

**Navie**
**Uniform Grid**
**Uniform Grid with Coherent Memory**

#### Q&A
- **Q1 For each implementation, how does changing the number of boids affect performance? Why do you think this is?**

For Navie method, since we have to compute the distance of each boid against others, the complexity is O(N^2). So if we increase the boid size N times from a previous one, the computation will be N^2 times. 

For Uniform Grid method, the performance will decrease linearly as the boid number increases. This is because when updating the pos and vel of each particle, we only have to search a fixed size of neighboring grids for neighbor particles. 

As for Coherent Uniform Grid method, the perfomance impact is similar to the Uniform Grid method. However, since the Coherent method has an improved memory access for accessing contiguous memory for particle in the same grid, the performance decrease will be less significant than the Uniform Grid method.

- **Q2 For each implementation, how does changing the block count and block size affect performance? Why do you think this is?**

For large numbers of boids, a larger block size increases the overall performance. However as the block size keeps increasing, the affect of performance improvement becomes less significant. The number of blocks determines the parallel utility when computing so increased block size will boost performance. However as the block size becomes too large, the time to access to registers and memory also increases.

- **Q3 For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?**

The coherent uniform grid method yeilds better performance when the boid number is very large (in my case it is around 100000). I expected a significant better performance in large boid number because the coherent memory for boid attributes can make the memory accessing more efficiently when updating the particle's velocity and position.


- **Q4 Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not?**
It may not affect performance. The purpose of checking neighoring cells is finding neighbor particles to update velocity. The number of neighbor particles is determined by the initiated random position as well as the grid.cell size and may vary for each simulation. Also with the contiguous memory, we cannot guarantee a better memory access of 8 girds compared to 27 grids.




