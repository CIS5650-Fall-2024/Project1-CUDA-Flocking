**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* ANNIE QIU
  * [LinkedIn](https://github.com/AnnieQiuuu/Project0-Getting-Started/blob/main/www.linkedin.com/in/annie-qiu-30531921a)
* Tested on: Windows 11, i9-12900H @2500 Mhz, 16GB, RTX 3070 Ti 8GB (Personal)

## Screenshots
### Screenshot 1
- Optimized coherent; number of boids = 10000; block size = 128
![](images/Coherent.gif)

![](images/Coherent1.gif)

### Screenshot 2
- Optimized coherent; number of boids = 500000; block size = 128
![](images/Coherent3.gif)

## Performance Analysis
### Framerate change with increasing # of boids
![](images/Boids.png)
- Descriptions: The dashed lines represent the performance without visualization, and the solid lines represent performance with visualization. I chose 1000, 5000, 10000, 20000, and 50000 boids for testing. The y-axis is the frames per second (FPS), and the x-axis is the number of boids. All the performances decreased as the number of boids increased. The Naive simulation is affected the most. And the scattered and coherent have a milder impact.

### Framerate change with increasing block size
![](images/BlockSize.png)
- Descriptions: I chose 8, 16, 32, 64, 128, 256 and 512 as block size. As the complexity of naivee simulation is high, even the increased of the block size may not have a big improvement on performance. And the scatted and coherent reach peak performance in 64 and 128 and get stable after that.

### Answers according to Analysis
1. For each implementation, how does changing the number of boids affect performance? Why do you think this is?
 -  As the number of boids increase, the fps decrease and the performance goes down.
 -  Naive simulation:  This is the slowest among the three implementations, and the FPS drops drastically as the number of boids increases. This is because the Naive Simulation requires looping through every single boid, resulting in O(N^2) complexity. As N increases, the speed becomes significantly slower.
 -  Scattered simulation:  By using a uniform grid, we can reduce the number of boids each boid has to check, so the decrease rate is much more mild compared to naive. When the number of boids becomes really big, it can also decrease a lot, since we have the access the unsorted boid data in every single loop.
 -  Coherent simulation: This is an optimized version of the scattered implementation. By sorting the pos and vel arrays, we reduced memory access times, while the number osf boids still affects the performance.
2. For each implementation, how does changing the block count and block size affect performance? Why do you think this is?
 - The low performance at block sizes smaller than 32 is due to not having enough threads to fit into the warp. If the block size is the multiples of 32, the perfomace can run efficiently in parallel. As the block size increases to a certain point like after 64 and 128, the performance stabilizes because the GPU's resources reach their limit. While larger block sizes mean more threads per block, the GPU’s resources like registers and shared memory are limited, so further increasing the block size doesn’t lead to additional performance gains.
 - Naive simulation: As the complexity is high, even the increased of the block size may not have a big improvement on performance.
 - Scattered simulation: The performace reaches a peak at block sizes of 32 and 64, and then stabilizes as the block size continues to increase. Not good as coherent, but much better than naive.
 - Coherent simulation:  It has the best performance over all simulations. The performance reaches the peak when the block sizes are 64 or 128. After that, it is getting stable as the scattered simulation.
3. For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
 - Yes. This is expected. Firstly, like in uniform grid we sorted the boids in the grids so that we can check our neighbors by cells. Secondly, cause the most time-consuming process is memory access. In the coherent uniform grid, I sorted the boids data, so I was no longer need to check particleArrayIndices to get index each time in doing the simulation. As the number of Boids increases, the memory access pattern becomes more continuous because the position and velocity data of the Boids have been rearranged in a grid. Therefore, as the number of Boids increases, the performance of the Coherent implementation decreases slower than Scattered. In my analysis screenshot, it shows that when the number of boids is 50000, the decrease rate of scatter is much sharper than coherent.
4. Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!
 - Checking 27 neighbors: The width of the grid cell is smaller, and the number of Boid within a single grid is smaller. However, due to the need to check 27 neighbors, the amount of calculation and memory access increases, and the performance decreases relatively. This is more obvious when the number of Boid is larger.
- Check 8 neighbors: The width of the grid cells is large, and each Boid only needs to check 8 neighbors. Although the number of neighbor checks is reduced, there may be more boids within a single grid cell, so performance may not improve significantly in certain cases, especially when boids are densely distributed.
