**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

Constance Wang  
- [LinkedIn](https://www.linkedin.com/in/conswang/)

Tested on AORUS 15P XD laptop with specs:
Windows 11 22000.856
11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz 2.30 GHz
NVIDIA GeForce RTX 3070 Laptop GPU

[](images/flocking_gif.gif)

### Analysis

#### For each implementation, how does changing the number of boids affect performance? Why do you think this is?
Increasing the number of boids causes performance (frame rate) to decrease.

#### For each implementation, how does changing the block count and block size affect performance? Why do you think this is?
Changing the block size by updating the blockSize macro does not have a large effect on performance. This is because block count is calculated as (N + blockSize - 1) / blockSize, meaning the total amount of parallelism will always be N + blockSize - 1, aka. launch enough blocks to run each calculations for each of N boids in parallel. The only way to achieve more parallelism might be to parallelize each boid checking its neighbouring grid cells - but we still have to collect the results to calculate average speed, velocity, etc. This is probably not worth the overhead, since each boid checks at most 8 neighbours.

I noticed that testing with coherent uniform grid, with visualization off, and number of boids = 5000, the frame rate increases slightly with increased block size.
[](images/framerateblocksize.png)

#### For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?

#### Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!



