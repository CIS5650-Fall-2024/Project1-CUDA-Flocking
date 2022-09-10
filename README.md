### **University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 1 - Flocking**

* Chang Liu
  * [LinkedIn](https://www.linkedin.com/in/chang-liu-0451a6208/)
  * [Personal website](https://hummawhite.github.io/)
* Tested on personal laptop:
  - Windows 11
  - i7-12700 @ 4.90GHz with 16GB RAM
  - RTX 3070 Ti Laptop 8GB

![](./images/256K.gif)

<center>256K particles</center>

![](./images/1M.gif)

<center>Running at 100+ fps with 1M particles</center>

## Performance Analysis

### Test Case Design

Since our three methods display huge performance difference and generate valid measurements only within specific ranges of boid number (e.g., 1K ~ 0.5M for naive method and 10K ~ 10M for coherent uniform grid), I let the number of boids increase exponentially by testcases. The number of boids starts at 1K and double each time, until the tested method is unable to work real-time (e.g., time of one simulation step > 1s)

The same approach is applied to measure the effect of block size. Tested block size starts at 32 (warp size) and ends at 1024 (hardware limit).

### Metrics

For more accurate measurement, I chose `cudaEventElapsedTime` to record the exact execution time of kernels that are only involved in simulation of boids, excluding the time spent for GL visualization. `cudaEventElapsedTime` provides metrics precise to 0.01 millisecond.

I wrote some code in the program to calculate average simulation time and framerate.

### Effect of Boid Number

Block size is set to 256 for all test results below.

#### Framerate Change

![](./images/fps_boid_number.png)

<center>Framerate change with increasing number of boids. X-axis: number of boids. Y-axis: framerate. Solid line: with visualization. Dashed line: with visualization</center>



For each method, the framerate decreases with increasing number of boids. All three methods show similar tendencies without visualization. And it's pretty straightforward to see that coherent uniform grid >> scattered uniform grid >>> naive searching in performance. 

When visualization is turned on, the average framerate doesn't reflect actual performance, because my laptop doesn't support direct presentation of frames from discrete GPU to monitor. It takes the integrated GPU ~1.5ms to forward frame buffers, so the framerate will always be limited under 700 (and it's *really* unstable to be precisely measured due to other factors in the OS).

With sufficiently large number of boids, framerates with and without visualization finally converge because in this case simulation time >>> vertex buffer filling  + drawing + frame presenting time.

#### Simulation Time Change

The graph of framerate provides us with a clear outline of boid number's effect on performance. To make one step further, I measured the execution time of function `Boid::stepSimulation...` of three methods.

Since the number of boids tested increased exponentially, I also plotted the logarithm of simulation time. By measuring the slope of log2(time) to log2(boid number / 1000), we are able to know basically the exponent of time complexity.

![](./images/log2_time_boid_number.png)

<center>Change of simulation time (dashed line), log2 of simulation time (solid line) with increasing number of boids.</center>

<center>X-axis: number of boids. Y-axis-L: log2(simulation time). Y-axis-R: simulation time (millisecond)</center>

For example, when boid number is sufficiently large, log2(simulation time) and log2(boid number / 1000) are almost linear for all of three methods. If we calculate the slope of blue solid line at 128K, we get 2, meaning that the time complexity from here on is $O(n^2)$ for the naive method.

All three algorithms have sublinear time complexity with small number of boids and superlinear time complexity with large number of boids.

### Effect of Block Size

The number of boids is 256K. I use two charts to present data because it is hard to plot them in a graph.

| Block Size    | 32   | 64   | 128  | 256  | 512  | 1024 |
| ------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| FPS Naive     | 1.5  | 2.0  | 2.0  | 2.0  | 2.0  | 2.0  |
| FPS Scattered | 210  | 185  | 183  | 186  | 188  | 181  |
| FPS Coherent  | 579  | 760  | 786  | 775  | 760  | 770  |

<center>Framerate change</center>

| Block Size     | 32   | 64   | 128  | 256  | 512  | 1024 |
| -------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| Time Naive     | 665  | 509  | 509  | 509  | 509  | 509  |
| Time Scattered | 4.21 | 4.80 | 4.86 | 4.80 | 4.71 | 4.96 |
| Time Coherent  | 1.28 | 0.87 | 0.83 | 0.84 | 0.87 | 0.84 |

<center>Simulation time change</center>

When block size is greater than 64, the change in performance is slight. This is easy to explain: changing block size doesn't change the pattern of calculation and memory access since we are using 1D thread hierarchy and parallel executing is guaranteed every 32 threads.

However, when block size is exactly the same as warp size, it becomes an interesting story. For naive method and coherent uniform grid, the performance is worse while for scattered uniform grid the performance is better.

For this phenomenon, I can only come up with one possible explanation: L1 cache hit rate of each SM. As we know, each block is assigned to an SM during execution, so an SM can hold more blocks if 

### Answers to Part 3 Questions

