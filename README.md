**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

![Boids Cover Image](images/Boids%20Cover.png)

* Megan Reddy
  * [LinkedIn](https://www.linkedin.com/in/meganr25a949125/), [personal website](https://meganr28.github.io/)
* Tested on: Windows 10, AMD Ryzen 9 5900HS with Radeon Graphics @ 3301 MHz 16GB, NVIDIA GeForce RTX 3060 Laptop GPU 6GB (Personal Computer)
* Compute Capability: 8.6

### Overview

| Coherent - 5000 boids             |  Coherent - 50000 boids |
:-------------------------:|:-------------------------:
![Boids 5,000 GIF](images/Boids%205k.gif)  |  ![Boids 50,000 GIF](images/Boids%2050k.gif)

[Boids](https://en.wikipedia.org/wiki/Boids) is an artificial life simulation developed by Craig Reynolds in 1986. The simulation represents the flocking behavior of birds or fish (boids).

This behavior follows three rules:
1. Cohesion - boids move towards the perceived center of mass of their neighbors
2. Separation - boids avoid getting to close to their neighbors
3. Alignment - boids generally try to move with the same direction and speed as
their neighbors

These three rules determine the **velocity change** for a particular boid at a certain timestep. Thus, the contribution from each rule must be computed and added to the boid's current velocity. 
For the Naive case, each boid must be checked against every other boid in the simulation. 

Here is psuedocode for computing the velocity contribution from each of the three rules:

#### Rule 1: Boids try to fly towards the centre of mass of neighbouring boids

```
function rule1(Boid boid)

    Vector perceived_center

    foreach Boid b:
        if b != boid and distance(b, boid) < rule1Distance then
            perceived_center += b.position
        endif
    end

    perceived_center /= number_of_neighbors

    return (perceived_center - boid.position) * rule1Scale
end
```

#### Rule 2: Boids try to keep a small distance away from other objects (including other boids).

```
function rule2(Boid boid)

    Vector c = 0

    foreach Boid b
        if b != boid and distance(b, boid) < rule2Distance then
            c -= (b.position - boid.position)
        endif
    end

    return c * rule2Scale
end
```

#### Rule 3: Boids try to match velocity with near boids.

```
function rule3(Boid boid)

    Vector perceived_velocity

    foreach Boid b
        if b != boid and distance(b, boid) < rule3Distance then
            perceived_velocity += b.velocity
        endif
    end

    perceived_velocity /= number_of_neighbors

    return perceived_velocity * rule3Scale
end
```

#### Compute the boid's new velocity and position

```
function computeNewVelocity(Boid boid)

    v1 = rule1(boid);
    v2 = rule2(boid);
    v3 = rule3(boid);
    
    boid.velocity += v1 + v2 + v3;
    boid.position += boid.velocity;
end
```

#### Optimizations

The method presented in the previous section can be inefficient, especially with a large number of boids and a small neighborhood distance.
One way to avoid checking every other boid is to "bin" each particle into a uniform grid. This way, we can limit the search to a certain number of surrounding grid cells.

In 3D:

* If `cell width` is 2x the neighborhood distance, you have to check 8 cells
* If `cell width` is 1x the neighborhood distance, you have to check 27 cells

2D Visualization:

![2D Uniform Grid Neighbor Search](images/Boids%20Ugrid%20neighbor%20search%20shown.png)

##### Scattered Uniform Grid

Calculating the uniform grid consists of these steps:

1. Assign each boid a `gridCellIndex`.
2. Sort the list of boids by `gridCellIndex` so that all boids in a cell are contiguous in memory.
3. In separate arrays, store the `start` and `end` indices of a particular grid cell.
4. Depending on the position of the boid within its cell, loop through possible neighbors and compare against boids in those cells.

The reason this version is called the "scattered" uniform grid is because the actual position and velocity data is not arranged contiguously in memory. 
This can be fixed using the method in the following section.

##### Coherent Uniform Grid

To make the position and velocity data coherent, we rearrange the data to match the order of the shuffled boid indices.
This allows us to have one less memory access in the neighbor search loop since the position and velocity data can be indexed
directly with `gridCellStartIndices` and `gridCellEndIndices`.

### Performance Analysis

To analyze the performance of each implementation (`Naive`, `Scattered Uniform Grid`, `Coherent Uniform Grid`), the frame rate (fps) was measured over 50 seconds.
These numbers were averaged to obtain the **average frame rate** for each data point on the following graphs. Additionally, each graph indicates whether
visualization was turned on (meaning boids were rendered on the screen) or off (just the simulation).

The follow data was recorded in **Release Mode** with **Vertical Sync** turned off.

#### Number of Boids vs. FPS (No visual)

![Number of Boids vs. FPS (No Vis)](images/Boids%20FPS%20No%20Vis.png)

#### Number of Boids vs. FPS (Visual)

![Number of Boids vs. FPS (Vis)](images/Boids%20FPS%20Vis.png)

Overall, changing the number of boids decreased the performance of each implementation. 
The Scattered and Coherent Grids performed faster overall despite the performance decrease
due to limiting neighbor checks. 

#### Block Size vs. FPS (No visual)

![Block Size vs. FPS (No Vis)](images/Block%20Size%20FPS%20No%20Vis.png)

#### Block Size vs. FPS (Visual)

![Block Size vs. FPS (Vis)](images/Block%20Size%20FPS%20Vis.png)

The block size graphs were measured using a fixed boid number of **50,000**. Block size did not have an obvious impact on performance as the numbers stayed relatively the
same within each implementation. A possible reason for this is discussed in the section below.

### Questions

**1. For each implementation, how does changing the number of boids affect
performance? Why do you think this is?** \
As the number of boids increase, the average FPS/performance per implementation goes down. 
This is most likely caused by the higher boid density per cell, making it necessary to check 
against more boids as the number increases. Even in optimized implementations such as the
Scattered and Coherent Grids, the number of boids has a significant impact on performance. 
However, since the search space is reduced by checking fewer cells, the average fps is higher for those implementations.
It is worth noting that the Naive implementation still achieves a good performance at lower boid counts (5000, 10000), 
but sharply decreases afterwards.

**2. For each implementation, how does changing the block count and block size
affect performance? Why do you think this is?** \
A GPU's main goal is to hide latency by switching between threads. Similarly, the GPU will keep the 
hardware busy by executing other warps (groups of 32 threads) when others are stalled. The number of active warps
relative to the maximum possible number of active warps at a time is known as occupancy. I chose multiples of 32 for 
my block sizes in order to maximize occupancy and increase performance, though this is not always the case as seen in the graphs.
There is a slight performance increase until 256-512 threads are reached for some implementations, but afterwards, the gains
from additional occupancy is negligible. As a whole, increasing block size did not have much of an impact on performance for the
chosen block sizes. If I had chosen block sizes that were not multiples of 32, I might
have seen a decrease in performance due to unused threads and lower occupancy. 

**3. For the coherent uniform grid: did you experience any performance improvements
with the more coherent uniform grid? Was this the outcome you expected?
Why or why not?** \
Yes, I did see a performance improvement with the coherent uniform grid. I expected this as the
outcome since there were a two memory reads that were removed, thus the performance should
have been better. Reading and writing from global memory is slow on the GPU, therefore less of these accesses
are more desirable. 

**4. Did changing cell width and checking 27 vs 8 neighboring cells affect performance?
Why or why not? Be careful: it is insufficient (and possibly incorrect) to say
that 27-cell is slower simply because there are more cells to check!** \
Yes, I (mostly) noticed a speed increase when running with 27 cells vs. running with 8 cells. This was especially noticeable
when running simulations with more than 10000 boids. When `cellWidth` is greater than `searchRadius`, this means that you have 
to search a larger portion of the grid, which will be slower (as is the case with searching 8 neighboring cells). When `cellWidth` 
is less than or equal `searchRadius`, the grid resolution is finer, therefore the search space is more confined (as is the case with 27 cells). 
However, this could mean looping through more cells, which could cause more divergence. 
