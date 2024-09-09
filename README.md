**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Matt Schwartz
  * [LinkedIn](https://www.linkedin.com/in/matthew-schwartz-37019016b/)
  * [Personal website](https://mattzschwartz.web.app/)
* Tested on: Windows 10 22H2, Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz, NVIDIA GeForce RTX 2060

# Reynold Boids Flocking

![Boids Demo](/images/BoidsDemo.gif)

![Boids Demo 2](/images/BoidsDemo2.gif)

## Background

This repository contains several implementations of Reynolds' famous Boids algorithm, in which multitudes of particles are simulated interacting with each other via three simple rules:

1. Cohesion - each Boid steers itself towards the perceived center of mass of the group.
1. Separation - each Boid maintains some small distance away from other Boids (to prevent crashes).
1. Alignment - each Boid attempts to align its velocity with the perceived group velocity.

Following only these rules, and tuning weights for each rule carefully, we can achieve a group motion that resembles flocking (of birds, fish, etc.), as seen above.

Because each Boid can be simulated at a given timestep independently of all other Boids, this algorithm lends itself well to parallelization via the GPU. In this project, I have implemented the three different GPU-based approaches as follows:


### Naive (Brute-force) 

In this approach, the kernel which simulates each Boid does so by comparing a given Boid against every other Boid (more or less - a target Boid could be out of range, and its impact excluded, but is still considered nonetheless).

This approach has the advantage of simplicity; no extra data structures beyond the positions and velocities of all Boids are necessary to optimize this approach. The downside, of course, is the time complexity required. As the number of Boids in the simulation grows, the simulation time grows with its square.

### Scattered Uniform Grid

Since each of our aforementioned rules has an associated distance of effect, we don't *need* to compare every Boid to every other. We only need to compare Boids within a given vicinity. Thus, the second approach buckets Boids into cells of a uniform spatial grid (see image below). With a well-chosen grid cell width, we can limit our neighboring Boid checks to only those within the surrounding cells.

In theory, this optimization should increase performance, as we have fewer Boids to check against. In practice, we'll see below that the performance gains significant, but that there is still much room for improvement. The data structures used in the approach, combined with a need for a preprocessing sorting step, lead to scattered global memory accesses, whose latency offsets some of the gains from the use of a uniform grid.

![Uniform spatial grid](/images/Boids%20Ugrid%20base.png)

### Coherent Uniform Grid

We can improve on the scattered grid approach by changing the underlying data structures involved. By sorting the Boid position and velocity arrays on each timestep, and cutting out a middleman look-up table into these arrays, each Boid's information can be stored and accessed in contiguous global memory. 

The downsides to this approach are the code complexity involved and extra memory usage for book-keeping buffers, but the resulting performance gains are notable.


## Performance Analysis

In this section, I will analyze how each of the above approaches performs with respect to two variables: number of Boids simulated, and number of threads per GPU block. The metric of performance I use will be average framerate; the greater the framerate, the better the performance of the implementation. During measurements, visualation will be turned off so that the framerate is only a factor of the simulation speed. V-sync will also be turned off.

### Performance as a function of number of Boids simulated

<p align="center">
  <img src="/images/numBoids.svg" alt="Description of Image">
</p>

Unsurpsingly, increasing the number of Boids in the simulation decreases the framerate across all implementations. This makes sense, as the GPU has more work to do as more Boids are simulated, which will slow it down. It also appears that the naive implementation suffers most drastically - as is evident from the concavity of its curve. This also makes sense, since the Naive algorithm has O(N^2) time complexity, so it will suffer the most of all implementations.

Another trend of interest: the coherent uniform grid outperforms the scattered uniform grid across all numbers of Boids. However, the difference in performance is vanishing as the number of Boids decreases. I have three potential theories to explain this (none of which are mutually exclusive):

1. With a small enough number of Boids, a larger portion of Boid data can fit in cache memory / fewer cache misses, negating the impact of random global memory access.
1. With a small enough number of Boids, a larger portion of Boids can be simulated in parallel, so the latency of global memory access is less noticeable.
1. The coherent grid requires an extra sort step, which entails overhead. The cost of this extra sort is more noticeable with a smaller number of boids.

### Performance as a function of number of threads per block

<p align="center">
  <img src="/images/threadsPerBlock.svg" alt="Description of Image">
</p>

Little to talk about here - (perhaps) surprisingly, the number of threads per block has very little impact on the performance of the simulation. The most apparent answer here, is that the GPU is already saturated at the minimum tested 64 threads/block, and so increasing the number of threads per block any further does not help performance.

### Bonus: variable cell width

<p align="center">
  <img src="/images/cellWidth.svg" alt="Description of Image">
</p>

This graph analyzes the impact of the uniform grid's cell width on performance. In the above plot, cell width is a ratio where the denominator is the maximum distance by which a Boid will be influenced by another Boid. At a ratio of 1x or greater, a Boid will have to check at most 27 neighboring cells for other Boids (usually less, though). At a ratio of < 1x, a Boid has to check more and more cells (for instance, 0.5x ratio -> up to 125 cell checks).

On the one hand, more cell checks means more work. On the other hand, smaller cells means fewer boids per cell (and thus fewer boid checks). *Too many* cells to check is not good for performance, but *too few, big* cells to check is also bad (in the limit, it approaches one big cell, i.e. no grid at all!). The balance point, intuitively, lies around where the Boid search range equals the cell width. At this point, we don't have to perform too many extraneous cell checks, and we also don't have to perform too many extraneous Boid checks.

This is confirmed by the above plot, where the optimal cell width is shown to be around 1-2x the maximum Boid search distance.