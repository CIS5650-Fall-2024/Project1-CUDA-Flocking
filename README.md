# University of Pennsylvania, CIS 5650: GPU Programming and Architecture  
**Project 1 - Flocking**

* **Your Name**: Yi Liu  
* **Tested on**: Windows 11, Intel i9-12900K @ 3.20GHz 32GB, NVIDIA RTX 4090 24GB (Moore 100 Lab)

---

## Overview

In this project, I implemented a GPU-based simulation of flocking behavior using CUDA. The simulation demonstrates how boids (bird-like objects) follow three basic rules:  
1. **Separation**: Boids avoid getting too close to their neighbors.  
2. **Alignment**: Boids generally try to move with the same direction and speed as their neighbors.  
3. **Cohesion**: Boids move towards the perceived center of mass of their neighbors.

---

## Features

- **CUDA-based parallel computation**: The simulation leverages the GPU to process thousands of boids in parallel.
- **Uniform Grid Optimization**: Implemented a uniform grid to partition space, reducing the number of interactions each boid must check, thereby improving performance.
- **Real-time visualization**: Utilized OpenGL to visualize boid movement in real-time, allowing interactive exploration of flocking behavior.

---

## Implementations

### 1. Naive Boids Algorithm Implementation
The naive approach calculates interactions between all pairs of boids, leading to a computational complexity of O(n^2).


### 2. Uniform Grid Scattered Search
This method uses a uniform grid to partition space and limit boid interactions to neighboring grid cells, improving efficiency.

### 3. Coherent Grid Memory Access
Optimized memory access by sorting boids based on their grid cells, enhancing data locality and overall performance.

---

## Visualization

![50000 Boids Flocking](N50000_CoherentGrid.gif)  
*Caption: 50,000 Boids Flocking using the Coherent Grid Approach*
