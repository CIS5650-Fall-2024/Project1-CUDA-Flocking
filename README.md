**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Christine Kneer
  * https://www.linkedin.com/in/christine-kneer/
  * https://www.christinekneer.com/
* Tested on: Windows 11, i7-13700HX @ 2.1GHz 32GB, RTX 4060 8GB (Personal Laptop)

## Part 1: Introduction

In the Boids flocking simulation, particles representing birds or fish
(boids) move around the simulation space according to three rules:

1. cohesion - boids move towards the perceived center of mass of their neighbors
2. separation - boids avoid getting to close to their neighbors
3. alignment - boids generally try to move with the same direction and speed as
their neighbors

These three rules specify a boid's velocity change in a timestep.

### 1.1 Naive Boids Simulation

In the naive simulation, a boid simply looks every other boid and compute the velocity
change contribution from each of the three rules (if within distance).

Example simulation:

|![](images/naive.gif)|
|:--:|
|*5,000 boids, scene scale = 100*|

### 1.2 Uniform Grid Boids Simulation

Instead of examining every other boid, we use **uniform spatial grid** to cull each boid's neighbors.
If the cell width is double the neighborhood distance, each boid only has to be
checked against other boids in 8 cells.

Example simulation:

|![](images/uniform.gif)|
|:--:|
|*10,000 boids, scene scale = 100*|

### 1.3 Uniform Grid Boids Simulation (with coherent boid data)

Rearranging the boid data itself so that all the velocities and positions of boids in one cell are also
contiguous in memory.

Example simulation:

|![](images/coherent.gif)|
|:--:|
|*500,000 boids, scene scale = 200*|

## Part 2: Performance Analysis