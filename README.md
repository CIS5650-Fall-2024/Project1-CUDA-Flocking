**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Joanna Fisch
  * [LinkedIn](https://www.linkedin.com/in/joanna-fisch-bb2979186/), [Website](https://sites.google.com/view/joannafischsportfolio/home)
* Tested on: Windows 11, i7-12700H @ 2.30GHz 16GB, NVIDIA GeForce RTX 3060 (Laptop)

### Introduction

In this project, I implement a flocking simulation based on the Reynolds Boids algorithm, along with two levels of optimization: a uniform grid, and a uniform grid with semi-coherent memory access. The flocking simulation uses 3 rules 

Rule 1: Cohesion - boids move towards the perceived center of mass of their neighbors

Rule 2: Separation - boids avoid getting to close to their neighbors

Rule 3: Alignment - boids generally try to move with the same direction and speed as their neighbors

We have three implementations of the algorithm Naive, uniform grid, and coherent grid.

Naive: To update the position of each boid in this implementation we check against each other boid causing a complexity of O(N^2). For small boid counts this implementation is more efficient because it avoids the preprocessing operations from sorting and setting up to read the grid. But for larger boid counts this implementation becomes much slower.

Uniform Grid: This implementation decreases the amount of boids we have to check each boid against by grouping the boids into cells and only checking the neighboring ones. This creates a slower preprocessing step which is less efficient on smaller boid counts but more efficient on larger boid counts.

Coherent Grid: This implementation works similary to the uniform grid with the added optimization of sorting the pos and vel arrays this allows a contiguous memory advantage which boosts its performance.

 <table>
  <tr>
    <td align="center"><b>Naive Flocking</b></td>
    <td align="center"><b>Uniform Grid-Based Flocking</b></td>
    <td align="center"><b>Coherent Grid-Based Flocking</b></td>
  </tr>
  <tr>
    <td><img src="images/naive.gif" /></td>
    <td><img src="images/uniformGrid.gif" /></td>
    <td><img src="images/coherentGrid.gif" /></td>
  </tr>
  <tr>
    <td colspan="3" align="center"><i>20,000 Boids, Screen Scale 100, 128 Blocks</i></td>
  </tr>
</table>

### Performance Analysis

* For each implementation, how does changing the number of boids affect performance? Why do you think this is?
<img src="images/boids_V.png" width=500>
<img src="images/boids_noV.png" width=500>

* For each implementation, how does changing the block count and block size affect performance? Why do you think this is?
<img src="images/blockSize.png" width=500>

* For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?

* Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!

