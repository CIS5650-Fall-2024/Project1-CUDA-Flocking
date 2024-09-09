**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Joanna Fisch
  * [LinkedIn](https://www.linkedin.com/in/joanna-fisch-bb2979186/), [Website](https://sites.google.com/view/joannafischsportfolio/home)
* Tested on: Windows 11, i7-12700H @ 2.30GHz 16GB, NVIDIA GeForce RTX 3060 (Laptop)

### Introduction

In this project, I implement a flocking simulation based on the Reynolds Boids algorithm, along with two levels of optimization: a uniform grid, and a uniform grid with semi-coherent memory access. The flocking simulation uses 3 rules 

Rule 1: Cohesion - boids fly towards perceived center of mass of local neighbors

Rule 2: Separation - boids are directed away from each other if they get too close

Rule 3: Alignment - boids try to match the velocity of their neighbors

![image](https://github.com/user-attachments/assets/d3629244-95a5-4c00-8fb3-d3069162f97a)

![image](https://github.com/user-attachments/assets/094207bd-2b86-4840-b8af-81e9b4da7d75)
