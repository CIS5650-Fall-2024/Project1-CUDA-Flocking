**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Nick Moon
  * [LinkedIn](https://www.linkedin.com/in/nick-moon1/), [personal website](https://nicholasmoon.github.io/)
* Tested on: Windows 10, AMD Ryzen 9 5900HS @ 3.0GHz 32GB, NVIDIA RTX 3060 Laptop 6GB (Personal Laptop)

### Results

![](images/5000000_boids.gif)
5000000 boids, 1.75 dt timestep, 400.0 scene scale, 1.5 max speed

![](images/5000_boids_02_dt.gif)
5000 boids, 0.2 dt timestep

![](images/5000_boids_1_dt.gif)
5000 boids, 1.0 dt timestep

![](images/500_boids_1_dt.gif)
500 boids, 1.0 dt timestep

![](images/500000_boids_1_dt.gif)
500000 boids, 1.0 dt timestep

![](images/quarter_distances_maxspeed.gif)
50000 boids, 1.0 dt timestep, 0.25 maxspeed, neighborhood search distances multiplied by 0.5

![](images/more_distance_half_maxspeed.gif)
50000 boids, 1.0 dt timestep, 0.5 maxspeed, neighborhood search distances multiplied by 1.35x

**Performance Analysis**

For each implementation, how does changing the number of boids affect performance? Why do you think this is?

![](images/boids_vs_fps.png)
![](images/boids_vs_fps_with_vis.png)

For each implementation, how does changing the block count and block size affect performance? Why do you think this is?

![](images/blocksize_vs_fps.png)

For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?


Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!

![](images/ratio_vs_fps.png)

### Bloopers

![](images/blooper_nightsky.PNG)
![](images/blooper_vacuum.PNG)
![](images/blooper_graydeath.PNG)
![](images/blooper1.gif)
![](images/blooper2.gif)
![](images/blooper3.gif)