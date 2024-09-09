<!-- <div style="background: rgba(100, 150, 223, 0.2); padding: 10px;">
<p>
  <b>
    University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
    Project 1 - Flocking
  </b>
</p>
<ul>
  <li>
    Michael Mason
    <ul>
      <li><a href="https://www.michaelmason.xyz/">Personal Website</a></li>
    </ul>
  </li>
  <li>Tested on: Windows 11, Ryzen 9 5900HS @ 3.00GHz 16GB, RTX 3080 (Laptop) 8192MB</li>
</ul>
</div> -->

> University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 1 - Flocking
> * Michael Mason
>   + [Personal Website](https://www.michaelmason.xyz/)
> * Tested on: Windows 11, Ryzen 9 5900HS @ 3.00GHz 16GB, RTX 3080 (Laptop) 8192MB 

# CUDA Boids

**This project is a CUDA implementation of a flocking simulation based on the Reynolds Boids algorithm, previewing two levels of optimization: using a uniform grid, and using a uniform grid with semi-coherent memory access. This was done as part of UPenn CIS 5650 (GPU Programming & Architecture).**

| *5,000 boids* | *50,000 boids* | *100,000 boids*
| :--: | :--: | :--:
| ![boids](boids5000.gif) | ![boids](boids50000.gif)  | ![boids](boids100000.gif) |



<!-- ## Table of Contents

- [TODO](#todo)
- [Analysis](#analysis) -->

## ☑️ Performance Results

#### Method Explanation

For all comparisons graphed below, the FPS was recorded over 12 seconds from program start time, then averaged.

### Boids vs FPS <span style="color: aqua;">*with visualization*</span>

![graph1](boids_vs_fps_with_vis.png)

Note: Block Size was 128. 

### Boids vs FPS <span style="color: crimson">*without visualization*</span>

![graph2](boids_vs_fps_without_vis.png)

Note: Block Size was 128. 

### Block Size vs FPS

![graph3](block_size_vs_fps.png)

Note: Used coherent grid with 100,000 boids, visualization on. 

## 📃 Performance Analysis

#### Q. For each implementation, how does changing the number of boids affect performance? Why do you think this is?
#### Q. For each implementation, how does changing the block count and block size affect performance? Why do you think this is?
#### Q. For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
#### Q. Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!