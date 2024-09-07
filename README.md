**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

## YUHAN LIU
  * [LinkedIn](https://www.linkedin.com/in/yuhan-liu-/), [personal website](https://liuyuhan.me/)
  * Tested on: Windows 11 Pro, Ultra 7 155H @ 1.40 GHz 32GB, RTX 4060 8192MB (Personal Laptop)

<p>
<img src="https://github.com/yuhanliu-tech/GPU-CUDA-Flocking/blob/main/images/screenshot.png" width="500"/>
<em>Screenshot of Boids simulation with 100,000 particles.</em>
</p>

## Performance Analysis

For each implementation, how does changing the number of boids affect performance? Why do you think this is?
* 

<img src="https://github.com/yuhanliu-tech/GPU-CUDA-Flocking/blob/main/images/FPS_boidnum.png" width="1000"/>

For each implementation, how does changing the block count and block size affect performance? Why do you think this is?
* 

<img src="https://github.com/yuhanliu-tech/GPU-CUDA-Flocking/blob/main/images/FPS_blocksize.png" width="500"/>

For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
* 

Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!
* 

<p>
<img src="https://github.com/yuhanliu-tech/GPU-CUDA-Flocking/blob/main/images/gif.gif" width="500"/>
<em>GIF of Boids simulation with 10,000 particles.</em>
</p>
