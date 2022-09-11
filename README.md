**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Di Lu
  * (TODO) [LinkedIn](), [personal website](), [twitter](), etc.
* Tested on: Windows 11, i7-12700H @ 2.30GHz 32GB, NVIDIA GeForce RTX 3050 Ti

## Introduction

In this project, I started to learn how to write CUDA kernel functions and how to invoke them. 
The project simulates flocking behavior based on Craig Reynold's artificial life program, for which
a SIGGRAPH paper was written in 1989.

## Part 1. Naive Boids Simulation



Include screenshots, analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

## Part 3: Performance Analysis
For this project, we will guide you through your performance analysis with some
basic questions. In the future, you will guide your own performance analysis -
but these simple questions will always be critical to answer. In general, we
want you to go above and beyond the suggested performance investigations and
explore how different aspects of your code impact performance as a whole.

The provided framerate meter (in the window title) will be a useful base
metric, but adding your own `cudaTimer`s, etc., will allow you to do more
fine-grained benchmarking of various parts of your code.

REMEMBER:
* Do your performance testing in `Release` mode!
* Turn off Vertical Sync in Nvidia Control Panel:
![Unlock FPS](images/UnlockFPS.png)
* Performance should always be measured relative to some baseline when
  possible. A GPU can make your program faster - but by how much?
* If a change impacts performance, show a comparison. Describe your changes.
* Describe the methodology you are using to benchmark.
* Performance plots are a good thing.

### Questions

There are two ways to measure performance:
* Disable visualization so that the framerate reported will be for the the
  simulation only, and not be limited to 60 fps. This way, the framerate
  reported in the window title will be useful.
  * To do this, change `#define VISUALIZE` to `0`.
* For tighter timing measurement, you can use CUDA events to measure just the
  simulation CUDA kernel. Info on this can be found online easily. You will
  probably have to average over several simulation steps, similar to the way
  FPS is currently calculated.

This section will not be graded for correctness, but please let us know your
hypotheses and insights.

**Answer these:**

* For each implementation, how does changing the number of boids affect
performance? Why do you think this is?
* For each implementation, how does changing the block count and block size
affect performance? Why do you think this is?
* For the coherent uniform grid: did you experience any performance improvements
with the more coherent uniform grid? Was this the outcome you expected?
Why or why not?
* Did changing cell width and checking 27 vs 8 neighboring cells affect performance?
Why or why not? Be careful: it is insufficient (and possibly incorrect) to say
that 27-cell is slower simply because there are more cells to check!

**NOTE: Nsight performance analysis tools *cannot* presently be used on the lab
computers, as they require administrative access.** If you do not have access
to a CUDA-capable computer, the lab computers still allow you to do timing
mesasurements! However, the tools are very useful for performance debugging.
