**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**
- Name: Tongwei Dai
	- [LinkedIn Page](https://www.linkedin.com/in/tongwei-dai-583350177/)
- Tested on: Windows 10, i7-8700 @ 3.20 GHz 16GB, RTX 2070

## Changes to Source Code Besides Requirement
- modified line 97 of the file `cmake/CUDAComputesList.cmake`
	- because my hardware or CUDA do not support `compute_30`
	- see the [Ed Question Thread](https://edstem.org/us/courses/28083/discussion/1723078) for more info
- modified `CMakeLists.txt`
	- added a new header `profiling.h` for performance data collection

## Performance Analysis
- Framerate is used as the primary metric to measure performance
	- The program is run for **10** seconds, and the average framerate is chosen.
- Two factors, the number of boids and thread block size, are investigated for their impact on the performance of the simulation
- Each section lists the framerate vs. factor graph for each of three implementations, i.e., Naive, Naive Uniform Grid, Coherent Uniform Grid.

### Performance Impact by The Number of Boids