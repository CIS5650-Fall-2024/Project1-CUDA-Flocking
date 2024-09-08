**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Logan Cho
  * [LinkedIn](https://www.linkedin.com/in/logan-cho/)
  * [Personal Website](https://www.logancho.com/)
* Tested on: Windows 11, 13th Gen Intel(R) Core(TM) i7-13700H, 2.40 GHz, RTX 4060 Laptop GPU
# Results

![boids6](https://github.com/user-attachments/assets/202a3911-a9ff-452c-bc4f-df4e9375f3d2)

# Analysis
## Charts:
![](images/Chart1.png)
![](images/Chart2.png)

#### *All Average FPS values were captured over 15, 2-second increments after an initial 60 second waiting period to allow for some level of convergence/stabilisation.

## Questions:
 * For each implementation, how does changing the number of boids affect performance? Why do you think this is?
   * For all of them, increasing the # of boids reduced performance/FPS. This is because of multiple reasons, including, an increased amount of work/instruction count required to be executed, and thus, more blocks and/or potentially more instructions per warp as well because of increased # of neighbor checks required from higher density. Additionally, a larger # of boids requires the buffers, such as pos and velocity buffers, to take up more global memory device space, which can restrict the amount of free memory, and also, increases the # of costly global memory operations.
 * For each implementation, how does changing the block count and block size affect performance? Why do you think this is?
   * From my personal results, changing the block size from 128 to 256 did not affect performance that much in any of the implementations (only decreasing very slightly.) This could be simply due to the testing range not being large enough to show a difference. That is, as we learnt from lecture, block size and block count usually play a big part in performance beyond a certain threshold where either or both variables are configured in an unoptimal way that forces serialization of blocks due to limited hardware resources on the available SMs.
 * For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
   * Yes, implementing 2.3 brought significant improvements to performance compared to the scattered, uniform grid approach. This was the outcome I expected. This is because, based on the context of this application, I believed that the benefits of contiguous memory reads would outweight any negatives from the extra instructions carried out to shuffle the boids, or the extra global memory taken up by the 'extraBuffer' I use as a swapbuffer to support the coherent reshuffling. 
 * Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!
   * Personally, I was not able to see a significant performance improvement between changing from checking 27 and 8 neighboring cells. I believe the reason for this was because there were more significant bottlenecks at play, such as global memory usage by Boid count, and etc.
   * Increasing cell width had the effect of more significantly reducing performance. This was in line with my expectations, since, increasing cell width beyond 2 * the max search radius reduced the usefulness of the grid system for reducing the # of boids to parse through. With a larger cell width, there are less cells, and more boids in each cell, meaning despite the reduced # of neighbor cells needed to be parsed through, we now have to parse through a much higher # of boids in the cell our target boid resides in, for every single boid, every frame. The minimum # of boids to be processed is increased, and thus, performance is reduced.

[FPS Data Spreadsheets](https://docs.google.com/spreadsheets/d/1qbjtnsCArbFfQOC4BTniKjXvcprzLOgFy3CWVulg-20/edit?usp=sharing)
