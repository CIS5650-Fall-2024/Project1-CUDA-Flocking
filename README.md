**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**
======================

* Wenqing Wang
  * [LinkedIn](https://www.linkedin.com/in/wenqingwang0910/) 
* Tested on: Windows 11, i7-11370H @ 3.30GHz 16.0 GB, GTX 3050 Ti

## Screenshots
* Coherent with 10K boids
![1](https://user-images.githubusercontent.com/33616958/189548869-6924fda7-1c0e-4308-952d-dffebb1ec029.gif)
* Coherent with 20K boids
![2](https://user-images.githubusercontent.com/33616958/189548865-61dd7752-f4a0-45aa-a383-f948aa85e920.gif)
* Coherent with 80K boids
![8](https://user-images.githubusercontent.com/33616958/189548868-0981d6cb-dec3-4cd5-b6ca-efe10a81c999.gif)



## Performance Analysis
![fps_w_v](https://user-images.githubusercontent.com/33616958/189547897-78ed6b50-76d0-4bb7-90e3-e1e491814548.png)

After disabling visualization, the framerates reported below are for the the simulation only:
![fps_wo_v](https://user-images.githubusercontent.com/33616958/189547898-3ca487ae-1ada-4b53-90f0-550108a8399c.png)

![fps_w_blocksize](https://user-images.githubusercontent.com/33616958/189547900-52a10a80-40e4-4ddc-af58-9eb90d97be9c.png)

* Questions 
1. For each implementation, how does changing the number of boids affect performance? Why do you think this is?
   - From the plots above, we can see that for all 3 methods, the average frame per second decreases as the # of boids increase. That's becasuse we'll need to process more data as the # of boids increases. When we switch from the naive method to the uniform grid search, the performance improves because instead of performing a brute force search to check every rule for every 2 boids, we check the 27 neighbor cells of each boid, which greatly reduces the simulation effort.
2. For each implementation, how does changing the block count and block size affect performance? Why do you think this is?
   - It seems that changing the block size doesn't have much impact on performance (at least no clear pattern). I think this is because it does not affect the total data we need to process or the number of threads needed to process these data.
3. For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
   - Yes, the performance imporves as we reranging the data buffer in the coherent uniform grid method. This is because we no longer need to get the boid index from the `dev_particleArrayIndices` buffer, which reduces the data access operations.
4. Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!
   - Changing the cell width of the uniform grid to be the neighborhood distance and check 27 cells instead of 8 improve the performance on my laptop. I suspect this is because although we checked more cells, since we reduced the cell width of the uniform grid to half the original size, we actually checked a smaller volume (contains less boids) each time.
