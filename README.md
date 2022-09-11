**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

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
From the plots below, we can see that for all 3 methods, the average frame per second decreases as the # of boids increase. That's becasuse we'll need to process more data as the # of boids increases. When we switch from the naive method to the uniform grid search, the performance improves because instead of performing a brute force search to check every rule for every 2 boids, we check the 27 neighbor cells of each boid, which greatly reduces the simulation effort. The performance was further improved after we optimized the data access method in the coherent method.
![fps_w_v](https://user-images.githubusercontent.com/33616958/189547897-78ed6b50-76d0-4bb7-90e3-e1e491814548.png)

After disabling visualization, the framerates reported below are for the the simulation only:
![fps_wo_v](https://user-images.githubusercontent.com/33616958/189547898-3ca487ae-1ada-4b53-90f0-550108a8399c.png)

Also, it seems that changing the block size doesn't have much impact on performance. I think this is because it does not affect the total data we need to process or the number of threads needed to process it.
![fps_w_blocksize](https://user-images.githubusercontent.com/33616958/189547900-52a10a80-40e4-4ddc-af58-9eb90d97be9c.png)

