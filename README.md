**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Shutong Wu
  * [LinkedIn](https://www.linkedin.com/in/shutong-wu-214043172/)
  * [Email](shutong@seas.uepnn.edu)
* Tested on: Windows 10, i7-10700K CPU @ 3.80GHz, RTX3080, SM8.6, Personal Computer 

### Showcase
-20K Boids Naive 
![20KNaive](./images/20knaive.gif)
-50K Boids Naive
![50KNaive](./images/50knaive.gif)
-200K Boids Uniform
![200KUniformGrid](./images/200kuni.gif)
-500K Boids Coherent
![500KCoherentGrid](./images/500kco.gif)

## Q&A

###  For each implementation, how does changing the number of boids affect performance? Why do you think this is?
In all, The performance goes down(FPS) as the number of boids become larger. 
When boids number are relatively small(like under 10K boids), the change of the number does not affect the performance that much, but from 10K to 50K then to 100K, the frame rate will drop significantly, from 500 to 100 and then to 10FPS. 
The reason is that with more boids, more computations are needed especially in terms of computing boids velocity, since there are so much more boids around one single boid in a 100K scenario than in a 10K scenario. 

###  For each implementation, how does changing the block count and block size affect performance? Why do you think this is?



###  For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
The performance improves slightly();

###  Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!





### Performance Analysis
####Mainly use FPS to test performance;
####

