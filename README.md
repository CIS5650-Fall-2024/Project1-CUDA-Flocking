**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Shutong Wu
  * [LinkedIn](https://www.linkedin.com/in/shutong-wu-214043172/)
  * [Email](shutong@seas.uepnn.edu)
* Tested on: Windows 10, i7-10700K CPU @ 3.80GHz, RTX3080, SM8.6, Personal Computer 

### Showcase
-5K Boids Naive 
![5KNaive](./images/5k.gif)
-50K Boids Naive
![50KNaive](./images/50knaive.gif)
-200K Boids Uniform
![200KUniformGrid](./images/200kuni.gif)
-500K Boids Coherent
![500KCoherentGrid](./images/500kco.gif)

###  For the questions

##  For each implementation, how does changing the number of boids affect performance? Why do you think this is?
The performance goes down(FPS) as the number of boids become larger; the performance will not significantly go down when the number of boids are not 
##  For each implementation, how does changing the block count and block size affect performance? Why do you think this is?
##  For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
##  Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!




### Performance Analysis

Include screenshots, analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)
