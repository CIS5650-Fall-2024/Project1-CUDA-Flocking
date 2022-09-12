**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**


* Yilin Liu
  * [LinkedIn](https://www.linkedin.com/in/yilin-liu-9538ba1a5/), [Personal Website](https://yilin.games/)
* Tested on: Windows 10, i7-10750H @ 2.59GHz 32GB, GeForce RTX 2070 with Max-Q 8GB (Personal Laptop) 

# Project 1 Results

##  Screenshots:
|![image](./images/1M_200Scale.gif)|
|:--:| 
| *1M boids with Uniform Coherent Grid* |


|![image](./images/200k%20uniform%20grid.gif)|
|:--:| 
| *200k boids with Unifrom Grid* |


## Analysis

|![image](./images/naive_table.png)|
|:--:| 
| *Naive method* |

|![image](./images/discrete_table.png)|
|:--:| 
| *Uniform Grid Scattered method* |

|![image](./images/coherent_table.png)|
|:--:| 
| *Uniform Grid Coherent method* |

|![image](./images/Framerate%20vs%20Methods.png)|
|:--:| 
| *Comparision of three methods* |

|![image](./images/block_Size.png)|
|:--:| 
| *Effects of Block SIze* |

## Answers:

- For each implementation, how does changing the number of boids affect performance? Why do you think this is?
  
  **Answer**: 
For the naïve implementation, increasing the number of boids will significantly affect performance. The complexity O(n^2) is exponential related to the number of boids since we need to calculate the distance between every boids
For uniform grid methods, the effect is not that obvious as the naive implementation since we only calculate the distance between boids among 8 out of 27 grids. 

- For each implementation, how does changing the block count and block size affect performance? Why do you think this is?
 
  **Answer**: For all the implementations, increasing the block size before 64 will increase the performance, However, after block size of 64, increasing the size will slightly reduce the performance. This could be explained by the idle thread wasted.

- For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
 
 **Answer**: 
  Coherent uniform grid's influence on performance improves as the number of boids increases. It can be explained by the increasing cost of data transfer for large number of boids. 

- Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!

 **Answer**: 
Changing to 27 cells will significantly improve the performance. This effect is especially obvious when boids are super-dense in the grid, because simply searching through 27 cells may be more efficient than calculating the nearest neighborhoods. 