#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include <device_launch_parameters.h>
#include "kernel.h"

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f // TEST-2.1.1 - 10.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;   // TEST-2.1.1 (int)(scene_scale / gridCellWidth)
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.

  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");
  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");
  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");
  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);

  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

__device__ glm::vec3 computeVelocityChangeRule1(int N, int iSelf, const glm::vec3* pos, const glm::vec3* vel) {
    // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
    int count = 0;
    glm::vec3 perceived_center(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < N; i++) {
        if (i != iSelf && glm::distance(pos[iSelf], pos[i]) < rule1Distance) {
            perceived_center += pos[i];
            count++;
        }
    }
    perceived_center /= count;
    return (perceived_center - pos[iSelf]) * rule1Scale;
}

__device__ glm::vec3 computeVelocityChangeRule2(int N, int iSelf, const glm::vec3* pos, const glm::vec3* vel) {
    // Rule 2: boids try to stay a distance d away from each other
    glm::vec3 c(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < N; i++) {
        if (i != iSelf && glm::distance(pos[iSelf], pos[i]) < rule2Distance) {
            c -= (pos[i] - pos[iSelf]);
        }
    }
    return c  * rule2Scale;
}

__device__ glm::vec3 computeVelocityChangeRule3(int N, int iSelf, const glm::vec3* pos, const glm::vec3* vel) {
    // Rule 3: boids try to match the speed of surrounding boids
    int count = 0;
    glm::vec3 perceived_velocity(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < N; i++) {
        if (i != iSelf && glm::distance(pos[iSelf], pos[i]) < rule3Distance) {
            perceived_velocity += vel[i];
            count++;
        }
    }
    perceived_velocity /= count;
    return perceived_velocity * rule3Scale;
}

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
    glm::vec3 dv_rule1 = computeVelocityChangeRule1(N, iSelf, pos, vel);
    glm::vec3 dv_rule2 = computeVelocityChangeRule2(N, iSelf, pos, vel);
    glm::vec3 dv_rule3 = computeVelocityChangeRule3(N, iSelf, pos, vel);
  return dv_rule1 + dv_rule2 + dv_rule3;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
    // Compute a new velocity based on pos and vel1
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }
    glm::vec3 thisVel(0.0f, 0.0f, 0.0f);
    thisVel = computeVelocityChange(N, index, pos, vel1) + vel1[index];

    // Clamp the speed
    float magnitude = glm::length(thisVel);
    if (magnitude > maxSpeed) {
        thisVel.x = (thisVel.x / magnitude) * maxSpeed;
        thisVel.y = (thisVel.y / magnitude) * maxSpeed;
        thisVel.z = (thisVel.z / magnitude) * maxSpeed;
    }

    // Record the new velocity into vel2. Question: why NOT vel1?
    vel2[index] = thisVel;
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    // - Label each boid with the index of its grid cell.
    int ix = 0, iy = 0, iz = 0;
    ix = floor((pos[index].x - gridMin.x) * inverseCellWidth);
    iy = floor((pos[index].y - gridMin.y) * inverseCellWidth);
    iz = floor((pos[index].z - gridMin.z) * inverseCellWidth);

    gridIndices[index] = gridIndex3Dto1D(ix, iy, iz, gridResolution);

    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
    indices[index] = index;
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) { // TEST 2.1.1 additional parameter int *idx
  
    /* ***********************************************************************
       THE FOLLOWING COMMENTED LOGIC DOES NOT WORK BECAUSE EACH index GETS
       COMPARED TO THE INITIALIZED VALUES PARALELLY
       FOR EXAMPLE, IN CASE OF END INDICES, EACH index GETS COMPARED TO -1
       AND THE FINAL VALUE SIMPLY DEPENDS ON WHICH THREAD GOT WRITE ACCESS FIRST

       ***********************************************************************
    */
    
    // TODO-2.1
    // Identify the start point of each cell in the gridIndices array.
    // This is basically a parallel unrolling of a loop that goes
    // "this index doesn't match the one before it, must be a new cell!"
    //int index = (blockIdx.x * blockDim.x) + threadIdx.x;    // 3
    //int gridNum = particleGridIndices[index];   // 5
    //// TEST-2.1.1 idx[index] = gridNum;
    //    
    //if (index < gridCellStartIndices[gridNum]) {    // 25
    //    gridCellStartIndices[gridNum] = index;      // startindex[5] = 25
    //}
    //if (index > gridCellEndIndices[gridNum]) {      // endindices  = - 1
    //    gridCellEndIndices[gridNum] = index;        // endindex[5] = 3
    //}


    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {   // N is Grid Cell
        return;
    }

    int thisGridIdxVal = particleGridIndices[index];
    if (index == 0) {
        int nextGridIdxVal = particleGridIndices[index + 1];

        gridCellStartIndices[thisGridIdxVal] = index;
        if (thisGridIdxVal != nextGridIdxVal) {
            gridCellEndIndices[thisGridIdxVal] = index;
        }
    }

    else if (index == (N - 1)) {
        int prevGridIdxVal = particleGridIndices[index - 1];

        gridCellEndIndices[thisGridIdxVal] = index;
        if (thisGridIdxVal != prevGridIdxVal) {
            gridCellStartIndices[thisGridIdxVal] = index;
        }
    }

    else {
        int prevGridIdxVal = particleGridIndices[index - 1];
        int nextGridIdxVal = particleGridIndices[index + 1];

        if (thisGridIdxVal != prevGridIdxVal) {
            gridCellStartIndices[thisGridIdxVal] = index;
        }

        if (thisGridIdxVal != nextGridIdxVal) {
            gridCellEndIndices[thisGridIdxVal] = index;
        }
    }

}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  
    /* ***********************************************************************
       THE FOLLOWING COMMENTED CODE DOES NOT WORK DUE TO SOME BUGS.
       CONCEPTUALLY, IT TRIES TO IDENTIFY THE QUADRANT IN WHICH EACH BOID
       IS PRESENT WITHIN A CELL BASED ON DIFFERENCE BETWEEN POSITION OF BOID
       AND THE GLOBAL CENTER COORDINATES OF ITS CELL.
       BASED ON THIS DIFFERENCE, THE DIRECTION IN WHICH NEIGHBORS SHOULD BE
       SEARCHED IS IDENTIFIED.

      ************************************************************************
    */

    /*
    // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
    // the number of boids that need to be checked.
    // - Identify the grid cell that this particle is in
    
      int thisBoidIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
      int thisBoid = particleArrayIndices[thisBoidIndex];
      if (thisBoidIndex >= N) {
          return;
      }
      int ix = 0, iy = 0, iz = 0;
      ix = floor((pos[thisBoid].x - gridMin.x) * inverseCellWidth);
      iy = floor((pos[thisBoid].y - gridMin.y) * inverseCellWidth);
      iz = floor((pos[thisBoid].z - gridMin.z) * inverseCellWidth);
      int totalCells = gridResolution * gridResolution * gridResolution;
      int gridNum = gridIndex3Dto1D(ix, iy, iz, gridResolution);
    
    // - Identify which cells may contain neighbors. This isn't always 8.
      glm::vec3 cellCenter(0.0, 0.0, 0.0);
      cellCenter.x = (ix * cellWidth + gridMin.x) + (cellWidth / 2);
      cellCenter.y = (iy * cellWidth + gridMin.y) + (cellWidth / 2);
      cellCenter.z = (iz * cellWidth + gridMin.z) + (cellWidth / 2);
    
      int dirx = 0, diry = 0, dirz = 0;
      dirx = (pos[thisBoid].x - cellCenter.x > 0) ? 1 : -1;
      diry = (pos[thisBoid].y - cellCenter.y > 0) ? 1 : -1;
      dirx = (pos[thisBoid].z - cellCenter.z > 0) ? 1 : -1;
    
      int neighbours[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
    
      neighbours[0] = gridIndex3Dto1D(ix, iy, iz, gridResolution);
      neighbours[1] = gridIndex3Dto1D(ix, iy, dirz + iz, gridResolution);
      neighbours[2] = gridIndex3Dto1D(ix, diry + iy, iz, gridResolution);
      neighbours[3] = gridIndex3Dto1D(ix, diry + iy, dirz + iz, gridResolution);
    
      neighbours[4] = gridIndex3Dto1D(dirx + ix, iy, iz, gridResolution);
      neighbours[5] = gridIndex3Dto1D(dirx + ix, iy, dirz + iz, gridResolution);
      neighbours[6] = gridIndex3Dto1D(dirx + ix, diry + iy, iz, gridResolution);
      neighbours[7] = gridIndex3Dto1D(dirx + ix, diry + iy, dirz + iz, gridResolution);
    
    // - For each cell, read the start/end indices in the boid pointer array.
    
      int gridStart = -1;
      int gridEnd = -1;
    
      glm::vec3 perceived_center(0.0f, 0.0f, 0.0f);
      glm::vec3 c(0.0f, 0.0f, 0.0f);
      glm::vec3 perceived_velocity(0.0f, 0.0f, 0.0f);
    
      int rule1_count = 0, rule3_count = 0;
    
      glm::vec3 v1(0.0, 0.0, 0.0);
      glm::vec3 v2(0.0, 0.0, 0.0);
      glm::vec3 v3(0.0, 0.0, 0.0);
      glm::vec3 thisVel(0.0f, 0.0f, 0.0f);
    
      for (int i = 0; i < 8; i++) {
          if (neighbours[i] >= 0 && neighbours[i] < totalCells) {
              gridStart = gridCellStartIndices[neighbours[i]];
              gridEnd = gridCellEndIndices[neighbours[i]];
    
              // - Access each boid in the cell and compute velocity change from
              //   the boids rules, if this boid is within the neighborhood distance.
    
              for (int j = gridStart; j < gridEnd; j++) {
                  // particleArrayIndices[j] gives me each boid in a gridcell
    
                  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
                  if (thisBoid != particleArrayIndices[j] && glm::distance(pos[thisBoid], pos[particleArrayIndices[j]]) < rule1Distance) {
                      perceived_center += pos[particleArrayIndices[j]];
                      rule1_count++;
                  }
    
                  // Rule 2: boids try to stay a distance d away from each other
                  if (thisBoid != particleArrayIndices[j] && glm::distance(pos[thisBoid], pos[particleArrayIndices[j]]) < rule2Distance) {
                      c -= (pos[particleArrayIndices[j]] - pos[thisBoid]);
                  }
    
                  // Rule 3: boids try to match the speed of surrounding boids
                  if (thisBoid != particleArrayIndices[j] && glm::distance(pos[thisBoid], pos[particleArrayIndices[j]]) < rule3Distance) {
                      perceived_velocity += vel1[particleArrayIndices[j]];
                      rule3_count++;
                  }
              }
          }
      }
      if (rule1_count > 0) {
          perceived_center /= rule1_count;
      }
      v1 = (perceived_center - pos[thisBoid]) * rule1Scale;
      v2 = c * rule2Scale;
      
      if (rule3_count > 0) {
          perceived_velocity /= rule3_count;
      }
    
      v3 = perceived_velocity * rule3Scale;
      thisVel = vel1[thisBoid] + v1 + v2 + v3;
    
      // - Clamp the speed change before putting the new speed in vel2
      float magnitude = glm::length(thisVel);
      if (magnitude > maxSpeed) {
          thisVel.x = (thisVel.x / magnitude) * maxSpeed;
          thisVel.y = (thisVel.y / magnitude) * maxSpeed;
          thisVel.z = (thisVel.z / magnitude) * maxSpeed;
      }
      vel2[thisBoid] = thisVel;
    */

    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    glm::vec3 gridIndex3D = glm::floor((pos[index] - gridMin) * inverseCellWidth);
    glm::vec3 cellCenter = (gridIndex3D * cellWidth) + gridMin + glm::vec3((cellWidth * 0.5), (cellWidth * 0.5), (cellWidth * 0.5));
    glm::vec3 minIdx = gridIndex3D;
    glm::vec3 increment = glm::vec3(2.f, 2.f, 2.f);

    // Updating min index while checking for min boundaries
    if (pos[index].x < cellCenter.x) {
        if (minIdx.x > 0)
            minIdx.x--;
        else
            increment.x--;
    }
    if (pos[index].y < cellCenter.y && minIdx.y > 0) {
        if (minIdx.y > 0)
            minIdx.y--;
        else
            increment.y--;
    }
    if (pos[index].z < cellCenter.z && minIdx.z > 0) {
        if (minIdx.z > 0)
            minIdx.z--;
        else
            increment.z--;
    }

    // Updating increment while checking for max boundaries
    if (minIdx.x + 2 > gridResolution)
    increment.x--;
    if (minIdx.y + 2 > gridResolution)
    increment.y--;
    if (minIdx.z + 2 > gridResolution)
    increment.z--;

    glm::vec3 velocityChange = glm::vec3(0.0f, 0.0f, 0.0f);

    glm::vec3 perceived_center = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 separation = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 perceived_velocity = glm::vec3(0.0f, 0.0f, 0.0f);
    int com_neighbors = 0;
    int avg_vel_neighbors = 0;

    for (int z = minIdx.z; z < minIdx.z + increment.z; z++) {
        for (int y = minIdx.y; y < minIdx.y + increment.y; y++) {
            for (int x = minIdx.x; x < minIdx.x + increment.x; x++) {

                // get start & indices of boid indices array
                int gridIndex = gridIndex3Dto1D(x, y, z, gridResolution);

                int startIdx = gridCellStartIndices[gridIndex];
                int endIdx = gridCellEndIndices[gridIndex];

                if (startIdx == -1) {
                    continue;
                }

                // for all boids in the cell accumulate values
                for (int i = startIdx; i <= endIdx; i++) {
                    int boidIndex = particleArrayIndices[i];

                    if (boidIndex == index) continue;

                    if (glm::length(pos[boidIndex] - pos[index]) < rule1Distance) {
                        perceived_center += pos[boidIndex];
                        com_neighbors++;
                    }

                    if (glm::length(pos[boidIndex] - pos[index]) < rule2Distance) {
                        separation -= (pos[boidIndex] - pos[index]);
                    }

                    if (glm::length(pos[boidIndex] - pos[index]) < rule3Distance) {
                        perceived_velocity += vel1[boidIndex];
                        avg_vel_neighbors++;
                    }
                }
            }
        }
    }

    // adding rule 1 vel change
    if (com_neighbors > 0) {
        perceived_center /= com_neighbors;
        velocityChange += ((perceived_center - pos[index]) * rule1Scale);
    }

    // adding rule 3 vel change
    if (avg_vel_neighbors > 0) {
        perceived_velocity /= avg_vel_neighbors;
        velocityChange += (perceived_velocity * rule3Scale);
    }

    // adding rule 2 vel change
    velocityChange += (separation * rule2Scale);

    // new velocity
    glm::vec3 newVelocity = vel1[index] + velocityChange;
    if (glm::length(newVelocity) > maxSpeed) {
        newVelocity = newVelocity / glm::length(newVelocity) * maxSpeed;
    }

    vel2[index] = newVelocity;
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
    
    // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    kernUpdateVelocityBruteForce << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos, dev_vel1, dev_vel2);
    checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");
    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel1);
    checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");
    
    // TODO-1.2 ping-pong the velocity buffers
    glm::vec3* temp;
    temp = dev_vel1;
    dev_vel1 = dev_vel2;
    dev_vel2 = temp;

}

void Boids::stepSimulationScatteredGrid(float dt) {
    // TODO-2.1
    // Uniform Grid Neighbor search using Thrust sort.
    // In Parallel:
    // - label each particle with its array index as well as its grid index.
    //   Use 2x width grids.
    
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    kernComputeIndices <<< fullBlocksPerGrid, blockSize >>> (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
                                                                dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
    checkCUDAErrorWithLine("kernComputeIndices failed!");

    //// - Unstable key sort using Thrust. A stable sort isn't necessary, but you
    ////   are welcome to do a performance comparison.
    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

    //// - Naively unroll the loop for finding the start and end indices of each
    ////   cell's data pointers in the array of boid indices
    
    dim3 fullBlocksPerGridCells((gridCellCount + blockSize - 1) / blockSize);
    kernResetIntBuffer << < fullBlocksPerGridCells, blockSize >> > (gridCellCount, dev_gridCellStartIndices, gridCellCount + 10);
    checkCUDAErrorWithLine("kernResetIntBuffer failed!");
    kernResetIntBuffer << < fullBlocksPerGridCells, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);
    checkCUDAErrorWithLine("kernResetIntBuffer failed!");

    kernIdentifyCellStartEnd <<< fullBlocksPerGrid, blockSize >>> (numObjects, dev_particleGridIndices,
                                                                    dev_gridCellStartIndices, dev_gridCellEndIndices);
    checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");
    
    //// - Perform velocity updates using neighbor search
    kernUpdateVelNeighborSearchScattered << < fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum,
                                                                                    gridInverseCellWidth, gridCellWidth,
                                                                                    dev_gridCellStartIndices, dev_gridCellEndIndices,
                                                                                    dev_particleArrayIndices,
                                                                                    dev_pos, dev_vel1, dev_vel2);

    checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

    //// - Update positions
    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel1);
    checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");
  
    //// - Ping-pong buffers as needed
    glm::vec3* temp;
    temp = dev_vel1;
    dev_vel1 = dev_vel2;
    dev_vel2 = temp;
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);

}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }


  //// test compute indices
  //glm::vec3* dev_test_pos;
  //int* dev_test_particleArrayIndices, * dev_test_particleGridIndices;
  //N = 3;

  //std::unique_ptr<int[]>test_particleArrayIndices{ new int[N] };
  //std::unique_ptr<int[]>test_particleGridIndices{ new int[N] };
  //std::unique_ptr<glm::vec3[]>test_pos{ new glm::vec3[N] };

  //test_pos[0].x = 1.1; test_pos[0].y = 1.2; test_pos[0].z = 1.3;
  //test_pos[0].x = 2.1; test_pos[0].y = 2.2; test_pos[0].z = 2.3;
  //test_pos[0].x = 3.1; test_pos[0].y = 3.2; test_pos[0].z = 3.3;

  //cudaMalloc((void**)&dev_test_pos, N * sizeof(glm::vec3));
  //checkCUDAErrorWithLine("cudaMalloc dev_test_pos failed!");
  //cudaMalloc((void**)&dev_test_particleArrayIndices, N * sizeof(int));
  //checkCUDAErrorWithLine("cudaMalloc dev_test_particleArrayIndices failed!");
  //cudaMalloc((void**)&dev_test_particleGridIndices, N * sizeof(int));
  //checkCUDAErrorWithLine("cudaMalloc dev_test_particleGridIndices failed!");

  //// How to copy data to the GPU
  //cudaMemcpy(dev_test_pos, test_pos.get(), sizeof(glm::vec3) * N, cudaMemcpyHostToDevice);

  //kernComputeIndices << < fullBlocksPerGrid, blockSize >> > (N, gridCellCount, gridMinimum, gridInverseCellWidth,
  //    dev_test_pos, dev_test_particleArrayIndices, dev_test_particleGridIndices);
  //checkCUDAErrorWithLine("kernComputeIndices failed!");
  //
  //// How to copy data back to the CPU side from the GPU
  //cudaMemcpy(test_particleArrayIndices.get(), dev_test_particleArrayIndices, sizeof(int) * N, cudaMemcpyDeviceToHost);
  //cudaMemcpy(test_particleGridIndices.get(), dev_test_particleGridIndices, sizeof(int) * N, cudaMemcpyDeviceToHost);

  //for (int i = 0; i < N; i++) {
  //    std::cout << "  particle: " << test_particleArrayIndices[i];
  //    std::cout << " grid: " << test_particleArrayIndices[i] << std::endl;
  //}

  //checkCUDAErrorWithLine("memcpy back failed!");

  /* //// TEST-2.1.1
    Output: 
      Start-end:
      start: 18 end: 83
      start: 12 end: 77
      start: 3 end: 79
      start: 13 end: 94
      start: 4 end: 68
      start: 15 end: 92
      start: 7 end: 95
      start: 14 end: 93
      idx: 1
      idx: 2
      idx: 2
      idx: 2
      idx: 4
      idx: 6
      idx: 6
      idx: 6
      idx: 7
      idx: 7
        */

      //numObjects = 10;
      //fullBlocksPerGrid = ((numObjects + blockSize - 1) / blockSize);
      //kernComputeIndices << < fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
      //    dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
      //checkCUDAErrorWithLine("kernComputeIndices failed!");

      ////// - Unstable key sort using Thrust. A stable sort isn't necessary, but you
      //////   are welcome to do a performance comparison.
      //thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

      ////// - Naively unroll the loop for finding the start and end indices of each
      //////   cell's data pointers in the array of boid indices

      //dim3 fullBlocksPerGridCells((gridCellCount + blockSize - 1) / blockSize);
      //kernResetIntBuffer << < fullBlocksPerGridCells, blockSize >> > (gridCellCount, dev_gridCellStartIndices, gridCellCount + 10);
      //checkCUDAErrorWithLine("kernResetIntBuffer failed!");
      //kernResetIntBuffer << < fullBlocksPerGridCells, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);
      //checkCUDAErrorWithLine("kernResetIntBuffer failed!");
      //
      //int* dev_idx;
      //cudaMalloc((void**)&dev_idx, numObjects * sizeof(int));

      //kernIdentifyCellStartEnd << < fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices,
      //    dev_gridCellStartIndices, dev_gridCellEndIndices, dev_idx);
      //checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

      //std::unique_ptr<int[]>test_gridCellStartIndices{ new int[gridCellCount] };
      //std::unique_ptr<int[]>test_gridCellEndIndices{ new int[gridCellCount] };
      //cudaMemcpy(test_gridCellStartIndices.get(), dev_gridCellStartIndices, sizeof(int) * gridCellCount, cudaMemcpyDeviceToHost);
      //cudaMemcpy(test_gridCellEndIndices.get(), dev_gridCellEndIndices, sizeof(int) * gridCellCount, cudaMemcpyDeviceToHost);
      //checkCUDAErrorWithLine("memcpy back failed!");

      //std::cout << "Start-end: " << std::endl;
      //for (int i = 0; i < gridCellCount; i++) {
      //    std::cout << "  start: " << test_gridCellStartIndices[i];
      //    std::cout << " end: " << test_gridCellEndIndices[i] << std::endl;
      //}

      //std::unique_ptr<int[]>test_idx{ new int[numObjects] };
      //cudaMemcpy(test_idx.get(), dev_idx, sizeof(int) * numObjects, cudaMemcpyDeviceToHost);
      //checkCUDAErrorWithLine("memcpy back failed!");
      //for (int i = 0; i < numObjects; i++) {
      //    std::cout << "  idx: " << test_idx[i] << std::endl;
      //}

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);

  /*cudaFree(dev_test_pos);
  cudaFree(dev_test_particleArrayIndices);
  cudaFree(dev_test_particleGridIndices);*/
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
