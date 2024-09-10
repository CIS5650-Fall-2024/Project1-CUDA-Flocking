#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
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
#define scene_scale 100.0f

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
// buffer containing a pointer for each boid to its data in dev_pos and dev_vel1 and dev_vel2
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle? 
// buffer containing the grid index of each boid 
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

// buffer containing a pointer for each cell to the beginning of its data in dev_particleArrayIndices
int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
// buffer containing a pointer for each cell to the end of its data in dev_particleArrayIndices
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

// coherentPos and coherentVel are the buffers that will hold the reshuffled position and velocity data
glm::vec3 *dev_coherentPos;
glm::vec3 *dev_coherentVel;

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
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.3 - Allocate additional buffers here.
  // Your uniform grid will probably look something like this in GPU memory:

    //dev_particleArrayIndices - buffer containing a pointer for each boid to its data in dev_pos and dev_vel1 and dev_vel2
    //dev_particleGridIndices - buffer containing the grid index of each boid
    //dev_gridCellStartIndices - buffer containing a pointer for each cell to the beginning of its data in dev_particleArrayIndices
    //dev_gridCellEndIndices - buffer containing a pointer for each cell to the end of its data in dev_particleArrayIndices.


  //allocate memory for the additional buffers as specified in instructions
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");
  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");
  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");
  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");
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

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
  // Rule 2: boids try to stay a distance d away from each other
  // Rule 3: boids try to match the speed of surrounding boids

  glm::vec3 perceivedCenter = glm::vec3(0.0f, 0.0f, 0.0f);
  glm::vec3 c(0.0f, 0.0f, 0.0f);
  glm::vec3 perceivedVelocity = glm::vec3(0.0f, 0.0f, 0.0f);

  int rule1Neighbors = 0;
  int rule3Neighbors = 0;

  glm::vec3 currentPos = pos[iSelf];
  glm::vec3 currentVel = vel[iSelf];

  for (int i = 0; i < N; i++) {
    if (i == iSelf) {
      continue;
    }

    glm::vec3 otherPos = pos[i];
    glm::vec3 otherVel = vel[i];
    float distance = glm::distance(currentPos, otherPos);

    // Rule 1: Cohesion (move towards the center of mass)
    if (distance < rule1Distance) {
      perceivedCenter += otherPos;
      rule1Neighbors++;
    }

    // Rule 2: Separation (avoid getting too close)
    if (distance < rule2Distance) {
      c -= (otherPos - currentPos);
    }

    // Rule 3: Alignment (match velocity with neighbors)
    if (distance < rule3Distance) {
      perceivedVelocity += otherVel;
      rule3Neighbors++;
    }
  }

  // Apply Rule 1: Cohesion
  glm::vec3 velocityChange = glm::vec3(0.0f, 0.0f, 0.0f);
  if (rule1Neighbors > 0) {
    perceivedCenter /= rule1Neighbors;
    velocityChange += (perceivedCenter - currentPos) * rule1Scale;
  }

  // Apply Rule 2: Separation
  velocityChange += c * rule2Scale;

  // Apply Rule 3: Alignment
  if (rule3Neighbors > 0) {
    perceivedVelocity /= rule3Neighbors;
    velocityChange += perceivedVelocity * rule3Scale;
  }

  return velocityChange;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  // Compute a new velocity based on pos and vel1
  // Clamp the speed
  // Record the new velocity into vel2. Question: why NOT vel1?

  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }

  // Compute the new velocity using computeVelocityChange
  glm::vec3 velocityChange = computeVelocityChange(N, index, pos, vel1);
  glm::vec3 newVelocity = vel1[index] + velocityChange;

  // Clamp the speed to a maximum speed (hardcoded to 1.0f)
  float speed = glm::length(newVelocity);
  if (speed > maxSpeed) {
    newVelocity *= maxSpeed / speed;
  }

  vel2[index] = newVelocity;
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

/** 
 * compute grid index using:
 * iX = (pos.x - gridMin.x) / cellWidth
 * iY = (pos.y - gridMin.y) / cellWidth
 * iZ = (pos.z - gridMin.z) / cellWidth
 * 
 * then convert 3D grid index to 1D grid index using gridIndex3Dto1D
 * 
 * indices: array of indices for each boid, is the index of the boid in dev_pos and dev_velX, etc
 * gridIndices: array of indices for each boid, is the grid index of the boid in the uniform grid
 */
__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {

    // Calculate the index of the current thread
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    // Check if the index is within bounds
    if (index >= N) return;

    glm::vec3 boidPos = pos[index];

    glm::vec3 gridPos = glm::floor((boidPos - gridMin) * inverseCellWidth);

    gridIndices[index] = gridIndex3Dto1D(gridPos.x, gridPos.y, gridPos.z, gridResolution);
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


/***
 * update the start and end indices of each grid cell in dev_gridCellStartIndices and dev_gridCellEndIndices
 * dev_gridCellStartIndices: buffer containing a pointer for each cell to the beginning of its data in dev_particleArrayIndices
 * dev_gridCellEndIndices: buffer containing a pointer for each cell to the end of its data in dev_particleArrayIndices
 * 
 */
__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {

  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= N) return;

  int currGridIdx = particleGridIndices[index];
  if (index == 0) {
    // Set the start index for the first boid
    gridCellStartIndices[currGridIdx] = 0;
  } else if (index > 0 && currGridIdx != particleGridIndices[index - 1]) {
    // New grid cell starts
    gridCellEndIndices[particleGridIndices[index - 1]] = index - 1;
    gridCellStartIndices[currGridIdx] = index;
  }

  // Handle the last boid in the array
  if (index == N - 1) {
    gridCellEndIndices[currGridIdx] = N - 1;
  }

}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  
  // index 
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  // bounds check
  if (index >= N) return;

  // insstantiate variables
  glm::vec3 perceivedCenter = glm::vec3(0.0f, 0.0f, 0.0f);
  glm::vec3 c(0.0f, 0.0f, 0.0f);
  glm::vec3 perceivedVelocity = glm::vec3(0.0f, 0.0f, 0.0f);
  glm::vec3 result = glm::vec3(0.0f, 0.0f, 0.0f);
  int rule1Neighbors = 0;
  int rule3Neighbors = 0;
  glm::vec3 currentPos = pos[index];
  glm::vec3 gridPos = glm::floor((currentPos - gridMin) * inverseCellWidth);
  
  // iterate over the neighboring cells
  for (int x = imax(gridPos.x - 1, 0); x < imin(gridPos.x + 1, gridResolution - 1); x++) {
    for (int y = imax(gridPos.y - 1, 0); y < imin(gridPos.y + 1, gridResolution - 1); y++) {
      for (int z = imax(gridPos.z - 1, 0); z < imin(gridPos.z + 1, gridResolution - 1); z++) {

        // get the grid index (flatten)
        int gridIdx = gridIndex3Dto1D(x, y, z, gridResolution);
        int startIdx = gridCellStartIndices[gridIdx];
        int endIdx = gridCellEndIndices[gridIdx];

        // check if the grid cell is empty
        if (startIdx == -1 || endIdx == -1) {
          continue;
        }

        // o/w iterate over the boids in the cell
        for (int i = startIdx; i <= endIdx; i++) {
          int boidIdx = particleArrayIndices[i];
          if (boidIdx == index) {
            continue;
          }

          glm::vec3 otherPos = pos[boidIdx];
          glm::vec3 otherVel = vel1[boidIdx];
          float distance = glm::distance(currentPos, otherPos);

          // Rule 1: Cohesion (move towards the center of mass)
          if (distance < rule1Distance) {
            perceivedCenter += otherPos;
            rule1Neighbors++;
          }

          // Rule 2: Separation (avoid getting too close)
          if (distance < rule2Distance) {
            c -= (otherPos - currentPos);
          }

          // Rule 3: Alignment (match velocity with neighbors)
          if (distance < rule3Distance) {
            perceivedVelocity += otherVel;
            rule3Neighbors++;
          }
        }
      }
    }
  }

  // Apply Rule 1: Cohesion
  if (rule1Neighbors > 0) {
    perceivedCenter /= rule1Neighbors;
    result += (perceivedCenter - currentPos) * rule1Scale;
  }
  
  // Apply Rule 2: Separation
  result += c * rule2Scale;

  // Apply Rule 3: Alignment  
  if (rule3Neighbors > 0) {
    perceivedVelocity /= rule3Neighbors;
    result += perceivedVelocity * rule3Scale;
  }

  result += vel1[index];
  float speed = glm::length(result);
  if (speed > maxSpeed) {
    result = glm::normalize(result) * maxSpeed;
  }

  vel2[index] = result;
}


__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  
  // index
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  // bounds check
  if (index >= N) return;

  // insstantiate variables
  glm::vec3 perceivedCenter = glm::vec3(0.0f, 0.0f, 0.0f);
  glm::vec3 c(0.0f, 0.0f, 0.0f);
  glm::vec3 perceivedVelocity = glm::vec3(0.0f, 0.0f, 0.0f);
  glm::vec3 result = glm::vec3(0.0f, 0.0f, 0.0f);
  int rule1Neighbors = 0;
  int rule3Neighbors = 0;

  glm::vec3 currentPos = pos[index];
  glm::vec3 gridPos = glm::floor((currentPos - gridMin) * inverseCellWidth);
  
  // iterate over the neighboring cells
  for (int x = imax(gridPos.x - 1, 0); x < imin(gridPos.x + 1, gridResolution - 1); x++) {
    for (int y = imax(gridPos.y - 1, 0); y < imin(gridPos.y + 1, gridResolution - 1); y++) {
      for (int z = imax(gridPos.z - 1, 0); z < imin(gridPos.z + 1, gridResolution - 1); z++) {
        int gridIdx = gridIndex3Dto1D(x, y, z, gridResolution);
        int startIdx = gridCellStartIndices[gridIdx];
        int endIdx = gridCellEndIndices[gridIdx];

        // check if the grid cell is empty
        if (startIdx == -1 || endIdx == -1) {
          continue;
        }

        // o/w iterate over the boids in the cell (this time using the coherent data)
        for (int i = startIdx; i <= endIdx; i++) {
          if (i == index) {
            continue;
          }

          glm::vec3 otherPos = pos[i];
          glm::vec3 otherVel = vel1[i];
          float distance = glm::distance(currentPos, otherPos);

          // Rule 1: Cohesion (move towards the center of mass)
          if (distance < rule1Distance) {
            perceivedCenter += otherPos;
            rule1Neighbors++;
          }

          // Rule 2: Separation (avoid getting too close)
          if (distance < rule2Distance) {
            c -= (otherPos - currentPos);
          }

          // Rule 3: Alignment (match velocity with neighbors)
          if (distance < rule3Distance) {
            perceivedVelocity += otherVel;
            rule3Neighbors++;
          }
        }
      }
    }
  }

  // Apply Rule 1: Cohesion
  if (rule1Neighbors > 0) {
    perceivedCenter /= rule1Neighbors;
    result += (perceivedCenter - currentPos) * rule1Scale;
  }
  
  // Apply Rule 2: Separation
  result += c * rule2Scale;

  // Apply Rule 3: Alignment
  if (rule3Neighbors > 0) {
    perceivedVelocity /= rule3Neighbors;
    result += perceivedVelocity * rule3Scale;
  }

  result += vel1[index];
  float speed = glm::length(result);
  if (speed > maxSpeed) {
    result = glm::normalize(result) * maxSpeed;
  }

  vel2[index] = result;
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  // TODO-1.2 ping-pong the velocity buffers
  
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  
  kernUpdateVelocityBruteForce<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, dev_vel1, dev_vel2);
  checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");

  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);
  checkCUDAErrorWithLine("kernUpdatePos failed!");

  std::swap(dev_vel1, dev_vel2);
}


void Boids::stepSimulationScatteredGrid(float dt) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  dim3 fullCellsPerGrid((gridCellCount + blockSize - 1) / blockSize);

  // Step 1: Compute grid index and boid array index.
  // Compute which grid cell each boid is in, and label each particle with its array index.
  kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount, 
                                                       gridMinimum, gridInverseCellWidth, 
                                                       dev_pos, dev_particleArrayIndices, 
                                                       dev_particleGridIndices);
  checkCUDAErrorWithLine("kernComputeIndices failed!");

  // Step 2: Sort particles by grid index.
  // Sort the boids based on the grid cell they are in using thrust::sort_by_key.
  thrust::device_ptr<int> dev_key(dev_particleGridIndices);
  thrust::device_ptr<int> dev_val(dev_particleArrayIndices);
  thrust::sort_by_key(dev_key, dev_key + numObjects, dev_val);

  // Step 3: Reset grid cell start/end index buffers.
  kernResetIntBuffer<<<fullCellsPerGrid, blockSize>>>(gridCellCount, dev_gridCellStartIndices, -1);
  kernResetIntBuffer<<<fullCellsPerGrid, blockSize>>>(gridCellCount, dev_gridCellEndIndices, -1);
  
  // Step 4: Identify start and end indices of grid cells.
  kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleGridIndices, 
                                                             dev_gridCellStartIndices, 
                                                             dev_gridCellEndIndices);
  checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

  // Step 5: Update velocities using neighbor search.
  kernUpdateVelNeighborSearchScattered<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount,
                                                                        gridMinimum, gridInverseCellWidth, 
                                                                        gridCellWidth, dev_gridCellStartIndices, 
                                                                        dev_gridCellEndIndices, 
                                                                        dev_particleArrayIndices, 
                                                                        dev_pos, dev_vel1, dev_vel2);
  checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

  // Step 6: Update positions based on new velocities.
  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);
  checkCUDAErrorWithLine("kernUpdatePos failed!");

  // Step 7: Ping-pong the velocity buffers.
  std::swap(dev_vel1, dev_vel2);
}

// custom kernel to reshuffle the position and velocity data to be coherent within cells for TODO-2.3
__global__ void kernReshuffle(int N, int *particleArrayIndices, glm::vec3 *pos, glm::vec3 *vel, glm::vec3 *coherentPos, glm::vec3 *coherentVel) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) return;

  int coherentIdx = particleArrayIndices[index];
  coherentPos[index] = pos[coherentIdx];
  coherentVel[index] = vel[coherentIdx];
}

void Boids::stepSimulationCoherentGrid(float dt) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  dim3 fullCellsPerGrid((gridCellCount + blockSize - 1) / blockSize);

  // Step 1: Compute grid index and boid array index.
  // Compute which grid cell each boid is in, and label each particle with its array index.
  kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount, 
                                                       gridMinimum, gridInverseCellWidth, 
                                                       dev_pos, dev_particleArrayIndices, 
                                                       dev_particleGridIndices);
  checkCUDAErrorWithLine("kernComputeIndices failed!");

  // Step 2: Sort particles by grid index.
  // Sort the boids based on the grid cell they are in using thrust::sort_by_key.
  thrust::device_ptr<int> dev_key(dev_particleGridIndices);
  thrust::device_ptr<int> dev_val(dev_particleArrayIndices);
  thrust::sort_by_key(dev_key, dev_key + numObjects, dev_val);

  // Step 3: Reset grid cell start/end index buffers.
  kernResetIntBuffer<<<fullCellsPerGrid, blockSize>>>(gridCellCount, dev_gridCellStartIndices, -1);
  kernResetIntBuffer<<<fullCellsPerGrid, blockSize>>>(gridCellCount, dev_gridCellEndIndices, -1);
  
  // Step 4: Identify start and end indices of grid cells.
  kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleGridIndices, 
                                                             dev_gridCellStartIndices, 
                                                             dev_gridCellEndIndices);
  checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

  // Step 5: Reshuffle boid data to be coherent.
  kernReshuffle<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleArrayIndices, 
                                                  dev_pos, dev_vel1, dev_coherentPos, dev_coherentVel);

  // Step 6: Update velocities using neighbor search.
  kernUpdateVelNeighborSearchCoherent<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount,
                                                                        gridMinimum, gridInverseCellWidth, 
                                                                        gridCellWidth, dev_gridCellStartIndices, 
                                                                        dev_gridCellEndIndices, 
                                                                        dev_particleArrayIndices, 
                                                                        dev_pos, dev_vel1, dev_vel2);
  checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

  // Step 7: Update positions based on new velocities.
  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);
  checkCUDAErrorWithLine("kernUpdatePos failed!");

  // Step 8: Ping-pong the velocity buffers (+ new coherent buffers).
  std::swap(dev_pos, dev_coherentPos);
  std::swap(dev_vel1, dev_coherentVel);
  std::swap(dev_vel1, dev_vel2);
}



void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.3 - Free any additional buffers here.
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

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
