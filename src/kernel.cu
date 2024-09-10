#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

// LOOKED-2.1 potentially useful for doing grid-based neighbor search
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

// LOOKED-1.2 Parameters for the boids algorithm.
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

// LOOKED-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOKED-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// DONE-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.
glm::vec3* dev_pos_sorted;
glm::vec3* dev_vel1_sorted;

// LOOKED-2.1 - Grid parameters based on simulation parameters.
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
* LOOKED-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOKED-1.2 - This is a basic CUDA kernel.
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

  // LOOKED-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOKED-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOKED-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // DONE-2.1 DONE-2.3 - Allocate additional buffers here.
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  cudaMalloc((void**)&dev_pos_sorted, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos_sorted failed!");

  cudaMalloc((void**)&dev_vel1_sorted, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1_sorted failed!");

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

/**
* LOOKED-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  
    glm::vec3 perceivedCenter(0.f, 0.f, 0.f);
    glm::vec3 perceivedVelocity(0.f, 0.f, 0.f);

    glm::vec3 delta1(0.f, 0.f, 0.f), delta2(0.f, 0.f, 0.f), delta3(0.f, 0.f, 0.f);

    // Keep track of the number of neighbors for rules 1 and 3
    int rule1Neighbors = 0;
    int rule3Neighbors = 0;

    // Final delta V
    glm::vec3 result;

    // Loop through all of the boids and exclude current boid
    glm::vec3 c = glm::vec3(0, 0, 0);
    for(int i = 0; i < N; i++)
    {
        if (i == iSelf)
        {
            // Exclude the current boid
            continue;
        }

        glm::vec3 neighborBoidPos = pos[i];
        float distance = glm::length(neighborBoidPos - pos[iSelf]);

        // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
        if (distance < rule1Distance)
        {
            perceivedCenter += neighborBoidPos;
            rule1Neighbors++;
        }

        // Rule 2: boids try to stay a distance d away from each other
        if (distance < rule2Distance)
        {
            c -= (pos[i] - pos[iSelf]);
        }

        // Rule 3: boids try to match the speed of surrounding boids
        if (distance < rule3Distance)
        {
            perceivedVelocity += vel[i];
            rule3Neighbors++;
        }

    }
                 
    // Check if any of the neighbors should influence the current boid
    // For rules 1 and 3
    if (rule1Neighbors > 0)
    {
        delta1 = (perceivedCenter / (float) rule1Neighbors - pos[iSelf]) * rule1Scale;
    }

    delta2 = c * rule2Scale;

    if (rule3Neighbors > 0)
    {
        delta3 = (perceivedVelocity / (float) rule3Neighbors) * rule3Scale;
    }

    result += delta1 + delta2 + delta3;

    return result;
}

/**
* DONE-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  
    // First figure out which boid we are working with
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N)
    {
        return;
    }

  // Compute a new velocity based on pos and vel1
  // vel1[index] represents current velocity
    glm::vec3 deltaV = computeVelocityChange(N, index, pos, vel1);
    glm::vec3 newVel = vel1[index] + deltaV;

  // Clamp the speed
    if (glm::length(newVel) > maxSpeed)
    {
        newVel = glm::normalize(newVel) * maxSpeed;
    }

  // Record the new velocity into vel2. Question: why NOT vel1?
  // Answer: Because we don't want to write over the data we're reading from (I.e. we need neighbor's data)
    vel2[index] = newVel;
}

/**
* LOOKED-1.2 Since this is pretty trivial, we implemented it for you.
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

// LOOKED-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOKED-2.3 Looking at this method, what would be the most memory efficient
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
    // DONE-2.1
    // First figure out which boid we are working with
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N)
    {
        return;
    }

    // - Label each boid with the index of its grid cell.
    // Calculate the index of the grid cell using the boid's position
    glm::vec3 offset = pos[index] - gridMin;
    glm::vec3 cellIndex = glm::floor(inverseCellWidth * offset);
    // Label this boid in the grid index table
    gridIndices[index] = gridIndex3Dto1D(cellIndex.x, cellIndex.y, cellIndex.z, gridResolution);

    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
    indices[index] = index;
}

// LOOKED-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // DONE-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"

    // First figure out which particle we are working with
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    // Find the grid index for this particle
    int particleGridIndex = particleGridIndices[index];

    // Find the start point for each cell in the gridIndices array
    // If this is the first index or if this index doesn't match the one before it
    // Then set this as a new cell!
    if ((index == 0) || (particleGridIndex != particleGridIndices[index - 1]))
    {
        gridCellStartIndices[particleGridIndex] = index;
    }

    // Find the end point for each cell in the gridIndices array
    // If this is the last index, or if this index doesn't match the next one,
    // Then set this as the end of the current cell!
    if ((index == N - 1) || (particleGridIndex != particleGridIndices[index + 1]))
    {
        gridCellEndIndices[particleGridIndex] = index;
    }
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // DONE-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  
  // First find the current boid's index
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N)
  {
      return;
  }

  // Find the max neighborhood distance
  float neighborhoodDistance = fmax(fmax(rule1Distance, rule2Distance), rule3Distance);

  // Figure out which grid cell this boid is in
  glm::vec3 boidPos = pos[index];
  glm::vec3 gridPos = boidPos - gridMin;
  int gridIdx = gridIndex3Dto1D(
      gridPos.x * inverseCellWidth,
      gridPos.y * inverseCellWidth, 
      gridPos.z * inverseCellWidth, 
      gridResolution);

  // - Identify which cells may contain neighbors. This isn't always 8.
  int minX = imax(0, static_cast<int>((gridPos.x - neighborhoodDistance) * inverseCellWidth));
  int maxX = imin(gridResolution - 1, static_cast<int>((gridPos.x + neighborhoodDistance) * inverseCellWidth));

  int minY = imax(0, static_cast<int>((gridPos.y - neighborhoodDistance) * inverseCellWidth));
  int maxY = imin(gridResolution - 1, static_cast<int>((gridPos.y + neighborhoodDistance) * inverseCellWidth));

  int minZ = imax(0, static_cast<int>((gridPos.z - neighborhoodDistance) * inverseCellWidth));
  int maxZ = imin(gridResolution - 1, static_cast<int>((gridPos.z + neighborhoodDistance) * inverseCellWidth));

  // - For each cell, read the start/end indices in the boid pointer array.
  glm::vec3 c(0.f, 0.f, 0.f);
  glm::vec3 deltaV(0.f, 0.f, 0.f);
  glm::vec3 perceivedCenter(0.f, 0.f, 0.f);
  glm::vec3 perceivedVelocity(0.f, 0.f, 0.f);

  // Keep track of the number of neighbors for rules 1 and 3
  int rule1Neighbors = 0;
  int rule3Neighbors = 0;

  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  for (int z = minZ; z <= maxZ; z++)
  {
      for (int y = minY; y <= maxY; y++)
      {
          for (int x = minX; x <= maxX; x++)
          {
              glm::vec3 cell(x * cellWidth, y * cellWidth, z * cellWidth);

              // Find the nearest point on current cell to our boid
              glm::vec3 nearest;
              nearest.x = fmax(cell.x, fmin(gridPos.x, cell.x + cellWidth));
              nearest.y = fmax(cell.y, fmin(gridPos.y, cell.y + cellWidth));
              nearest.z = fmax(cell.z, fmin(gridPos.z, cell.z + cellWidth));
              
              // If the nearest point is within neighborhood distance, consider the neighb boids in that cell
              if (glm::distance(gridPos, nearest) <= neighborhoodDistance) {
                  int startEndIdx = gridIndex3Dto1D(x, y, z, gridResolution);

                  int gridCellStart = gridCellStartIndices[startEndIdx];
                  int gridCellEnd = gridCellEndIndices[startEndIdx];

                  if (gridCellStart == -1) {
                      continue;
                  }

                  for (int cellIndex = gridCellStart; cellIndex <= gridCellEnd; cellIndex++) {
                      int boidIdx = particleArrayIndices[cellIndex];
                      glm::vec3 bPos = pos[boidIdx];
                      glm::vec3 bVel = vel1[boidIdx];

                      if (boidIdx != index) {

                          // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
                          if (glm::distance(bPos, boidPos) < rule1Distance) {
                              rule1Neighbors++;
                              perceivedCenter += bPos;
                          }

                          // Rule 2: boids try to stay a distance d away from each other
                          if (glm::distance(bPos, boidPos) < rule2Distance) {
                              c -= (bPos - boidPos);
                          }

                          // Rule 3: boids try to match the speed of surrounding boids
                          if (glm::distance(bPos, boidPos) < rule3Distance) {
                              rule3Neighbors++;
                              perceivedVelocity += bVel;
                          }
                      }
                  }
              }
          }
      }
  }

  glm::vec3 rule1(0.f);
  glm::vec3 rule2(0.f);
  glm::vec3 rule3(0.f);

  if (rule1Neighbors > 0) {
      perceivedCenter /= glm::vec3(rule1Neighbors);
      rule1 = (perceivedCenter - boidPos) * rule1Scale;
  }

  rule2 = c * rule2Scale;

  if (rule3Neighbors > 0) {
      perceivedVelocity /= glm::vec3(rule3Neighbors);
      rule3 = perceivedVelocity * rule3Scale;
  }

  // - Clamp the speed change before putting the new speed in vel2
  glm::vec3 newV = vel1[index] + (rule1 + rule2 + rule3);
  if (glm::length(newV) > maxSpeed) {
      newV = glm::normalize(newV) * maxSpeed;
  }

  vel2[index] = newV;

}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // DONE-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
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

    // First find the current boid's index
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N)
    {
        return;
    }

    // Figure out which grid cell this boid is in
    glm::vec3 boidPos = pos[index];
    glm::vec3 gridPos = boidPos - gridMin;
    int gridIdx = gridIndex3Dto1D(
        gridPos.x * inverseCellWidth,
        gridPos.y * inverseCellWidth,
        gridPos.z * inverseCellWidth,
        gridResolution);

    // Find the max neighborhood distance
    float neighborhoodDistance = fmax(fmax(rule1Distance, rule2Distance), rule3Distance);

    // - Identify which cells may contain neighbors. This isn't always 8.
    // Calculate the mins in the x, y, and z directions
    int minX = imax(0, static_cast<int>((gridPos.x - neighborhoodDistance) * inverseCellWidth));
    int minY = imax(0, static_cast<int>((gridPos.y - neighborhoodDistance) * inverseCellWidth));
    int minZ = imax(0, static_cast<int>((gridPos.z - neighborhoodDistance) * inverseCellWidth));
    // Calculate the maxes in the x, y, and z directions
    int maxX = imin(gridResolution - 1, static_cast<int>((gridPos.x + neighborhoodDistance) * inverseCellWidth));
    int maxY = imin(gridResolution - 1, static_cast<int>((gridPos.y + neighborhoodDistance) * inverseCellWidth));
    int maxZ = imin(gridResolution - 1, static_cast<int>((gridPos.z + neighborhoodDistance) * inverseCellWidth));

    // - For each cell, read the start/end indices in the boid pointer array.
    glm::vec3 c(0.f, 0.f, 0.f);
    glm::vec3 deltaV(0.f, 0.f, 0.f);
    glm::vec3 perceivedCenter(0.f, 0.f, 0.f);
    glm::vec3 perceivedVelocity(0.f, 0.f, 0.f);

    // Keep track of the number of neighbors for rules 1 and 3
    int rule1Neighbors = 0;
    int rule3Neighbors = 0;

    // - Access each boid in the cell and compute velocity change from
    //   the boids rules, if this boid is within the neighborhood distance.
    for (int z = minZ; z <= maxZ; z++)
    {
        for (int y = minY; y <= maxY; y++)
        {
            for (int x = minX; x <= maxX; x++)
            {
                glm::vec3 cell(x * cellWidth, y * cellWidth, z * cellWidth);

                // Find the nearest point on current cell to our boid
                glm::vec3 nearest;
                nearest.x = fmax(cell.x, fmin(gridPos.x, cell.x + cellWidth));
                nearest.y = fmax(cell.y, fmin(gridPos.y, cell.y + cellWidth));
                nearest.z = fmax(cell.z, fmin(gridPos.z, cell.z + cellWidth));

                if (glm::distance(gridPos, nearest) <= neighborhoodDistance) 
                {
                    // Find the grid start and end indices
                    int startEndIndex = gridIndex3Dto1D(x, y, z, gridResolution);

                    // Only proceed with a valid start
                    int gridCellStart = gridCellStartIndices[startEndIndex];
                    if (gridCellStart == -1) {
                        continue;
                    }

                    // Find the grid cell end
                    int gridCellEnd = gridCellEndIndices[startEndIndex];

                    // Iterate through the neighboids (get it? neighboring boids? :D)
                    for (int neighboidIndex = gridCellStart; neighboidIndex <= gridCellEnd; neighboidIndex++) {
                        
                        // Get the position and velocity for the current boid
                        glm::vec3 neighboidPos = pos[neighboidIndex];
                        glm::vec3 neighboidVel = vel1[neighboidIndex];

                        // Exclude the current boid, we only care about neighbors
                        if (neighboidIndex == index) 
                        {
                            continue;
                        }

                            // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
                            if (glm::distance(boidPos, boidPos) < rule1Distance) {
                                rule1Neighbors++;
                                perceivedCenter += boidPos;
                            }

                            // Rule 2: boids try to stay a distance d away from each other
                            if (glm::distance(boidPos, neighboidPos) < rule2Distance) {
                                c -= (neighboidPos - boidPos);
                            }

                            // Rule 3: boids try to match the speed of surrounding boids
                            if (glm::distance(neighboidPos, boidPos) < rule3Distance) {
                                rule3Neighbors++;
                                perceivedVelocity += neighboidVel;
                            }
                    }
                }
            }
        }
    }

    glm::vec3 delta1(0.f), delta2(0.f), delta3(0.f);
    
    // Rule 1
    if (rule1Neighbors > 0) {
        perceivedCenter /= glm::vec3(rule1Neighbors);
        delta1 = (perceivedCenter - boidPos) * rule1Scale;
    }
    
    // Rule 2
    delta2 = c * rule2Scale;
    
    // Rule3
    if (rule3Neighbors > 0) {
        perceivedVelocity /= glm::vec3(rule3Neighbors);
        delta3 = perceivedVelocity * rule3Scale;
    }

    // - Clamp the speed change before putting the new speed in vel2
    glm::vec3 newV = vel1[index] + (delta1 + delta2 + delta3);
    if (glm::length(newV) > maxSpeed) 
    {
        newV = glm::normalize(newV) * maxSpeed;
    }

    vel2[index] = newV;
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // DONE-1.2 - use the kernels you wrote to step the simulation forward in time.
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    
    // Update the boid's velocity
    kernUpdateVelocityBruteForce << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos, dev_vel1, dev_vel2);
    
    // Update the boid's position
    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);

  // DONE-1.2 ping-pong the velocity buffers
    glm::vec3* temp = dev_vel2;
    dev_vel2 = dev_vel1;
    dev_vel1 = temp;
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // DONE-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // Calculate blocks per grid
    int blocksPerGrid = (numObjects + threadsPerBlock.x - 1) / threadsPerBlock.x;

  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
    kernComputeIndices << <blocksPerGrid, threadsPerBlock >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);
  
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
    dim3 gridCellBlocksPerGrid((gridCellCount + threadsPerBlock.x - 1) / threadsPerBlock.x);

    kernResetIntBuffer << <gridCellBlocksPerGrid, threadsPerBlock >> > (gridCellCount, dev_gridCellStartIndices, -1);
    checkCUDAErrorWithLine("kernResetIntBuffer failed!");
    kernResetIntBuffer << <gridCellBlocksPerGrid, threadsPerBlock >> > (gridCellCount, dev_gridCellEndIndices, -1);
    checkCUDAErrorWithLine("kernResetIntBuffer failed!");

    kernIdentifyCellStartEnd << <blocksPerGrid, threadsPerBlock >> > (numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
    checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");
  
#if 0
    std::vector<int> host_particleGridIndices;
    std::vector<int> host_particleArrayIndices;
    std::vector<int> host_start;
    std::vector<int> host_end;

    host_particleGridIndices.resize(numObjects);
    host_particleArrayIndices.resize(numObjects);
    host_start.resize(gridCellCount);
    host_end.resize(gridCellCount);

    cudaMemcpy(host_particleGridIndices.data(), dev_particleGridIndices, numObjects * sizeof(int), cudaMemcpyDeviceToHost);
    checkCUDAErrorWithLine("Memcpy host_particleGridIndices.data failed!");

    cudaMemcpy(host_particleArrayIndices.data(), dev_particleArrayIndices, numObjects * sizeof(int), cudaMemcpyDeviceToHost);
    checkCUDAErrorWithLine("Memcpy host_particleArrayIndices.data failed!");

    cudaMemcpy(host_start.data(), dev_gridCellStartIndices, gridCellCount * sizeof(int), cudaMemcpyDeviceToHost);
    checkCUDAErrorWithLine("Memcpy start.data failed!");

    cudaMemcpy(host_end.data(), dev_gridCellEndIndices, gridCellCount * sizeof(int), cudaMemcpyDeviceToHost);
    checkCUDAErrorWithLine("Memcpy end.data failed!");
#endif

  // - Perform velocity updates using neighbor search
    kernUpdateVelNeighborSearchScattered << <blocksPerGrid, threadsPerBlock >> > (
        numObjects,
        gridSideCount,
        gridMinimum,
        gridInverseCellWidth,
        gridCellWidth,
        dev_gridCellStartIndices,
        dev_gridCellEndIndices,
        dev_particleArrayIndices,
        dev_pos,
        dev_vel1,
        dev_vel2
        );
    checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

  // - Update positions
    kernUpdatePos << <blocksPerGrid, threadsPerBlock >> > (numObjects, dt, dev_pos, dev_vel2);
    checkCUDAErrorWithLine("kernUpdatePos failed!");

  // - Ping-pong buffers as needed
    cudaMemcpy(dev_vel1, dev_vel2, numObjects * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
    checkCUDAErrorWithLine("Memcpy (dev_vel2 to dev_vel1) failed!");
}

// Helper function to copy the position and velocity into new buffers with sorted indices based on grid index
__global__ void kernReshufflePosAndVel(
    int N, glm::vec3* pos, glm::vec3* vel, glm::vec3* pos_sorted,
    glm::vec3* vel_sorted, int* particleArrayIndices) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) {
        return;
    }

    pos_sorted[index] = pos[particleArrayIndices[index]];
    vel_sorted[index] = vel[particleArrayIndices[index]];
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // DONE-2.3 - start by copying Boids::stepSimulationNaiveGrid
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
  //==================================================================================
  
  // First calculate the number of blocks per grid
    int blocksPerGrid = (numObjects + threadsPerBlock.x - 1) / threadsPerBlock.x;

  // - Label each particle with its array index as well as its grid index. Use 2x width grids
    kernComputeIndices << <blocksPerGrid, threadsPerBlock >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
    checkCUDAErrorWithLine("kernComputeIndices failed!");

  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you are welcome to do a performance comparison.
    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);
    kernReshufflePosAndVel << <blocksPerGrid, threadsPerBlock >> > (numObjects, dev_pos, dev_vel1, dev_pos_sorted, dev_vel1_sorted, dev_particleArrayIndices);

  // - Naively unroll the loop for finding the start and end indices of each cell's data pointers in the array of boid indices
    dim3 gridCellBlocksPerGrid((gridCellCount + threadsPerBlock.x - 1) / threadsPerBlock.x);
  // Reset start index buffer
    kernResetIntBuffer << <gridCellBlocksPerGrid, threadsPerBlock >> > (gridCellCount, dev_gridCellStartIndices, -1);
    checkCUDAErrorWithLine("kernResetIntBuffer failed!");
  // Reset ending index buffer
    kernResetIntBuffer << <gridCellBlocksPerGrid, threadsPerBlock >> > (gridCellCount, dev_gridCellEndIndices, -1);
    checkCUDAErrorWithLine("kernResetIntBuffer failed!");
  // Run kernel to try and identify cell start and end indices
    kernIdentifyCellStartEnd << <blocksPerGrid, threadsPerBlock >> > (numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
    checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search

    kernUpdateVelNeighborSearchCoherent << <blocksPerGrid, threadsPerBlock >> > (
        numObjects,
        gridSideCount,
        gridMinimum,
        gridInverseCellWidth,
        gridCellWidth,
        dev_gridCellStartIndices,
        dev_gridCellEndIndices,
        dev_pos_sorted,
        dev_vel1_sorted,
        dev_vel2
        );

    // - Update positions
    kernUpdatePos << <blocksPerGrid, threadsPerBlock >> > (numObjects, dt, dev_pos_sorted, dev_vel2);
    checkCUDAErrorWithLine("kernUpdatePos failed!");

    // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
    cudaMemcpy(dev_pos, dev_pos_sorted, numObjects * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
    checkCUDAErrorWithLine("Memcpy (dev_pos_sorted to dev_pos) failed!");

    cudaMemcpy(dev_vel1, dev_vel2, numObjects * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
    checkCUDAErrorWithLine("Memcpy (dev_vel2 to dev_vel1) failed!");
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // DONE-2.1 DONE-2.3 - Free any additional buffers here.
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
}

void Boids::unitTest() {
  // LOOKED-1.2 Feel free to write additional tests here.
  

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
  // LOOKED-2.1 Example for using thrust::sort_by_key
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

  // Cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
