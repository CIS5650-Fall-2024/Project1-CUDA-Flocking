#define GLM_FORCE_CUDA
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"
#include <thrust/gather.h>

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

/*! Defines how big a cell is compared to the maximum rule distance */
#define cellWidthToSearchDistanceRatio 2.0f
#define inverseCellWidthToSearchDistanceRatio (1.0f / cellWidthToSearchDistanceRatio)

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
glm::vec3 *dev_pos_sorted;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;
glm::vec3* dev_vel1_sorted;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // This array answers the question: What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices;  // This array answers the question: What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;
thrust::device_ptr<glm::vec3> dev_thrust_pos;
thrust::device_ptr<glm::vec3> dev_thrust_vel;
thrust::device_ptr<glm::vec3> dev_thrust_pos_sorted;
thrust::device_ptr<glm::vec3> dev_thrust_vel1_sorted;

int *dev_gridCellStartIndices; // Together, these two arrays answer the question: 
int *dev_gridCellEndIndices;   // What part of dev_particleArrayIndices belongs to this cell?

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

  cudaMalloc((void**)&dev_pos_sorted, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos_sorted failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  cudaMalloc((void**)&dev_vel1_sorted, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1_sorted failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = cellWidthToSearchDistanceRatio * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndicies failed!");

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
  dev_thrust_pos = thrust::device_ptr<glm::vec3>(dev_pos);
  dev_thrust_vel = thrust::device_ptr<glm::vec3>(dev_vel1);
  dev_thrust_pos_sorted = thrust::device_ptr<glm::vec3>(dev_pos_sorted);
  dev_thrust_vel1_sorted = thrust::device_ptr<glm::vec3>(dev_vel1_sorted);

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

// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
__device__ glm::vec3 computeCenterOfMassVelChange(int N, int iSelf, const glm::vec3* pos) {
    glm::vec3 perceived_center(0, 0, 0);
    const glm::vec3 pos_self = pos[iSelf];
    int neighborCount = 0;

    for (int i = 0; i < N; ++i) {
        if (i == iSelf) continue;

        const glm::vec3& pos_i = *(pos + i);
        if (glm::length(pos_i - pos_self) > rule1Distance) continue;

        ++neighborCount;
        perceived_center += pos_i;
    }
    
    if (neighborCount > 0) {
        perceived_center /= neighborCount;
        return (perceived_center - pos_self) * rule1Scale;
    }
    return glm::vec3(0, 0, 0);
}

// Rule 2: boids try to stay a distance d away from each other
__device__ glm::vec3 computeMaintainDistanceVelChange(int N, int iSelf, const glm::vec3* pos) {
    glm::vec3 adjustment_velocity(0, 0, 0);
    const glm::vec3 pos_self = pos[iSelf];

    for (int i = 0; i < N; ++i) {
        if (i == iSelf) continue;

        const glm::vec3 pos_i = *(pos + i);
        const glm::vec3 pos_diff = pos_i - pos_self;

        if (glm::length(pos_diff) > rule2Distance) continue;

        adjustment_velocity -= pos_diff;
    }

    return adjustment_velocity * rule2Scale;
}

// Rule 3: boids try to match the speed of surrounding boids
__device__ glm::vec3 computeVelocityMatchVelChange(int N, int iSelf, const glm::vec3* pos, const glm::vec3* vel) {
    glm::vec3 perceived_velocity(0, 0, 0);
    const glm::vec3 pos_self = pos[iSelf];
    int neighborCount = 0;

    for (int i = 0; i < N; ++i) {
        if (i == iSelf) continue;
        const glm::vec3 pos_i = *(pos + i);
        if (glm::length(pos_i - pos_self) > rule3Distance) continue;

        const glm::vec3 vel_i = *(vel + i);
        ++neighborCount;
        perceived_velocity += vel_i;
    }

    if (neighborCount > 0) {
        perceived_velocity /= neighborCount;
        return perceived_velocity * rule3Scale;
    }

    return glm::vec3(0, 0, 0);
}

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  glm::vec3 deltaV1 = computeCenterOfMassVelChange(N, iSelf, pos);
  glm::vec3 deltaV2 = computeMaintainDistanceVelChange(N, iSelf, pos);
  glm::vec3 deltaV3 = computeVelocityMatchVelChange(N, iSelf, pos, vel);
  
  return deltaV1 + deltaV2 + deltaV3;
}

/**
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }
    
    const glm::vec3 deltaV = computeVelocityChange(N, index, pos, vel1);
    glm::vec3 updatedVelocity = vel1[index] + deltaV;
    const float updatedSpeed = glm::length(updatedVelocity);
    if (updatedSpeed > maxSpeed) {
        updatedVelocity *= (maxSpeed / updatedSpeed);
    }

    // We record the new velocity into vel2 because we don't want to affect threads running in parallel that need the pre-update velocity of each boid.
    vel2[index] = updatedVelocity;
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
  const glm::vec3 *pos, int *indices, int *gridIndices) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return; 

    indices[index] = index; // This seems useless right now, but it's necessary for sorting later since we don't
                            // have map data structures in CUDA.

    // Determine what grid cell we're in.
    const glm::vec3 pos_i = pos[index];
    const glm::vec3 gridCell3D_i = glm::floor((pos_i - gridMin) * inverseCellWidth);
    int gridCell1D_i = gridIndex3Dto1D(gridCell3D_i.x, gridCell3D_i.y, gridCell3D_i.z, gridResolution);

    gridIndices[index] = gridCell1D_i;

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
  int *gridCellStartIndices, int *gridCellEndIndices) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= N) return;

  int gridCell_i = particleGridIndices[index];
  
  int previousGridCell = (index - 1 < 0) ? -1 : particleGridIndices[index - 1];
  int nextGridCell = (index + 1 >= N) ? -1 : particleGridIndices[index + 1];

  if (previousGridCell != gridCell_i) {
      gridCellStartIndices[gridCell_i] = index;
  }

  if (nextGridCell != gridCell_i) {
      gridCellEndIndices[gridCell_i] = index;
  }
}

__device__ glm::vec3 wrapIndices(const glm::vec3& index, int size) {
    return glm::vec3(
        (static_cast<int>(index.x) % size + size) % size,
        (static_cast<int>(index.y) % size + size) % size,
        (static_cast<int>(index.z) % size + size) % size
    );
}

__device__ bool isBoidWithinRadiusOfGridCell(const glm::vec3& pos, const glm::vec3& gridCellMinPoint, float cellWidth, float radius) {
    const glm::vec3 maxBounds = gridCellMinPoint + glm::vec3(cellWidth);
    glm::vec3 closestPoint = glm::clamp(pos, gridCellMinPoint, maxBounds);

    return glm::distance(pos, closestPoint) <= radius;
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  const glm::vec3 *pos, const glm::vec3 *vel1, glm::vec3 *vel2) {

  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= N) return;

  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  const glm::vec3 pos_i = pos[index];
  const glm::vec3 gridCell3D_i = glm::floor((pos_i - gridMin) * inverseCellWidth);

  // Initialize tracking variables for different rules
  glm::vec3 perceived_center(0, 0, 0);
  glm::vec3 separation_velocity_adjustment(0, 0, 0);
  glm::vec3 perceived_velocity(0, 0, 0);
  glm::vec3 rule1DeltaV(0, 0, 0);
  glm::vec3 rule2DeltaV(0, 0, 0);
  glm::vec3 rule3DeltaV(0, 0, 0);
  int rule1NeighborCount = 0;
  int rule3NeighborCount = 0;

  // Search over neighboring cells within the neighborhood distance of the boid
  int searchRange = ceil(imax(inverseCellWidthToSearchDistanceRatio, 1.0f));
  for (int x = -searchRange; x <= searchRange; ++x) {
      for (int y = -searchRange; y <= searchRange; ++y) {
          for (int z = -searchRange; z <= searchRange; ++z) {
              const glm::vec3 neighborCell3DUnwrapped = gridCell3D_i + glm::vec3(x, y, z);
              const glm::vec3 neighborCell3DWrapped = wrapIndices(neighborCell3DUnwrapped, gridResolution);
              const glm::vec3 neighborGridCellMinPoint = gridMin + (neighborCell3DUnwrapped * cellWidth); // Note the use of the *unwrapped* neighbor here

              // We can quickly eliminate some neighboring cells whose nearest point is not in range
              if (!isBoidWithinRadiusOfGridCell(pos_i, neighborGridCellMinPoint, cellWidth, inverseCellWidthToSearchDistanceRatio * cellWidth)) continue;

              // - For each cell, read the start/end indices in the boid pointer array.
              int neighborCell1D = gridIndex3Dto1D(neighborCell3DWrapped.x, neighborCell3DWrapped.y, neighborCell3DWrapped.z, gridResolution);
              int cellStart = gridCellStartIndices[neighborCell1D];
              int cellEnd = gridCellEndIndices[neighborCell1D];
              if (cellStart == -1 || cellEnd == -1) continue;

              // - Collect pos and vel information from boids in neighboring cells and apply rules
              for (int i = cellStart; i <= cellEnd; ++i) {
                  int neighborIdx = particleArrayIndices[i];
                  if (neighborIdx == index) continue;

                  const glm::vec3 neighborPos = pos[neighborIdx];
                  const glm::vec3 neighborVel = vel1[neighborIdx];
                  float neighborDist = glm::distance(neighborPos, pos_i);

                  if (neighborDist < rule1Distance) {
                      perceived_center += neighborPos;
                      ++rule1NeighborCount;
                  }

                  if (neighborDist < rule2Distance) {
                      rule2DeltaV -= (neighborPos - pos_i);
                  }

                  if (neighborDist < rule3Distance) {
                      perceived_velocity += neighborVel;
                      ++rule3NeighborCount;
                  }
              }

          }
      }
  }

  if (rule1NeighborCount > 0) {
      perceived_center /= rule1NeighborCount;
      rule1DeltaV = (perceived_center - pos_i) * rule1Scale;
  }

  rule2DeltaV *= rule2Scale;

  if (rule3NeighborCount > 0) {
      perceived_velocity /= rule3NeighborCount;
      rule3DeltaV = (perceived_velocity * rule3Scale);
  }

  const glm::vec3 deltaV = rule1DeltaV + rule2DeltaV + rule3DeltaV;
  glm::vec3 updatedVelocity = vel1[index] + deltaV;
  const float updatedSpeed = glm::length(updatedVelocity);
  if (updatedSpeed > maxSpeed) {   // - Clamp the speed change before putting the new speed in vel2
      updatedVelocity *= (maxSpeed / updatedSpeed);
  }

  // We record the new velocity into vel2 because we don't want to affect threads running in parallel that need the pre-update velocity of each boid.
  vel2[index] = updatedVelocity;
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;

    // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
    // the number of boids that need to be checked.
    // - Identify the grid cell that this particle is in
    const glm::vec3 pos_i = pos[index];
    const glm::vec3 gridCell3D_i = glm::floor((pos_i - gridMin) * inverseCellWidth);

    // Initialize tracking variables for different rules
    glm::vec3 perceived_center(0, 0, 0);
    glm::vec3 separation_velocity_adjustment(0, 0, 0);
    glm::vec3 perceived_velocity(0, 0, 0);
    glm::vec3 rule1DeltaV(0, 0, 0);
    glm::vec3 rule2DeltaV(0, 0, 0);
    glm::vec3 rule3DeltaV(0, 0, 0);
    int rule1NeighborCount = 0;
    int rule3NeighborCount = 0;

    // Search over neighboring cells within the neighborhood distance of the boid
    // Note that the outermost loop is over the z-direction. This aligns our computation pattern with our boid layout in data, for optimal coherency.
    int searchRange = ceil(imax(inverseCellWidthToSearchDistanceRatio, 1.0f));
    for (int z = -searchRange; z <= searchRange; ++z) {
        for (int y = -searchRange; y <= searchRange; ++y) {
            for (int x = -searchRange; x <= searchRange; ++x) {
                const glm::vec3 neighborCell3DUnwrapped = gridCell3D_i + glm::vec3(x, y, z);
                const glm::vec3 neighborCell3DWrapped = wrapIndices(neighborCell3DUnwrapped, gridResolution);
                const glm::vec3 neighborGridCellMinPoint = gridMin + (neighborCell3DUnwrapped * cellWidth); // Note the use of the *unwrapped* neighbor here

                // We can quickly eliminate some neighboring cells whose nearest point is not in range
                if (!isBoidWithinRadiusOfGridCell(pos_i, neighborGridCellMinPoint, cellWidth, inverseCellWidthToSearchDistanceRatio * cellWidth)) continue;

                // - For each cell, read the start/end indices in the boid pointer array.
                int neighborCell1D = gridIndex3Dto1D(neighborCell3DWrapped.x, neighborCell3DWrapped.y, neighborCell3DWrapped.z, gridResolution);
                int cellStart = gridCellStartIndices[neighborCell1D];
                int cellEnd = gridCellEndIndices[neighborCell1D];
                if (cellStart == -1 || cellEnd == -1) continue;

                // - Collect pos and vel information from boids in neighboring cells and apply rules
                for (int i = cellStart; i <= cellEnd; ++i) {
                    if (i == index) continue;

                    const glm::vec3 neighborPos = pos[i];
                    const glm::vec3 neighborVel = vel1[i];
                    float neighborDist = glm::distance(neighborPos, pos_i);

                    if (neighborDist < rule1Distance) {
                        perceived_center += neighborPos;
                        ++rule1NeighborCount;
                    }

                    if (neighborDist < rule2Distance) {
                        rule2DeltaV -= (neighborPos - pos_i);
                    }

                    if (neighborDist < rule3Distance) {
                        perceived_velocity += neighborVel;
                        ++rule3NeighborCount;
                    }
                }

            }
        }
    }

    if (rule1NeighborCount > 0) {
        perceived_center /= rule1NeighborCount;
        rule1DeltaV = (perceived_center - pos_i) * rule1Scale;
    }

    rule2DeltaV *= rule2Scale;

    if (rule3NeighborCount > 0) {
        perceived_velocity /= rule3NeighborCount;
        rule3DeltaV = (perceived_velocity * rule3Scale);
    }

    const glm::vec3 deltaV = rule1DeltaV + rule2DeltaV + rule3DeltaV;
    glm::vec3 updatedVelocity = vel1[index] + deltaV;
    const float updatedSpeed = glm::length(updatedVelocity);
    if (updatedSpeed > maxSpeed) {   // - Clamp the speed change before putting the new speed in vel2
        updatedVelocity *= (maxSpeed / updatedSpeed);
    }

    // We record the new velocity into vel2 because we don't want to affect threads running in parallel that need the pre-update velocity of each boid.
    vel2[index] = updatedVelocity;
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  kernUpdateVelocityBruteForce<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, dev_vel1, dev_vel2);
  
  cudaDeviceSynchronize();
  checkCUDAErrorWithLine("Error during execution of velocity update kernel!");

  kernUpdatePos<<<fullBlocksPerGrid, blockSize >>>(numObjects, dt, dev_pos, dev_vel2);
  
  cudaDeviceSynchronize();
  checkCUDAErrorWithLine("Error during execution of position update kernel!");

  glm::vec3* tmp_dev_vel = dev_vel1;
  dev_vel1 = dev_vel2;
  dev_vel2 = tmp_dev_vel;

}

void Boids::stepSimulationScatteredGrid(float dt) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  // First, reset the cell start and end indices to be populated with -1, so we know when a cell is empty.
  dim3 gridIndicesResetBlocks((gridCellCount + blockSize - 1) / blockSize);
  kernResetIntBuffer<<<gridIndicesResetBlocks, blockSize >>>(gridCellCount, dev_gridCellStartIndices, -1);
  kernResetIntBuffer<<<gridIndicesResetBlocks, blockSize >>>(gridCellCount, dev_gridCellEndIndices, -1);

  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, 
                                                        dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
 
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleGridIndices, 
                                                             dev_gridCellStartIndices, dev_gridCellEndIndices);
  
  // - Perform velocity updates using neighbor search
  kernUpdateVelNeighborSearchScattered<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount, gridMinimum, 
                                                                         gridInverseCellWidth, gridCellWidth,
                                                                         dev_gridCellStartIndices, dev_gridCellEndIndices, 
                                                                         dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);

  // - Update positions
  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);

  // - Ping-pong buffers as needed
  glm::vec3* tmp_dev_vel = dev_vel1;
  dev_vel1 = dev_vel2;
  dev_vel2 = tmp_dev_vel;
}

void Boids::stepSimulationCoherentGrid(float dt) {
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

    // First, reset the cell start and end indices to be populated with -1, so we know when a cell is empty.
    dim3 gridIndicesResetBlocks((gridCellCount + blockSize - 1) / blockSize);
    kernResetIntBuffer<<<gridIndicesResetBlocks, blockSize>>> (gridCellCount, dev_gridCellStartIndices, -1);
    kernResetIntBuffer<<<gridIndicesResetBlocks, blockSize>>> (gridCellCount, dev_gridCellEndIndices, -1);

    kernComputeIndices<<<fullBlocksPerGrid, blockSize>>> (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
        dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

    // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
    //   are welcome to do a performance comparison.
    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);
    thrust::gather(dev_thrust_particleArrayIndices, dev_thrust_particleArrayIndices + numObjects, dev_thrust_pos, dev_thrust_pos_sorted);
    thrust::gather(dev_thrust_particleArrayIndices, dev_thrust_particleArrayIndices + numObjects, dev_thrust_vel, dev_thrust_vel1_sorted);

    // - Naively unroll the loop for finding the start and end indices of each
    //   cell's data pointers in the array of boid indices
    kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleGridIndices,
        dev_gridCellStartIndices, dev_gridCellEndIndices);

    // - Perform velocity updates using neighbor search
    kernUpdateVelNeighborSearchCoherent<<<fullBlocksPerGrid, blockSize>>> (numObjects, gridSideCount, gridMinimum, 
        gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_pos_sorted, dev_vel1_sorted, dev_vel2);

    // - Update positions
    kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos_sorted, dev_vel2);

    // - Ping-pong buffers as needed
    glm::vec3* tmp_dev_vel = dev_vel1;
    dev_vel1 = dev_vel2;
    dev_vel2 = tmp_dev_vel;

    glm::vec3* tmp_dev_pos = dev_pos;
    dev_pos = dev_pos_sorted;
    dev_pos_sorted = tmp_dev_pos;

    // And update thrust pointers
    dev_thrust_pos = thrust::device_ptr<glm::vec3>(dev_pos);
    dev_thrust_pos_sorted = thrust::device_ptr<glm::vec3>(dev_pos_sorted);
    dev_thrust_vel = thrust::device_ptr<glm::vec3>(dev_vel1);
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_vel1_sorted);
  cudaFree(dev_pos);
  cudaFree(dev_pos_sorted);

  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellEndIndices);
  cudaFree(dev_gridCellStartIndices);
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
