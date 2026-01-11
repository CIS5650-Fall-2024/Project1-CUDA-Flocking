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

// smaller cell width = the neighbourhood distance
// larger cell width = 2 * the neighbourhood distance
#define USE_LARGER_CELL_WIDTH true
#define USE_SHARED_MEMORY true

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
#define blockSize 128 // Default is 128

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
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.
glm::vec3 *dev_pos_coherent;
glm::vec3 *dev_vel1_coherent;
glm::vec3 *dev_vel2_coherent;

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float neighborDistance;
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
  // Don't forget to cudaFree in Boids::endSimulation.

  // Note: We don't pass dev_pos because this means we write the result (i.e. the GPU mem address we've allocated) to
  // where dev_pos is pointed to, which is garbage. Instead, we pass &dev_pos, which means we write the result to
  // wherever &dev_pos is pointed to, which is an initialised address (because we've initialised the variable and it has an address).
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
  neighborDistance = std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
#if USE_LARGER_CELL_WIDTH
  gridCellWidth = 2.0f * neighborDistance;
#else
  gridCellWidth = neighborDistance;
#endif

  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
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

  dev_thrust_particleArrayIndices = thrust::device_pointer_cast(dev_particleArrayIndices);  

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  dev_thrust_particleGridIndices = thrust::device_pointer_cast(dev_particleGridIndices);

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  cudaMalloc((void**)&dev_pos_coherent, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos_coherent failed!");

  cudaMalloc((void**)&dev_vel1_coherent, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1_coherent failed!");

  cudaMalloc((void**)&dev_vel2_coherent, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2_coherent failed!");

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
  glm::vec3 perceived_center = glm::vec3(0.0f, 0.0f, 0.0f);
  glm::vec3 c = glm::vec3(0.0f, 0.0f, 0.0f);
  glm::vec3 perceived_velocity = glm::vec3(0.0f, 0.0f, 0.0f);

  int num_neighbors_r1 = 0;
  int num_neighbors_r3 = 0;

  for (int i = 0; i < N; i++) {
    if (i != iSelf) {
      float distance = glm::distance(pos[i], pos[iSelf]);
      if (distance < rule1Distance) {
        perceived_center += pos[i];
        num_neighbors_r1++;
      }

      if (distance < rule2Distance) {
        c -= pos[i] - pos[iSelf];
      }

      if (distance < rule3Distance) {
        perceived_velocity += vel[i];
        num_neighbors_r3++;
      }
    }
  }

  if (num_neighbors_r1 > 0) {
    perceived_center /= num_neighbors_r1;
  }

  if (num_neighbors_r3 > 0) {
    perceived_velocity /= num_neighbors_r3;
  }

  glm::vec3 v1 = (perceived_center - pos[iSelf]) * rule1Scale;
  glm::vec3 v2 = c * rule2Scale;
  glm::vec3 v3 = perceived_velocity * rule3Scale;
  
  return vel[iSelf] + v1 + v2 + v3;
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
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 vel = computeVelocityChange(N, index, pos, vel1);
    vel = glm::length(vel) > maxSpeed ? glm::normalize(vel) * maxSpeed : vel;
    vel2[index] = vel;
  }
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
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
      return;
    }

    indices[index] = index;

    gridIndices[index] = gridIndex3Dto1D(
      (pos[index].x - gridMin.x) * inverseCellWidth,
      (pos[index].y - gridMin.y) * inverseCellWidth,
      (pos[index].z - gridMin.z) * inverseCellWidth,
      gridResolution
    ); 
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(
  int N, const int* sortedGridIdx,
  int* cellStart, int* cellEnd)
{
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"
  // At this point, the particleGridIndices has already been sorted.
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  int cur  = sortedGridIdx[i];
  int prev = (i > 0)     ? sortedGridIdx[i - 1] : -1;
  int next = (i < N - 1) ? sortedGridIdx[i + 1] : -1;
  if (i == 0     || cur != prev) cellStart[cur] = i;
  if (i == N - 1 || cur != next) cellEnd[cur]   = i;
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float neighborDistance,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;

  // Process boids in the same order as the sorted grid arrays (better locality)
  const int self = particleArrayIndices[tid];

  // ----- compute integer cell of this boid -----
  const glm::vec3 rel = (pos[self] - gridMin) * inverseCellWidth;
  const int ix = max(0, min(gridResolution - 1, (int)floorf(rel.x)));
  const int iy = max(0, min(gridResolution - 1, (int)floorf(rel.y)));
  const int iz = max(0, min(gridResolution - 1, (int)floorf(rel.z)));

  // ----- map [pos - R, pos + R] to index ranges (inclusive) -----
  const float R = neighborDistance;

  int ixMin = (int)floorf((pos[self].x - R - gridMin.x) * inverseCellWidth);
  int ixMax = (int)floorf((pos[self].x + R - gridMin.x) * inverseCellWidth);
  int iyMin = (int)floorf((pos[self].y - R - gridMin.y) * inverseCellWidth);
  int iyMax = (int)floorf((pos[self].y + R - gridMin.y) * inverseCellWidth);
  int izMin = (int)floorf((pos[self].z - R - gridMin.z) * inverseCellWidth);
  int izMax = (int)floorf((pos[self].z + R - gridMin.z) * inverseCellWidth);

  ixMin = max(0, min(gridResolution - 1, ixMin));
  ixMax = max(0, min(gridResolution - 1, ixMax));
  iyMin = max(0, min(gridResolution - 1, iyMin));
  iyMax = max(0, min(gridResolution - 1, iyMax));
  izMin = max(0, min(gridResolution - 1, izMin));
  izMax = max(0, min(gridResolution - 1, izMax));

  // ----- accumulators -----
  glm::vec3 perceived_center(0.0f);
  glm::vec3 c(0.0f);
  glm::vec3 perceived_velocity(0.0f);
  int num_r1 = 0, num_r3 = 0;

  const float r1sq = rule1Distance * rule1Distance;
  const float r2sq = rule2Distance * rule2Distance;
  const float r3sq = rule3Distance * rule3Distance;

  // ----- iterate only the needed neighbor cells (inclusive ranges) -----
  for (int z = izMin; z <= izMax; ++z) {
    for (int y = iyMin; y <= iyMax; ++y) {
      for (int x = ixMin; x <= ixMax; ++x) {
        const int cell1D = gridIndex3Dto1D(x, y, z, gridResolution);

        const int start = gridCellStartIndices[cell1D];
        if (start == -1) continue;  // empty cell

        const int end = gridCellEndIndices[cell1D];

        // Indirection for scattered layout: turn sorted slot -> original boid index
        for (int b = start; b <= end; ++b) {
          const int j = particleArrayIndices[b];
          if (j == self) continue;

          const glm::vec3 d = pos[j] - pos[self];
          const float dist2 = glm::dot(d, d);

          if (dist2 < r1sq) { perceived_center += pos[j]; ++num_r1; }
          if (dist2 < r2sq) { c -= d; }
          if (dist2 < r3sq) { perceived_velocity += vel1[j]; ++num_r3; }
        }
      }
    }
  }

  if (num_r1 > 0) perceived_center /= (float)num_r1;
  if (num_r3 > 0) perceived_velocity /= (float)num_r3;

  glm::vec3 v1 = (perceived_center - pos[self]) * rule1Scale;
  glm::vec3 v2 = c * rule2Scale;
  glm::vec3 v3 = perceived_velocity * rule3Scale;

  glm::vec3 v = vel1[self] + v1 + v2 + v3;
  float speed = glm::length(v);
  if (speed > maxSpeed) v *= (maxSpeed / speed);

  vel2[self] = v;
}

__global__ void kernGenerateCoherentPosVal(int N, int *dev_particleArrayIndices, glm::vec3 *dev_pos, glm::vec3 *dev_vel1, glm::vec3 *dev_pos_coherent, glm::vec3 *dev_vel1_coherent) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }

  int array_index = dev_particleArrayIndices[index];
  dev_pos_coherent[index] = dev_pos[array_index];
  dev_vel1_coherent[index] = dev_vel1[array_index];
}

__global__ void kernRestorePosValFromCoherent(int N, int *dev_particleArrayIndices, glm::vec3 *dev_pos, glm::vec3 *dev_vel2, glm::vec3 *dev_pos_coherent, glm::vec3 *dev_vel2_coherent) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }

  int array_index = dev_particleArrayIndices[index];
  dev_pos[array_index] = dev_pos_coherent[index];
  dev_vel2[array_index] = dev_vel2_coherent[index];
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float neighborDistance,
  const int *gridCellStartIndices, const int *gridCellEndIndices,
  const glm::vec3 *pos, const glm::vec3 *vel1, glm::vec3 *vel2) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  // Axis-wise min/max indices from [pos - R, pos + R]
  auto toCell = [&](float x, float gmin) {
    return (int)floorf((x - gmin) * inverseCellWidth);
  };
  int ixMin = max(0, toCell(pos[i].x - neighborDistance, gridMin.x));
  int ixMax = min(gridResolution - 1, toCell(pos[i].x + neighborDistance, gridMin.x));
  int iyMin = max(0, toCell(pos[i].y - neighborDistance, gridMin.y));
  int iyMax = min(gridResolution - 1, toCell(pos[i].y + neighborDistance, gridMin.y));
  int izMin = max(0, toCell(pos[i].z - neighborDistance, gridMin.z));
  int izMax = min(gridResolution - 1, toCell(pos[i].z + neighborDistance, gridMin.z));

  // ----- accumulators -----
  glm::vec3 perceived_center(0.0f);
  glm::vec3 c(0.0f);
  glm::vec3 perceived_velocity(0.0f);
  int num_r1 = 0, num_r3 = 0;

  const float R1sq = rule1Distance * rule1Distance;
  const float R2sq = rule2Distance * rule2Distance;
  const float R3sq = rule3Distance * rule3Distance;

  // Iterate only the needed cells (1–8 when cellWidth=2R)
  for (int z = izMin; z <= izMax; ++z) {
    for (int y = iyMin; y <= iyMax; ++y) {
      for (int x = ixMin; x <= ixMax; ++x) {
        int cell1D = gridIndex3Dto1D(x, y, z, gridResolution);
        int start  = gridCellStartIndices[cell1D];
        if (start == -1) continue;
        int end    = gridCellEndIndices[cell1D];

        // NOTE: coherent layout → direct indices b
        for (int b = start; b <= end; ++b) {
          if (b == i) continue;

          glm::vec3 d = pos[b] - pos[i];
          float dist2 = glm::dot(d, d);

          if (dist2 < R1sq) { perceived_center += pos[b]; ++num_r1; }
          if (dist2 < R2sq) { c -= d; }
          if (dist2 < R3sq) { perceived_velocity += vel1[b]; ++num_r3; }
        }
      }
    }
  }

  if (num_r1 > 0) perceived_center /= (float)num_r1;
  if (num_r3 > 0) perceived_velocity /= (float)num_r3;

  glm::vec3 v1 = (perceived_center - pos[i]) * rule1Scale;
  glm::vec3 v2 = c * rule2Scale;
  glm::vec3 v3 = perceived_velocity * rule3Scale;

  glm::vec3 v = vel1[i] + v1 + v2 + v3;
  float speed = glm::length(v);
  if (speed > maxSpeed) v = v * (maxSpeed / speed);

  vel2[i] = v;
}

__global__ void kernUpdateVelCoherentShared(
  int gridResolution,
  glm::vec3 gridMin,
  float invCellWidth,
  float neighborDistance,
  const int* __restrict__ cellStart,
  const int* __restrict__ cellEnd,
  const glm::vec3* __restrict__ posCo, // coherent
  const glm::vec3* __restrict__ velCo, // coherent
  glm::vec3* __restrict__ outVelCo // coherent
) {
  // We use float4 because it’s the most GPU-friendly way to store 3D vectors for CUDA kernels, especially with shared-memory tiling.
  extern __shared__ float4 shmem[];   // 2*blockDim.x float4's
  float4* shPos = shmem;
  float4* shVel = shmem + blockDim.x;

  const int cell = blockIdx.x;
  const int start = cellStart[cell];
  if (start == -1) return;
  const int end   = cellEnd[cell];

  // This block processes boids in [start..end]
  // Loop in case the cell has more boids than blockDim.x
  // (increment by blockDim.x because boid with index < multiple of it will be processed by other threads)
  for (int selfIdx = start + threadIdx.x; selfIdx <= end; selfIdx += blockDim.x) {
    // Per-thread accumulators (registers)
    glm::vec3 selfPos = posCo[selfIdx];
    glm::vec3 v1acc(0);
    glm::vec3 v2acc(0);
    glm::vec3 v3acc(0);
    int n1 = 0, n3 = 0;

    // Compute neighbor-cell index ranges from [pos+-R]
    const float R = neighborDistance;
    auto toCell = [&](float x, float gmin) {
      return (int)floorf((x - gmin) * invCellWidth);
    };

    int ixMin = max(0, toCell(selfPos.x - R, gridMin.x));
    int ixMax = min(gridResolution - 1, toCell(selfPos.x + R, gridMin.x));
    int iyMin = max(0, toCell(selfPos.y - R, gridMin.y));
    int iyMax = min(gridResolution - 1, toCell(selfPos.y + R, gridMin.y));
    int izMin = max(0, toCell(selfPos.z - R, gridMin.z));
    int izMax = min(gridResolution - 1, toCell(selfPos.z + R, gridMin.z));

    const float r1sq = rule1Distance * rule1Distance;
    const float r2sq = rule2Distance * rule2Distance;
    const float r3sq = rule3Distance * rule3Distance;

    // Loop all neighbor cells (inclusive ranges)
    for (int z = izMin; z <= izMax; ++z) {
      for (int y = iyMin; y <= iyMax; ++y) {
        for (int x = ixMin; x <= ixMax; ++x) {
          int nCell = gridIndex3Dto1D(x, y, z, gridResolution);
          int ns = cellStart[nCell];
          if (ns == -1) continue;
          int ne = cellEnd[nCell];

          // Tile the neighbor cell into shared memory
          // Note that this is run for every neighbour cell. The loop makes sure that all neighboour boids are loaded.
          for (int tile = ns; tile <= ne; tile += blockDim.x) {
            int j = tile + threadIdx.x;

            // Cooperative load (coalesced)
            if (j <= ne) {
              shPos[threadIdx.x] = make_float4(posCo[j].x, posCo[j].y, posCo[j].z, 0);
              shVel[threadIdx.x] = make_float4(velCo[j].x, velCo[j].y, velCo[j].z, 0);
            }
            __syncthreads();

            int count = min(blockDim.x, ne - tile + 1);
            // Consume the tile from shared memory
            #pragma unroll
            for (int t = 0; t < count; ++t) {
              int idx = tile + t;
              if (idx == selfIdx) continue;

              float3 d;
              d.x = shPos[t].x - selfPos.x;
              d.y = shPos[t].y - selfPos.y;
              d.z = shPos[t].z - selfPos.z;
              float dist2 = d.x*d.x + d.y*d.y + d.z*d.z;

              if (dist2 < r1sq) { v1acc.x += shPos[t].x; v1acc.y += shPos[t].y; v1acc.z += shPos[t].z; ++n1; }
              if (dist2 < r2sq) { v2acc.x -= d.x; v2acc.y -= d.y; v2acc.z -= d.z; }
              if (dist2 < r3sq) { v3acc.x += shVel[t].x; v3acc.y += shVel[t].y; v3acc.z += shVel[t].z; ++n3; }
            }
            __syncthreads();
          }
        }
      }
    }

    if (n1 > 0) { v1acc /= (float)n1; v1acc -= selfPos; }
    if (n3 > 0) { v3acc /= (float)n3; }

    // Scales
    v1acc *= rule1Scale;
    v2acc *= rule2Scale;
    v3acc *= rule3Scale;

    glm::vec3 v = velCo[selfIdx];
    v += v1acc + v2acc + v3acc;

    // Clamp speed
    float speed2 = glm::dot(v, v);
    if (speed2 > maxSpeed * maxSpeed) {
      float inv = maxSpeed * rsqrtf(speed2);
      v *= inv;
    }

    outVelCo[selfIdx] = glm::vec3(v.x, v.y, v.z);
  }
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  // TODO-1.2 ping-pong the velocity buffers
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernUpdateVelocityBruteForce << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, dev_vel1, dev_vel2);
  checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");

  kernUpdatePos << <fullBlocksPerGrid, blockSize >> >(numObjects, dt, dev_pos, dev_vel2);
  checkCUDAErrorWithLine("kernUpdatePos failed!");

  std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  dim3 fullBlocksPerCell((gridCellCount + blockSize - 1) / blockSize);

  kernComputeIndices << <fullBlocksPerGrid, blockSize >> >(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

  thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

  kernResetIntBuffer << <fullBlocksPerCell, blockSize >> >(gridCellCount, dev_gridCellStartIndices, -1);

  kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

  kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, blockSize >> >(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, neighborDistance, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);

  kernUpdatePos << <fullBlocksPerGrid, blockSize >> >(numObjects, dt, dev_pos, dev_vel2);

  std::swap(dev_vel1, dev_vel2);
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
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  dim3 fullBlocksPerCell((gridCellCount + blockSize - 1) / blockSize);

  kernComputeIndices << <fullBlocksPerGrid, blockSize >> >(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
  
  thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

  kernResetIntBuffer << <fullBlocksPerCell, blockSize >> >(gridCellCount, dev_gridCellStartIndices, -1);

  kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

  kernGenerateCoherentPosVal << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_particleArrayIndices, dev_pos, dev_vel1, dev_pos_coherent, dev_vel1_coherent);

#if USE_SHARED_MEMORY
  // grid dim defines the number of blocks and block dim defines the number of threads
  dim3 grid(gridCellCount);
  int block = 32;
  size_t shmem = 2 * block * sizeof(float4);
  kernUpdateVelCoherentShared <<<grid, block, shmem>>> (
      gridSideCount,
      gridMinimum,
      gridInverseCellWidth,
      neighborDistance,
      dev_gridCellStartIndices,
      dev_gridCellEndIndices,
      dev_pos_coherent,
      dev_vel1_coherent,
      dev_vel2_coherent
    );
#else
  kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> >(
      numObjects,
      gridSideCount,
      gridMinimum,
      gridInverseCellWidth,
      neighborDistance,
      dev_gridCellStartIndices,
      dev_gridCellEndIndices,
      dev_pos_coherent,
      dev_vel1_coherent,
      dev_vel2_coherent
    );
#endif

  kernUpdatePos << <fullBlocksPerGrid, blockSize >> >(numObjects, dt, dev_pos_coherent, dev_vel2_coherent);

  kernRestorePosValFromCoherent << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_particleArrayIndices, dev_pos, dev_vel2, dev_pos_coherent, dev_vel2_coherent);

  std::swap(dev_vel1, dev_vel2);
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
  cudaFree(dev_pos_coherent);
  cudaFree(dev_vel1_coherent);
  cudaFree(dev_vel2_coherent);
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
