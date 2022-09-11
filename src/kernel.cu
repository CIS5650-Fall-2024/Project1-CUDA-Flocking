#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"
#include <device_launch_parameters.h>

// LOOK-2.1 potentially useful for doing grid-based neighbor search
// Di: Returns max of the two or min of the two elements for comparison.
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
#define rule1Distance 5.0f // 5.0f
#define rule2Distance 3.0f // 3.0f
#define rule3Distance 5.0f // 5.0f

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
// Di all initialized on GPU
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?

// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.
// glm::vec3* dev_posSortedPrev;
glm::vec3* dev_posSorted;
glm::vec3* dev_vel1Sorted;
// thrust::device_ptr<int> dev_thrust_particleArrayIndicesValue;


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
  // some threads are extra and thus won't enter this case.
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

  // di added
  cudaMalloc((void**)&dev_posSorted, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_posSorted failed!");

  cudaMalloc((void**)&dev_vel1Sorted, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_posSortedVel1 failed!");

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

  // dev_posSortedPrev = dev_pos;

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

  kernCopyPositionsToVBO <<<fullBlocksPerGrid, blockSize >>>(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO <<<fullBlocksPerGrid, blockSize >>>(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

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
  //// return glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 velocityChange = glm::vec3(0.f);
    glm::vec3 selfPos = pos[iSelf];
 
    // Rule 1
    // find the average position of all the birds
    glm::vec3 center = glm::vec3(0.f);
    int numValid1 = 0;

    glm::vec3 c = glm::vec3(0.f);

    glm::vec3 perceivedVelocity = glm::vec3(0.f);
    int numValid3 = 0;

    for (int i = 0; i < N; i++) {
        glm::vec3 bPos = pos[i];
        float dist = glm::distance(selfPos, bPos);

        // Rule 1
        if (i != iSelf && dist < rule1Distance) {
            center += pos[i];
            numValid1 += 1;
        }

        // Rule 2
        if (i != iSelf && dist < rule2Distance) {
            c -= (bPos - selfPos);
        }

        // Rule 3
        if (i != iSelf && dist < rule3Distance) {
            perceivedVelocity += vel[i];
            numValid3 += 1;
        }

    }

    if (numValid1 != 0) {
        center /= numValid1;
        velocityChange += (center - selfPos) * rule1Scale;
    }

    velocityChange += c * rule2Scale;

    if (numValid3 != 0) {
        perceivedVelocity /= numValid3;
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
  // Record the new velocity into vel2. Question: why NOT vel1? -> 
 
    // we don't want to overwrite original velocities because other boids still need to refer to it to compute their vels.
    // and vel1 and vel2 are all being accessed at the same time so if info in vel1 gets changed while another thread is accessing
    // that would be bad.
    // once we finish computing these vels, vel1 becomes our "write" array and vel2 becomes our "read" array
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }
    glm::vec3 velChange = computeVelocityChange(N, index, pos, vel1);
    glm::vec3 finalVel = vel1[index] + velChange;
    float speed = glm::length(finalVel);
    if (speed > maxSpeed) {
        finalVel = finalVel * maxSpeed / speed;
    }
    vel2[index] = finalVel;
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

  // debug value
  glm::vec3 thisVel = vel[index];

  thisPos += vel[index] * dt;

  // printf("vel x: %f, vel y: %f, vel z: %f", vel[index].x, vel[index].y, vel[index].z);

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
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridSideCount) {
  return x + y * gridSideCount + z * gridSideCount * gridSideCount;
}

__global__ void kernComputeIndices(int N, int gridSideCount,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
    
    // this kern function is called in parallel by each boid so it only needs to be 
    int boidIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (boidIdx < N) {
        // compute components
        glm::vec3 boidPos = pos[boidIdx];

        int ix = (int) (std::floor((boidPos.x - gridMin.x) * inverseCellWidth));
        int iy = (int) (std::floor((boidPos.y - gridMin.y) * inverseCellWidth));
        int iz = (int) (std::floor((boidPos.z - gridMin.z) * inverseCellWidth));

        // compute 1D cell idx from 3D
        int cellIdx = gridIndex3Dto1D(ix, iy, iz, gridSideCount);

        // fill dev_particleGridIndices at boidIdx
        gridIndices[boidIdx] = cellIdx;

        // fill dev_particleArrayIndices, starts exactly as boidIdx initially.
        indices[boidIdx] = boidIdx;
    }

}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
// pass in an invalid array value such as -1 for cells that don't contain boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"

    // post sorting
    // also looks like it's called in parallel by boids.
    // note that idx is NOT boidIdx
    // for current idx, check its own cell in GridIndices and one before it.
    // if idx 0, must be start cell, and set its start location to itself. then check for end cell status
    // if idx N - 1, must be end cell, and set its end location to itself. then check for start cell status
    // otherwise, if different from idx - 1, set as start cell.
    // if different from idx + 1, set as end cell.
    // not sure if most efficient way to do it honestly.

    int accessIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (accessIdx < N) {
        int gridCell = particleGridIndices[accessIdx];

        if (accessIdx == 0) {
            gridCellStartIndices[gridCell] = accessIdx;

            // check for next grid cell as well
            int nextGridCell = particleGridIndices[accessIdx + 1];
            if (gridCell != nextGridCell) {
                gridCellEndIndices[gridCell] = accessIdx;
            }
        }
        else if (accessIdx == N - 1) {
            gridCellEndIndices[gridCell] = accessIdx;

            // check if gridCell == prevGridCell. 
            int prevGridCell = particleGridIndices[accessIdx - 1];
            if (gridCell != prevGridCell) {
                gridCellStartIndices[gridCell] = accessIdx;
            }
        }
        else {
            int prevGridCell = particleGridIndices[accessIdx - 1];
            int nextGridCell = particleGridIndices[accessIdx + 1];
            if (gridCell != prevGridCell) {
                gridCellStartIndices[gridCell] = accessIdx;
            }
            if (gridCell != nextGridCell) {
                gridCellEndIndices[gridCell] = accessIdx;
            }
        }
    }
}

// device helper function
__device__ glm::vec3 kernComputeVelocityChangeScattered(int N, glm::vec3 gridMin, 
    float cellWidth, int iSelf, glm::vec3* pos, 
    glm::vec3* vel1, int* gridCellStartIndices, 
    int* gridCellEndIndices, int* particleArrayIndices, int sideCount) {

    int selfIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (selfIdx < N) {
        glm::vec3 selfPos = pos[selfIdx];
        glm::vec3 velocityChange = glm::vec3(0.f);

        float radius = imax(imax(rule1Distance, rule2Distance), rule3Distance);

        int numValid1 = 0;
        int numValid3 = 0;

        glm::vec3 center = glm::vec3(0.f);
        glm::vec3 c = glm::vec3(0.f);
        glm::vec3 perceivedVelocity = glm::vec3(0.f);

        // find maximum of entire grid in cell space coordinates.
        glm::vec3 cellSpaceMax = glm::vec3(sideCount - 1, sideCount - 1, sideCount - 1);

        // create a bounding box based on the location of boid within the grid in cell space coordinates.
        float xMax, xMin, yMax, yMin, zMax, zMin = 0;

        // xMax and co. are all cell space coordinates and not world space coordaintes.
        // find the grid cell associated with radMaxx, radMaxy, radMaxz, radMinx, radMiny, radMinz
        // set xMax, xMin, yMax, yMin, zMax, zMin etc by clamping between that and 
        int ixRadMax = std::floor((selfPos.x + radius - gridMin.x) / cellWidth);
        int iyRadMax = std::floor((selfPos.y + radius - gridMin.y) / cellWidth);
        int izRadMax = std::floor((selfPos.z + radius - gridMin.z) / cellWidth);
        int ixRadMin = std::floor((selfPos.x - radius - gridMin.x) / cellWidth);
        int iyRadMin = std::floor((selfPos.y - radius - gridMin.y) / cellWidth);
        int izRadMin = std::floor((selfPos.z - radius - gridMin.z) / cellWidth);

        xMax = imin(ixRadMax, cellSpaceMax.x);
        yMax = imin(iyRadMax, cellSpaceMax.y);
        zMax = imin(izRadMax, cellSpaceMax.z);
        xMin = imax(ixRadMin, gridMin.x);
        yMin = imax(iyRadMin, gridMin.y);
        zMin = imax(izRadMin, gridMin.z);

        // loop within the bounding box.
        // these coordinates have already been clamped
        // without consideration of which axis is best to check first
        // started with x,y,z

        for (float z = zMin; z <= zMax; z += 1) {
            for (float y = yMin; y <= yMax; y += 1) {
                for (float x = xMin; x <= xMax; x += 1) {
                    int gridIdx = gridIndex3Dto1D(x, y, z, sideCount);
                    int startIdx = gridCellStartIndices[gridIdx];
                    int endIdx = gridCellEndIndices[gridIdx];

                    if (startIdx > -1) {
                        // if start idx is > -1, then boids exist
                        for (int curIdx = startIdx; curIdx <= endIdx; curIdx++) {
                            int nBoidIdx = particleArrayIndices[curIdx];
                            glm::vec3 nBoidPos = pos[nBoidIdx];

                            float dist = glm::distance(selfPos, nBoidPos);

                            // Rule 1
                            if (nBoidIdx != selfIdx && dist < rule1Distance) {
                                center += pos[nBoidIdx];
                                numValid1 += 1;
                            }

                            // Rule 2
                            if (nBoidIdx != selfIdx && dist < rule2Distance) {
                                c -= (nBoidPos - selfPos);
                            }

                            // Rule 3
                            if (nBoidIdx != selfIdx && dist < rule3Distance) {
                                perceivedVelocity += vel1[nBoidIdx];
                                numValid3 += 1;
                            }
                        }
                    }
                }
            }
        }

        if (numValid1 != 0) {
            center /= numValid1;
            velocityChange += (center - selfPos) * rule1Scale;
        }

        velocityChange += c * rule2Scale;

        if (numValid3 != 0) {
            perceivedVelocity /= numValid3;
            velocityChange += perceivedVelocity * rule3Scale;
        }

        return velocityChange;
    }
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridSideCount, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
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

    // for a given boid, try all cells first, then optimize by doing quadrant search later.
    // 1. split cell into octants, identify the octant it belongs in
    // 2. get 8 neighboring octants. figure out which octants contain cells. --> get all filled cells regardless of distance
    // 2. Scan the appropriate 8 or so octants based on the result
    // 3. for each cell, read the start/end endices and read boid in each cell. apply velocity changes.
    // 4. clamp speed as usual

    int selfIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (selfIdx < N) {
        glm::vec3 velocityChange = kernComputeVelocityChangeScattered(N, gridMin, cellWidth, selfIdx, pos, vel1, gridCellStartIndices, 
            gridCellEndIndices, particleArrayIndices, gridSideCount);

        // clamp speed and set new velocity
        glm::vec3 finalVel = vel1[selfIdx] + velocityChange;
        float speed = glm::length(finalVel);
        if (speed > maxSpeed) {
            finalVel = finalVel * maxSpeed / speed;
        }
        // printf("finalVel: x: %f, y: %f, z: %f \n", finalVel.x, finalVel.y, finalVel.z);

        vel2[selfIdx] = finalVel;
    }
}

// helper device
__device__ glm::vec3 kernComputeVelocityChangeCoherent(int N, glm::vec3 gridMin,
    float cellWidth, int iSelf, glm::vec3* posSorted,
    glm::vec3* vel1Sorted, int* gridCellStartIndices,
    int* gridCellEndIndices, int sideCount) {
    // int selfIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (iSelf < N) {
        glm::vec3 selfPos = posSorted[iSelf];
        glm::vec3 velocityChange = glm::vec3(0.f);

        float radius = imax(imax(rule1Distance, rule2Distance), rule3Distance);

        int numValid1 = 0;
        int numValid3 = 0;

        glm::vec3 center = glm::vec3(0.f);
        glm::vec3 c = glm::vec3(0.f);
        glm::vec3 perceivedVelocity = glm::vec3(0.f);

        // find maximum of entire grid in cell space coordinates.
        glm::vec3 cellSpaceMax = glm::vec3(sideCount - 1, sideCount - 1, sideCount - 1);

        // create a bounding box based on the location of boid within the grid in cell space coordinates.
        float xMax, xMin, yMax, yMin, zMax, zMin = 0;

        // xMax and co. are all cell space coordinates and not world space coordaintes.
        // find the grid cell associated with radMaxx, radMaxy, radMaxz, radMinx, radMiny, radMinz
        // set xMax, xMin, yMax, yMin, zMax, zMin etc by clamping between that and 
        int ixRadMax = std::floor((selfPos.x + radius - gridMin.x) / cellWidth);
        int iyRadMax = std::floor((selfPos.y + radius - gridMin.y) / cellWidth);
        int izRadMax = std::floor((selfPos.z + radius - gridMin.z) / cellWidth);
        int ixRadMin = std::floor((selfPos.x - radius - gridMin.x) / cellWidth);
        int iyRadMin = std::floor((selfPos.y - radius - gridMin.y) / cellWidth);
        int izRadMin = std::floor((selfPos.z - radius - gridMin.z) / cellWidth);

        xMax = imin(ixRadMax, cellSpaceMax.x);
        yMax = imin(iyRadMax, cellSpaceMax.y);
        zMax = imin(izRadMax, cellSpaceMax.z);
        xMin = imax(ixRadMin, gridMin.x);
        yMin = imax(iyRadMin, gridMin.y);
        zMin = imax(izRadMin, gridMin.z);

        // loop within the bounding box.
        // these coordinates have already been clamped
        // todo consider which axis is the best for checking
        for (float x = xMin; x <= xMax; x += 1) {
            for (float y = yMin; y <= yMax; y += 1) {
                for (float z = zMin; z <= zMax; z += 1) {
                    int gridIdx = gridIndex3Dto1D(x, y, z, sideCount);
                    int startIdx = gridCellStartIndices[gridIdx];
                    int endIdx = gridCellEndIndices[gridIdx];

                    if (startIdx > -1) {
                        // if start idx is > -1, then boids exist
                        for (int curIdx = startIdx; curIdx <= endIdx; curIdx++) {
                            // now curIdx is the same as nBoidIdx
                            glm::vec3 nBoidPos = posSorted[curIdx];
                            glm::vec3 vel = vel1Sorted[curIdx];

                            float dist = glm::distance(selfPos, nBoidPos);
                            //printf("dist: %f \n", dist);

                            // Rule 1
                            if (curIdx != iSelf && dist < rule1Distance) {
                                center += posSorted[curIdx];
                                numValid1 += 1;
                            }

                            // Rule 2
                            if (curIdx != iSelf && dist < rule2Distance) {
                                c -= (nBoidPos - selfPos);
                            }

                            // Rule 3
                            if (curIdx != iSelf && dist < rule3Distance) {
                                perceivedVelocity += vel1Sorted[curIdx];
                                numValid3 += 1;
                            }
                        }
                    }
                }
            }
        }

        if (numValid1 != 0) {
            center /= numValid1;
            velocityChange += (center - selfPos) * rule1Scale;
        }

        velocityChange += c * rule2Scale;

        if (numValid3 != 0) {
            perceivedVelocity /= numValid3;
            velocityChange += perceivedVelocity * rule3Scale;
        }

        return velocityChange;
    }
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridSideCount, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *posSorted, glm::vec3 *vel1Sorted, glm::vec3 *vel2) {
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
    int selfIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (selfIdx < N) {
        glm::vec3 velocityChange = kernComputeVelocityChangeCoherent(N, gridMin, 
            cellWidth, selfIdx, posSorted, 
            vel1Sorted, gridCellStartIndices,
            gridCellEndIndices, gridSideCount);

        // clamp speed and set new velocity
        glm::vec3 finalVel = vel1Sorted[selfIdx] + velocityChange;
        float speed = glm::length(finalVel);
        if (speed > maxSpeed) {
            finalVel = finalVel * maxSpeed / speed;
        }
        // printf("finalVel: x: %f, y: %f, z: %f \n", finalVel.x, finalVel.y, finalVel.z);

        // vel2 is still the write buffer
        vel2[selfIdx] = finalVel;
    }
}

// helper global which rearranges information in pos and vel based on particleArrayIndices O(N) time

__global__ void kernRearrangePosVel(int N, int* particleArrayIndices, glm::vec3* posSorted, glm::vec3* vel1Sorted, glm::vec3* pos, glm::vec3* vel1) {
    int selfIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (selfIdx < N) {
        int requiredIdx = particleArrayIndices[selfIdx];

        //printf("vel x: %f, y: %f, z: %f \n ", vel1Prev[requiredIdx].x, vel1Prev[requiredIdx].y, vel1Prev[requiredIdx].z);

        // essentially swap the contents
        glm::vec3 requiredPos = pos[requiredIdx];
        glm::vec3 requiredVel = vel1[requiredIdx];

        posSorted[selfIdx] = requiredPos;
        vel1Sorted[selfIdx] = requiredVel;
    }
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  // TODO-1.2 ping-pong the velocity buffers
    int fullBlocksPerGrid = (numObjects + blockSize - 1) / blockSize;

    kernUpdateVelocityBruteForce<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, dev_vel1, dev_vel2);
    kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel1);

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
    int numCells = gridSideCount * gridSideCount * gridSideCount;

    // For each boid, label its array index and grid index
    int fullBlocksPerGrid = (numObjects + blockSize - 1) / blockSize;
    int fullCellsPerGrid = (numCells + blockSize - 1) / blockSize;

    // fill with -1s as default values
    // Per cell basis
    kernResetIntBuffer<<<fullCellsPerGrid, blockSize>>>(numCells, dev_gridCellStartIndices, -1);
    kernResetIntBuffer<<<fullCellsPerGrid, blockSize>>>(numCells, dev_gridCellEndIndices, -1);
    
    // per boid basis
    kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
        dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

    // pointer to first key, pointer to last key, pointer to first value
    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

    kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

    // per boid basis
    kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, blockSize >> >(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
        dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);

    // update position per boid
    kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel1);

    // ping pong
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
    int numCells = gridSideCount * gridSideCount * gridSideCount;

    // For each boid, label its array index and grid index
    int fullBlocksPerGrid = (numObjects + blockSize - 1) / blockSize;
    int fullCellsPerGrid = (numCells + blockSize - 1) / blockSize;

    // fill with -1s as default values
    // Per cell basis
    kernResetIntBuffer << <fullCellsPerGrid, blockSize >> > (numCells, dev_gridCellStartIndices, -1);
    kernResetIntBuffer << <fullCellsPerGrid, blockSize >> > (numCells, dev_gridCellEndIndices, -1);

    // per boid basis
    kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
        dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

    // sort dev_array_indices values
    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

    // rearrange pos and vel
    // we never change vel1 and pos. only use them as reads.
    kernRearrangePosVel << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleArrayIndices, dev_posSorted, dev_vel1Sorted, dev_pos, dev_vel1);

    // find start and end indices
// per boid basis because we read from dev_particleGridIndices
    kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

    // per boid basis  
    kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, 
        gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, 
        dev_gridCellEndIndices, dev_posSorted, dev_vel1Sorted, dev_vel2);

    // update position per boid
    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_posSorted, dev_vel2);

    // ping pong
    std::swap(dev_vel1, dev_vel2);
    std::swap(dev_posSorted, dev_pos);
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

  // cannot free the thrust pointers
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
