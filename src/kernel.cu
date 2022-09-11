#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/epsilon.hpp>
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
void checkCUDAError(const char* msg, int line = -1) {
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

#define maxSpeed 4.0f
#define epsilon 1e-4f

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
glm::vec3* dev_pos1;
glm::vec3* dev_pos2; //ping pong buffer for 2.3
glm::vec3* dev_vel1;
glm::vec3* dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int* dev_boidArrayIndices; // What index in dev_pos and dev_velX represents this boid?
int* dev_boidGridIndices; // What grid cell is this boid in?
// needed for use with thrust
std::unique_ptr<thrust::device_ptr<int>> dev_thrust_boidArrayIndices;
std::unique_ptr<thrust::device_ptr<int>> dev_thrust_boidGridIndices;

int* dev_gridCellStartIndices; // What part of dev_boidArrayIndices belongs
int* dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int halfGridSideCount;
int gridSideCount;
float gridCellWidth;
float halfGridCellWidth;
float gridinverseCellWidth;
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
__host__ __device__ glm::vec3 generateRandomVec3(int seed, int index, float scale = 1.f) {
	thrust::default_random_engine rng(hash(index * seed));
	if (scale < 0) scale = -scale;
	thrust::uniform_real_distribution<float> unitDistrib(-scale, scale);

	return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomArray(int seed, int N, glm::vec3* arr, float scale) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		arr[index] = generateRandomVec3(seed, index, scale);
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
	cudaMalloc((void**)&dev_pos1, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

	cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

	cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

	// LOOK-1.2 - This is a typical CUDA kernel invocation.
	kernGenerateRandomArray <<<fullBlocksPerGrid, blockSize>>> (/*time(NULL)*/1, numObjects,
		dev_pos1, scene_scale);
	checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");
	kernGenerateRandomArray <<<fullBlocksPerGrid, blockSize>>> (/*time(NULL)*/1, numObjects,
		dev_vel1, maxSpeed);
	checkCUDAErrorWithLine("kernGenerateRandomVelArray failed!");

	// LOOK-2.1 computing grid params
	halfGridCellWidth = std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
	gridCellWidth = 2.0f * halfGridCellWidth;
	halfGridSideCount = (int)(scene_scale / gridCellWidth) + 1;
	gridSideCount = 2 * halfGridSideCount;

	gridCellCount = gridSideCount * gridSideCount * gridSideCount;
	gridinverseCellWidth = 1.0f / gridCellWidth;
	float halfGridWidth = gridCellWidth * halfGridSideCount;
	gridMinimum.x -= halfGridWidth;
	gridMinimum.y -= halfGridWidth;
	gridMinimum.z -= halfGridWidth;

	// TODO-2.1 TODO-2.3 - Allocate additional buffers here.

	cudaMalloc((void**)&dev_boidArrayIndices, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_boidArrayIndices failed!");
	dev_thrust_boidArrayIndices = std::make_unique<thrust::device_ptr<int>>(dev_boidArrayIndices);

	cudaMalloc((void**)&dev_boidGridIndices, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_boidGridIndices failed!");
	dev_thrust_boidGridIndices = std::make_unique<thrust::device_ptr<int>>(dev_boidGridIndices);

	cudaMalloc((void**)&dev_gridCellStartIndices, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

	cudaMalloc((void**)&dev_gridCellEndIndices, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

	cudaMalloc((void**)&dev_pos2, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_pos2 failed!");

	cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3* pos, float* vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	float c_scale = -1.0f / s_scale;

	if (index < N) {
		vbo[4 * index + 0] = pos[index].x * c_scale;
		vbo[4 * index + 1] = pos[index].y * c_scale;
		vbo[4 * index + 2] = pos[index].z * c_scale;
		vbo[4 * index + 3] = 1.0f;
	}
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3* vel, float* vbo, float s_scale) {
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
void Boids::copyBoidsToVBO(float* vbodptr_positions, float* vbodptr_velocities) {
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

	kernCopyPositionsToVBO <<<fullBlocksPerGrid, blockSize>>> (numObjects, dev_pos1, vbodptr_positions, scene_scale);
	kernCopyVelocitiesToVBO <<<fullBlocksPerGrid, blockSize>>> (numObjects, dev_vel1, vbodptr_velocities, scene_scale);

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
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3* pos, const glm::vec3* vel) {
	// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
	// Rule 2: boids try to stay a distance d away from each other
	// Rule 3: boids try to match the speed of surrounding boids

	// https://vergenet.net/~conrad/boids/pseudocode.html
	float num1 = 0.f, /*num2 = 0.f, */num3 = 0.f;
	glm::vec3 thisPos = pos[iSelf];
	glm::vec3 thisVel = vel[iSelf];
	glm::vec3 rule1Center, rule2Vel, rule3Vel;
	for (int i = 0; i < N; i++) {
		if (i == iSelf) continue;
		glm::vec3 thatPos = pos[i];
		glm::vec3 thatVel = vel[i];
		float dist = glm::distance(thatPos, thisPos);
		if (dist < rule1Distance) {
			num1 += 1.f;
			rule1Center += thatPos;
		}
		if (dist < rule2Distance) {
			rule2Vel -= thatPos - thisPos;
		}
		if (dist < rule3Distance) {
			num3 += 1.f;
			rule3Vel += thatVel;
		}
	}
	glm::vec3 outVel;
	if (!glm::epsilonEqual(num1, 0.f, epsilon)) outVel += (rule1Center / num1 - thisPos) * rule1Scale;
	outVel += rule2Vel * rule2Scale;
	if (!glm::epsilonEqual(num3, 0.f, epsilon)) outVel += (rule3Vel / num3/* - thisVel*/) * rule3Scale;
	return outVel;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3* pos,
	glm::vec3* vel1, glm::vec3* vel2) {
	// Compute a new velocity based on pos and vel1
	// Clamp the speed
	// Record the new velocity into vel2. Question: why NOT vel1?

	int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	if (idx >= N) return;
	glm::vec3 outSpeed = vel1[idx] + computeVelocityChange(N, idx, pos, vel1);

	//if (glm::length(outSpeed) > maxSpeed)
	//	vel2[idx] = glm::normalize(outSpeed) * maxSpeed;
	//else vel2[idx] = outSpeed;
	vel2[idx] = glm::clamp(outSpeed, -maxSpeed, maxSpeed); // faster but not accurate
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3* pos, glm::vec3* vel) {
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
__device__ int gridIndex3Dto1D(int x, int y, int z, int sideCount) {
	return x + y * sideCount + z * sideCount * sideCount;
}
__device__ glm::ivec3 gridIndex1Dto3D(int id, int sideCount) {
	int x = id % sideCount;
	int y = ((id - x) / sideCount) % sideCount;
	int z = (id - x - y * sideCount) / (sideCount * sideCount);
	return glm::ivec3(x, y, z);
}

__global__ void kernComputeIndices(int N, int sideCount,
	glm::vec3 gridMin, float inverseCellWidth,
	glm::vec3* pos, int* boidArrayIndices, int* boidGridIndices) {
	// TODO-2.1
	// - Label each boid with the index of its grid cell.
	// - Set up a parallel array of integer indices as pointers to the actual
	//   boid data in pos and vel1/vel2

	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= N) return;
	boidArrayIndices[idx] = idx;
	glm::vec3 vec = pos[idx] - gridMin;
	int x = floor(vec.x * inverseCellWidth);
	int y = floor(vec.y * inverseCellWidth);
	int z = floor(vec.z * inverseCellWidth);
	boidGridIndices[idx] = gridIndex3Dto1D(x, y, z, sideCount);
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int* intBuffer, int value) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		intBuffer[index] = value;
	}
}

__global__ void kernIdentifyCellStartEnd(int N, int* boidGridIndices,
	int* gridCellStartIndices, int* gridCellEndIndices) {
	// TODO-2.1
	// Identify the start point of each cell in the gridIndices array.
	// This is basically a parallel unrolling of a loop that goes
	// "this index doesn't match the one before it, must be a new cell!"

	// must be called after the sort

	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= N) return;
	if (idx == 0) {
		gridCellStartIndices[boidGridIndices[0]] = 0;
	}
	else if (boidGridIndices[idx] != boidGridIndices[idx - 1])
		gridCellStartIndices[boidGridIndices[idx]] = idx;
	if (idx == N - 1) {
		gridCellEndIndices[boidGridIndices[N - 1]] = N - 1;
	}
	else if (boidGridIndices[idx] != boidGridIndices[idx + 1])
		gridCellEndIndices[boidGridIndices[idx]] = idx;
}

__device__ glm::vec3 gridComputeVelocityChangeScattered(
	int idx, int idxTrue, int* boidArrayIndices, int serchCell, int* gridCellStartIndices, int* gridCellEndIndices, const glm::vec3* pos, const glm::vec3* vel) {
	float num1 = 0.f, /*num2 = 0.f, */num3 = 0.f;
	glm::vec3 thisPos = pos[idxTrue];
	glm::vec3 thisVel = vel[idxTrue];
	glm::vec3 rule1Center, rule2Vel, rule3Vel;
	int startIdx = gridCellStartIndices[serchCell];
	int endIdx = gridCellEndIndices[serchCell];
	if (startIdx == -1 || endIdx == -1) return glm::vec3(); //cell doesn't have boids
	for (int ii = startIdx; ii <= endIdx; ii++) {
		if (ii == idx) continue;
		glm::vec3 thatPos = pos[boidArrayIndices[ii]];
		glm::vec3 thatVel = vel[boidArrayIndices[ii]];
		float dist = glm::distance(thatPos, thisPos);
		if (dist < rule1Distance) {
			num1 += 1.f;
			rule1Center += thatPos;
		}
		if (dist < rule2Distance) {
			rule2Vel -= thatPos - thisPos;
		}
		if (dist < rule3Distance) {
			num3 += 1.f;
			rule3Vel += thatVel;
		}
	}
	glm::vec3 outVel;
	if (!glm::epsilonEqual(num1, 0.f, epsilon)) outVel += (rule1Center / num1 - thisPos) * rule1Scale;
	outVel += rule2Vel * rule2Scale;
	if (!glm::epsilonEqual(num3, 0.f, epsilon)) outVel += (rule3Vel / num3/* - thisVel*/) * rule3Scale;
	return outVel;
}

__global__ void kernUpdateVelNeighborSearchScattered(
	int N, int sideCount, glm::vec3 gridMin,
	float inverseCellWidth, float cellWidth, float halfCellWidth,
	int* gridCellStartIndices, int* gridCellEndIndices,
	int* boidArrayIndices, int* boidGridIndices,
	glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2) {
	// TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
	// the number of boids that need to be checked.
	// - Identify the grid cell that this boid is in
	// - Identify which cells may contain neighbors. This isn't always 8.
	// - For each cell, read the start/end indices in the boid pointer array.
	// - Access each boid in the cell and compute velocity change from
	//   the boids rules, if this boid is within the neighborhood distance.
	// - Clamp the speed change before putting the new speed in vel2

	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= N) return;
	int idxTrue = boidArrayIndices[idx];
	glm::vec3 thisPos = pos[idxTrue];
	int cellIdx = boidGridIndices[idx];
	glm::ivec3 cellCoord = gridIndex1Dto3D(cellIdx, sideCount);
	glm::vec3 vec = thisPos - gridMin;
	float x = vec.x - cellWidth * cellCoord.x;
	float y = vec.y - cellWidth * cellCoord.y;
	float z = vec.z - cellWidth * cellCoord.z;

	int searchX = 0;
	int searchY = 0;
	int searchZ = 0;

	glm::vec3 outSpeed = vel1[idxTrue];
	// search itself
	outSpeed += gridComputeVelocityChangeScattered(idx, idxTrue, boidArrayIndices, cellIdx, gridCellStartIndices, gridCellEndIndices, pos, vel1);

	if (x <  halfCellWidth && cellCoord.x > 0)           searchX = -1;
	if (x >= halfCellWidth && cellCoord.x < sideCount-1) searchX = 1;
	if (y <  halfCellWidth && cellCoord.y > 0)           searchY = -1;
	if (y >= halfCellWidth && cellCoord.y < sideCount-1) searchY = 1;
	if (z <  halfCellWidth && cellCoord.z > 0)           searchZ = -1;
	if (z >= halfCellWidth && cellCoord.z < sideCount-1) searchZ = 1;

	float sideCount2 = sideCount*sideCount;
	if (searchX != 0)
		outSpeed += gridComputeVelocityChangeScattered(idx, idxTrue, boidArrayIndices, cellIdx + searchX, gridCellStartIndices, gridCellEndIndices, pos, vel1);
	else if (searchX != 0 && searchY != 0)
		outSpeed += gridComputeVelocityChangeScattered(idx, idxTrue, boidArrayIndices, cellIdx + searchX + searchY*sideCount, gridCellStartIndices, gridCellEndIndices, pos, vel1);
	else if (searchX != 0 && searchY != 0 && searchZ != 0)
		outSpeed += gridComputeVelocityChangeScattered(idx, idxTrue, boidArrayIndices, cellIdx + searchX + searchY*sideCount + searchZ * sideCount2, gridCellStartIndices, gridCellEndIndices, pos, vel1);
	else if (searchX != 0 && searchZ != 0)
		outSpeed += gridComputeVelocityChangeScattered(idx, idxTrue, boidArrayIndices, cellIdx + searchX + searchZ * sideCount2, gridCellStartIndices, gridCellEndIndices, pos, vel1);
	else if (searchY != 0)
		outSpeed += gridComputeVelocityChangeScattered(idx, idxTrue, boidArrayIndices, cellIdx + searchY * sideCount, gridCellStartIndices, gridCellEndIndices, pos, vel1);
	else if (searchZ != 0 && searchY != 0)
		outSpeed += gridComputeVelocityChangeScattered(idx, idxTrue, boidArrayIndices, cellIdx + searchY * sideCount + searchZ * sideCount2, gridCellStartIndices, gridCellEndIndices, pos, vel1);
	else if (searchZ != 0)
		outSpeed += gridComputeVelocityChangeScattered(idx, idxTrue, boidArrayIndices, cellIdx + searchZ * sideCount2, gridCellStartIndices, gridCellEndIndices, pos, vel1);

	vel2[idxTrue] = glm::clamp(outSpeed, -maxSpeed, maxSpeed);
}

__device__ glm::vec3 gridComputeVelocityChangeCoherent(
	int idx, int serchCell, int* gridCellStartIndices, int* gridCellEndIndices, const glm::vec3* pos, const glm::vec3* vel) {
	float num1 = 0.f, /*num2 = 0.f, */num3 = 0.f;
	glm::vec3 thisPos = pos[idx];
	glm::vec3 thisVel = vel[idx];
	glm::vec3 rule1Center, rule2Vel, rule3Vel;
	int startIdx = gridCellStartIndices[serchCell];
	int endIdx = gridCellEndIndices[serchCell];
	if (startIdx == -1 || endIdx == -1) return glm::vec3(); //cell doesn't have boids
	for (int ii = startIdx; ii <= endIdx; ii++) {
		if (ii == idx) continue;
		glm::vec3 thatPos = pos[ii];
		glm::vec3 thatVel = vel[ii];
		float dist = glm::distance(thatPos, thisPos);
		if (dist < rule1Distance) {
			num1 += 1.f;
			rule1Center += thatPos;
		}
		if (dist < rule2Distance) {
			rule2Vel -= thatPos - thisPos;
		}
		if (dist < rule3Distance) {
			num3 += 1.f;
			rule3Vel += thatVel;
		}
	}
	glm::vec3 outVel;
	if (!glm::epsilonEqual(num1, 0.f, epsilon)) outVel += (rule1Center / num1 - thisPos) * rule1Scale;
	outVel += rule2Vel * rule2Scale;
	if (!glm::epsilonEqual(num3, 0.f, epsilon)) outVel += (rule3Vel / num3/* - thisVel*/) * rule3Scale;
	return outVel;
}


__global__ void kernUpdateVelNeighborSearchCoherent(
	int N, int sideCount, glm::vec3 gridMin,
	float inverseCellWidth, float cellWidth, float halfCellWidth,
	int* gridCellStartIndices, int* gridCellEndIndices, int* boidGridIndices,
	glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2) {
	// TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
	// except with one less level of indirection.
	// This should expect gridCellStartIndices and gridCellEndIndices to refer
	// directly to pos and vel1.
	// - Identify the grid cell that this boid is in
	// - Identify which cells may contain neighbors. This isn't always 8.
	// - For each cell, read the start/end indices in the boid pointer array.
	//   DIFFERENCE: For best results, consider what order the cells should be
	//   checked in to maximize the memory benefits of reordering the boids data.
	// - Access each boid in the cell and compute velocity change from
	//   the boids rules, if this boid is within the neighborhood distance.
	// - Clamp the speed change before putting the new speed in vel2

	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= N) return;
	glm::vec3 thisPos = pos[idx];
	int cellIdx = boidGridIndices[idx];
	glm::ivec3 cellCoord = gridIndex1Dto3D(cellIdx, sideCount);
	glm::vec3 vec = thisPos - gridMin;
	float x = vec.x - cellWidth * cellCoord.x;
	float y = vec.y - cellWidth * cellCoord.y;
	float z = vec.z - cellWidth * cellCoord.z;

	int searchX = 0;
	int searchY = 0;
	int searchZ = 0;

	glm::vec3 outSpeed = vel1[idx];
	
	outSpeed += gridComputeVelocityChangeCoherent(idx, cellIdx, gridCellStartIndices, gridCellEndIndices, pos, vel1);
	//x = glm::value_ptr(outSpeed)[0];
	//y = glm::value_ptr(outSpeed)[1];
	//z = glm::value_ptr(outSpeed)[2];
	//float x0 = glm::value_ptr(vel1[idx])[0];
	//float y0 = glm::value_ptr(vel1[idx])[1];
	//float z0 = glm::value_ptr(vel1[idx])[2];

	if (x < halfCellWidth && cellCoord.x > 0)              searchX = -1;
	if (x >= halfCellWidth && cellCoord.x < sideCount - 1) searchX = 1;
	if (y < halfCellWidth && cellCoord.y > 0)              searchY = -1;
	if (y >= halfCellWidth && cellCoord.y < sideCount - 1) searchY = 1;
	if (z < halfCellWidth && cellCoord.z > 0)              searchZ = -1;
	if (z >= halfCellWidth && cellCoord.z < sideCount - 1) searchZ = 1;

	float sideCount2 = sideCount * sideCount;
	if (searchX != 0)
		outSpeed += gridComputeVelocityChangeCoherent(idx, cellIdx + searchX, gridCellStartIndices, gridCellEndIndices, pos, vel1);
	else if (searchX != 0 && searchY != 0)
		outSpeed += gridComputeVelocityChangeCoherent(idx, cellIdx + searchX + searchY * sideCount, gridCellStartIndices, gridCellEndIndices, pos, vel1);
	else if (searchX != 0 && searchY != 0 && searchZ != 0)
		outSpeed += gridComputeVelocityChangeCoherent(idx, cellIdx + searchX + searchY * sideCount + searchZ * sideCount2, gridCellStartIndices, gridCellEndIndices, pos, vel1);
	else if (searchX != 0 && searchZ != 0)
		outSpeed += gridComputeVelocityChangeCoherent(idx, cellIdx + searchX + searchZ * sideCount2, gridCellStartIndices, gridCellEndIndices, pos, vel1);
	else if (searchY != 0)
		outSpeed += gridComputeVelocityChangeCoherent(idx, cellIdx + searchY * sideCount, gridCellStartIndices, gridCellEndIndices, pos, vel1);
	else if (searchZ != 0 && searchY != 0)
		outSpeed += gridComputeVelocityChangeCoherent(idx, cellIdx + searchY * sideCount + searchZ * sideCount2, gridCellStartIndices, gridCellEndIndices, pos, vel1);
	else if (searchZ != 0)
		outSpeed += gridComputeVelocityChangeCoherent(idx, cellIdx + searchZ * sideCount2, gridCellStartIndices, gridCellEndIndices, pos, vel1);

	vel2[idx] = glm::clamp(outSpeed, -maxSpeed, maxSpeed);
}

__global__ void kernRearrangePosVel(
	int N, int* boidArrayIndices, glm::vec3 *pos1, glm::vec3* pos2, glm::vec3* vel1, glm::vec3* vel2) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= N) return;
	int idxTrue = boidArrayIndices[idx];
	pos2[idx] = pos1[idxTrue];
	vel2[idx] = vel1[idxTrue];
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
	// TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
	// TODO-1.2 ping-pong the velocity buffers
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
	kernUpdateVelocityBruteForce <<<fullBlocksPerGrid, blockSize>>> (numObjects, dev_pos1, dev_vel1, dev_vel2);
	kernUpdatePos <<<fullBlocksPerGrid, blockSize>>> (numObjects, dt, dev_pos1, dev_vel2);
	std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationScatteredGrid(float dt) {
	// TODO-2.1
	// Uniform Grid Neighbor search using Thrust sort.
	// In Parallel:
	// - label each boid with its array index as well as its grid index.
	//   Use 2x width grids.
	// - Unstable key sort using Thrust. A stable sort isn't necessary, but you
	//   are welcome to do a performance comparison.
	// - Naively unroll the loop for finding the start and end indices of each
	//   cell's data pointers in the array of boid indices
	// - Perform velocity updates using neighbor search
	// - Update positions
	// - Ping-pong buffers as needed

	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
	kernComputeIndices <<<fullBlocksPerGrid, blockSize >>> (
		numObjects, 
		gridSideCount, 
		gridMinimum, 
		gridinverseCellWidth, 
		dev_pos1, 
		dev_boidArrayIndices, 
		dev_boidGridIndices);

	//int    keys[6] = { 1,   4,   2,   8,   5,   7 };
	//char values[6] = { 'a', 'b', 'c', 'd', 'e', 'f' };
	//thrust::sort_by_key(thrust::host, keys, keys + 6, values);
	// keys is now   {  1,   2,   4,   5,   7,   8}
	// values is now {'a', 'c', 'b', 'e', 'f', 'd'}
	thrust::sort_by_key(*dev_thrust_boidGridIndices, *dev_thrust_boidGridIndices + numObjects, *dev_thrust_boidArrayIndices);
	kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_gridCellStartIndices, -1);
	kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_gridCellEndIndices, -1);
	kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_boidGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

	kernUpdateVelNeighborSearchScattered <<<fullBlocksPerGrid, blockSize>>> (
		numObjects, gridSideCount, gridMinimum, gridinverseCellWidth, gridCellWidth, halfGridCellWidth, dev_gridCellStartIndices,
		dev_gridCellEndIndices, dev_boidArrayIndices, dev_boidGridIndices, dev_pos1, dev_vel1, dev_vel2);
	kernUpdatePos <<<fullBlocksPerGrid, blockSize >>> (numObjects, dt, dev_pos1, dev_vel2);
	std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationCoherentGrid(float dt) {
	// TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
	// Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
	// In Parallel:
	// - Label each boid with its array index as well as its grid index.
	//   Use 2x width grids
	// - Unstable key sort using Thrust. A stable sort isn't necessary, but you
	//   are welcome to do a performance comparison.
	// - Naively unroll the loop for finding the start and end indices of each
	//   cell's data pointers in the array of boid indices
	// - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
	//   the boid data in the simulation array.
	//   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
	// - Perform velocity updates using neighbor search
	// - Update positions
	// - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.

		dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
	kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (
		numObjects,
		gridSideCount,
		gridMinimum,
		gridinverseCellWidth,
		dev_pos1,
		dev_boidArrayIndices,
		dev_boidGridIndices);

	thrust::sort_by_key(*dev_thrust_boidGridIndices, *dev_thrust_boidGridIndices + numObjects, *dev_thrust_boidArrayIndices);
	kernRearrangePosVel << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_boidArrayIndices, dev_pos1, dev_pos2, dev_vel1, dev_vel2);
	std::swap(dev_vel1, dev_vel2);
	std::swap(dev_pos1, dev_pos2);

	kernResetIntBuffer <<<fullBlocksPerGrid, blockSize>>> (numObjects, dev_gridCellStartIndices, -1);
	kernResetIntBuffer <<<fullBlocksPerGrid, blockSize>>> (numObjects, dev_gridCellEndIndices, -1);
	kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_boidGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

	kernUpdateVelNeighborSearchCoherent <<<fullBlocksPerGrid, blockSize >>> (
		numObjects, gridSideCount, gridMinimum, gridinverseCellWidth, gridCellWidth, halfGridCellWidth, dev_gridCellStartIndices,
		dev_gridCellEndIndices, dev_boidGridIndices, dev_pos1, dev_vel1, dev_vel2);
	kernUpdatePos <<<fullBlocksPerGrid, blockSize>>> (numObjects, dt, dev_pos1, dev_vel2);
	std::swap(dev_vel1, dev_vel2);
}

void Boids::endSimulation() {
	cudaFree(dev_vel1);
	cudaFree(dev_vel2);
	cudaFree(dev_pos1);

	// TODO-2.1 TODO-2.3 - Free any additional buffers here.

	cudaFree(dev_boidArrayIndices);
	cudaFree(dev_boidGridIndices);
	cudaFree(dev_gridCellStartIndices);
	cudaFree(dev_gridCellEndIndices);
	cudaFree(dev_pos2);
}

void Boids::unitTest() {
	// LOOK-1.2 Feel free to write additional tests here.

	// test unstable sort
	int* dev_intKeys;
	int* dev_intValues;
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
