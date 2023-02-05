//#include <cutil_inline.h>
//#include "cutil_math.h"
#include <helper_math.h>
#include <helper_cuda.h>
#include "math_constants.h"

#include <assert.h>

float4 *pCudaPos0 = NULL; float4 *pCudaPos1 = NULL;
float4 *pCudaPosOld0 = NULL; float4 *pCudaPosOld1 = NULL;   

float4 * pPosIn, *pPosOut;
float4 * pPosOldIn, *pPosOldOut;

int iters = 0;
								
extern __global__ void verlet(	float4 * pos_vbo, float4 * nor_vbo, float4 * g_pos_in, float4 * g_pos_old_in, float4 * g_pos_out, float4 * g_pos_old_out, 
								int side, float stiffness, float damp, float inverse_mass, int coll_primitives );

// size is the total number of float (= number of particles * 4)
void InitCuda(const int size)
{  
	const unsigned int num_threads = size / 4;
	//cutilCondition(0 == (num_threads % 4));	
	const unsigned int mem_size = sizeof(float4) * num_threads;

	// allocate device memory for float4 version
	checkCudaErrors(cudaMalloc((void**) &pCudaPos0, mem_size));	// positions
	checkCudaErrors(cudaMalloc((void**) &pCudaPos1, mem_size));	// positions
	checkCudaErrors(cudaMalloc((void**) &pCudaPosOld0, mem_size));	// old positions
	checkCudaErrors(cudaMalloc((void**) &pCudaPosOld1, mem_size));	// old positions

	iters = 0;
}

//Round a / b to nearest higher integer value
uint iDivUp(uint a, uint b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
    numThreads = min(blockSize, n); 
    numBlocks = iDivUp(n, numThreads);
}    
     
void UploadCuda(float * positions, float * positions_old, const int size)
{
	assert(pCudaPos0 != NULL); 
	assert(pCudaPosOld0 != NULL); 

	const unsigned int num_threads = size / 4;
	//cutilCondition(0 == (num_threads % 4));
	const unsigned int mem_size = sizeof(float4) * num_threads;

	// copy host memory to device
	// NOTE: it is not necessary to copy in each iteration, just swap the buffers -> huge save of computation time

	if ((iters % 2) == 0)   
	{     
		pPosIn = pCudaPos0;			pPosOut = pCudaPos1;
		pPosOldIn = pCudaPosOld0;	pPosOldOut = pCudaPosOld1;
	}
	else
	{
		pPosIn = pCudaPos1;			pPosOut = pCudaPos0;
		pPosOldIn = pCudaPosOld1;	pPosOldOut = pCudaPosOld0; 
	}

	if (iters == 0)
	{
		checkCudaErrors(cudaMemcpy(pPosIn, positions,  mem_size, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(pPosOldIn, positions_old, mem_size, cudaMemcpyHostToDevice)); 
	} 

	iters++;
}

void VerletCuda(float4 * pos_vbo, float4* nor_vbo, float * positions, float * positions_old, const int size, const int & side, const float & stiffness, 
					const float & damp, const float & inverse_mass, const int & coll_primitives)
{   
	// setup execution parameters 
	uint numThreads, numBlocks;
	uint numParticles = size / 4;

	computeGridSize(numParticles, 256, numBlocks, numThreads);

	// execute the kernel
	//	printf("numParticles: %d,   numThreads: %d   numBlocks: %d\n", numParticles, numThreads, numBlocks);
	verlet<<< numBlocks, numThreads >>>(pos_vbo, nor_vbo, pPosIn, pPosOldIn, pPosOut, pPosOldOut, side, stiffness, damp, inverse_mass, coll_primitives);

	// stop the CPU until the kernel has been executed
	cudaThreadSynchronize();

	// check if kernel execution generated and error
	checkCudaErrors(cudaGetLastError());
}

void DownloadCuda(float * positions, float * positions_old, const int size)
{ 
/*
/// DEPRECATED CODE
  
	const unsigned int num_threads = size / 4;
	cutilCondition(0 == (num_threads % 4));
	const unsigned int mem_size = sizeof(float4) * num_threads;

	// copy results from device to host
	cutilSafeCall(cudaMemcpy(positions,		pPosOut,		mem_size, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(positions_old, pPosOldOut,		mem_size, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(normals,		pCudaNor,		mem_size, cudaMemcpyDeviceToHost));
	cutilCheckMsg("Cuda memory copy device to host failed.");
*/
}

void ResetCuda()
{
	// cleanup memory
	if (pCudaPos0 != NULL) 
	{
		checkCudaErrors(cudaFree(pCudaPos0));
		checkCudaErrors(cudaFree(pCudaPos1));
		pCudaPos0 = NULL;
		pCudaPos1 = NULL;
	}

	if (pCudaPosOld0 != NULL)
	{
		checkCudaErrors(cudaFree(pCudaPosOld0));
		checkCudaErrors(cudaFree(pCudaPosOld1));
		pCudaPosOld0 = NULL;
		pCudaPosOld1 = NULL;
	}
}



