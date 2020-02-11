#include "framework.h"

__global__ void test_generate(char *pArray, size_t pitch, size_t width, size_t height, size_t count)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	int *element = (int*)(pArray + tidy * pitch + tidx * sizeof(int));
	atomicAdd(element, (int)count & tidx & tidy);
}

cudaError_t cuda_test_generate(PVOID pArray, size_t pitch, size_t width, size_t height, size_t count)
{
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

	test_generate << <Dg, Db >> > ((char*)pArray, pitch, width, height, 0xffffff);
	error = cudaGetLastError();
	if (error != cudaSuccess) return error;

	cudaDeviceSynchronize();
	error = cudaGetLastError();
	return error;
}
