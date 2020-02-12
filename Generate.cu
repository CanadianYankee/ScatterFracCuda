#include "framework.h"
#include "CudaUtil.h"

__global__ void test_iterate(unsigned char *pArray, UINT *pMax, SIZE_2D arr_size, size_t count)
{
	UINT tidx = blockIdx.x * blockDim.x + threadIdx.x;
	UINT tidy = blockIdx.y * blockDim.y + threadIdx.y;
	if (tidx >= arr_size.nWidth || tidy >= arr_size.nHeight) return;
	
	UINT *element = (UINT*)(pArray + tidy * arr_size.nPitch + tidx * sizeof(UINT));
	UINT val = (UINT)count & tidx & tidy;
	UINT old = atomicAdd(element, val);
	atomicMax(pMax, old + val);
}

__global__ void render_texture(unsigned char* pTex, SIZE_2D tex_size, unsigned char* pArray, SIZE_2D arr_size, float* pScale, UINT iAntiAlias)
{
	UINT texx = blockIdx.x * blockDim.x + threadIdx.x;
	UINT texy = blockIdx.y * blockDim.y + threadIdx.y;
	if (texx >= tex_size.nWidth || texy >= tex_size.nHeight) return;

	float *pixel = (float*)(pTex + texy * tex_size.nPitch) + 4 * texx;

	UINT arrx = texx * iAntiAlias;
	UINT arry = texy * iAntiAlias;

	float color = 0.0f;
	for (UINT j = 0; j < iAntiAlias; j++)
	{
		UINT* pRow = (UINT*)(pArray + (arry + j) * arr_size.nPitch);
		for (UINT i = 0; i < iAntiAlias; i++)
		{
			color += (float)(pRow[arrx + i]) * (*pScale);
		}
	}
	color /= (float)(iAntiAlias * iAntiAlias);
	pixel[0] = color;
	pixel[1] = color;
	pixel[2] = color;
	pixel[3] = 1.0f;
}

cudaError_t cuda_test_generate(PVOID pArray, PVOID pMax, SIZE_2D size, size_t count)
{
	cudaError_t error = cudaSuccess;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int accumMax = *(int*)pMax;

	dim3 Db = dim3(16, 16);   
	dim3 Dg = dim3(((UINT)size.nWidth + Db.x - 1) / Db.x, ((UINT)size.nHeight + Db.y - 1) / Db.y);

	cudaEventRecord(start);
	test_iterate << <Dg, Db >> > ((unsigned char*)pArray, (UINT*)pMax, size, count);
	cudaEventRecord(stop);
	error = cudaGetLastError();
	if (error != cudaSuccess) return error;

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	accumMax = *(int*)pMax;
	error = cudaGetLastError();

	return error;
}

cudaError_t cuda_render_texture(PVOID pTexture, SIZE_2D size_tex, PVOID pArray, SIZE_2D size_arr, PVOID pScale, UINT iAntiAlias)
{
	cudaError_t error = cudaSuccess;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 Db = dim3(16, 16);   
	dim3 Dg = dim3(((UINT)size_tex.nWidth + Db.x - 1) / Db.x, ((UINT)size_tex.nHeight + Db.y - 1) / Db.y);

	cudaEventRecord(start);
	render_texture <<<Dg, Db>>> ((unsigned char*)pTexture, size_tex, (unsigned char*)pArray, size_arr, (float*)pScale, iAntiAlias);
	cudaEventRecord(stop);
	error = cudaGetLastError();
	if (error != cudaSuccess) return error;

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	error = cudaGetLastError();

	return error;
}
