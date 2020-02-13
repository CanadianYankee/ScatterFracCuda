#include "framework.h"
#include "AccumData.h"

__global__ void test_iterate(GPU_ARRAY_2D arrAccum, PVOID pStats, ACCUM_PARAMS params)
{
	UINT tidx = blockIdx.x * blockDim.x + threadIdx.x;
	UINT tidy = blockIdx.y * blockDim.y + threadIdx.y;
	if (tidx >= arrAccum.nWidth || tidy >= arrAccum.nHeight) return;
	unsigned char* pArray = (unsigned char *)(arrAccum.pArray);
	
	COUNT_COLOR *element = (COUNT_COLOR*)(pArray + tidy * arrAccum.nPitch + tidx * sizeof(COUNT_COLOR));
	element->nCount = tidx + tidy + 2;
	element->r = (tidx & 256) / 256.0;
	element->g = (tidy & 256) / 256.0;
	element->b = (((tidx >> 2)& (tidy >> 2)) & 256) / 256.0;
	atomicMax(&((ACCUM_STATS *)pStats)->nMaxCount, element->nCount);
}

__global__ void render_texture(GPU_ARRAY_2D texture, GPU_ARRAY_2D arrAccum, const RENDER_PARAMS params)
{
	UINT texx = blockIdx.x * blockDim.x + threadIdx.x;
	UINT texy = blockIdx.y * blockDim.y + threadIdx.y;
	if (texx >= texture.nWidth || texy >= texture.nHeight) return;

	float *pixel = (float*)((unsigned char *)(texture.pArray) + texy * texture.nPitch) + 4 * texx;

	int iAntiAlias = max(1, params.iAntiAlias);
	UINT arrx = texx * iAntiAlias;
	UINT arry = texy * iAntiAlias;

	float r = 0.0f, g = 0.0f, b = 0.0f, a = 0.0f;
	for (UINT j = 0; j < iAntiAlias; j++)
	{
		COUNT_COLOR* pRow = (COUNT_COLOR*)((unsigned char *)arrAccum.pArray + (arry + j) * arrAccum.nPitch);
		for (UINT i = 0; i < iAntiAlias; i++)
		{
			COUNT_COLOR* pItem = &pRow[arrx + i];
			a += (float)(pItem->nCount) * params.fCountScale;
			r += pItem->r;
			g += pItem->g;
			b += pItem->b;
		}
	}
	a /= (float)(iAntiAlias * iAntiAlias);
	pixel[0] = r / (float)(iAntiAlias * iAntiAlias) * a;
	pixel[1] = g / (float)(iAntiAlias * iAntiAlias) * a;
	pixel[2] = b / (float)(iAntiAlias * iAntiAlias) * a;
	pixel[3] = 1.0f;
}

cudaError_t cuda_test_generate(GPU_ARRAY_2D& arrAccum, PVOID pStats, const ACCUM_PARAMS& params)
{
	cudaError_t error = cudaSuccess;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int accumMax = ((ACCUM_STATS*)pStats)->nMaxCount;

	dim3 Db = dim3(16, 16);   
	dim3 Dg = dim3(((UINT)arrAccum.nWidth + Db.x - 1) / Db.x, ((UINT)arrAccum.nHeight + Db.y - 1) / Db.y);

	cudaEventRecord(start);
	test_iterate << <Dg, Db >> > (arrAccum, pStats, params);
	cudaEventRecord(stop);
	error = cudaGetLastError();
	if (error != cudaSuccess) return error;

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	accumMax = ((ACCUM_STATS*)pStats)->nMaxCount;
	error = cudaGetLastError();

	return error;
}

cudaError_t cuda_render_texture(GPU_ARRAY_2D& texture, GPU_ARRAY_2D& arrAccum, const RENDER_PARAMS& params)
{
	cudaError_t error = cudaSuccess;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 Db = dim3(16, 16);   
	dim3 Dg = dim3(((UINT)texture.nWidth + Db.x - 1) / Db.x, ((UINT)texture.nHeight + Db.y - 1) / Db.y);

	cudaEventRecord(start);
	render_texture <<<Dg, Db>>> (texture, arrAccum, params);
	cudaEventRecord(stop);
	error = cudaGetLastError();
	if (error != cudaSuccess) return error;

	cudaEventSynchronize(stop);
	error = cudaGetLastError();
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	return error;
}
