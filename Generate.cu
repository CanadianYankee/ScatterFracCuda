#include "framework.h"
#include "AccumData.h"

__global__ void test_iterate(unsigned char *pArray, unsigned char *pStats, SIZE_2D arr_size, size_t count)
{
	UINT tidx = blockIdx.x * blockDim.x + threadIdx.x;
	UINT tidy = blockIdx.y * blockDim.y + threadIdx.y;
	if (tidx >= arr_size.nWidth || tidy >= arr_size.nHeight) return;
	
	COUNT_COLOR *element = (COUNT_COLOR*)(pArray + tidy * arr_size.nPitch + tidx * sizeof(COUNT_COLOR));
	element->nCount = tidx + tidy;
	element->r = (tidx & 256) / 256.0; 
	element->g = (tidy & 256) / 256.0;
	element->b = (((tidx >> 2) & (tidy >> 2)) & 256) / 256.0;
	atomicMax(&((ACCUM_STATS *)pStats)->nMaxCount, element->nCount);
}

__global__ void render_texture(unsigned char* pTex, SIZE_2D tex_size, unsigned char* pArray, SIZE_2D arr_size, float* pScale, UINT iAntiAlias)
{
	UINT texx = blockIdx.x * blockDim.x + threadIdx.x;
	UINT texy = blockIdx.y * blockDim.y + threadIdx.y;
	if (texx >= tex_size.nWidth || texy >= tex_size.nHeight) return;

	float *pixel = (float*)(pTex + texy * tex_size.nPitch) + 4 * texx;

	UINT arrx = texx * iAntiAlias;
	UINT arry = texy * iAntiAlias;

	float r = 0.0f, g = 0.0f, b = 0.0f, a = 0.0f;
	for (UINT j = 0; j < iAntiAlias; j++)
	{
		COUNT_COLOR* pRow = (COUNT_COLOR*)(pArray + (arry + j) * arr_size.nPitch);
		for (UINT i = 0; i < iAntiAlias; i++)
		{
			a += (float)(pRow[arrx + i].nCount) * (*pScale);
			r += pRow[arrx + i].r;
			g += pRow[arrx + i].g;
			b += pRow[arrx + i].b;
		}
	}
	a /= (float)(iAntiAlias * iAntiAlias);
	pixel[0] = r / (float)(iAntiAlias * iAntiAlias) * a;
	pixel[1] = g / (float)(iAntiAlias * iAntiAlias) * a;
	pixel[2] = b / (float)(iAntiAlias * iAntiAlias) * a;
	pixel[3] = 1.0f;
}

cudaError_t cuda_test_generate(PVOID pArray, PVOID pStats, SIZE_2D size, size_t count)
{
	cudaError_t error = cudaSuccess;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int accumMax = ((ACCUM_STATS*)pStats)->nMaxCount;

	dim3 Db = dim3(16, 16);   
	dim3 Dg = dim3(((UINT)size.nWidth + Db.x - 1) / Db.x, ((UINT)size.nHeight + Db.y - 1) / Db.y);

	cudaEventRecord(start);
	test_iterate << <Dg, Db >> > ((unsigned char*)pArray, (unsigned char*)pStats, size, count);
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
