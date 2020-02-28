#include "framework.h"
#include "AccumData.h"

__device__ void transform(ITERATOR* iter);

__device__ __forceinline__ float atomicMinFloat(float* addr, float value) {
	float old;
	old = (value >= 0) ? __int_as_float(atomicMin((int*)addr, __float_as_int(value))) :
		__uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));

	return old;
}

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
	float old;
	old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
		__uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

	return old;
}

__global__ void iterate(ACCUM_PARAMS params, GPU_ARRAY_2D arrIter, GPU_ARRAY_2D arrAccum, PVOID pStats)
{
	UINT idx = blockIdx.x * blockDim.x + threadIdx.x;
	ITERATOR* iter = &(((ITERATOR*)(arrIter.pArray))[idx]);
	ACCUM_STATS* accumStats = (ACCUM_STATS*)pStats;

	if (accumStats->bAbort) return;
	
	// On intialize, seed the random number generators
	if (params.bInit)
	{
		iter->rand.init(idx);
	}

	for (UINT i = 0; i < params.nSteps; i++)
	{
		transform(iter);

		if (!isfinite(iter->x) || !isfinite(iter->y))
		{
			accumStats->bAbort = TRUE;
			break;
		}

		if (!params.bInit)
		{
			int i = int(iter->x * params.rect.fScale + params.rect.fOffsetX);
			int j = int(iter->y * params.rect.fScale + params.rect.fOffsetY);
			if (i >= 0 && i < (int)arrAccum.nWidth && j >= 0 && j < (int)arrAccum.nHeight)
			{
				FLOAT_COLOR* element = (FLOAT_COLOR*)((unsigned char *)(arrAccum.pArray) + j * arrAccum.nPitch + i * sizeof(FLOAT_COLOR));
				if (params.bHitPercent)
				{
					atomicAdd(&(accumStats->nHitRect), 1);
					if (element->IsZero())
						atomicAdd(&(accumStats->nNewHits), 1);
				}
				atomicAdd(&(element->r), iter->clr.r);
				atomicAdd(&(element->g), iter->clr.g);
				atomicAdd(&(element->b), iter->clr.b);
				atomicMax(&(accumStats->nMaxColorElement), (UINT)element->Max());
			}
		}
	}

	// On initialize, adjust the bounding box based on the zeroth block
	if (params.bInit && (blockIdx.x == 0))
	{
		atomicMaxFloat(&(accumStats->xMax), iter->x);
		atomicMaxFloat(&(accumStats->yMax), iter->y);
		atomicMinFloat(&(accumStats->xMin), iter->x);
		atomicMinFloat(&(accumStats->yMin), iter->y);
	}
}

__device__ void transform(ITERATOR* iter)
{
	float rnd = iter->rand.frand();
	FLOAT_COLOR clr;
	if (rnd < 0.33333f)
	{
		iter->x = iter->x * 0.5f;
		iter->y = iter->y * 0.5f + 0.5f;
		clr.r = 1.0f;
		clr.g = 1.0f;
		clr.b = 0.0f;
	}
	else if (rnd < 0.66666f)
	{
		iter->x = iter->x * 0.5f + 0.433f;
		iter->y = iter->y * 0.5f - 0.25f;
		clr.r = 1.0f;
		clr.g = 0.0f;
		clr.b = 1.0f;
	}
	else
	{
		iter->x = iter->x * 0.5f - 0.433f;
		iter->y = iter->y * 0.5f - 0.25f;
		clr.r = 0.0f;
		clr.g = 1.0f;
		clr.b = 1.0f;
	}
	iter->clr.Tint(clr, 3.0f);
}

__global__ void render_texture(const RENDER_PARAMS params, GPU_ARRAY_2D texture, GPU_ARRAY_2D arrAccum)
{
	UINT texx = blockIdx.x * blockDim.x + threadIdx.x;
	UINT texy = blockIdx.y * blockDim.y + threadIdx.y;
	if (texx >= texture.nWidth || texy >= texture.nHeight) return;

	float *pixel = (float*)((unsigned char *)(texture.pArray) + texy * texture.nPitch) + 4 * texx;

	int iAntiAlias = max(1, params.iAntiAlias);
	UINT arrx = texx * iAntiAlias;
	UINT arry = texy * iAntiAlias;

	float r = 0.0f, g = 0.0f, b = 0.0f;
	for (UINT j = 0; j < iAntiAlias; j++)
	{
		FLOAT_COLOR* pRow = (FLOAT_COLOR*)((unsigned char *)arrAccum.pArray + (arry + j) * arrAccum.nPitch);
		for (UINT i = 0; i < iAntiAlias; i++)
		{
			FLOAT_COLOR* pItem = &pRow[arrx + i];
			if(pItem->r) r += logf(pItem->r) * params.fLogColorScale;
			if(pItem->b) b += logf(pItem->b) * params.fLogColorScale;
			if(pItem->g) g += logf(pItem->g) * params.fLogColorScale;
		}
	}
	pixel[0] = r / (float)(iAntiAlias * iAntiAlias);
	pixel[1] = g / (float)(iAntiAlias * iAntiAlias);
	pixel[2] = b / (float)(iAntiAlias * iAntiAlias);
	pixel[3] = 1.0f;
}

cudaError_t cuda_iterate(const ACCUM_PARAMS& params, GPU_ARRAY_2D& arrIter, GPU_ARRAY_2D& arrAccum, PVOID pStats)
{
	cudaError_t error = cudaSuccess;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	iterate << < arrIter.nWidth, arrIter.nHeight >> > (params, arrIter, arrAccum, pStats);
	cudaEventRecord(stop);
	error = cudaGetLastError();
	if (error != cudaSuccess) return error;

	cudaEventSynchronize(stop);
	error = cudaGetLastError();
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	return error;
}

cudaError_t cuda_render_texture(const RENDER_PARAMS& params, GPU_ARRAY_2D& texture, GPU_ARRAY_2D& arrAccum)
{
	cudaError_t error = cudaSuccess;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 Db = dim3(16, 16);   
	dim3 Dg = dim3(((UINT)texture.nWidth + Db.x - 1) / Db.x, ((UINT)texture.nHeight + Db.y - 1) / Db.y);

	cudaEventRecord(start);
	render_texture <<<Dg, Db>>> (params, texture, arrAccum);
	cudaEventRecord(stop);
	error = cudaGetLastError();
	if (error != cudaSuccess) return error;

	cudaEventSynchronize(stop);
	error = cudaGetLastError();
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	return error;
}
