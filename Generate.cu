#include "framework.h"
#include "AccumData.h"
#include "CudaArray.h"

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

__global__ void iterate(ACCUM_PARAMS params, CCudaArray1D<ITERATOR> arrIter, CCudaArray2D<ACCUM> arrAccum, PVOID pStats)
{
	UINT idx = blockIdx.x * blockDim.x + threadIdx.x;
	ITERATOR* iter = arrIter.GetAt(idx);
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

		if (!isfinite(iter->pos[0]) || !isfinite(iter->pos[1]))
		{
			accumStats->bAbort = TRUE;
			break;
		}

		if (!params.bInit)
		{
			int i = int(iter->pos[0] * params.rect.fScale + params.rect.fOffsetX);
			int j = int(iter->pos[1] * params.rect.fScale + params.rect.fOffsetY);
			if (arrAccum.ValidIndex(i, j))
			{
				ACCUM* element = arrAccum.GetAt(i, j);
				UINT nCount = atomicAdd(&(element->nCount), 1);
				atomicMax(&(accumStats->nMaxCount), nCount);
				if (params.bHitPercent)
				{
					atomicAdd(&(accumStats->nHitRect), 1);
					if (nCount == 0)
						atomicAdd(&(accumStats->nNewHits), 1);
				}
				atomicAdd(&(element->clrAccum.r), iter->clr.r);
				atomicAdd(&(element->clrAccum.g), iter->clr.g);
				atomicAdd(&(element->clrAccum.b), iter->clr.b);
				atomicMaxFloat(&(accumStats->fMaxColorElement), element->clrAccum.Max());
			}
		}
	}

	// On initialize, adjust the bounding box based on the zeroth block
	if (params.bInit && (blockIdx.x == 0))
	{
		atomicMaxFloat(&(accumStats->xMax), iter->pos[0]);
		atomicMaxFloat(&(accumStats->yMax), iter->pos[1]);
		atomicMinFloat(&(accumStats->xMin), iter->pos[0]);
		atomicMinFloat(&(accumStats->yMin), iter->pos[1]);
	}
}

__device__ void transform(ITERATOR* iter)
{
	float rnd = iter->rand.frand();
	FLOAT_COLOR clr;
	if (rnd < 0.5f)
	{
		iter->pos *= 0.5f;
		iter->pos += CVector2D(0.0f, 0.5f);
		clr.r = 1.0f;
		clr.g = 1.0f;
		clr.b = 0.0f;
	}
	else if (rnd < 0.9f)
	{
		iter->pos *= 0.5f;
		iter->pos += CVector2D(0.433f, -0.25f);
		clr.r = 1.0f;
		clr.g = 0.0f;
		clr.b = 1.0f;
	}
	else
	{
		iter->pos *= 0.5f;
		iter->pos += CVector2D(-0.433f, -0.25f);
		clr.r = 0.0f;
		clr.g = 1.0f;
		clr.b = 1.0f;
	}
	iter->clr.Tint(clr, 3.0f);
}

cudaError_t cuda_iterate(const ACCUM_PARAMS& params, CCudaArray1D<ITERATOR>& arrIter, CCudaArray2D<ACCUM>& arrAccum, PVOID pStats)
{
	assert(params.nBlocks * params.nThreads == arrIter.Length());
	cudaError_t error = cudaSuccess;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	iterate << < params.nBlocks, params.nThreads >> > (params, arrIter, arrAccum, pStats);
	cudaEventRecord(stop);
	error = cudaGetLastError();
	if (error != cudaSuccess) return error;

	cudaEventSynchronize(stop);
	error = cudaGetLastError();
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	return error;
}

