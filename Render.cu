#include "framework.h"
#include "AccumData.h"
#include "CudaArray.h"

// Checks for out-of-range and does atomic adds
__device__ inline void AddFiltered(CCudaArray2D<FILTERED>& arrFiltered, int x, int y, const FLOAT_COLOR& clr)
{
	if (arrFiltered.ValidIndex(x, y))
	{
		FILTERED* pFiltered = arrFiltered.GetAt(x, y);
		if (clr.r) atomicAdd(&(pFiltered->r), clr.r);
		if (clr.g) atomicAdd(&(pFiltered->g), clr.g);
		if (clr.b) atomicAdd(&(pFiltered->b), clr.b);
	}
}

__device__ inline float gaussian(float x, float stddev)
{
	float x0 = x / stddev;
	return exp(-0.5f * x0 * x0) / (sqrt(6.2831853f) * stddev);
}

__global__ void rescale_filter(const RENDER_PARAMS params, CCudaArray2D<FILTERED> arrFiltered, CCudaArray2D<ACCUM> arrAccum)
{
	int arrx = blockIdx.x * blockDim.x + threadIdx.x;
	int arry = blockIdx.y * blockDim.y + threadIdx.y;
	if (!arrAccum.ValidIndex(arrx, arry)) return;

	ACCUM* pAccum = arrAccum.GetAt(arrx, arry);
	int rad = params.iKernelRadius * params.iAntiAlias;
	int fx = arrx - rad;
	int fy = arry - rad;
	float fCutoff = rad ? 0.001f / (float)(rad * rad) : 0.0f;
	if (!pAccum->clrAccum.IsZero())
	{
		FLOAT_COLOR clr = pAccum->clrAccum;
		clr.LogScale(params.fLogColorScale);
		if (!clr.IsZero())
		{
			float h, s, v;
			clr.ToHSV(h, s, v);
			v = powf(v, params.fValuePower);
			if (params.fSaturPower) s = powf(s, params.fSaturPower);
			clr.FromHSV(h, s, v);
			if (rad)
			{
				float fHalfRad = 0.5f * (float)rad;
				float stddev = params.fFilterScale / pow((float)(pAccum->nCount), params.fKernelAlpha);
				if (stddev > fHalfRad) stddev = fHalfRad;
				if (stddev < 0.5f)
				{
					// No dispersion
					AddFiltered(arrFiltered, fx, fy, clr);
				}
				else
				{
					// Calculate one-eighth(ish) of the filter and use symmetry to get the rest
					for (int ix = 0; ix <= rad; ix++)
					{
						float attx = gaussian((float)ix, stddev);
						for (int iy = 0; iy <= ix; iy++)
						{
							float att = attx * gaussian((float)iy, stddev);
							if (att < fCutoff) continue;	// Don't bother if numbers are tiny
							FLOAT_COLOR attClr = att * clr;
							AddFiltered(arrFiltered, fx + ix, fy + iy, attClr);
							if (ix || iy)
							{
								if (iy == 0)
								{
									AddFiltered(arrFiltered, fx - ix, fy, attClr);
									AddFiltered(arrFiltered, fx, fy + ix, attClr);
									AddFiltered(arrFiltered, fx, fy - ix, attClr);
								}
								else
								{
									AddFiltered(arrFiltered, fx - ix, fy + iy, attClr);
									AddFiltered(arrFiltered, fx + ix, fy - iy, attClr);
									AddFiltered(arrFiltered, fx - ix, fy - iy, attClr);
									if (ix != iy)
									{
										AddFiltered(arrFiltered, fx + iy, fy + ix, attClr);
										AddFiltered(arrFiltered, fx - iy, fy + ix, attClr);
										AddFiltered(arrFiltered, fx + iy, fy - ix, attClr);
										AddFiltered(arrFiltered, fx - iy, fy - ix, attClr);
									}
								}
							}
						}
					}
				}
			}
			else
			{
				// Just scaling, no actual filtering
				FILTERED* pFiltered = arrFiltered.GetAt(fx, fy);
				*pFiltered = clr;
			}
		}
	}
}

__global__ void render_texture(const RENDER_PARAMS params, CCudaTexture2D texture, CCudaArray2D<FILTERED> arrFiltered)
{
	UINT texx = blockIdx.x * blockDim.x + threadIdx.x;
	UINT texy = blockIdx.y * blockDim.y + threadIdx.y;
	if (!texture.ValidIndex(texx, texy)) return;

	float* pixel = texture.GetAt(texx, texy);

	UINT iAntiAlias = max(1, params.iAntiAlias);
	UINT arrx = texx * iAntiAlias;
	UINT arry = texy * iAntiAlias;

	float r = 0.0f, g = 0.0f, b = 0.0f;
	for (UINT j = 0; j < iAntiAlias; j++)
	{
		FILTERED* pRow = arrFiltered.GetRow(arry + j);
		for (UINT i = 0; i < iAntiAlias; i++)
		{
			FILTERED* pItem = &pRow[arrx + i];
			if (!pItem->IsZero())
			{
				r += pItem->r;
				g += pItem->g;
				b += pItem->b;
			}
		}
	}
	pixel[0] = r / (float)(iAntiAlias * iAntiAlias);
	pixel[1] = g / (float)(iAntiAlias * iAntiAlias);
	pixel[2] = b / (float)(iAntiAlias * iAntiAlias);
	pixel[3] = 1.0f;
}

cudaError_t cuda_render_texture(const RENDER_PARAMS& params, CCudaTexture2D& texture, CCudaArray2D<FILTERED>& arrFiltered, CCudaArray2D<ACCUM>& arrAccum)
{
	cudaError_t error = cudaSuccess;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// First run the log-scale and density estimation to go from arrAccum -> arrFiltered
	dim3 Db = dim3(16, 16);
	dim3 Dg = arrAccum.BlockDim(Db);
	error = arrFiltered.Zero();
	if (error != cudaSuccess) return error;

	cudaEventRecord(start);
	rescale_filter << <Dg, Db >> > (params, arrFiltered, arrAccum);
	cudaEventRecord(stop);
	error = cudaGetLastError();
	if (error != cudaSuccess) return error;
	cudaEventSynchronize(stop);
	error = cudaGetLastError();
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	if (error != cudaSuccess) return error;

	// Next, do anti-aliasing and final conversion to texture to go from arrFiltered -> texture
	Dg = texture.BlockDim(Db);

	cudaEventRecord(start);
	render_texture << <Dg, Db >> > (params, texture, arrFiltered);
	cudaEventRecord(stop);
	error = cudaGetLastError();
	if (error != cudaSuccess) return error;

	cudaEventSynchronize(stop);
	error = cudaGetLastError();
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	return error;
}
