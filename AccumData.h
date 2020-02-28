#pragma once

#include "randgen.h"

inline void CudaFree(PVOID& ptr) { if (ptr) { cudaFree(ptr); ptr = nullptr; } }

// Struct to encapsulate a memory-aligned 2D array on the GPU device
struct GPU_ARRAY_2D
{
	GPU_ARRAY_2D() : pArray(nullptr), nWidth(0), nHeight(0), nPitch(0) {}
	GPU_ARRAY_2D(PVOID ptr, UINT w, UINT h, UINT p = 0) : pArray(ptr), nWidth(w), nHeight(h), nPitch(p) {}
	PVOID pArray;
	UINT nWidth;
	UINT nHeight;
	UINT nPitch;
};

// Each element in the 2D accum array is of this type
struct FLOAT_COLOR
{
	__device__ __host__ FLOAT_COLOR() : r(0), g(0), b(0) {}
	__device__ __host__ FLOAT_COLOR(float r0, float g0, float b0) : r(r0), g(g0), b(b0) {}
	__device__ bool IsZero() { return r == 0 && g == 0 && b == 0; }
	__device__ float Max() { return fmax(fmax(r, g), b); }
	__device__ void Tint(FLOAT_COLOR clr, float frac) { r = (frac * r + clr.r) / (frac + 1.0f); 
														g = (frac * g + clr.g) / (frac + 1.0f);
														b = (frac * b + clr.b) / (frac + 1.0f);	}

	float r, g, b;
};

// Each thread gets an iterator, which has a random number generator, a position, and a color
struct ITERATOR
{
	float x, y;
	FLOAT_COLOR clr;
	CRandgen rand;
};

// Scaling and offset for 2D window
struct RECT_SCALE
{
	RECT_SCALE() : fScale(1.0f), fOffsetX(0), fOffsetY(0) {}
	float fScale, fOffsetX, fOffsetY;
};

// Parameters passed to each accum thread
struct ACCUM_PARAMS
{
	ACCUM_PARAMS() : bInit(FALSE), bHitPercent(FALSE), nSteps(0) {}
	BOOL bInit;			// When true, do initializtion and not accumlation
	BOOL bHitPercent;	// When true (on first cycle), do hit count percentage tracking
	UINT nSteps;		// Number of iterations to do in this cycle (per thread)
	RECT_SCALE rect;	// Scaling and offset to fit points in window
};

// Global statistics kept across all accum threads (in global GPU memory)
struct ACCUM_STATS
{ 
	ACCUM_STATS() : nMaxColorElement(0), xMin(0), yMin(0), yMax(0), nHitRect(0), nNewHits(0), bAbort(FALSE) {}
	UINT nMaxColorElement;
	float xMin, yMin, xMax, yMax; 
	UINT nHitRect;
	UINT nNewHits;
	BOOL bAbort;
};

// Parameters passed to each render thread
struct RENDER_PARAMS
{
	float fLogColorScale;	// Scale factor for count (based on MaxColorElement)
	UINT iAntiAlias;		// AntiAlias factor
};
