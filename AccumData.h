#pragma once

#include "randgen.h"
#include "FloatColor.h"

// Struct to encapsulate a memory-aligned 2D array on the GPU device
struct GPU_ARRAY_2D_OLD
{
	GPU_ARRAY_2D_OLD() : pArray(nullptr), nWidth(0), nHeight(0), nPitch(0) {}
	GPU_ARRAY_2D_OLD(PVOID ptr, UINT w, UINT h, UINT p = 0) : pArray(ptr), nWidth(w), nHeight(h), nPitch(p) {}
	PVOID pArray;
	UINT nWidth;
	UINT nHeight;
	UINT nPitch;
};

// Each element in the 2D accum array has a count and a color
struct ACCUM
{
	UINT nCount;
	FLOAT_COLOR clrAccum;
};

// Each element in the 2D filtered array is just a color
typedef FLOAT_COLOR FILTERED;

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
	UINT nThreads;		// Number of threads for the iterator bundle
	UINT nBlocks;		// Number of blocks for the iterator bundle
	RECT_SCALE rect;	// Scaling and offset to fit points in window
};

// Global statistics kept across all accum threads (in global GPU memory)
struct ACCUM_STATS
{ 
	ACCUM_STATS() : nMaxCount(0), fMaxColorElement(0), xMin(0), yMin(0), yMax(0), nHitRect(0), nNewHits(0), bAbort(FALSE) {}
	UINT nMaxCount;
	float fMaxColorElement;
	float xMin, yMin, xMax, yMax; 
	UINT nHitRect;
	UINT nNewHits;
	BOOL bAbort;
};

// Parameters passed to each render thread
struct RENDER_PARAMS
{
	float fLogColorScale;	// Scale factor for count (based on MaxColorElement)
	float fFilterScale;		// Density filter scale (derived from MaxCount)
	UINT iAntiAlias;		// AntiAlias factor
	UINT iKernelRadius;		// Density estimation maximum kernel blur size
	float fKernelAlpha;		// Kernel scaling alpha
	float fValuePower;		// Value power
	float fSaturPower;		// Saturation power
};
