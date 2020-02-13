#pragma once

struct SIZE_2D
{
	__host__ __device__ SIZE_2D() : nWidth(0), nHeight(0), nPitch(0) {}
	__host__ __device__ SIZE_2D(UINT w, UINT h, UINT p = 0) : nWidth(w), nHeight(h), nPitch(p) {}
	UINT nWidth;
	UINT nHeight;
	UINT nPitch;
};

struct COUNT_COLOR
{
	__host__ __device__ COUNT_COLOR() : nCount(0), r(0), g(0), b(0) {}
	UINT nCount;
	float r, g, b;
};

struct ACCUM_STATS
{
	UINT nMaxCount;
	float xMin, yMin, xMax, yMax; 

};