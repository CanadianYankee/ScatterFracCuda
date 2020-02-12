#pragma once

struct SIZE_2D
{
	__host__ __device__ SIZE_2D() : nWidth(0), nHeight(0), nPitch(0) {}
	__host__ __device__ SIZE_2D(UINT w, UINT h, UINT p = 0) : nWidth(w), nHeight(h), nPitch(p) {}
	UINT nWidth;
	UINT nHeight;
	UINT nPitch;
};
