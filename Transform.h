#pragma once

#include "FloatColor.h"
#include "VectorMatrix.h"


class CTransform
{
public:
	__host__  float& Weight() { return *(params + WEIGHT); }
	__device__ float Weight() const { return *(params + WEIGHT); }
	__host__ FLOAT_COLOR &Color() { return *(FLOAT_COLOR::From(params + COLOR)); }
	__device__ FLOAT_COLOR Color() const { return *(FLOAT_COLOR::From(params + COLOR)); }
	__host__ CMatrix2D& Matrix0() { return *(CMatrix2D::From(params + MATRIX0));  }
	__host__ CVector2D& Offset0() { return *(CVector2D::From(params + OFFSET0)); }
	__device__ CMatrix2D Matrix0() const { return *(CMatrix2D::From(params + MATRIX0)); }
	__device__ CVector2D Offset0() const { return *(CVector2D::From(params + OFFSET0)); }

protected:
	float params[10];

	// Index where each parameter set beginss
	enum TRIDX
	{
		WEIGHT = 0
		, COLOR = 1
		, MATRIX0 = 4
		, OFFSET0 = 8
	};
};
