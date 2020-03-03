#pragma once

struct FLOAT_COLOR
{
	__device__ __host__ FLOAT_COLOR() : r(0), g(0), b(0) {}
	__device__ __host__ FLOAT_COLOR(float r0, float g0, float b0) : r(r0), g(g0), b(b0) {}
	__device__ bool IsZero() { return r == 0 && g == 0 && b == 0; }
	__device__ float Max() { return fmax(fmax(r, g), b); }
	__device__ float Min() { return fmin(fmin(r, g), b); }
	__device__ void Tint(FLOAT_COLOR clr, float frac) {
		r = (frac * r + clr.r) / (frac + 1.0f);
		g = (frac * g + clr.g) / (frac + 1.0f);
		b = (frac * b + clr.b) / (frac + 1.0f);
	}
	__device__ void LogScale(float fLogScale) {
		r = r ? max(0.0f, logf(r + 1.0f) * fLogScale) : 0;
		g = g ? max(0.0f, logf(g + 1.0f) * fLogScale) : 0;
		b = b ? max(0.0f, logf(b + 1.0f) * fLogScale) : 0;
	}
	__device__ void ToHSV(float& h, float& s, float& v);
	__device__ void FromHSV(float h, float s, float v);

	float r, g, b;
};

__device__
inline FLOAT_COLOR operator*(float t, const FLOAT_COLOR& clr)
{
	return FLOAT_COLOR(t * clr.r, t * clr.g, t * clr.b);
}

__device__ inline void FLOAT_COLOR::ToHSV(float& h, float& s, float& v)
{
	float cmax = Max();
	float cmin = Min();

	if (cmax == cmin)
		h = 0;
	else if (cmax == r)
		h = fmod((g - b) / (6.0f * (cmax - cmin)) + 1.0f, 1.0f);
	else if (cmax == g)
		h = fmod((b - r) / (6.0f * (cmax - cmin)) + 1.0f / 3.0f, 1.0f);
	else
		h = fmod((r - g) / (6.0f * (cmax - cmin)) + 2.0f / 3.0f, 1.0f);

	s = (cmax == 0) ? 0 : 1.0f - cmin / cmax;
	v = cmax;
}

__device__ inline void FLOAT_COLOR::FromHSV(float h, float s, float v)
{
	int hi = (int)floor(h * 6.0f) % 6;

	float f = h * 6.0f - floor(h * 6.0f);
	float p = v * (1 - s);
	float q = v * (1 - f * s);
	float t = v * (1 - (1 - f) * s);

	switch (hi)
	{
	case 0:
		r = v; g = t; b = p; break;
	case 1:
		r = q; g = v; b = p; break;
	case 2:
		r = p; g = v; b = t; break;
	case 3:
		r = p; g = q; b = v; break;
	case 4:
		r = t; g = p; b = v; break;
	case 5:
		r = v; g = p; b = q; break;
	default:
		r = g = b = 0;
	}
}
