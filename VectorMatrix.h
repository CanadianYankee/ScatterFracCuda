#pragma once

class CVector2D
{
public:
	__host__ __device__ CVector2D() { v[0] = v[1] = 0.0f; }
	__host__ __device__ CVector2D(float x, float y) { v[0] = x; v[1] = y; }
	__host__ __device__ CVector2D(float* vin) { v[0] = vin[0]; v[1] = vin[1]; }

	__host__ __device__ static CVector2D* From(float* fptr) { return reinterpret_cast<CVector2D*>(fptr); }
	__host__ __device__ static const CVector2D* From(const float* fptr) { return reinterpret_cast<const CVector2D*>(fptr); }

	__host__ __device__ inline float operator [](UINT idx) const { return v[idx]; }
	__host__ __device__ inline float &operator [](UINT idx) { return v[idx]; }

	__host__ __device__ inline CVector2D operator -() const { return CVector2D(-v[0], -v[1]); }

	__host__ __device__ inline CVector2D operator +(const CVector2D &ov) const { return CVector2D(v[0] + ov[0], v[1] + ov[1]); }
	__host__ __device__ inline CVector2D operator -(const CVector2D &ov) const { return CVector2D(v[0] - ov[0], v[1] - ov[1]); }

	__host__ __device__ inline CVector2D operator +=(const CVector2D &ov) { v[0] += ov[0]; v[1] += ov[1]; return *this; }
	__host__ __device__ inline CVector2D operator -=(const CVector2D &ov) { v[0] -= ov[0]; v[1] -= ov[1]; return *this; }
	__host__ __device__ inline CVector2D operator *=(float a) { v[0] *= a; v[1] *= a; return *this; }

	__host__ __device__ inline float operator *(const CVector2D &ov) const { return v[0] * ov[0] + v[1] * ov[1];  }

	__host__ __device__ inline CVector2D Rotate(float fRadians) const {
		float c = cos(fRadians); float s = sin(fRadians);
		return CVector2D(c * v[0] - s * v[1], s * v[0] + c * v[1]);
	}

protected:
	float v[2];
};

__host__ __device__
inline CVector2D operator *(const CVector2D& v, float a) { return CVector2D(v[0] * a, v[1] * a); }
__host__ __device__
inline CVector2D operator *(float a, const CVector2D& v) { return CVector2D(v[0] * a, v[1] * a); }
__host__ __device__
inline CVector2D operator /(const CVector2D& v, float a) { return CVector2D(v[0] / a, v[1] / a); }

class CMatrix2D
{
public:
	__host__ __device__ CMatrix2D() { m[0] = m[1] = m[2] = m[3] =  0.0f; }
	__host__ __device__ CMatrix2D(float m00, float m01, float m10, float m11) { m[0] = m00; m[1] = m01; m[2] = m10; m[3] = m11; }
	__host__ __device__ CMatrix2D(float* min) { m[0] = min[0]; m[1] = min[1]; m[2] = min[2]; m[3] = min[3]; }

	__host__ __device__ static CMatrix2D* From(float* fptr) { return reinterpret_cast<CMatrix2D*>(fptr); }
	__host__ __device__ static const CMatrix2D* From(const float* fptr) { return reinterpret_cast<const CMatrix2D*>(fptr); }

	__host__ __device__ inline float operator [](UINT idx) const { return m[idx]; }
	__host__ __device__ inline float& operator [](UINT idx) { return m[idx]; }

	__host__ __device__ inline CMatrix2D operator -() const { return CMatrix2D(-m[0], -m[1], -m[2], -m[3]); }

	__host__ __device__ inline CMatrix2D operator +(const CMatrix2D &om) const { return CMatrix2D(m[0] + om[0], m[1] + om[1], m[2] + om[2], m[3] + om[3]); }
	__host__ __device__ inline CMatrix2D operator -(const CMatrix2D &om) const { return CMatrix2D(m[0] - om[0], m[1] - om[1], m[2] - om[2], m[3] - om[3]); }

	__host__ __device__ inline CMatrix2D operator +=(const CMatrix2D &om) { m[0] += om[0]; m[1] += om[1]; m[2] += om[2]; m[3] += om[3]; return *this; }
	__host__ __device__ inline CMatrix2D operator -=(const CMatrix2D &om) { m[0] -= om[0]; m[1] -= om[1]; m[2] -= om[2]; m[3] -= om[3]; return *this; }

	__host__ __device__ inline CMatrix2D operator *(const CMatrix2D& om) const {
		return CMatrix2D(m[0] * om[0] + m[1] * om[2], m[0] * om[1] + m[1] * om[3], m[2] * om[0] + m[3] * om[2], m[2] * om[1] + m[3] * om[3]);
	}

	__host__ __device__ inline CMatrix2D operator *=(const CMatrix2D& om) {
		*this = *this * om; return *this;
	}

	__host__ __device__ inline static CMatrix2D Rotation(float fRadians) {
		float c = cos(fRadians); float s = sin(fRadians);
		return CMatrix2D(c, -s, s, c);
	}

protected:
	float m[4];
};

__host__ __device__
inline CMatrix2D operator *(const CMatrix2D& m, float a) { return CMatrix2D(m[0] * a, m[1] * a, m[2] * a, m[3] * a); }
__host__ __device__
inline CMatrix2D operator *(float a, const CMatrix2D& m) { return CMatrix2D(m[0] * a, m[1] * a, m[2] * a, m[3] * a); }
__host__ __device__
inline CMatrix2D operator /(const CMatrix2D& m, float a) { return CMatrix2D(m[0] / a, m[1] / a, m[2] / a, m[3] / a); }

__host__ __device__
inline CVector2D operator *(const CMatrix2D& m, const CVector2D& v) { 
	return CVector2D(m[0] * v[0] + m[1] * v[1], m[2] * v[0] + m[3] * v[1]);
}
