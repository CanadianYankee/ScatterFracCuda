#pragma once

inline void CudaFree(PVOID& ptr) { if (ptr) { cudaFree(ptr); ptr = nullptr; } }

// This wraps a 1D array that's allocated on the host, but only accessed by the GPU
template <typename T>
class CCudaArray1D
{
public:
	CCudaArray1D() : m_pArray(nullptr), m_nLength(0) {}
	CCudaArray1D(PVOID ptr, UINT len) : m_pArray(ptr), m_nLength(len) {}

	__host__ __device__ inline UINT Length() const { return m_nLength; }

	__host__ __device__ size_t ElementSize() const { return sizeof(T); }

	__host__ __device__ inline bool ValidIndex(UINT idx) const { return idx < m_nLength; }
	__host__ __device__ inline bool ValidIndex(int idx) const { return idx >= 0 && idx < (int)m_nLength; }

	__host__ cudaError_t Malloc(UINT length, bool bZero = true) {
		assert(!m_pArray);
		cudaError_t err = cudaSuccess;
		err = cudaMalloc(&(m_pArray), length * sizeof(T));
		if (err != cudaSuccess) return err;
		m_nLength = length;
		if (bZero) err = Zero();
		return err;
	}

	__host__ inline cudaError_t Zero() {
		return m_pArray ? cudaMemset(m_pArray, 0, m_nLength * ElementSize()) : cudaSuccess;
	}

	__host__ void Free() { CudaFree(m_pArray); m_nLength = 0; }

	__host__ cudaError_t CopyTo(PVOID dest, cudaMemcpyKind kind) const {
		return cudaMemcpy(dest, m_pArray, Length() * ElementSize(), kind);
	}

	__host__ cudaError_t CopyFrom(PVOID src, cudaMemcpyKind kind) {
		return cudaMemcpy(m_pArray, src, Length() * ElementSize(), kind);
	}

	__device__ T* GetAt(UINT idx) {
		return (T*)((unsigned char*)m_pArray + idx * sizeof(T));
	}

	__device__ const T* GetAt(UINT idx) const {
		return (const T*)((unsigned char*)m_pArray + idx * sizeof(T));
	}

protected:
	PVOID m_pArray;
	UINT m_nLength;
};

// This wraps a 2D, memory-aligned array that's allocated on the host, but 
// only accessed by the GPU
template <typename T>
class CCudaArray2D 
{
public:
	CCudaArray2D() : m_pArray(nullptr), m_nWidth(0), m_nHeight(0), m_nPitch(0) {}
	CCudaArray2D(PVOID ptr, UINT w, UINT h, UINT p = 0) : m_pArray(ptr), m_nWidth(w), m_nHeight(h), m_nPitch(p) {}

	__host__ __device__ inline UINT Width() const { return m_nWidth; }
	__host__ __device__ inline UINT Height() const { return m_nHeight; }
	__host__ __device__ inline UINT Pitch() const { return m_nPitch; }

	__host__ dim3 BlockDim(dim3 threadDim) const {
		assert(threadDim.z == 1);
		return dim3((Width() + threadDim.x - 1) / threadDim.x, (Height() + threadDim.y - 1) / threadDim.y);
	}

	__host__ __device__ size_t ElementSize() const { return sizeof(T); }

	__host__ __device__ inline bool ValidIndex(UINT idx, UINT idy) const { return idx < m_nWidth && idy < m_nHeight; }
	__host__ __device__ inline bool ValidIndex(int idx, int idy) const { return idx >= 0 && idy >= 0 && idx < (int)m_nWidth && idy < (int)m_nHeight; }

	__host__ cudaError_t MallocPitch(UINT width, UINT height, bool bZero = true) {
		assert(!m_pArray);
		cudaError_t err = cudaSuccess;
		size_t pitch;
		err = cudaMallocPitch(&(m_pArray), &pitch, width * sizeof(T), height);
		if (err != cudaSuccess) return err;
		m_nPitch = (UINT)pitch;
		m_nWidth = width; m_nHeight = height;
		if (bZero) err = Zero();
		return err;
	}

	__host__ inline cudaError_t Zero() 	{
		return m_pArray ? cudaMemset2D(m_pArray, m_nPitch, 0, m_nWidth * ElementSize(), m_nHeight) : cudaSuccess;
	}

	__host__ void Free() { CudaFree(m_pArray); m_nWidth = m_nHeight = m_nPitch = 0; }

	__device__ T* GetAt(UINT idx, UINT idy) {
		return (T*)((unsigned char*)m_pArray + idy * m_nPitch + idx * sizeof(T));
	}

	__device__ T* GetRow(UINT idy) {
		return (T*)((unsigned char*)m_pArray + idy * m_nPitch);
	}

protected:
	PVOID m_pArray;
	UINT m_nWidth;
	UINT m_nHeight;
	UINT m_nPitch;
};

// A specialized CCudaArray2D for graphics textures of type R32G32B32A32_FLOAT
class CCudaTexture2D : protected CCudaArray2D<float>
{
public:
	CCudaTexture2D() : CCudaArray2D<float>() {}

	__host__ __device__ inline UINT Width() const { return CCudaArray2D<float>::Width() / 4; }
	__host__ __device__ inline UINT Height() const { return CCudaArray2D<float>::Height(); }
	__host__ __device__ inline UINT Pitch() const { return CCudaArray2D<float>::Pitch(); }

	__host__ __device__ inline bool ValidIndex(UINT idx, UINT idy) const { return CCudaArray2D<float>::ValidIndex(idx * 4, idy); }
	__host__ __device__ inline bool ValidIndex(int idx, int idy) const { return CCudaArray2D<float>::ValidIndex(idx * 4, idy); }

	__host__ dim3 BlockDim(dim3 threadDim) const {
		assert(threadDim.z == 1);
		return dim3((Width() * 4 + threadDim.x - 1) / threadDim.x, (Height() + threadDim.y - 1) / threadDim.y);
	}

	__host__ cudaError_t MallocPitch(UINT width, UINT height, bool bZero = true) {
		return CCudaArray2D<float>::MallocPitch(width * 4, height, bZero);
	}

	__host__ void Zero() { CCudaArray2D<float>::Zero(); }
	__host__ void Free() { CCudaArray2D<float>::Free(); }
	__host__ bool IsEmpty() const { return !m_pArray; }

	__host__ cudaError_t CopyTo(cudaArray_t dest, cudaMemcpyKind kind) const {
		return cudaMemcpy2DToArray(dest, 0, 0, m_pArray, Pitch(),
			(size_t)Width() * 4 * sizeof(float), Height(), kind);
	}

	__device__ float* GetAt(UINT idx, UINT idy) { return CCudaArray2D<float>::GetAt(idx * 4, idy); }
	__device__ float* GetRow(UINT idy) { return CCudaArray2D<float>::GetRow(idy); }
};
