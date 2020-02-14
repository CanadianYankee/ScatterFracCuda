#pragma once

class CRandgen
{
public:
	__device__ CRandgen() : state(963852741) {};
	__device__ UINT init(UINT seed);
	__device__ UINT rand();
	__device__ float frand();

protected:
	UINT state; 
};

__device__ inline UINT CRandgen::init(UINT seed)
{
	state = (seed != 0) ? seed : 321654987;
	state = (state ^ 61) ^ (state >> 16);
	state *= 9;
	state = state ^ (state >> 4);
	state *= 0x27d4eb2d;
	state = state ^ (state >> 15);
	return state;
}

__device__ inline UINT CRandgen::rand()
{
	// Xorshift algorithm from George Marsaglia's paper
	if (state == 0)
		state = 784951623;
	state ^= (state << 13);
	state ^= (state >> 17);
	state ^= (state << 5);
	return state;
}

__device__ inline float CRandgen::frand()
{
	UINT num = rand();
	return float(num) / float(UINT_MAX);
}
