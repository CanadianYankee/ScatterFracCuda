#pragma once

#include "CudaUtil.h"

struct cudaGraphicsResource;

class CDX11CudaTexture
{
public:
	CDX11CudaTexture(UINT nWidth, UINT nHeight);
	~CDX11CudaTexture();

	HRESULT Initialize(ComPtr<ID3D11Device> pD3DDevice);
	void LoadPS(ComPtr<ID3D11DeviceContext> pDeviceContext);
	float AspectRatio() { return m_fAspectRatio; }
	cudaError_t MapToCudaArray(PVOID *pCudaMemory);
	cudaError_t UnmapFromCudaArray();
	SIZE_2D Size() { return m_size; }

protected:
	SIZE_2D m_size;
	float m_fAspectRatio;

	ComPtr<ID3D11Texture2D> m_pD3DTexture;
	ComPtr<ID3D11ShaderResourceView> m_pD3DTextureSRV;
	ComPtr<ID3D11SamplerState> m_pSamplerState;
	cudaGraphicsResource* m_pCudaResource;
	PVOID m_pCudaMemory;
	cudaArray* m_pCudaArray;
};

