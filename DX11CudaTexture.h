#pragma once

#include "AccumData.h"

struct cudaGraphicsResource;

class CDX11CudaTexture
{
public:
	CDX11CudaTexture(UINT nWidth, UINT nHeight);
	~CDX11CudaTexture();

	HRESULT Initialize(ComPtr<ID3D11Device> pD3DDevice);
	void LoadPS(ComPtr<ID3D11DeviceContext> pDeviceContext);
	float AspectRatio() { return m_fAspectRatio; }
	cudaError_t MapToCudaArray(GPU_ARRAY_2D &cudaTexture);
	cudaError_t UnmapFromCudaArray();
	GPU_ARRAY_2D &GpuArray() { return m_GpuArray; }

protected:
	float m_fAspectRatio;

	ComPtr<ID3D11Texture2D> m_pD3DTexture;
	ComPtr<ID3D11ShaderResourceView> m_pD3DTextureSRV;
	ComPtr<ID3D11SamplerState> m_pSamplerState;
	cudaGraphicsResource* m_pCudaResource;
	GPU_ARRAY_2D m_GpuArray;
	cudaArray* m_pCudaArray;
};

