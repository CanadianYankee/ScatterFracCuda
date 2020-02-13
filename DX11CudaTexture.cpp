#include "framework.h"
#include "DXUtils.h"
#include "DX11CudaTexture.h"

CDX11CudaTexture::CDX11CudaTexture(UINT nWidth, UINT nHeight) :
	  m_fAspectRatio((float)nWidth/(float)nHeight)
	, m_pCudaResource(nullptr)
	, m_pCudaArray(nullptr)
{
	m_GpuArray.nWidth = nWidth;
	m_GpuArray.nHeight = nHeight;
}

CDX11CudaTexture::~CDX11CudaTexture()
{
	if (m_pCudaResource)
	{
		cudaGraphicsUnregisterResource(m_pCudaResource);
		m_pCudaResource = nullptr;
	}
	CudaFree(m_GpuArray.pArray);
}

HRESULT CDX11CudaTexture::Initialize(ComPtr<ID3D11Device> pD3DDevice)
{
	HRESULT hr = S_OK;

	assert(m_GpuArray.nWidth && m_GpuArray.nHeight);

	// Create the texture for drawing
	D3D11_TEXTURE2D_DESC desc;
	ZeroMemory(&desc, sizeof(desc));
	desc.Width = m_GpuArray.nWidth;
	desc.Height = m_GpuArray.nHeight;
	desc.MipLevels = 1;
	desc.ArraySize = 1;
	desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	desc.SampleDesc.Count = 1;
	desc.Usage = D3D11_USAGE_DEFAULT;
	desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

	hr = pD3DDevice->CreateTexture2D(&desc, NULL, &m_pD3DTexture);
	if (FAILED(hr)) return hr;
	D3DDEBUGNAME(m_pD3DTexture, "Texture");

	hr = pD3DDevice->CreateShaderResourceView(m_pD3DTexture.Get(), NULL, &m_pD3DTextureSRV);
	if (FAILED(hr)) return hr;
	D3DDEBUGNAME(m_pD3DTextureSRV, "Texture SRV");

	// Create the texture sampler
	CD3D11_SAMPLER_DESC SamplerDesc;
	ZeroMemory(&SamplerDesc, sizeof(SamplerDesc));
	SamplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_BORDER;
	SamplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_BORDER;
	SamplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_BORDER;
	SamplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
	hr = pD3DDevice->CreateSamplerState(&SamplerDesc, &m_pSamplerState);
	if (FAILED(hr)) return hr;
	D3DDEBUGNAME(m_pSamplerState, "Texture Sampler");

	// Register the Texture as a CUDA resource 
	cudaGraphicsD3D11RegisterResource(&m_pCudaResource, m_pD3DTexture.Get(), cudaGraphicsRegisterFlagsNone);
	if (cudaGetLastError() != cudaSuccess) return E_FAIL;
	size_t pitch;
	cudaMallocPitch(&m_GpuArray.pArray, &pitch, m_GpuArray.nWidth * sizeof(float) * 4, m_GpuArray.nHeight);
	if (cudaGetLastError() != cudaSuccess) return E_FAIL;
	cudaMemset(m_GpuArray.pArray, 0, pitch * m_GpuArray.nHeight);
	m_GpuArray.nPitch = (UINT)pitch;

	return hr;
}

cudaError_t CDX11CudaTexture::MapToCudaArray(GPU_ARRAY_2D &cudaTexture)
{
	cudaError_t err = cudaSuccess;

	assert(!(cudaTexture.pArray) && !m_pCudaArray);

	err = cudaGraphicsMapResources(1, &m_pCudaResource);
	if (err != cudaSuccess) return err;

	err = cudaGraphicsSubResourceGetMappedArray(&m_pCudaArray, m_pCudaResource, 0, 0);
	if (err != cudaSuccess) return err;

	cudaTexture = m_GpuArray;

	return err;
}

cudaError_t CDX11CudaTexture::UnmapFromCudaArray()
{
	cudaError_t err = cudaSuccess;

	assert(m_pCudaArray);

	err = cudaMemcpy2DToArray(m_pCudaArray, 0, 0, m_GpuArray.pArray, m_GpuArray.nPitch, 
		(size_t)m_GpuArray.nWidth * 4 * sizeof(float), m_GpuArray.nHeight, cudaMemcpyDeviceToDevice);
	if (err != cudaSuccess) return err;

	cudaGraphicsUnmapResources(1, &m_pCudaResource);
	if (err != cudaSuccess) return err;

	m_pCudaArray = nullptr;
	
	return err;
}

void CDX11CudaTexture::LoadPS(ComPtr<ID3D11DeviceContext> pD3DContext)
{
	pD3DContext->PSSetShaderResources(0, 1, m_pD3DTextureSRV.GetAddressOf());
	pD3DContext->PSSetSamplers(0, 1, m_pSamplerState.GetAddressOf());
}
