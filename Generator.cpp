#include "framework.h"
#include "Generator.h"

extern cudaError_t cuda_test_generate(PVOID pArray, PVOID pStats, SIZE_2D size, size_t count);
extern cudaError_t cuda_render_texture(PVOID pTexture, SIZE_2D size_tex, PVOID pArray, SIZE_2D size_arr, PVOID pScale, UINT iAntiAlias);

CGenerator::CGenerator(const CONFIG_DATA &config) : 
	m_config(config)
	, m_nAccumMargin(0)
	, m_pAccumArray(nullptr)
	, m_pAccumStats(nullptr)
	, m_pDrawScale(nullptr)
{

}

CGenerator::~CGenerator()
{
	if (m_pAccumArray)
	{
		cudaFree(m_pAccumArray);
		m_pAccumArray = nullptr;
	}
	if (m_pAccumStats)
	{
		cudaFree(m_pAccumStats);
		m_pAccumStats = nullptr;
	}
	if (m_pDrawScale)
	{
		cudaFree(m_pDrawScale);
		m_pDrawScale = nullptr;
	}
}

HRESULT CGenerator::Initialize(ComPtr<ID3D11Device> pD3DDevice)
{
	HRESULT hr = S_OK;
	cudaError_t err = cudaSuccess;

	// Create the D3D11 texture that Cuda will draw to 
	m_pTexture = std::make_unique<CDX11CudaTexture>(m_config.nDrawWidth, m_config.nDrawHeight);
	hr = m_pTexture->Initialize(pD3DDevice);

	// Create the array of count/colors for fractal generation 
	m_sizeAccum.nWidth = m_config.nDrawWidth;
	m_sizeAccum.nHeight = m_config.nDrawHeight;
	if (m_config.bAntiAlias && m_config.iAntiAliasLevel > 1)
	{
		m_sizeAccum.nWidth *= m_config.iAntiAliasLevel;
		m_sizeAccum.nHeight *= m_config.iAntiAliasLevel;
	}
	m_nAccumMargin = 0;
	size_t pitch;
	err = cudaMallocPitch(&m_pAccumArray, &pitch, m_sizeAccum.nWidth * sizeof(COUNT_COLOR), m_sizeAccum.nHeight);
	if (err != cudaSuccess) return E_FAIL;
	err = cudaMemset2D(m_pAccumArray, pitch, 0, m_sizeAccum.nWidth * sizeof(COUNT_COLOR), m_sizeAccum.nHeight);
	if (err != cudaSuccess) return E_FAIL;
	m_sizeAccum.nPitch = (UINT)pitch;	

	err = cudaMallocManaged(&m_pAccumStats, sizeof(UINT));
	if (err != cudaSuccess) return E_FAIL;
	err = cudaMemset(m_pAccumStats, 0, sizeof(UINT));
	if (err != cudaSuccess) return E_FAIL;

	err = cudaMallocManaged(&m_pDrawScale, sizeof(float));
	if (err != cudaSuccess) return E_FAIL;

	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) return E_FAIL;

	return hr;
}

HRESULT CGenerator::Iterate()
{
	HRESULT hr = S_OK;
	cudaError_t err = cudaSuccess;

	err = cuda_test_generate(m_pAccumArray, m_pAccumStats, m_sizeAccum, 0xffffff);
	if (err != cudaSuccess) return E_FAIL;

	((float*)m_pDrawScale)[0] = 1.0f / (float)((UINT*)m_pAccumStats)[0];

	PVOID pTexData = nullptr;
	err = m_pTexture->MapToCudaArray(&pTexData);
	if (err != cudaSuccess) return E_FAIL;

	UINT iAntiAlias = m_config.bAntiAlias && m_config.iAntiAliasLevel > 1 ? (UINT)m_config.iAntiAliasLevel : 1;
	err = cuda_render_texture(pTexData, m_pTexture->Size(), m_pAccumArray, m_sizeAccum, m_pDrawScale, iAntiAlias);
	if (err != cudaSuccess) return E_FAIL;

	err = m_pTexture->UnmapFromCudaArray();
	if (err != cudaSuccess) return E_FAIL;

	return hr;
}
