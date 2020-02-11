#include "framework.h"
#include "Generator.h"

extern cudaError_t cuda_test_generate(PVOID pArray, size_t pitch, size_t width, size_t height, size_t count);

CGenerator::CGenerator(const CONFIG_DATA &config) : 
	m_config(config)
	, m_nAccumWidth(0)
	, m_nAccumHeight(0)
	, m_nAccumPitch(0)
	, m_nAccumMargin(0)
	, m_pAccumArray(nullptr)
{

}

CGenerator::~CGenerator()
{
	if (m_pAccumArray)
	{
		cudaFree(m_pAccumArray);
		m_pAccumArray = nullptr;
	}
}

HRESULT CGenerator::Initialize(ComPtr<ID3D11Device> pD3DDevice)
{
	HRESULT hr = S_OK;
	cudaError_t err = cudaSuccess;

	m_pTexture = std::make_unique<CDX11CudaTexture>(m_config.nDrawWidth, m_config.nDrawHeight);
	hr = m_pTexture->Initialize(pD3DDevice);

	m_nAccumWidth = (size_t)m_config.nDrawWidth;
	m_nAccumWidth = (size_t)m_config.nDrawHeight;
	if (m_config.bAntiAlias && m_config.iAntiAliasLevel > 1)
	{
		m_nAccumWidth *= m_config.iAntiAliasLevel;
		m_nAccumHeight *= m_config.iAntiAliasLevel;
	}
	m_nAccumMargin = 0;
	err = cudaMallocPitch(&m_pAccumArray, &m_nAccumPitch, m_nAccumWidth * sizeof(int), m_nAccumHeight);
	if (err != cudaSuccess) return E_FAIL;

	err = cudaMemset2D(m_pAccumArray, m_nAccumPitch, 0, m_nAccumWidth, m_nAccumHeight);
	if (err != cudaSuccess) return E_FAIL;

	return hr;
}
