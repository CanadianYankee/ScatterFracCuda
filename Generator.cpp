#include "framework.h"
#include "Generator.h"
#include "randgen.h"

extern cudaError_t cuda_intialize(const ACCUM_PARAMS &params, GPU_ARRAY_2D &randgen, PVOID pStats);
extern cudaError_t cuda_test_generate(const ACCUM_PARAMS &params, GPU_ARRAY_2D &randgen, GPU_ARRAY_2D &arrAccum, PVOID pStats);
extern cudaError_t cuda_render_texture(const RENDER_PARAMS &params, GPU_ARRAY_2D &texture, GPU_ARRAY_2D &arrAccum);

CGenerator::CGenerator(const CONFIG_DATA &config) : 
	m_config(config)
	, m_pAccumStats(nullptr)
{

}

CGenerator::~CGenerator()
{
	CudaFree(m_AccumArray.pArray);
	CudaFree(m_pAccumStats);
	CudaFree(m_Randgen.pArray);
	m_pTexture.reset();

	cudaDeviceReset();
}

HRESULT CGenerator::Initialize(ComPtr<ID3D11Device> pD3DDevice)
{
	HRESULT hr = S_OK;
	cudaError_t err = cudaSuccess;

	// Create the D3D11 texture that Cuda will draw to 
	m_pTexture = std::make_unique<CDX11CudaTexture>(m_config.nDrawWidth, m_config.nDrawHeight);
	hr = m_pTexture->Initialize(pD3DDevice);

	// Create the array of count/colors for fractal generation 
	assert(!m_AccumArray.pArray);
	m_AccumArray.nWidth = m_config.nDrawWidth;
	m_AccumArray.nHeight = m_config.nDrawHeight;
	if (m_config.bAntiAlias && m_config.iAntiAliasLevel > 1)
	{
		m_AccumArray.nWidth *= m_config.iAntiAliasLevel;
		m_AccumArray.nHeight *= m_config.iAntiAliasLevel;
	}
	size_t pitch;
	err = cudaMallocPitch(&(m_AccumArray.pArray), &pitch, m_AccumArray.nWidth * sizeof(COUNT_COLOR), m_AccumArray.nHeight);
	if (err != cudaSuccess) return E_FAIL;
	err = cudaMemset2D(m_AccumArray.pArray, pitch, 0, m_AccumArray.nWidth * sizeof(COUNT_COLOR), m_AccumArray.nHeight);
	if (err != cudaSuccess) return E_FAIL;
	m_AccumArray.nPitch = (UINT)pitch;

	// Stats gathered during generation (managed memory for easy CPU access
	err = cudaMallocManaged(&m_pAccumStats, sizeof(UINT));
	if (err != cudaSuccess) return E_FAIL;
	err = cudaMemset(m_pAccumStats, 0, sizeof(UINT));
	if (err != cudaSuccess) return E_FAIL;

	// One random number generator for each thread
	m_Randgen.nWidth = m_nAccumBlocks;
	m_Randgen.nHeight = m_nAccumThreads;
	err = cudaMallocManaged(&(m_Randgen.pArray), m_nAccumThreads * m_nAccumBlocks * sizeof(CRandgen));
	if (err != cudaSuccess) return E_FAIL;
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) return E_FAIL;

	// Initialize all of the generators
	ACCUM_PARAMS paramsAccum;
	paramsAccum.bInit = TRUE;
	paramsAccum.nSteps = 128;
	err = cuda_intialize(paramsAccum, m_Randgen, m_pAccumStats);
	if (err != cudaSuccess) return E_FAIL;

	return hr;
}

HRESULT CGenerator::Iterate(BOOL bRender)
{
	HRESULT hr = S_OK;
	cudaError_t err = cudaSuccess;

	ACCUM_PARAMS paramsAccum;
	paramsAccum.nSteps = 0xffffff;
	paramsAccum.bInit = FALSE;
	err = cuda_test_generate(paramsAccum, m_Randgen, m_AccumArray, m_pAccumStats);
	if (err != cudaSuccess) return E_FAIL;

	if (bRender)
	{
		GPU_ARRAY_2D texture;
		err = m_pTexture->MapToCudaArray(texture);
		if (err != cudaSuccess) return E_FAIL;

		RENDER_PARAMS paramsRender;
		paramsRender.fCountScale = 1.0f / (float)((UINT*)m_pAccumStats)[0];
		paramsRender.iAntiAlias = m_config.bAntiAlias && m_config.iAntiAliasLevel > 1 ? (UINT)m_config.iAntiAliasLevel : 1;
		err = cuda_render_texture(paramsRender, texture, m_AccumArray);
		if (err != cudaSuccess) return E_FAIL;

		err = m_pTexture->UnmapFromCudaArray();
		if (err != cudaSuccess) return E_FAIL;
	}

	return hr;
}
