#include "framework.h"
#include "Generator.h"

extern cudaError_t cuda_iterate(const ACCUM_PARAMS& params, GPU_ARRAY_2D& arrIter, GPU_ARRAY_2D& arrAccum, PVOID pStats);
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
	CudaFree(m_IterArray.pArray);
	m_pTexture.reset();

	cudaDeviceReset();
}

HRESULT CGenerator::Initialize(ComPtr<ID3D11Device> pD3DDevice, BOOL& bFailed)
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
	m_IterArray.nWidth = m_nAccumBlocks;
	m_IterArray.nHeight = m_nAccumThreads;
	UINT szIters = m_nAccumThreads * m_nAccumBlocks * sizeof(ITERATOR);
	err = cudaMalloc(&(m_IterArray.pArray), szIters);
	if (err != cudaSuccess) return E_FAIL;
	err = cudaMemset(m_IterArray.pArray, 0, szIters);

	// Initialize all of the generators
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) return E_FAIL;
	ACCUM_PARAMS paramsAccum;
	paramsAccum.bInit = TRUE;
	paramsAccum.nSteps = 128;
	ACCUM_STATS* pAccumStats = (ACCUM_STATS*)m_pAccumStats;
	pAccumStats->xMin = pAccumStats->yMin = FLT_MAX;
	pAccumStats->xMax = pAccumStats->yMax = -FLT_MAX;
	GPU_ARRAY_2D arrDummy; 
	err = cuda_iterate(paramsAccum, m_IterArray, arrDummy, m_pAccumStats);
	if (err != cudaSuccess) return E_FAIL;

	// Check for failure (blowing up to infinity)
	pAccumStats = (ACCUM_STATS*)m_pAccumStats;
	if (pAccumStats->bAbort)
		bFailed = TRUE;
	else
	{
		float dx = (pAccumStats->xMax - pAccumStats->xMin) * 1.1f;
		float dy = (pAccumStats->yMax - pAccumStats->yMin) * 1.1f;
		float cx = 0.5f * (pAccumStats->xMax + pAccumStats->xMin);
		float cy = 0.5f * (pAccumStats->yMax + pAccumStats->yMin);
		m_rectScale.fScale = min((float)m_AccumArray.nWidth / dx, (float)m_AccumArray.nHeight / dy);
		m_rectScale.fOffsetX = 0.5f * (float)m_AccumArray.nWidth - m_rectScale.fScale * cx;
		m_rectScale.fOffsetY = 0.5f * (float)m_AccumArray.nHeight - m_rectScale.fScale * cy;
	}

	return hr;
}

HRESULT CGenerator::Iterate(BOOL bRender)
{
	HRESULT hr = S_OK;
	cudaError_t err = cudaSuccess;

	ACCUM_PARAMS paramsAccum;
	paramsAccum.nSteps = 1024;
	paramsAccum.rect = m_rectScale;
//	paramsAccum.bHitPercent = TRUE;
	err = cuda_iterate(paramsAccum, m_IterArray, m_AccumArray, m_pAccumStats);

//	ACCUM_STATS *pAccumStats = (ACCUM_STATS*)m_pAccumStats;
//	float fPercent = (float)(pAccumStats->nNewHits) / (float)(pAccumStats->nHitRect);


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
