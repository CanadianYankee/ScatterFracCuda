#include "framework.h"
#include "Generator.h"

extern cudaError_t cuda_iterate(const ACCUM_PARAMS& params, GPU_ARRAY_2D& arrIter, GPU_ARRAY_2D& arrAccum, PVOID pStats);
extern cudaError_t cuda_render_texture(const RENDER_PARAMS &params, GPU_ARRAY_2D &texture, GPU_ARRAY_2D& arrFiltered, GPU_ARRAY_2D &arrAccum);

CGenerator::CGenerator(const CONFIG_DATA &config) : 
	m_config(config)
	, m_pAccumStats(nullptr)
	, m_nTotalIter(0)
	, m_nCycleIter(0)
	, m_nIterComplete(0)
{

}

CGenerator::~CGenerator()
{
	CudaFree(m_AccumArray.pArray);
	CudaFree(m_FilteredArray.pArray);
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
	m_AccumArray.nWidth += m_config.KernelRadius() * 2;
	m_AccumArray.nHeight += m_config.KernelRadius() * 2;
	m_AccumArray.nWidth *= m_config.AntiAlias();
	m_AccumArray.nHeight *= m_config.AntiAlias();

	size_t pitch;
	err = cudaMallocPitch(&(m_AccumArray.pArray), &pitch, m_AccumArray.nWidth * sizeof(ACCUM), m_AccumArray.nHeight);
	if (err != cudaSuccess) return E_FAIL;
	err = cudaMemset2D(m_AccumArray.pArray, pitch, 0, m_AccumArray.nWidth * sizeof(ACCUM), m_AccumArray.nHeight);
	if (err != cudaSuccess) return E_FAIL;
	m_AccumArray.nPitch = (UINT)pitch;

	// Create the array for the filtered and log-scaled results
	assert(!m_FilteredArray.pArray);
	m_FilteredArray.nWidth = m_config.nDrawWidth;
	m_FilteredArray.nHeight = m_config.nDrawHeight;
	m_FilteredArray.nWidth *= m_config.AntiAlias();
	m_FilteredArray.nHeight *= m_config.AntiAlias();

	err = cudaMallocPitch(&(m_FilteredArray.pArray), &pitch, m_FilteredArray.nWidth * sizeof(FILTERED), m_FilteredArray.nHeight);
	if (err != cudaSuccess) return E_FAIL;
	err = cudaMemset2D(m_FilteredArray.pArray, pitch, 0, m_FilteredArray.nWidth * sizeof(FILTERED), m_FilteredArray.nHeight);
	if (err != cudaSuccess) return E_FAIL;
	m_FilteredArray.nPitch = (UINT)pitch;

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

	// Initialize all of the generators and do a short run to find window scale
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
		// Set window scale
		float dx = (pAccumStats->xMax - pAccumStats->xMin) * 1.1f;
		float dy = (pAccumStats->yMax - pAccumStats->yMin) * 1.1f;
		float cx = 0.5f * (pAccumStats->xMax + pAccumStats->xMin);
		float cy = 0.5f * (pAccumStats->yMax + pAccumStats->yMin);
		m_rectScale.fScale = min((float)m_AccumArray.nWidth / dx, (float)m_AccumArray.nHeight / dy);
		m_rectScale.fOffsetX = 0.5f * (float)m_AccumArray.nWidth - m_rectScale.fScale * cx;
		m_rectScale.fOffsetY = 0.5f * (float)m_AccumArray.nHeight - m_rectScale.fScale * cy;
	}

	m_nTotalIter = m_config.iIterationLevel * (m_config.AntiAlias() * m_config.AntiAlias()) * m_AccumArray.nHeight * m_AccumArray.nWidth / 25;
	m_nCycleIter = m_nTotalIter / (16 * m_config.iIterationLevel);
	m_nIterComplete = 0;

	return hr;
}

HRESULT CGenerator::Iterate(BOOL bRender)
{
	HRESULT hr = S_OK;
	cudaError_t err = cudaSuccess;

	if (m_nIterComplete < m_nTotalIter)
	{
		ACCUM_PARAMS paramsAccum;
		paramsAccum.nSteps = max(1, m_nCycleIter / (m_IterArray.nHeight * m_IterArray.nWidth)); // m_nTotalIter / (m_IterArray.nHeight * m_IterArray.nWidth);
		paramsAccum.rect = m_rectScale;
		paramsAccum.bHitPercent = (m_nIterComplete == 0);
		err = cuda_iterate(paramsAccum, m_IterArray, m_AccumArray, m_pAccumStats);
		m_nIterComplete += paramsAccum.nSteps * m_IterArray.nHeight * m_IterArray.nWidth;

		ACCUM_STATS* pAccumStats = (ACCUM_STATS*)m_pAccumStats;
		float fPercent = (float)(pAccumStats->nNewHits) / (float)(pAccumStats->nHitRect);
		
		if (bRender)
		{
			GPU_ARRAY_2D texture;
			err = m_pTexture->MapToCudaArray(texture);
			if (err != cudaSuccess) return E_FAIL;

			RENDER_PARAMS paramsRender;
			ACCUM_STATS* pAccumStats = (ACCUM_STATS*)m_pAccumStats;
			paramsRender.fLogColorScale = 1.0f / log((float)(pAccumStats->nMaxColorElement));
			paramsRender.iAntiAlias = m_config.AntiAlias();
			paramsRender.iKernelRadius = m_config.KernelRadius();
			paramsRender.fFilterScale = m_config.KernelRadius() ?
				0.25f * (float)m_config.KernelRadius() * log((float)(pAccumStats->nMaxCount)) : 0.0f;
			paramsRender.fValuePower = 1.0f / m_config.fGammaValue;
			paramsRender.fSaturPower = m_config.fGammaSatur ? 1.0f / m_config.fGammaSatur : 0.0f;
			err = cuda_render_texture(paramsRender, texture, m_FilteredArray, m_AccumArray);
			if (err != cudaSuccess) return E_FAIL;

			err = m_pTexture->UnmapFromCudaArray();
			if (err != cudaSuccess) return E_FAIL;
		}
	}

	return hr;
}
