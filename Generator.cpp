#include "framework.h"
#include "Generator.h"
#include "Transform.h"

extern cudaError_t cuda_iterate(const ACCUM_PARAMS& params, CCudaArray1D<CTransform>& arrTransforms, CCudaArray1D<ITERATOR>& arrIter, CCudaArray2D<ACCUM>& arrAccum, PVOID pStats);
extern cudaError_t cuda_render_texture(const RENDER_PARAMS &params, CCudaTexture2D &texture, CCudaArray2D<FILTERED>& arrFiltered, CCudaArray2D<ACCUM> &arrAccum);

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
	m_AccumArray.Free();
	m_FilteredArray.Free();
	m_IterArray.Free();
	m_TransformArray.Free();
	CudaFree(m_pAccumStats);
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
	UINT nAccWidth = m_config.nDrawWidth;
	UINT nAccHeight = m_config.nDrawHeight;
	nAccWidth += m_config.KernelRadius() * 2;
	nAccHeight += m_config.KernelRadius() * 2;
	nAccWidth *= m_config.AntiAlias();
	nAccHeight *= m_config.AntiAlias();

	err = m_AccumArray.MallocPitch(nAccWidth, nAccHeight);
	if (err != cudaSuccess) return E_FAIL;

	// Create the array for the filtered and log-scaled results
	UINT nFilteredWidth = m_config.nDrawWidth * m_config.AntiAlias();
	UINT nFilteredHeight = m_config.nDrawHeight * m_config.AntiAlias();
	err = m_FilteredArray.MallocPitch(nFilteredWidth, nFilteredHeight);
	if (err != cudaSuccess) return E_FAIL;

	// Stats gathered during generation (managed memory for easy CPU access
	err = cudaMallocManaged(&m_pAccumStats, sizeof(UINT));
	if (err != cudaSuccess) return E_FAIL;
	err = cudaMemset(m_pAccumStats, 0, sizeof(UINT));
	if (err != cudaSuccess) return E_FAIL;

	// One random number generator for each thread
	err = m_IterArray.Malloc(m_nIterBlocks * m_nIterThreads);
	if (err != cudaSuccess) return E_FAIL;

	// Create the set of transforms and copy to the GPU
	err = RandomizeTransforms();
	if (err != cudaSuccess) return E_FAIL;

	// Initialize all of the generators and do a short run to find window scale
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) return E_FAIL;
	ACCUM_PARAMS paramsAccum;
	paramsAccum.bInit = TRUE;
	paramsAccum.nSteps = 128;
	paramsAccum.nThreads = m_nIterThreads;
	paramsAccum.nBlocks = m_nIterBlocks;
	ACCUM_STATS* pAccumStats = (ACCUM_STATS*)m_pAccumStats;
	pAccumStats->xMin = pAccumStats->yMin = FLT_MAX;
	pAccumStats->xMax = pAccumStats->yMax = -FLT_MAX;
	CCudaArray2D<ACCUM> arrDummy; 
	err = cuda_iterate(paramsAccum, m_TransformArray, m_IterArray, arrDummy, m_pAccumStats);
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
		m_rectScale.fScale = min((float)nFilteredWidth / dx, (float)nFilteredHeight / dy);
		m_rectScale.fOffsetX = 0.5f * (float)nFilteredWidth - m_rectScale.fScale * cx;
		m_rectScale.fOffsetY = 0.5f * (float)nFilteredHeight - m_rectScale.fScale * cy;
	}

	m_nTotalIter = m_config.iIterationLevel * (m_config.AntiAlias() * m_config.AntiAlias()) * nFilteredHeight * nFilteredWidth / 25;
	m_nCycleIter = m_nTotalIter / (16 * m_config.iIterationLevel);
	m_nIterComplete = 0;

	return hr;
}

cudaError_t CGenerator::RandomizeTransforms()
{
	cudaError_t err = cudaSuccess;

	UINT nTransforms = (UINT)random(MAXTRANSFORMS - 1) + 2;
	UINT nSymmetry = m_config.bRotation ? (randomFlip() ? random(MAXSYMMETRY) + 1 : 1) : 1;
	bool bMirror = (m_config.bMirror) ? randomFlip() : false;

	std::vector<CTransform> vecTrans;
	vecTrans.resize(nTransforms + nSymmetry - 1 + (bMirror ? 1 : 0));

	// Higher powers encourage more symmetry
	float fPow = frand();

	float fWeightPower = frand() * 2.0f;	// Higher weight powers give more even coverage
	bool bElongate = randomFlip();
	bool bSkew = randomFlip();

	float fTotalWeight = 0.0f;

	for (UINT i = 0; i < nTransforms; i++)
	{
		float fScaleFactor = 1.0f / pow(i + 1.0f, fPow);

		float fScale = frand() * fWeightPower;

		float fElong = 1.0f;
		if (bElongate)
		{
			fElong = 1.0f - frand() * fScale;
			if (randomFlip())
				fElong = 1.0f / fElong;
		}
		float fSx = fScale * fElong;
		float fSy = fScale / fElong;

		float fRot = (frand() - 0.5f) * 2.0f * 3.14169f;
		CMatrix2D mat = CMatrix2D::Rotation(fRot) * CMatrix2D(fSx, 0, 0, fSy);

		float fSkew = 0.0f;
		if (bSkew)
		{
			fSkew = (frand() - 0.5f) * (1.0f - fScale);
			mat *= CMatrix2D(1.0f, fSkew, fSkew, 1.0f) / sqrt(1.0f - fSkew * fSkew);
		}

		vecTrans[i].Matrix0() = mat;

		CVector2D vec = CVector2D(1.0f, 0.0f);
		vec = vec.Rotate(frand() * 2.0f * 3.14159f);
		float fVecLen = frand();
		vecTrans[i].Offset0() = fVecLen * vec;

		vecTrans[i].Weight() = pow(fScale, fWeightPower);
		vecTrans[i].Color() = FLOAT_COLOR(frand(), frand(), frand());

		fTotalWeight += vecTrans[i].Weight();
	}

	float fCumWeight = 0.0f;
	for (size_t i = 0; i < vecTrans.size(); i++)
	{
		fCumWeight += vecTrans[i].Weight();
		vecTrans[i].Weight() = fCumWeight / fTotalWeight;
	}

	// Copy transform information to GPU
	err = m_TransformArray.Malloc(nTransforms);
	if (err != cudaSuccess) return err;
	err = m_TransformArray.CopyFrom((PVOID)(vecTrans.data()), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) return err;

	return err;
}

HRESULT CGenerator::Iterate(BOOL bRender)
{
	HRESULT hr = S_OK;
	cudaError_t err = cudaSuccess;

	if (m_nIterComplete < m_nTotalIter)
	{
		ACCUM_PARAMS paramsAccum;
		paramsAccum.nSteps = max(1, m_nCycleIter / (m_IterArray.Length())); // m_nTotalIter / (m_IterArray.nHeight * m_IterArray.nWidth);
		paramsAccum.rect = m_rectScale;
		paramsAccum.bHitPercent = (m_nIterComplete == 0);
		paramsAccum.nThreads = m_nIterThreads;
		paramsAccum.nBlocks = m_nIterBlocks;
		err = cuda_iterate(paramsAccum, m_TransformArray, m_IterArray, m_AccumArray, m_pAccumStats);
		m_nIterComplete += paramsAccum.nSteps * m_IterArray.Length();

		ACCUM_STATS* pAccumStats = (ACCUM_STATS*)m_pAccumStats;
		float fPercent = (float)(pAccumStats->nNewHits) / (float)(pAccumStats->nHitRect);
		
		if (bRender)
		{
			CCudaTexture2D texture;
			err = m_pTexture->MapToCudaArray(texture);
			if (err != cudaSuccess) return E_FAIL;

			RENDER_PARAMS paramsRender;
			ACCUM_STATS* pAccumStats = (ACCUM_STATS*)m_pAccumStats;
			paramsRender.fLogColorScale = 1.0f / log(pAccumStats->fMaxColorElement + 1.0f);
			paramsRender.iAntiAlias = m_config.AntiAlias();
			paramsRender.iKernelRadius = m_config.KernelRadius();
			paramsRender.fFilterScale = m_config.KernelRadius() ?
				0.1f * (float)m_config.KernelRadius() * pow((float)(pAccumStats->nMaxCount), m_config.fKernelAlpha) : 0.0f;
			paramsRender.fKernelAlpha = m_config.fKernelAlpha;
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
