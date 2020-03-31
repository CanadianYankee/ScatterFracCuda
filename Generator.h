#pragma once

#include "ConfigData.h"
#include "DX11CudaTexture.h"
#include "AccumData.h"
#include "CudaArray.h"
#include "Transform.h"
#include "Randomizer.h"

class CGenerator : private CRandomizer
{
public:
	CGenerator(const CONFIG_DATA &config);
	~CGenerator();

	HRESULT Initialize(ComPtr<ID3D11Device> pD3DDevice);
	cudaError_t RandomizeTransforms();
	HRESULT Iterate(BOOL bRender = TRUE);
	bool IsIncomplete() { return m_nIterComplete < m_nTotalIter; }
	bool RestartNeeded() { return m_bRestartNeeded;  }
	float DrawAspectRatio() { return m_pTexture->AspectRatio(); }
	void LoadDrawPS(ComPtr<ID3D11DeviceContext> pD3DContext) { m_pTexture->LoadPS(pD3DContext); } 

protected:
	void RandomAffine(float fScaleFactor, float fScale, bool bElongate, bool bSkew, CMatrix2D& matTrans, CVector2D& vecOffset);

	CONFIG_DATA m_config;
	const UINT MAXSYMMETRY = 12;
	const UINT MAXTRANSFORMS = 10;
	std::unique_ptr<CDX11CudaTexture> m_pTexture;
	CCudaArray2D<ACCUM> m_AccumArray;
	CCudaArray2D<FILTERED> m_FilteredArray;
	CCudaArray1D<ITERATOR> m_IterArray;
	CCudaArray1D<CTransform> m_TransformArray;
	PVOID m_pAccumStats;
	bool m_bRestartNeeded;
	RECT_SCALE m_rectScale;
	const UINT m_nIterThreads = 128;
	const UINT m_nIterBlocks = 128;
	UINT m_nTotalIter;
	UINT m_nCycleIter;
	UINT m_nIterComplete;
};
