#pragma once

#include "ConfigData.h"
#include "DX11CudaTexture.h"
#include "AccumData.h"
#include "CudaArray.h"

class CGenerator
{
public:
	CGenerator(const CONFIG_DATA &config);
	~CGenerator();

	HRESULT Initialize(ComPtr<ID3D11Device> pD3DDevice, BOOL &bFailed);
	HRESULT Iterate(BOOL bRender = TRUE);
	bool IsIncomplete() { return m_nIterComplete < m_nTotalIter; }
	float DrawAspectRatio() { return m_pTexture->AspectRatio(); }
	void LoadDrawPS(ComPtr<ID3D11DeviceContext> pD3DContext) { m_pTexture->LoadPS(pD3DContext); } 

protected:
	CONFIG_DATA m_config;
	std::unique_ptr<CDX11CudaTexture> m_pTexture;
	CCudaArray2D<ACCUM> m_AccumArray;
	CCudaArray2D<FILTERED> m_FilteredArray;
	CCudaArray1D<ITERATOR> m_IterArray;
	PVOID m_pAccumStats;
	RECT_SCALE m_rectScale;
	const UINT m_nIterThreads = 128;
	const UINT m_nIterBlocks = 128;
	UINT m_nTotalIter;
	UINT m_nCycleIter;
	UINT m_nIterComplete;
};
