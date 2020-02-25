#pragma once

#include "ConfigData.h"
#include "DX11CudaTexture.h"
#include "AccumData.h"

class CGenerator
{
public:
	CGenerator(const CONFIG_DATA &config);
	~CGenerator();

	HRESULT Initialize(ComPtr<ID3D11Device> pD3DDevice, BOOL &bFailed);
	HRESULT Iterate(BOOL bRender = TRUE);
	float DrawAspectRatio() { return m_pTexture->AspectRatio(); }
	void LoadDrawPS(ComPtr<ID3D11DeviceContext> pD3DContext) { m_pTexture->LoadPS(pD3DContext); } 

protected:
	CONFIG_DATA m_config;
	std::unique_ptr<CDX11CudaTexture> m_pTexture;
	GPU_ARRAY_2D m_AccumArray;
	GPU_ARRAY_2D m_IterArray;
	PVOID m_pAccumStats;
	RECT_SCALE m_rectScale;
	const UINT m_nAccumThreads = 128;
	const UINT m_nAccumBlocks = 128;
	UINT m_nTotalIter;
};
