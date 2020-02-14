#pragma once

#include "ConfigData.h"
#include "DX11CudaTexture.h"
#include "AccumData.h"

class CGenerator
{
public:
	CGenerator(const CONFIG_DATA &config);
	~CGenerator();

	HRESULT Initialize(ComPtr<ID3D11Device> pD3DDevice);
	HRESULT Iterate(BOOL bRender = TRUE);
	float DrawAspectRatio() { return m_pTexture->AspectRatio(); }
	void LoadDrawPS(ComPtr<ID3D11DeviceContext> pD3DContext) { m_pTexture->LoadPS(pD3DContext); } 

protected:
	CONFIG_DATA m_config;
	std::unique_ptr<CDX11CudaTexture> m_pTexture;
	GPU_ARRAY_2D m_AccumArray;
	GPU_ARRAY_2D m_Randgen;
	PVOID m_pAccumStats;
	const UINT m_nAccumThreads = 16;
	const UINT m_nAccumBlocks = 16;
};
