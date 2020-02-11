#pragma once

#include "ConfigData.h"
#include "DX11CudaTexture.h"

class CGenerator
{
public:
	CGenerator(const CONFIG_DATA &config);
	~CGenerator();

	HRESULT Initialize(ComPtr<ID3D11Device> pD3DDevice);

	float DrawAspectRatio() { return m_pTexture->AspectRatio(); }
	void LoadDrawPS(ComPtr<ID3D11DeviceContext> pD3DContext) { m_pTexture->LoadPS(pD3DContext); } 

protected:
	CONFIG_DATA m_config;
	std::unique_ptr<CDX11CudaTexture> m_pTexture;
	size_t m_nAccumWidth;
	size_t m_nAccumHeight;
	size_t m_nAccumPitch;
	size_t m_nAccumMargin;
	PVOID m_pAccumArray;
};
