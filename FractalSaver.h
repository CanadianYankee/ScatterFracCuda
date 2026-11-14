#pragma once

#include "SaverBase.h"

class CGenerator;

// One fractal generator drawing into one saver window. Each instance owns its own
// CGenerator and runs its CUDA work on the GPU behind its window's D3D device.
class CFractalSaver : public CSaverBase
{
public:
	CFractalSaver();
	virtual ~CFractalSaver();

	bool OnCommand(WORD wmId) override;

protected:
	BOOL InitSaverData() override;
	BOOL IterateSaver() override;
	BOOL PauseSaver() override { return TRUE; }
	BOOL ResumeSaver() override { return TRUE; }
	BOOL OnResizeSaver() override;
	void CleanUpSaver() override;
	bool AcceptAdapter(IDXGIAdapter *pAdapter) override;

	struct VS_VARIABLES
	{
		VS_VARIABLES() : g_fXScale(1.0f), g_fYScale(1.0f), fDummy1(0), fDummy2(0) {}
		float g_fXScale;
		float g_fYScale;
		float fDummy1;
		float fDummy2;
	};

	BOOL StartGenerator();
	void UpdateAspect();
	BOOL RenderScene();

	int m_iCudaDevice;
	std::unique_ptr<CGenerator> m_pGenerator;

	ComPtr<ID3D11VertexShader> m_pVertexShader;
	ComPtr<ID3D11PixelShader> m_pPixelShader;
	VS_VARIABLES m_sVSVariables;
	ComPtr<ID3D11Buffer> m_pCBVSVariables;
};
