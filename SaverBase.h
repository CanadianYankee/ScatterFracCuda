#pragma once

#include "WindowCluster.h"

// Base class for one saver instance: one window, one D3D11 device on that window's
// monitor's adapter, and one swap chain. Ported from InteractDX11's CSaverBase.
class CSaverBase
{
public:
	CSaverBase(void);
	virtual ~CSaverBase(void);

	BOOL Initialize(const CWindowCluster::SAVER_WINDOW &myWindow, int iSaverIndex, int iNumSavers);
	bool Tick();	// Returns true if the saver did any work this pass
	bool IsActive() const { return m_bRunning && !m_bOccluded; }
	void Pause();
	void Resume();
	void ResumeResized(int cx, int cy);
	void SetSyncInterval(UINT uSyncInterval) { m_uSyncInterval = uSyncInterval; }
	void CleanUp();

	// Menu commands from the windowed-mode window. Return true if handled.
	virtual bool OnCommand(WORD wmId) { UNREFERENCED_PARAMETER(wmId); return false; }

protected:
	// Override these functions to implement your saver - return FALSE to error out and shut down
	virtual BOOL InitSaverData() = 0;
	virtual BOOL IterateSaver() = 0;
	virtual BOOL PauseSaver() = 0;
	virtual BOOL ResumeSaver() = 0;
	virtual BOOL OnResizeSaver() = 0;
	virtual void CleanUpSaver() = 0;

	// Return false if the saver can't run on this adapter. The base class then falls back
	// to the first adapter that is accepted.
	virtual bool AcceptAdapter(IDXGIAdapter *pAdapter) { UNREFERENCED_PARAMETER(pAdapter); return true; }

	// Present the back buffer; handles DXGI_STATUS_OCCLUDED. Returns FALSE on a real failure.
	BOOL PresentFrame();

private:
	HRESULT InitDirect3D();
	HRESULT GetMyAdapter(IDXGIAdapter **ppAdapter);
	BOOL OnResize();

protected:
	ComPtr<ID3D11Device> m_pD3DDevice;
	ComPtr<ID3D11DeviceContext> m_pD3DContext;
	ComPtr<IDXGISwapChain> m_pSwapChain;
	ComPtr<ID3D11RenderTargetView> m_pRenderTargetView;
	D3D11_VIEWPORT m_ScreenViewport;

	HWND m_hMyWindow;
	int m_iSaverIndex;
	int m_iNumSavers;
	int m_iClientWidth;
	int m_iClientHeight;
	bool m_bFullScreen;
	bool m_bNested;
	float m_fAspectRatio;
	bool m_bRunning;
	UINT m_uSyncInterval = 1;	// Present() sync interval; see CWindowCluster::Run

	// When nothing we present is visible (locked session, display off, window fully covered or
	// minimized) Present returns DXGI_STATUS_OCCLUDED without waiting for vsync. We then stop
	// iterating and only re-test visibility every OCCLUSION_POLL_MS.
	static const ULONGLONG OCCLUSION_POLL_MS = 250;
	bool m_bOccluded = false;
	ULONGLONG m_ullNextOcclusionTest = 0;
};
