#include "framework.h"
#include "DXUtils.h"
#include "SaverBase.h"

CSaverBase::CSaverBase(void) :
	m_hMyWindow(NULL),
	m_iSaverIndex(0),
	m_iNumSavers(0),
	m_iClientWidth(0),
	m_iClientHeight(0),
	m_bFullScreen(false),
	m_bNested(false),
	m_fAspectRatio(1.0f),
	m_bRunning(false)
{
	ZeroMemory(&m_ScreenViewport, sizeof(m_ScreenViewport));
}

CSaverBase::~CSaverBase(void)
{
}

BOOL CSaverBase::Initialize(const CWindowCluster::SAVER_WINDOW &myWindow, int iSaverIndex, int iNumSavers)
{
	if(!myWindow.hWnd)
	{
		assert(false);
		return FALSE;
	}

	m_hMyWindow = myWindow.hWnd;
	m_iSaverIndex = iSaverIndex;
	m_iNumSavers = iNumSavers;
	// Size the back buffer from the window's actual client area (physical pixels, since the
	// process is Per-Monitor DPI aware) rather than the requested size, which can differ.
	RECT rcClient = {};
	if(GetClientRect(m_hMyWindow, &rcClient) && rcClient.right > rcClient.left && rcClient.bottom > rcClient.top)
	{
		m_iClientWidth = rcClient.right - rcClient.left;
		m_iClientHeight = rcClient.bottom - rcClient.top;
	}
	else
	{
		m_iClientWidth = max(1L, myWindow.sClientArea.cx);
		m_iClientHeight = max(1L, myWindow.sClientArea.cy);
	}
	m_fAspectRatio = (float)m_iClientWidth / (float)m_iClientHeight;
	m_bFullScreen = myWindow.bSaverMode;
	m_bNested = myWindow.bNested;

	HRESULT hr = InitDirect3D();

	BOOL bSuccess = SUCCEEDED(hr);
	if(bSuccess)
	{
		bSuccess = InitSaverData();
		if(bSuccess)
			m_bRunning = true;
	}

	return bSuccess;
}

// Create D3D device, context, and swap chain
HRESULT CSaverBase::InitDirect3D()
{
	HRESULT hr = S_OK;

	UINT createDeviceFlags = 0;
#if defined(DEBUG) || defined(_DEBUG)
	createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

	ComPtr<IDXGIAdapter> pAdapter;
	hr = GetMyAdapter(&pAdapter);
	if(SUCCEEDED(hr))
	{
		D3D_FEATURE_LEVEL featureLevel;
		hr = D3D11CreateDevice(pAdapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, NULL,
			createDeviceFlags, NULL, 0, D3D11_SDK_VERSION, &m_pD3DDevice, &featureLevel,
			&m_pD3DContext);

		if(SUCCEEDED(hr))
		{
			D3DDEBUGNAME(m_pD3DDevice, "Device");
			D3DDEBUGNAME(m_pD3DContext, "Device Context");
			if(featureLevel < D3D_FEATURE_LEVEL_11_0)
				hr = E_FAIL;
		}
		assert(SUCCEEDED(hr));
	}

	if(SUCCEEDED(hr))
	{
		DXGI_SWAP_CHAIN_DESC sd;
		ZeroMemory(&sd, sizeof(sd));
		sd.BufferDesc.Width = m_iClientWidth;
		sd.BufferDesc.Height = m_iClientHeight;
		sd.BufferDesc.RefreshRate.Numerator = 60;
		sd.BufferDesc.RefreshRate.Denominator = 1;
		sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		sd.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
		sd.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
		sd.SampleDesc.Count = 1;
		sd.SampleDesc.Quality = 0;
		sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
		sd.BufferCount = 1;
		sd.OutputWindow = m_hMyWindow;
		sd.Windowed = TRUE;
		sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
		sd.Flags = 0;

		ComPtr<IDXGIFactory> dxgiFactory;
		hr = pAdapter->GetParent(__uuidof(IDXGIFactory), &dxgiFactory);

		if(SUCCEEDED(hr))
		{
			hr = dxgiFactory->CreateSwapChain(m_pD3DDevice.Get(), &sd, &m_pSwapChain);
		}

		if(SUCCEEDED(hr))
		{
			D3DDEBUGNAME(m_pSwapChain, "Swap Chain");
			hr = dxgiFactory->MakeWindowAssociation(m_hMyWindow, DXGI_MWA_NO_WINDOW_CHANGES);
		}
	}

	if(SUCCEEDED(hr))
	{
		hr = OnResize() ? S_OK : E_FAIL;
	}

	return hr;
}

// Look up the DXGI adapter that drives the monitor hosting our window. If the saver can't
// run on that adapter (see AcceptAdapter), fall back to the first adapter it accepts;
// Windows copies the frames across adapters when presenting.
HRESULT CSaverBase::GetMyAdapter(IDXGIAdapter **ppAdapter)
{
	*ppAdapter = nullptr;

	assert(m_hMyWindow);
	HMONITOR hMonitor = MonitorFromWindow(m_hMyWindow, MONITOR_DEFAULTTOPRIMARY);
	MONITORINFOEX monitorInfo;
	ZeroMemory(&monitorInfo, sizeof(monitorInfo));
	monitorInfo.cbSize = sizeof(monitorInfo);
	GetMonitorInfo(hMonitor, &monitorInfo);

	ComPtr<IDXGIFactory> pFactory;
	HRESULT hr = CreateDXGIFactory(__uuidof(IDXGIFactory), &pFactory);
	if(FAILED(hr))
		return hr;

	ComPtr<IDXGIAdapter> pMonitorAdapter;
	ComPtr<IDXGIAdapter> pFallbackAdapter;
	ComPtr<IDXGIAdapter> pAdapter;
	for(UINT iAdapter = 0; !pMonitorAdapter && SUCCEEDED(pFactory->EnumAdapters(iAdapter, &pAdapter)); iAdapter++)
	{
		if(!AcceptAdapter(pAdapter.Get()))
			continue;

		if(!pFallbackAdapter)
			pFallbackAdapter = pAdapter;

		ComPtr<IDXGIOutput> pOutput;
		for(UINT iOutput = 0; SUCCEEDED(pAdapter->EnumOutputs(iOutput, &pOutput)); iOutput++)
		{
			DXGI_OUTPUT_DESC outputDesc;
			if(SUCCEEDED(pOutput->GetDesc(&outputDesc)) && (_tcscmp(outputDesc.DeviceName, monitorInfo.szDevice) == 0))
			{
				pMonitorAdapter = pAdapter;
				break;
			}
		}
	}

	if(pMonitorAdapter)
	{
		*ppAdapter = pMonitorAdapter.Detach();
		return S_OK;
	}
	if(pFallbackAdapter)
	{
		OutputDebugString(_T("Monitor's adapter not usable; using fallback adapter\n"));
		*ppAdapter = pFallbackAdapter.Detach();
		return S_OK;
	}

	return E_FAIL;
}

bool CSaverBase::Tick()
{
	if(!m_bRunning)
		return false;

	if(m_bOccluded)
	{
		// Cheaply check whether we've become visible again, without rendering anything.
		ULONGLONG ullNow = GetTickCount64();
		if(ullNow < m_ullNextOcclusionTest)
			return false;

		if(m_pSwapChain->Present(0, DXGI_PRESENT_TEST) == DXGI_STATUS_OCCLUDED)
		{
			m_ullNextOcclusionTest = ullNow + OCCLUSION_POLL_MS;
			return false;
		}

		m_bOccluded = false;
	}

	if(!IterateSaver())
	{
		PostMessage(m_hMyWindow, WM_DESTROY, 0, 0);
		return false;
	}

	return true;
}

BOOL CSaverBase::PresentFrame()
{
	HRESULT hr = m_pSwapChain->Present(m_uSyncInterval, 0);

	if(hr == DXGI_STATUS_OCCLUDED)
	{
		// Not a failure: nothing was displayed. Stop iterating until visible again.
		m_bOccluded = true;
		m_ullNextOcclusionTest = GetTickCount64() + OCCLUSION_POLL_MS;
		return TRUE;
	}

	return SUCCEEDED(hr);
}

void CSaverBase::Pause()
{
	if(m_bRunning)
	{
		if(!PauseSaver())
		{
			PostMessage(m_hMyWindow, WM_DESTROY, 0, 0);
		}
		m_bRunning = false;
	}
}

void CSaverBase::Resume()
{
	if(!m_bRunning)
	{
		if(!ResumeSaver())
		{
			PostMessage(m_hMyWindow, WM_DESTROY, 0, 0);
		}
		m_bRunning = true;
	}
}

void CSaverBase::ResumeResized(int cx, int cy)
{
	if(m_bRunning)
		Pause();

	m_iClientWidth = max(1, cx);
	m_iClientHeight = max(1, cy);
	m_fAspectRatio = (float)m_iClientWidth / (float)m_iClientHeight;

	if(OnResize())
	{
		Resume();
	}
	else
	{
		PostMessage(m_hMyWindow, WM_DESTROY, 0, 0);
	}
}

// Create/recreate the render target based on the new size
BOOL CSaverBase::OnResize()
{
	HRESULT hr = S_OK;

	if(m_pSwapChain)
	{
		// Release the old view, as it holds a reference to the buffer we will be destroying.
		m_pRenderTargetView = nullptr;

		hr = m_pSwapChain->ResizeBuffers(1, m_iClientWidth, m_iClientHeight, DXGI_FORMAT_R8G8B8A8_UNORM, 0);

		if(SUCCEEDED(hr))
		{
			ComPtr<ID3D11Texture2D> pBackBuffer;
			hr = m_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), &pBackBuffer);
			if(SUCCEEDED(hr))
			{
				hr = m_pD3DDevice->CreateRenderTargetView(pBackBuffer.Get(), 0, &m_pRenderTargetView);
			}
		}

		if(SUCCEEDED(hr))
		{
			m_pD3DContext->OMSetRenderTargets(1, m_pRenderTargetView.GetAddressOf(), nullptr);

			m_ScreenViewport.TopLeftX = 0;
			m_ScreenViewport.TopLeftY = 0;
			m_ScreenViewport.Width = static_cast<float>(m_iClientWidth);
			m_ScreenViewport.Height = static_cast<float>(m_iClientHeight);
			m_ScreenViewport.MinDepth = 0.0f;
			m_ScreenViewport.MaxDepth = 1.0f;

			m_pD3DContext->RSSetViewports(1, &m_ScreenViewport);
		}
	}

	assert(SUCCEEDED(hr));
	if(FAILED(hr))
		return FALSE;

	return OnResizeSaver();
}

void CSaverBase::CleanUp()
{
	if(m_bRunning)
		Pause();

	CleanUpSaver();
}
