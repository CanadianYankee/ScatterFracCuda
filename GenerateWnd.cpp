#include "framework.h"
#include "Resource.h"
#include "DXUtils.h"
#include "GenerateWnd.h"
#include "DX11CudaTexture.h"
#include "DXUtils.h"

CGenerateWnd* g_pMainWnd = nullptr;

//
//  FUNCTION: WndProc(HWND, UINT, WPARAM, LPARAM)
//
//  PURPOSE: Processes messages for the main window.
//
//  WM_COMMAND  - process the application menu
//  WM_PAINT    - Paint the main window
//  WM_DESTROY  - post a quit message and return
//
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    return g_pMainWnd->WndProc(hWnd, message, wParam, lParam);
}

CGenerateWnd::CGenerateWnd() :
    m_hWnd(NULL)
    , m_iClientWidth(0)
    , m_iClientHeight(0)
    , m_fAspectRatio(1.0f)
{}

CGenerateWnd::~CGenerateWnd()
{}

CGenerateWnd* CGenerateWnd::GetMainWnd()
{
    if (!g_pMainWnd)
    {
        g_pMainWnd = new CGenerateWnd();
    }

    return g_pMainWnd;
}

void CGenerateWnd::DestroyMainWnd()
{
	if (g_pMainWnd)
	{
		delete g_pMainWnd;
		g_pMainWnd = nullptr;
	}
}

BOOL CGenerateWnd::Initialize(HINSTANCE hInstance, const WCHAR* szTitle, const WCHAR* szWindowClass, int nCmdShow)
{
    HRESULT hr = S_OK;

    WNDCLASSEXW wcex;

    wcex.cbSize = sizeof(WNDCLASSEX);

    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = ::WndProc;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;
    wcex.hInstance = hInstance;
    wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_SCATTERFRACCUDA));
    wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wcex.lpszMenuName = MAKEINTRESOURCEW(IDC_SCATTERFRACCUDA);
    wcex.lpszClassName = szWindowClass;
    wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

    RegisterClassExW(&wcex);

    m_hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, nullptr, nullptr, hInstance, nullptr);

    if (!m_hWnd)
        return FALSE;

    ShowWindow(m_hWnd, nCmdShow);
    RECT rcClient;
    GetClientRect(m_hWnd, &rcClient);
    m_iClientWidth = rcClient.right - rcClient.left;
    m_iClientHeight = rcClient.bottom - rcClient.top;
    m_fAspectRatio = (float)m_iClientWidth / (float)m_iClientHeight;

    hr = InitDirect3D();
    if (FAILED(hr)) return FALSE;

    hr = InitD3DResources();
    if (FAILED(hr)) return FALSE;

//    hr = InitWorld();
//    if (FAILED(hr)) return FALSE;

    // Do the CUDA calculation
//    hr = DoCUDA();
//    if (FAILED(hr)) return FALSE;

    UpdateWindow(m_hWnd);

    return SUCCEEDED(hr);
}

HRESULT CGenerateWnd::InitDirect3D()
{
    HRESULT hr = S_OK;

    UINT createDeviceFlags = 0;
#if defined(DEBUG) || defined(_DEBUG)  
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    // Fill out a DXGI_SWAP_CHAIN_DESC to describe our swap chain.
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
    sd.OutputWindow = m_hWnd;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    sd.Flags = 0;

    D3D_FEATURE_LEVEL featureLevel;
    hr = D3D11CreateDeviceAndSwapChain(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL,
        createDeviceFlags, NULL, 0, D3D11_SDK_VERSION, &sd, &m_pSwapChain,
        &m_pD3DDevice, &featureLevel, &m_pD3DContext);

    if (SUCCEEDED(hr))
    {
        D3DDEBUGNAME(m_pD3DDevice, "Device");
        D3DDEBUGNAME(m_pD3DContext, "Device Context");
        D3DDEBUGNAME(m_pSwapChain, "Swap Chain");
        if (featureLevel != D3D_FEATURE_LEVEL_11_0)
            hr = E_FAIL;
    }
    assert(SUCCEEDED(hr));

    return OnResize();
}

HRESULT CGenerateWnd::InitD3DResources()
{
	HRESULT hr = S_OK;

	m_pTexture = std::make_unique<CDX11CudaTexture>(640, 400);
	hr = m_pTexture->Initialize(m_pD3DDevice);
	if (FAILED(hr)) return hr;

	// Load the Vertex and Pixel shaders
	ComPtr<ID3D11DeviceChild> pShader;
	hr = DXUtils::LoadShader(m_pD3DDevice, DXUtils::ShaderType::VertexShader, L"TextureVS.cso", nullptr, &pShader);
	if (SUCCEEDED(hr))
	{
		hr = pShader.As<ID3D11VertexShader>(&m_pVertexShader);
	}
	if (FAILED(hr)) return hr;
	D3DDEBUGNAME(m_pVertexShader, "Vertex Shader");

	hr = DXUtils::LoadShader(m_pD3DDevice, DXUtils::ShaderType::PixelShader, L"TexturePS.cso", nullptr, &pShader);
	if (SUCCEEDED(hr))
	{
		hr = pShader.As<ID3D11PixelShader>(&m_pPixelShader);
	}
	if (FAILED(hr)) return hr;
	D3DDEBUGNAME(m_pPixelShader, "Pixel Shader");

	return hr;
}

HRESULT CGenerateWnd::OnResize()
{
    HRESULT hr = S_OK;

    if (m_pSwapChain)
    {
        // Release the old views, as they hold references to the buffers we
        // will be destroying. 
        m_pRenderTargetView = nullptr;

        // Resize the swap chain and recreate the render target view.
        hr = m_pSwapChain->ResizeBuffers(1, m_iClientWidth, m_iClientHeight, DXGI_FORMAT_R8G8B8A8_UNORM, 0);

        if (SUCCEEDED(hr))
        {
            ComPtr<ID3D11Texture2D> pBackBuffer;
            hr = m_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), &pBackBuffer);
            if (SUCCEEDED(hr))
            {
                hr = m_pD3DDevice->CreateRenderTargetView(pBackBuffer.Get(), 0, &m_pRenderTargetView);
            }
        }

        if (SUCCEEDED(hr))
        {
            // Bind the render target view to the pipeline.
            m_pD3DContext->OMSetRenderTargets(1, m_pRenderTargetView.GetAddressOf(), nullptr);

            // Set the viewport transform.
            D3D11_VIEWPORT vp;
            vp.TopLeftX = 0;
            vp.TopLeftY = 0;
            vp.Width = static_cast<float>(m_iClientWidth);
            vp.Height = static_cast<float>(m_iClientHeight);
            vp.MinDepth = 0.0f;
            vp.MaxDepth = 1.0f;

            m_pD3DContext->RSSetViewports(1, &vp);
        }
    }

    assert(SUCCEEDED(hr));

    return hr;
}

HRESULT CGenerateWnd::RenderScene()
{
    HRESULT hr = S_OK;

    if (!m_pD3DContext || !m_pSwapChain)
    {
        assert(false);
        return FALSE;
    }

    const float background[4] = { 1.0f, 0.0f, 1.0f, 1.0f };

    m_pD3DContext->ClearRenderTargetView(m_pRenderTargetView.Get(), background);

    m_pD3DContext->IASetInputLayout(0);
    m_pD3DContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

    m_pD3DContext->VSSetShader(m_pVertexShader.Get(), NULL, 0);

	m_pD3DContext->PSSetShader(m_pPixelShader.Get(), NULL, 0);

	m_pTexture->LoadPS(m_pD3DContext);

    m_pD3DContext->Draw(4, 0);

    hr = m_pSwapChain->Present(1, 0);

    return hr;
}

void CGenerateWnd::Cleanup()
{
    if (g_pMainWnd)
    {
        delete g_pMainWnd;
        g_pMainWnd = nullptr;
    }
}

LRESULT CGenerateWnd::WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{

    switch (message)
    {
    case WM_COMMAND:
    {
        int wmId = LOWORD(wParam);
        // Parse the menu selections:
        switch (wmId)
        {
        case IDM_EXIT:
            DestroyWindow(hWnd);
            break;
        default:
            return DefWindowProc(hWnd, message, wParam, lParam);
        }
    }
    break;

    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hWnd, &ps);
        RenderScene();
        EndPaint(hWnd, &ps);
    }
    break;

    case WM_SIZE:
    {
        // Save the new client area dimensions.
        RECT rcClient;
        GetClientRect(m_hWnd, &rcClient);
        m_iClientWidth = rcClient.right - rcClient.left;
        m_iClientHeight = rcClient.bottom - rcClient.top;
        m_fAspectRatio = (float)m_iClientWidth / (float)m_iClientHeight;

        OnResize();
    }
    break;

    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}
