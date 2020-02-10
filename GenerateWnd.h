#pragma once

class CDX11CudaTexture;

class CGenerateWnd
{
private:
	CGenerateWnd();
	~CGenerateWnd();

public:
	static CGenerateWnd* GetMainWnd();  // Get the singleton object
	static void DestroyMainWnd(); // Destroy the sigleton object
	static void Cleanup();

	BOOL Initialize(HINSTANCE hInstance, const WCHAR* szTitle, const WCHAR* szWindowClass, int nCmdShow);
	LRESULT WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

protected:
	HRESULT InitDirect3D();
	HRESULT InitD3DResources();
	HRESULT OnResize();
	HRESULT RenderScene();

	HWND m_hWnd;
	LONG m_iClientWidth;
	LONG m_iClientHeight;
	float m_fAspectRatio;

	ComPtr<ID3D11Device> m_pD3DDevice;
	ComPtr<ID3D11DeviceContext> m_pD3DContext;
	ComPtr<IDXGISwapChain> m_pSwapChain;
	ComPtr<ID3D11RenderTargetView> m_pRenderTargetView;

	std::unique_ptr<CDX11CudaTexture> m_pTexture;

	ComPtr<ID3D11VertexShader> m_pVertexShader;
	ComPtr<ID3D11PixelShader> m_pPixelShader;
};

