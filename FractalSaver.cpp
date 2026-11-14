#include "framework.h"
#include "Resource.h"
#include "DXUtils.h"
#include "FractalSaver.h"
#include "Generator.h"

CFractalSaver::CFractalSaver() :
	m_iCudaDevice(0)
{
}

CFractalSaver::~CFractalSaver()
{
	CleanUpSaver();
}

// CUDA/D3D11 interop only works when the D3D device lives on a CUDA-capable GPU.
bool CFractalSaver::AcceptAdapter(IDXGIAdapter *pAdapter)
{
	int iDevice = 0;
	if(cudaD3D11GetDevice(&iDevice, pAdapter) != cudaSuccess)
	{
		cudaGetLastError();	// Clear the error so later cudaGetLastError checks don't see it
		return false;
	}
	return true;
}

BOOL CFractalSaver::InitSaverData()
{
	HRESULT hr = S_OK;

	// Run all of this saver's CUDA work on the GPU that owns its D3D device. With several
	// monitors on different GPUs, each saver has a different device; IterateSaver selects it.
	ComPtr<IDXGIDevice> pDXGIDevice;
	ComPtr<IDXGIAdapter> pAdapter;
	hr = m_pD3DDevice.As(&pDXGIDevice);
	if(SUCCEEDED(hr))
		hr = pDXGIDevice->GetAdapter(&pAdapter);
	if(FAILED(hr)) return FALSE;
	if(cudaD3D11GetDevice(&m_iCudaDevice, pAdapter.Get()) != cudaSuccess) return FALSE;
	if(cudaSetDevice(m_iCudaDevice) != cudaSuccess) return FALSE;

	// Load the Vertex and Pixel shaders
	ComPtr<ID3D11DeviceChild> pShader;
	hr = DXUtils::LoadShader(m_pD3DDevice, DXUtils::ShaderType::VertexShader, L"TextureVS.cso", nullptr, &pShader);
	if(SUCCEEDED(hr))
		hr = pShader.As<ID3D11VertexShader>(&m_pVertexShader);
	if(FAILED(hr)) return FALSE;
	D3DDEBUGNAME(m_pVertexShader, "Vertex Shader");

	hr = DXUtils::LoadShader(m_pD3DDevice, DXUtils::ShaderType::PixelShader, L"TexturePS.cso", nullptr, &pShader);
	if(SUCCEEDED(hr))
		hr = pShader.As<ID3D11PixelShader>(&m_pPixelShader);
	if(FAILED(hr)) return FALSE;
	D3DDEBUGNAME(m_pPixelShader, "Pixel Shader");

	// Constant buffer for the vertex shader
	D3D11_SUBRESOURCE_DATA cbData;
	cbData.pSysMem = &m_sVSVariables;
	cbData.SysMemPitch = 0;
	cbData.SysMemSlicePitch = 0;
	D3D11_BUFFER_DESC Desc;
	Desc.Usage = D3D11_USAGE_DYNAMIC;
	Desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	Desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	Desc.MiscFlags = 0;
	Desc.StructureByteStride = 0;
	Desc.ByteWidth = sizeof(VS_VARIABLES);
	hr = m_pD3DDevice->CreateBuffer(&Desc, &cbData, &m_pCBVSVariables);
	if(FAILED(hr)) return FALSE;
	D3DDEBUGNAME(m_pCBVSVariables, "Vertex Shader CB");

	// The generator itself is created on the first tick
	return TRUE;
}

BOOL CFractalSaver::IterateSaver()
{
	if(cudaSetDevice(m_iCudaDevice) != cudaSuccess)
		return FALSE;

	if(!m_pGenerator || m_pGenerator->RestartNeeded())
	{
		return StartGenerator();
	}

	if(m_pGenerator->IsIncomplete())
	{
		if(FAILED(m_pGenerator->Iterate()))
		{
			OutputDebugString(_T("CGenerator::Iterate failed\n"));
			return FALSE;
		}
		return RenderScene();
	}

	// Finished; a new fractal starts on the next tick
	m_pGenerator.reset();
	return TRUE;
}

BOOL CFractalSaver::StartGenerator()
{
	// Free the old generator's GPU memory before allocating the new one
	m_pGenerator.reset();

	CONFIG_DATA config;
	if(m_bFullScreen || m_bNested || !config.nDrawWidth || !config.nDrawHeight)
	{
		// The saver and preview draw at the window's native resolution, which also matches
		// the monitor's aspect ratio. Windowed (debug) mode uses the configured size.
		config.nDrawWidth = m_iClientWidth;
		config.nDrawHeight = m_iClientHeight;
	}
	if(m_bNested)
	{
		// The preview is so small that its iteration total would be used up in a few frames.
		// Give it the frame count CGenerator aims for at full size (16 cycles per level).
		config.iMinFrames = 16 * config.iIterationLevel;
	}

	m_pGenerator = std::make_unique<CGenerator>(config);
	if(FAILED(m_pGenerator->Initialize(m_pD3DDevice)))
	{
		OutputDebugString(_T("CGenerator::Initialize failed\n"));
		m_pGenerator.reset();
		return FALSE;
	}

	UpdateAspect();
	return TRUE;
}

// Letterbox the generator's texture into the window
void CFractalSaver::UpdateAspect()
{
	if(!m_pGenerator || !m_pCBVSVariables)
		return;

	float fTexAspect = m_pGenerator->DrawAspectRatio();
	m_sVSVariables.g_fXScale = m_sVSVariables.g_fYScale = 1.0f;
	if(m_fAspectRatio > fTexAspect)
	{
		m_sVSVariables.g_fXScale = fTexAspect / m_fAspectRatio;
	}
	else if(m_fAspectRatio < fTexAspect)
	{
		m_sVSVariables.g_fYScale = m_fAspectRatio / fTexAspect;
	}

	D3D11_MAPPED_SUBRESOURCE MappedResource;
	if(SUCCEEDED(m_pD3DContext->Map(m_pCBVSVariables.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource)))
	{
		CopyMemory(MappedResource.pData, &m_sVSVariables, sizeof(VS_VARIABLES));
		m_pD3DContext->Unmap(m_pCBVSVariables.Get(), 0);
	}
}

BOOL CFractalSaver::OnResizeSaver()
{
	UpdateAspect();
	return TRUE;
}

BOOL CFractalSaver::RenderScene()
{
	const float background[4] = { 0.0f, 0.0f, 0.0f, 1.0f };

	m_pD3DContext->ClearRenderTargetView(m_pRenderTargetView.Get(), background);

	if(m_pGenerator)
	{
		m_pD3DContext->IASetInputLayout(nullptr);
		m_pD3DContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

		m_pD3DContext->VSSetShader(m_pVertexShader.Get(), nullptr, 0);
		m_pD3DContext->VSSetConstantBuffers(0, 1, m_pCBVSVariables.GetAddressOf());

		m_pD3DContext->PSSetShader(m_pPixelShader.Get(), nullptr, 0);

		m_pGenerator->LoadDrawPS(m_pD3DContext);

		m_pD3DContext->Draw(4, 0);
	}

	return PresentFrame();
}

bool CFractalSaver::OnCommand(WORD wmId)
{
	if(wmId == IDM_FILE_NEW)
	{
		// Drop the current fractal; the next tick starts a new one
		cudaSetDevice(m_iCudaDevice);
		m_pGenerator.reset();
		return true;
	}

	return false;
}

void CFractalSaver::CleanUpSaver()
{
	// The generator unregisters its texture from CUDA, so it must go before the D3D device
	if(m_pGenerator)
	{
		cudaSetDevice(m_iCudaDevice);
		m_pGenerator.reset();
	}
}
