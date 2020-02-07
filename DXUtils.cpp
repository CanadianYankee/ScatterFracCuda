#include "framework.h"
#include "DXUtils.h"

std::wstring g_strBasePath;

void FindPaths()
{
	TCHAR path[_MAX_PATH];
	TCHAR drive[_MAX_DRIVE];
	TCHAR dir[_MAX_DIR];
	TCHAR fname[_MAX_FNAME];
	TCHAR ext[_MAX_EXT];
	GetModuleFileName(NULL, path, _MAX_PATH);
	_tsplitpath_s(path, drive, _MAX_DRIVE, dir, _MAX_DIR, fname, _MAX_FNAME, ext, _MAX_EXT);
	_tmakepath_s(path, _MAX_PATH, drive, dir, NULL, NULL);
	g_strBasePath = path;
}

HRESULT DXUtils::LoadShader(ComPtr<ID3D11Device>& pD3DDevice, ShaderType type, const std::wstring& strFileName, ComPtr<ID3D11ClassLinkage> pClassLinkage, ComPtr<ID3D11DeviceChild>* pShader, VS_INPUTLAYOUTSETUP* pILS)
{
	HRESULT hr = S_OK;

	if (g_strBasePath.empty())
	{
		FindPaths();
	}

	std::wstring strFullPath = g_strBasePath + strFileName;

	std::vector<char> buffer;
	hr = DXUtils::LoadBinaryFile(strFullPath, buffer);

	if (SUCCEEDED(hr))
	{
		switch (type)
		{
		case ShaderType::VertexShader:
		{
			ComPtr<ID3D11VertexShader> pVertexShader;
			hr = pD3DDevice->CreateVertexShader(buffer.data(), buffer.size(), pClassLinkage.Get(), &pVertexShader);
			if (SUCCEEDED(hr) && (pILS))
			{
				hr = pD3DDevice->CreateInputLayout(pILS->pInputDesc, pILS->NumElements,
					buffer.data(), buffer.size(), &pILS->pInputLayout);
			}
			if (SUCCEEDED(hr))
			{
				hr = pVertexShader.As<ID3D11DeviceChild>(pShader);
			}
			break;
		}

		case ShaderType::PixelShader:
		{
			ComPtr<ID3D11PixelShader> pPixelShader;
			hr = pD3DDevice->CreatePixelShader(buffer.data(), buffer.size(), pClassLinkage.Get(), &pPixelShader);
			if (SUCCEEDED(hr))
			{
				hr = pPixelShader.As<ID3D11DeviceChild>(pShader);
			}
			break;
		}

		case ShaderType::GeometryShader:
		{
			ComPtr<ID3D11GeometryShader> pGeometryShader;
			hr = pD3DDevice->CreateGeometryShader(buffer.data(), buffer.size(), pClassLinkage.Get(), &pGeometryShader);
			if (SUCCEEDED(hr))
			{
				hr = pGeometryShader.As<ID3D11DeviceChild>(pShader);
			}
			break;
		}

		case ShaderType::ComputeShader:
		{
			ComPtr<ID3D11ComputeShader> pComputeShader;
			hr = pD3DDevice->CreateComputeShader(buffer.data(), buffer.size(), pClassLinkage.Get(), &pComputeShader);
			if (SUCCEEDED(hr))
			{
				hr = pComputeShader.As<ID3D11DeviceChild>(pShader);
			}
			break;
		}

		case ShaderType::HullShader:
		{
			ComPtr<ID3D11HullShader> pHullShader;
			hr = pD3DDevice->CreateHullShader(buffer.data(), buffer.size(), pClassLinkage.Get(), &pHullShader);
			if (SUCCEEDED(hr))
			{
				hr = pHullShader.As<ID3D11DeviceChild>(pShader);
			}
			break;
		}

		case ShaderType::DomainShader:
		{
			ComPtr<ID3D11DomainShader> pDomainShader;
			hr = pD3DDevice->CreateDomainShader(buffer.data(), buffer.size(), pClassLinkage.Get(), &pDomainShader);
			if (SUCCEEDED(hr))
			{
				hr = pDomainShader.As<ID3D11DeviceChild>(pShader);
			}
			break;
		}
		}
	}

	return hr;
}

HRESULT DXUtils::LoadBinaryFile(const std::wstring& strPath, std::vector<char>& buffer)
{
	HRESULT hr = S_OK;

	std::ifstream fin(strPath, std::ios::binary);

	fin.seekg(0, std::ios_base::end);
	int size = (int)fin.tellg();
	if (size > 0)
	{
		fin.seekg(0, std::ios_base::beg);
		buffer.resize(size);
		fin.read(buffer.data(), size);
	}
	else
	{
		hr = E_FAIL;
	}
	fin.close();

	return hr;
}
