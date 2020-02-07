#pragma once

#if (defined(DEBUG) || defined(_DEBUG))
#define D3DDEBUGNAME(pobj, name) pobj->SetPrivateData(WKPDID_D3DDebugObjectName, lstrlenA(name), name)
#else
#define D3DDEBUGNAME(pobj, name)
#endif 

class DXUtils
{
public:
	enum class ShaderType
	{
		VertexShader, PixelShader, GeometryShader, ComputeShader, HullShader, DomainShader
	};

	struct VS_INPUTLAYOUTSETUP
	{
		const D3D11_INPUT_ELEMENT_DESC* pInputDesc;
		UINT NumElements;
		ID3D11InputLayout* pInputLayout;
	};

	static HRESULT LoadShader(ComPtr<ID3D11Device>& pD3DDevice, ShaderType type, const std::wstring& strFileName, ComPtr<ID3D11ClassLinkage> pClassLinkage, ComPtr<ID3D11DeviceChild>* pShader, VS_INPUTLAYOUTSETUP* pILS = NULL);

private:
	static HRESULT LoadBinaryFile(const std::wstring& strPath, std::vector<char>& buffer);
};

