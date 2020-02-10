// header.h : include file for standard system include files,
// or project specific include files
//

#pragma once

#include "targetver.h"
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
// Windows Header Files
#include <windows.h>
// C RunTime Header Files
#include <stdlib.h>
#include <malloc.h>
#include <memory.h>
#include <tchar.h>
#include <string>
#include <vector>
#include <fstream>
#include <memory>

#include <d3d11.h>
#include <DirectXMath.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

using Microsoft::WRL::ComPtr;
using namespace DirectX;
