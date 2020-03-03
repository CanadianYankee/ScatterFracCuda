#pragma once

struct CONFIG_DATA
{
	UINT nDrawWidth;
	UINT nDrawHeight;
	BOOL bAntiAlias;
	UINT iAntiAliasLevel;
	BOOL bKernelBlur;
	UINT iKernelRadius;
	float fKernelAlpha;
	UINT iIterationLevel;
	float fGammaValue;
	float fGammaSatur;

	CONFIG_DATA() :
		nDrawWidth(640)
		, nDrawHeight(400)
		, bAntiAlias(TRUE)
		, iAntiAliasLevel(3)
		, bKernelBlur(TRUE)
		, iKernelRadius(9)
		, fKernelAlpha(0.4f)
		, iIterationLevel(8)
		, fGammaValue(3.0f)
		, fGammaSatur(1.5f)
	{}

	inline UINT AntiAlias() { return bAntiAlias && iAntiAliasLevel > 1 ? iAntiAliasLevel : 1; }
	inline UINT KernelRadius() { return bKernelBlur && iKernelRadius > 0 ? iKernelRadius : 0; }
};