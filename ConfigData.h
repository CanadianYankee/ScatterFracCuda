#pragma once

struct CONFIG_DATA
{
	UINT nDrawWidth;
	UINT nDrawHeight;
	BOOL bRotation;
	BOOL bMirror;
	BOOL bAntiAlias;
	UINT iAntiAliasLevel;
	BOOL bKernelBlur;
	UINT iKernelRadius;
	float fKernelAlpha;
	UINT iIterationLevel;
	UINT iHitPercent;
	float fGammaValue;
	float fGammaSatur;

	CONFIG_DATA() :
		nDrawWidth(3840)
		, nDrawHeight(2160)
		, bRotation(TRUE)
		, bMirror(TRUE)
		, bAntiAlias(FALSE)
		, iAntiAliasLevel(3)
		, bKernelBlur(TRUE)
		, iKernelRadius(9)
		, fKernelAlpha(0.4f)
		, iIterationLevel(4)
		, iHitPercent(50)
		, fGammaValue(3.0f)
		, fGammaSatur(1.5f)
	{}

	inline UINT AntiAlias() { return bAntiAlias && iAntiAliasLevel > 1 ? iAntiAliasLevel : 1; }
	inline UINT KernelRadius() { return bKernelBlur && iKernelRadius > 0 ? iKernelRadius : 0; }
};