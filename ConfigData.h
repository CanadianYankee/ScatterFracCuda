#pragma once

struct CONFIG_DATA
{
	UINT nDrawWidth;
	UINT nDrawHeight;
	BOOL bAntiAlias;
	UINT iAntiAliasLevel;
	UINT iIterationLevel;
	float fGammaValue;
	float fGammaSatur;

	CONFIG_DATA() :
		nDrawWidth(320)
		, nDrawHeight(200)
		, bAntiAlias(TRUE)
		, iAntiAliasLevel(3)
		, iIterationLevel(8)
		, fGammaValue(3.0f)
		, fGammaSatur(1.5f)
	{}

	inline UINT AntiAlias() { return bAntiAlias && iAntiAliasLevel > 1 ? iAntiAliasLevel : 1;  }
};