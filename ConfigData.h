#pragma once

struct CONFIG_DATA
{
	UINT nDrawWidth;
	UINT nDrawHeight;
	BOOL bAntiAlias;
	UINT iAntiAliasLevel;
	UINT iIterationLevel;

	CONFIG_DATA() :
		nDrawWidth(3840)
		, nDrawHeight(2160)
		, bAntiAlias(TRUE)
		, iAntiAliasLevel(5)
		, iIterationLevel(8)
	{}

	inline UINT AntiAlias() { return bAntiAlias && iAntiAliasLevel > 1 ? iAntiAliasLevel : 1;  }
};