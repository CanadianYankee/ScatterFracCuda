#pragma once

struct CONFIG_DATA
{
	UINT nDrawWidth;
	UINT nDrawHeight;
	BOOL bAntiAlias;
	int iAntiAliasLevel;

	CONFIG_DATA() :
		nDrawWidth(3840)
		, nDrawHeight(2160)
		, bAntiAlias(TRUE)
		, iAntiAliasLevel(3)
	{}
};