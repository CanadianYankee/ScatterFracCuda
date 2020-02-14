#pragma once

struct CONFIG_DATA
{
	UINT nDrawWidth;
	UINT nDrawHeight;
	BOOL bAntiAlias;
	int iAntiAliasLevel;

	CONFIG_DATA() :
		nDrawWidth(1920)
		, nDrawHeight(1080)
		, bAntiAlias(TRUE)
		, iAntiAliasLevel(5)
	{}
};