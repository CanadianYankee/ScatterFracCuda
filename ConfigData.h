#pragma once

struct CONFIG_DATA
{
	UINT nDrawWidth;
	UINT nDrawHeight;
	BOOL bAntiAlias;
	int iAntiAliasLevel;

	CONFIG_DATA() :
		nDrawWidth(320)
		, nDrawHeight(200)
		, bAntiAlias(TRUE)
		, iAntiAliasLevel(1)
	{}
};