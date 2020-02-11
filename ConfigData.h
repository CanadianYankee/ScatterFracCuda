#pragma once

struct CONFIG_DATA
{
	UINT nDrawWidth;
	UINT nDrawHeight;
	BOOL bAntiAlias;
	int iAntiAliasLevel;

	CONFIG_DATA() :
		nDrawWidth(640)
		, nDrawHeight(400)
		, bAntiAlias(FALSE)
		, iAntiAliasLevel(1)
	{}
};