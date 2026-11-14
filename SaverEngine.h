#pragma once

class CWindowCluster;

// Decides the display mode from the executable's extension and the standard screen-saver
// command line switches, then sets up the windows. Ported from InteractDX11.
class CSaverEngine
{
public:
	enum SaverDisplayMode
	{
		sdmUnknown = 0,
		sdmNormalWindow,	// .exe with no arguments: resizable window, for debugging
		sdmSaverWindow,		// /s: full screen on every monitor
		sdmPreviewWindow,	// /p <hwnd>: nested in the Screen Saver Settings preview
		sdmConfigOnly		// /c[:hwnd] or .scr with no arguments: settings
	};

	CSaverEngine(HINSTANCE hAppInstance);
	~CSaverEngine(void);

	SaverDisplayMode GetMode() const { return m_sdmMode; }
	HINSTANCE GetAppInstance() const { return m_hAppInstance; }

	BOOL Initialize(LPCTSTR lpCmdLine);
	BOOL PerMonitorInit(HMONITOR hMonitor);

	void Run();
	void DoConfig();

	void CleanUp();

private:
	void ParseForOptions(LPCTSTR lpCmdLine);
	bool ParseOption(LPCTSTR lpCmdLine, TCHAR &tcOpt, HANDLE &hParam) const;
	BOOL SetupWindows();

	HINSTANCE m_hAppInstance;
	HWND m_hParentWnd;
	CWindowCluster *m_pWindowCluster;
	SaverDisplayMode m_sdmMode;
};
