#pragma once

class CSaverEngine;
class CSaverBase;

// Owns the set of saver windows (one per monitor in full-screen saver mode, otherwise one),
// the shared window procedure, and the message loop. Ported from InteractDX11.
class CWindowCluster
{
public:
	CWindowCluster(CSaverEngine *pEngine);
	virtual ~CWindowCluster(void);

	static CWindowCluster *s_pTheCluster;
	LRESULT ClusterWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

	BOOL IsMultiRender() const { return m_bMultiRender; }
	BOOL AddPrimary(LPRECT lpRect, BOOL bSystemPrimary);
	BOOL AddSecondary(LPRECT lpRect);
	BOOL AddNested(HWND hParent);

	BOOL CreateAll();
	void Run();
	void DestroyAll();

	typedef struct tagSAVER_WINDOW
	{
		RECT rcCreateArea;
		SIZE sClientArea;
		DWORD dwCreateFlags;
		bool bSaverMode;
		bool bNested;
		bool bPrimary;
		HWND hParentWnd;
		HWND hWnd;
		CSaverBase *pSaver;
	} SAVER_WINDOW;

protected:
	typedef std::vector<SAVER_WINDOW> WindowVector;
	typedef WindowVector::iterator WindowIter;

	void RegisterWindowClass(const SAVER_WINDOW &windowInfo, const std::wstring &strClass);
	WindowIter LookupWindow(HWND hWnd);

	CSaverEngine *m_pParentEngine;
	WindowVector m_vecWindows;
	BOOL m_bMultiRender;

	void OnDisplayStateChange(DWORD dwState);
	void OnSessionChange(WPARAM wParam);
	void OnHiddenChanged(bool bWasHidden);
	bool IsHidden() const { return m_bDisplaysOff || m_bSessionLocked; }

	static const DWORD IDLE_POLL_MS = 100;	// Max sleep in Run() when nothing is rendering
	HPOWERNOTIFY m_hPowerNotify = NULL;
	bool m_bSessionNotify = false;
	bool m_bDisplaysOff = false;
	bool m_bSessionLocked = false;

	HBRUSH m_blackBrush;
	POINT m_ptMouse;
	bool m_bResizing;
};
