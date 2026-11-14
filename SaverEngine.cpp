#include "framework.h"
#include "Resource.h"
#include "SaverEngine.h"
#include "WindowCluster.h"

static BOOL CALLBACK MonitorEnumFn(HMONITOR hMonitor, HDC hDC, LPRECT pRect, LPARAM lParam)
{
	UNREFERENCED_PARAMETER(hDC);
	UNREFERENCED_PARAMETER(pRect);
	return reinterpret_cast<CSaverEngine *>(lParam)->PerMonitorInit(hMonitor);
}

CSaverEngine::CSaverEngine(HINSTANCE hAppInstance) :
	m_hAppInstance(hAppInstance),
	m_hParentWnd(nullptr),
	m_pWindowCluster(nullptr),
	m_sdmMode(sdmUnknown)
{
}

CSaverEngine::~CSaverEngine(void)
{
	CleanUp();
}

void CSaverEngine::CleanUp()
{
	if(m_pWindowCluster)
	{
		m_pWindowCluster->DestroyAll();
		delete m_pWindowCluster;
		m_pWindowCluster = nullptr;
	}
}

BOOL CSaverEngine::Initialize(LPCTSTR lpCmdLine)
{
	// Figure out what our display/windowing mode should be
	ParseForOptions(lpCmdLine);
	if(m_sdmMode == sdmUnknown)
		return FALSE;

	if(m_sdmMode == sdmConfigOnly)
	{
		DoConfig();
		return FALSE;	// Everything is done, no need to run the saver.
	}

	return SetupWindows();
}

void CSaverEngine::ParseForOptions(LPCTSTR lpCmdLine)
{
	// Figure out the file extension ("scr" is a screen saver)
	bool bSCR = false;
	TCHAR fname[MAX_PATH];
	DWORD fnlen = ::GetModuleFileName(NULL, fname, MAX_PATH);
	if(fnlen > 4 && fnlen < MAX_PATH)
	{
		bSCR = _tcsicmp(fname + (fnlen - 4), _T(".scr")) == 0;
	}

	// Set for either "saver" or "windowed" mode; may change after parsing options
	m_sdmMode = bSCR ? sdmSaverWindow : sdmNormalWindow;

	TCHAR tcOption;
	HANDLE hParam;
	bool bParsed = ParseOption(lpCmdLine, tcOption, hParam);
	if(bSCR && !bParsed)
	{
		// Windows runs a .scr with no arguments for "Configure" from Explorer's context menu
		tcOption = _T('c');
		bParsed = true;
	}
	if(bParsed)
	{
		switch(tcOption)
		{
		case _T('c'):
			m_sdmMode = sdmConfigOnly;
			m_hParentWnd = (HWND)hParam;
			break;

		case _T('s'):
			m_sdmMode = sdmSaverWindow;
			break;

		case _T('p'):
			m_sdmMode = sdmPreviewWindow;
			m_hParentWnd = (HWND)hParam;
			break;
		}
	}
}

// Parses "/x", "-x", "/x 1234" or "/x:1234". The number is a decimal HWND.
bool CSaverEngine::ParseOption(LPCTSTR lpCmdLine, TCHAR &tcOpt, HANDLE &hParam) const
{
	tcOpt = _T(' ');
	hParam = nullptr;

	LPCTSTR lpsz = lpCmdLine;
	while(*lpsz == _T(' ') || *lpsz == _T('\t'))
		lpsz++;

	if(*lpsz == _T('-') || *lpsz == _T('/'))
		lpsz++;

	if(!IsCharAlpha(*lpsz))
		return false;

	tcOpt = static_cast<TCHAR>(_totlower(*lpsz));
	lpsz++;

	while(*lpsz == _T(' ') || *lpsz == _T(':'))
		lpsz++;

	unsigned long long ullParam = 0;
	if(_stscanf_s(lpsz, _T("%llu"), &ullParam) == 1)
		hParam = reinterpret_cast<HANDLE>(static_cast<ULONG_PTR>(ullParam));

	return true;
}

BOOL CSaverEngine::SetupWindows()
{
	assert(!m_pWindowCluster);
	m_pWindowCluster = new CWindowCluster(this);

	BOOL bSuccess = FALSE;

	if(m_sdmMode == sdmPreviewWindow)
	{
		// Create a nested, frameless child window inside the provided parent
		bSuccess = m_hParentWnd && m_pWindowCluster->AddNested(m_hParentWnd);
	}
	else
	{
		// One saver window per monitor in saver mode; one normal window on the primary
		// monitor otherwise.
		bSuccess = EnumDisplayMonitors(NULL, NULL, MonitorEnumFn, reinterpret_cast<LPARAM>(this));
	}

	if(bSuccess)
	{
		bSuccess = m_pWindowCluster->CreateAll();
	}

	return bSuccess;
}

BOOL CSaverEngine::PerMonitorInit(HMONITOR hMonitor)
{
	MONITORINFO monitorInfo;
	ZeroMemory(&monitorInfo, sizeof(monitorInfo));
	monitorInfo.cbSize = sizeof(monitorInfo);
	if(!GetMonitorInfo(hMonitor, &monitorInfo))
		return FALSE;

	if(IsRectEmpty(&monitorInfo.rcMonitor))
		return TRUE;

	BOOL bSystemPrimary = (monitorInfo.dwFlags & MONITORINFOF_PRIMARY) ? TRUE : FALSE;
	BOOL bSuccess = TRUE;
	if(bSystemPrimary || m_pWindowCluster->IsMultiRender())
	{
		bSuccess = m_pWindowCluster->AddPrimary(&monitorInfo.rcMonitor, bSystemPrimary);
	}
	else if(m_sdmMode == sdmSaverWindow)
	{
		bSuccess = m_pWindowCluster->AddSecondary(&monitorInfo.rcMonitor);
	}

	return bSuccess;
}

void CSaverEngine::Run()
{
	m_pWindowCluster->Run();
}

void CSaverEngine::DoConfig()
{
	TCHAR szTitle[100];
	LoadString(m_hAppInstance, IDS_APP_TITLE, szTitle, 100);
	MessageBox(m_hParentWnd, _T("This screen saver has no settings yet."), szTitle, MB_OK | MB_ICONINFORMATION);
}
