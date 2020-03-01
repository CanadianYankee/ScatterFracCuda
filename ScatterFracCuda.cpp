// ScatterFracCuda.cpp : Defines the entry point for the application.
//

#include "framework.h"
#include "ScatterFracCuda.h"
#include "GenerateWnd.h"

#define MAX_LOADSTRING 100

// Global Variables:
HINSTANCE hInst;                                // current instance
WCHAR szTitle[MAX_LOADSTRING];                  // The title bar text
WCHAR szWindowClass[MAX_LOADSTRING];            // the main window class name

int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
                     _In_opt_ HINSTANCE hPrevInstance,
                     _In_ LPWSTR    lpCmdLine,
                     _In_ int       nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    // Initialize global strings
    LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
    LoadStringW(hInstance, IDC_SCATTERFRACCUDA, szWindowClass, MAX_LOADSTRING);

    CGenerateWnd *pWnd = CGenerateWnd::GetMainWnd();

    // Perform window initialization:
    if (!pWnd->Initialize(hInstance, szTitle, szWindowClass, nCmdShow))
    {
        return FALSE;
    }

	// Run the main message loop
	int iRet = pWnd->Run(hInstance);

	CGenerateWnd::DestroyMainWnd();

    return iRet;
}
