// ScatterFracCuda.cpp : Defines the entry point for the application.
//
// Run as ScatterFracCuda.exe for a normal window (debugging), or as ScatterFracCuda.scr
// with the standard screen saver switches (/s, /p <hwnd>, /c). See CSaverEngine.

#include "framework.h"
#include "SaverEngine.h"

int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
                     _In_opt_ HINSTANCE hPrevInstance,
                     _In_ LPWSTR    lpCmdLine,
                     _In_ int       nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(nCmdShow);

    CSaverEngine engine(hInstance);

    if (engine.Initialize(lpCmdLine))
    {
        engine.Run();
    }

    engine.CleanUp();

    return 0;
}
