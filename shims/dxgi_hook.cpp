#include <windows.h>
#include <dxgi1_6.h>
#include <stdio.h>
#include <stdlib.h>

#pragma comment(lib, "dxgi.lib")

typedef HRESULT (STDMETHODCALLTYPE *PFN_EnumAdapters)(IDXGIFactory* This, UINT Adapter, IDXGIAdapter** ppAdapter);
typedef HRESULT (STDMETHODCALLTYPE *PFN_EnumAdapters1)(IDXGIFactory1* This, UINT Adapter, IDXGIAdapter1** ppAdapter);
typedef HRESULT (STDMETHODCALLTYPE *PFN_EnumAdapterByGpuPreference)(IDXGIFactory6* This, UINT Adapter, DXGI_GPU_PREFERENCE GpuPreference, REFIID riid, void** ppvAdapter);

static PFN_EnumAdapters g_pfnRealEnumAdapters = NULL;
static PFN_EnumAdapters1 g_pfnRealEnumAdapters1 = NULL;
static PFN_EnumAdapterByGpuPreference g_pfnRealEnumAdapterByGpuPreference = NULL;

static UINT g_targetRealIndex = 1;
static BOOL g_hookInstalled = FALSE;

HRESULT STDMETHODCALLTYPE HookedEnumAdapters(IDXGIFactory* This, UINT Adapter, IDXGIAdapter** ppAdapter) {
    if (Adapter == 0) {
        return g_pfnRealEnumAdapters(This, g_targetRealIndex, ppAdapter);
    }
    return DXGI_ERROR_NOT_FOUND;
}

HRESULT STDMETHODCALLTYPE HookedEnumAdapters1(IDXGIFactory1* This, UINT Adapter, IDXGIAdapter1** ppAdapter) {
    if (Adapter == 0) {
        return g_pfnRealEnumAdapters1(This, g_targetRealIndex, ppAdapter);
    }
    return DXGI_ERROR_NOT_FOUND;
}

HRESULT STDMETHODCALLTYPE HookedEnumAdapterByGpuPreference(IDXGIFactory6* This, UINT Adapter, DXGI_GPU_PREFERENCE GpuPreference, REFIID riid, void** ppvAdapter) {
    if (Adapter == 0) {
        return g_pfnRealEnumAdapters1((IDXGIFactory1*)This, g_targetRealIndex, (IDXGIAdapter1**)ppvAdapter);
    }
    return DXGI_ERROR_NOT_FOUND;
}

extern "C" __declspec(dllexport) void InstallDxgiHook(int targetIndex) {
    if (g_hookInstalled) return;

    g_targetRealIndex = (UINT)targetIndex;
    IDXGIFactory6* pFactory = NULL;
    HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory6), (void**)&pFactory);
    if (FAILED(hr)) return;

    // Log available adapters
    printf("[LiteRT-LM DXGI Shim] Interceptando adaptadores de GPU:\n");
    UINT i = 0;
    IDXGIAdapter1* pTest = NULL;
    while (pFactory->EnumAdapters1(i, &pTest) != DXGI_ERROR_NOT_FOUND) {
        DXGI_ADAPTER_DESC1 d;
        pTest->GetDesc1(&d);
        wprintf(L"  - GPU %u: %s (VRAM: %llu MB) %s\n", 
            i, d.Description, d.DedicatedVideoMemory / (1024 * 1024),
            (i == g_targetRealIndex) ? L"<= [SELECIONADA]" : L"");
        pTest->Release();
        i++;
    }

    void** vtbl = *(void***)pFactory;
    DWORD oldProtect;
    VirtualProtect(vtbl, sizeof(void*) * 32, PAGE_EXECUTE_READWRITE, &oldProtect);

    g_pfnRealEnumAdapters = (PFN_EnumAdapters)vtbl[7];
    g_pfnRealEnumAdapters1 = (PFN_EnumAdapters1)vtbl[12];
    g_pfnRealEnumAdapterByGpuPreference = (PFN_EnumAdapterByGpuPreference)vtbl[29];

    vtbl[7] = (void*)HookedEnumAdapters;
    vtbl[12] = (void*)HookedEnumAdapters1;
    vtbl[29] = (void*)HookedEnumAdapterByGpuPreference;

    VirtualProtect(vtbl, sizeof(void*) * 32, oldProtect, &oldProtect);
    pFactory->Release();
    g_hookInstalled = TRUE;
    printf("[LiteRT-LM DXGI Shim] Hook ativo: isolando GPU %u como unico dispositivo visivel no processo.\n\n", g_targetRealIndex);
}

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpReserved) {
    if (fdwReason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(hinstDLL);
        const char* envGpu = getenv("LITERT_GPU_INDEX");
        if (!envGpu) envGpu = getenv("LITERT_TARGET_GPU");
        if (envGpu) {
            int idx = atoi(envGpu);
            InstallDxgiHook(idx);
        }
    }
    return TRUE;
}
