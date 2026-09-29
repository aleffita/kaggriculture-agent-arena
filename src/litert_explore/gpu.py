"""GPU management and DXGI vtable interceptor for LiteRT-LM."""

from __future__ import annotations

import contextlib
import ctypes
import os
import pathlib
import sys
from typing import Generator, List, Dict, Any

def get_shim_dll_path() -> pathlib.Path:
    """Locates the dxgi_hook.dll binary."""
    # Check current package dir
    local_path = pathlib.Path(__file__).parent / "dxgi_hook.dll"
    if local_path.exists():
        return local_path

    # Check project shims/ dir
    repo_shims = pathlib.Path(__file__).resolve().parent.parent.parent / "shims" / "dxgi_hook.dll"
    if repo_shims.exists():
        return repo_shims

    # Check global/uv tool path
    appdata_path = (
        pathlib.Path(os.environ.get("APPDATA", ""))
        / "uv"
        / "tools"
        / "litert-lm"
        / "Lib"
        / "site-packages"
        / "litert_lm"
        / "dxgi_hook.dll"
    )
    if appdata_path.exists():
        return appdata_path

    raise FileNotFoundError("dxgi_hook.dll could not be located in local or repository paths.")


def install_dxgi_hook(target_index: int) -> bool:
    """Installs the DXGI vtable hook to isolate target GPU index as Adapter 0."""
    if sys.platform != "win32":
        return False
    try:
        shim_path = get_shim_dll_path()
        dll = ctypes.CDLL(str(shim_path))
        dll.InstallDxgiHook.argtypes = [ctypes.c_int]
        dll.InstallDxgiHook.restype = None
        dll.InstallDxgiHook(target_index)
        return True
    except Exception as exc:
        print(f"[Warning] Failed to install DXGI hook: {exc}", file=sys.stderr)
        return False


def list_gpus() -> List[Dict[str, Any]]:
    """Enumerates available physical GPUs on the Windows system using DXGI ctypes."""
    if sys.platform != "win32":
        return []

    gpus = []
    try:
        dxgi = ctypes.oledll.LoadLibrary("dxgi.dll")
        # Try powershell CIM fallback if ctypes DXGI interfaces are complex
        import subprocess
        ps_cmd = 'Get-CimInstance Win32_VideoController | Select-Object Name, DeviceID, AdapterRAM | ConvertTo-Json'
        res = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True)
        if res.returncode == 0:
            import json
            data = json.loads(res.stdout)
            if isinstance(data, dict):
                data = [data]
            for item in data:
                name = item.get("Name", "Unknown")
                if "Virtual" not in name:
                    gpus.append({
                        "name": name,
                        "device_id": item.get("DeviceID"),
                        "vram_mb": (item.get("AdapterRAM", 0) or 0) // (1024 * 1024),
                    })
    except Exception:
        pass
    return gpus


@contextlib.contextmanager
def select_gpu(target: str | int) -> Generator[None, None, None]:
    """Context manager setting environment variable and installing DXGI hook for a GPU target.
    
    Target can be:
      0, '0', '2060', 'rtx2060' -> RTX 2060 (Adapter 0)
      1, '1', '1050', '1050ti'  -> GTX 1050 Ti (Adapter 1)
    """
    target_str = str(target).lower().strip()
    if target_str in ("1", "1050", "1050ti", "gtx1050ti", "gtx 1050 ti"):
        target_idx = 1
    else:
        target_idx = 0

    old_env = os.environ.get("LITERT_GPU_INDEX")
    os.environ["LITERT_GPU_INDEX"] = str(target_idx)
    install_dxgi_hook(target_idx)
    try:
        yield
    finally:
        if old_env is None:
            os.environ.pop("LITERT_GPU_INDEX", None)
        else:
            os.environ["LITERT_GPU_INDEX"] = old_env
