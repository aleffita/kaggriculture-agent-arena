"""
Hardware Profiler for Heterogeneous HPC & Benchmark Evaluation.
Directly interfaces with NVIDIA NVML (nvml.dll) and Host OS metrics (psutil) to track:
- VRAM per GPU (RTX 2060 + GTX 1050 Ti) in MB
- Host RAM usage in MB
- Disk / NVMe I/O Throughput (MB/s)
"""
import ctypes
import time
import threading
from typing import Dict, Any, Optional
import psutil

class NVMLMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]

class NVMLUtilization(ctypes.Structure):
    _fields_ = [
        ("gpu", ctypes.c_uint),
        ("memory", ctypes.c_uint),
    ]

class HardwareProfiler:
    def __init__(self):
        self.nvml_available = False
        self._nvml = None
        self._handles = []
        self._device_names = []
        self._init_nvml()

        self._sampling = False
        self._thread: Optional[threading.Thread] = None
        self._samples = []
        self._initial_disk = None

    def _init_nvml(self):
        try:
            self._nvml = ctypes.CDLL("nvml.dll")
            self._nvml.nvmlInit()
            count = ctypes.c_uint()
            self._nvml.nvmlDeviceGetCount(ctypes.byref(count))
            for i in range(count.value):
                handle = ctypes.c_void_p()
                self._nvml.nvmlDeviceGetHandleByIndex(i, ctypes.byref(handle))
                name_buf = ctypes.create_string_buffer(64)
                self._nvml.nvmlDeviceGetName(handle, name_buf, 64)
                self._handles.append(handle)
                self._device_names.append(name_buf.value.decode(errors="ignore"))
            self.nvml_available = True
        except Exception:
            self.nvml_available = False

    def snapshot(self) -> Dict[str, Any]:
        """Captura um snapshot instantâneo do hardware."""
        data = {
            "timestamp": time.time(),
            "host_ram_used_mb": psutil.virtual_memory().used // (1024 * 1024),
            "host_ram_total_mb": psutil.virtual_memory().total // (1024 * 1024),
            "host_ram_percent": psutil.virtual_memory().percent,
            "gpus": []
        }

        if self.nvml_available:
            for idx, handle in enumerate(self._handles):
                mem = NVMLMemory()
                self._nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem))
                util = NVMLUtilization()
                try:
                    self._nvml.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(util))
                    gpu_util = util.gpu
                except Exception:
                    gpu_util = 0

                data["gpus"].append({
                    "id": idx,
                    "name": self._device_names[idx],
                    "vram_used_mb": mem.used // (1024 * 1024),
                    "vram_total_mb": mem.total // (1024 * 1024),
                    "gpu_util_pct": gpu_util
                })

        return data

    def start_sampling(self, interval_sec: float = 0.05):
        """Inicia amostragem em thread assíncrona."""
        self._sampling = True
        self._samples = []
        try:
            self._initial_disk = psutil.disk_io_counters()
        except Exception:
            self._initial_disk = None
        self._t0 = time.perf_counter()

        def _loop():
            while self._sampling:
                self._samples.append(self.snapshot())
                time.sleep(interval_sec)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop_sampling(self) -> Dict[str, Any]:
        """Finaliza a amostragem e retorna métricas de pico e vazão."""
        self._sampling = False
        if self._thread:
            self._thread.join(timeout=1.0)
        elapsed = max(0.001, time.perf_counter() - self._t0)

        # Cálculo de I/O de disco
        disk_read_mb = 0.0
        disk_read_rate_mbs = 0.0
        try:
            if self._initial_disk:
                final_disk = psutil.disk_io_counters()
                delta_bytes = final_disk.read_bytes - self._initial_disk.read_bytes
                disk_read_mb = delta_bytes / (1024 * 1024)
                disk_read_rate_mbs = disk_read_mb / elapsed
        except Exception:
            pass

        if not self._samples:
            snap = self.snapshot()
            self._samples.append(snap)

        peak_vram_gpu0 = 0
        peak_vram_gpu1 = 0
        peak_ram = 0

        for s in self._samples:
            peak_ram = max(peak_ram, s["host_ram_used_mb"])
            if len(s["gpus"]) > 0:
                peak_vram_gpu0 = max(peak_vram_gpu0, s["gpus"][0]["vram_used_mb"])
            if len(s["gpus"]) > 1:
                peak_vram_gpu1 = max(peak_vram_gpu1, s["gpus"][1]["vram_used_mb"])

        return {
            "elapsed_sec": round(elapsed, 4),
            "samples_count": len(self._samples),
            "peak_gpu0_vram_mb": peak_vram_gpu0,
            "peak_gpu1_vram_mb": peak_vram_gpu1,
            "peak_host_ram_mb": peak_ram,
            "disk_read_mb": round(disk_read_mb, 2),
            "nvme_read_rate_mbs": round(disk_read_rate_mbs, 2),
            "timeline": self._samples
        }

    def close(self):
        if self.nvml_available and self._nvml:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
