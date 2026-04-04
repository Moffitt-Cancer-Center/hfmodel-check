"""
hardware.py — Detect GPU VRAM and system memory across NVIDIA, AMD, Apple Silicon.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    name: str
    vram_mb: int
    is_unified_memory: bool = False

    def __str__(self) -> str:
        mem = _fmt_mb(self.vram_mb)
        if self.is_unified_memory:
            return f"{self.name} (Unified Memory: {mem})"
        return f"{self.name} ({mem} VRAM)"


@dataclass
class HardwareInfo:
    gpus: List[GPUInfo] = field(default_factory=list)
    system_ram_mb: int = 0
    cpu_name: str = "Unknown CPU"

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def best_gpu_vram_mb(self) -> int:
        return max((g.vram_mb for g in self.gpus), default=0)

    @property
    def total_vram_mb(self) -> int:
        return sum(g.vram_mb for g in self.gpus)

    @property
    def has_unified_memory(self) -> bool:
        return any(g.is_unified_memory for g in self.gpus)

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def homogeneous_gpu_count(self) -> int:
        """
        Number of GPUs that share the same VRAM size as the best GPU.
        Used for tensor-parallel sharding estimates.
        On heterogeneous nodes (rare) only the matching GPUs are counted.
        """
        if not self.gpus:
            return 0
        best_vram = self.best_gpu_vram_mb
        return sum(1 for g in self.gpus if g.vram_mb == best_vram)

    @property
    def effective_memory_mb(self) -> int:
        """
        Memory available for model inference.
        On unified-memory systems (Apple Silicon) the model competes with all
        system RAM.  On discrete-GPU systems use the largest GPU's VRAM.
        Use system RAM as a floor when no GPU is present.
        """
        if self.has_unified_memory:
            return self.system_ram_mb
        if self.gpus:
            return self.best_gpu_vram_mb
        return self.system_ram_mb  # CPU-only fallback

    @property
    def inference_device(self) -> str:
        if not self.gpus:
            return "CPU"
        if self.has_unified_memory:
            return "Apple MPS (Unified Memory)"
        names = ", ".join(g.name for g in self.gpus)
        return f"CUDA ({names})"

    def summary(self) -> str:
        lines = [
            f"CPU  : {self.cpu_name}",
            f"RAM  : {_fmt_mb(self.system_ram_mb)}",
        ]
        if self.gpus:
            for g in self.gpus:
                lines.append(f"GPU  : {g}")
        else:
            lines.append("GPU  : None detected — inference will use CPU")
        lines.append(f"Device : {self.inference_device}")
        lines.append(f"Effective memory : {_fmt_mb(self.effective_memory_mb)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fmt_mb(mb: int) -> str:
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb} MB"


def _run(cmd: List[str]) -> Optional[str]:
    """Run a command and return stdout, or None on failure."""
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
        pass
    return None


def _parse_mem_string(s: str) -> Optional[int]:
    """Parse '16 GB', '32768 MB', '16384' → MB."""
    s = s.strip()
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([GgMm][Bb]?)?$", s)
    if not m:
        return None
    value = float(m.group(1))
    unit = (m.group(2) or "").upper()
    if unit.startswith("G"):
        return int(value * 1024)
    if unit.startswith("M") or unit == "":
        return int(value)
    return None


# ---------------------------------------------------------------------------
# NVIDIA
# ---------------------------------------------------------------------------

def _detect_nvidia() -> List[GPUInfo]:
    out = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return []
    gpus: List[GPUInfo] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                gpus.append(GPUInfo(name=parts[0], vram_mb=int(parts[1])))
            except ValueError:
                pass
    return gpus


# ---------------------------------------------------------------------------
# AMD / ROCm
# ---------------------------------------------------------------------------

def _detect_amd() -> List[GPUInfo]:
    out = _run(["rocm-smi", "--showmeminfo", "vram", "--json"])
    if not out:
        return []
    try:
        data = json.loads(out)
        gpus: List[GPUInfo] = []
        for card_id, card_data in data.items():
            if not isinstance(card_data, dict):
                continue
            vram_bytes_str = card_data.get("VRAM Total Memory (B)", "")
            if not vram_bytes_str:
                continue
            vram_mb = int(vram_bytes_str) // (1024 * 1024)
            name = card_data.get("Card series") or f"AMD GPU {card_id}"
            gpus.append(GPUInfo(name=name, vram_mb=vram_mb))
        return gpus
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    return []


# ---------------------------------------------------------------------------
# Apple Silicon (unified memory)
# ---------------------------------------------------------------------------

def _detect_apple_silicon() -> List[GPUInfo]:
    out = _run(["system_profiler", "SPHardwareDataType", "-json"])
    if not out:
        return []
    try:
        data = json.loads(out)
        hw_list = data.get("SPHardwareDataType", [{}])
        hw = hw_list[0] if hw_list else {}

        chip = hw.get("chip_type", "") or hw.get("cpu_type", "")
        # Only flag as unified-memory Apple Silicon
        if not re.search(r"Apple\s+M\d", chip):
            return []

        mem_str = hw.get("physical_memory", "")
        vram_mb = _parse_mem_string(mem_str)
        if vram_mb is None:
            return []
        return [GPUInfo(name=chip, vram_mb=vram_mb, is_unified_memory=True)]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return []


# ---------------------------------------------------------------------------
# System RAM
# ---------------------------------------------------------------------------

def _detect_system_ram() -> int:
    system = platform.system()
    if system == "Darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        if out:
            try:
                return int(out) // (1024 * 1024)
            except ValueError:
                pass
    elif system == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) // 1024
        except (OSError, ValueError):
            pass
    elif system == "Windows":
        out = _run(["wmic", "computersystem", "get", "TotalPhysicalMemory"])
        if out:
            for line in out.splitlines():
                line = line.strip()
                if line.isdigit():
                    return int(line) // (1024 * 1024)
    return 0


# ---------------------------------------------------------------------------
# CPU name
# ---------------------------------------------------------------------------

def _detect_cpu() -> str:
    system = platform.system()
    if system == "Darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out
        # Apple Silicon — brand_string may be empty
        out = _run(["sysctl", "-n", "hw.model"])
        if out:
            return out
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif system == "Windows":
        out = _run(["wmic", "cpu", "get", "name"])
        if out:
            lines = out.splitlines()
            if len(lines) > 1:
                return lines[1].strip()
    return platform.processor() or "Unknown CPU"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_hardware() -> HardwareInfo:
    """Detect GPU(s), VRAM, and system RAM. Returns a HardwareInfo instance."""
    gpus: List[GPUInfo] = []

    gpus.extend(_detect_nvidia())
    if not gpus:
        gpus.extend(_detect_amd())
    if not gpus and platform.system() == "Darwin":
        gpus.extend(_detect_apple_silicon())

    return HardwareInfo(
        gpus=gpus,
        system_ram_mb=_detect_system_ram(),
        cpu_name=_detect_cpu(),
    )
