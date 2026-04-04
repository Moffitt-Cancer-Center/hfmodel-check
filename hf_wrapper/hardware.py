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
    is_mig_slice: bool = False

    def __str__(self) -> str:
        mem = _fmt_mb(self.vram_mb)
        if self.is_unified_memory:
            return f"{self.name} (Unified Memory: {mem})"
        if self.is_mig_slice:
            return f"{self.name} (MIG slice: {mem})"
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
    def has_mig(self) -> bool:
        """True when the reported GPU unit is a MIG slice, not a full GPU."""
        return any(g.is_mig_slice for g in self.gpus)

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def homogeneous_gpu_count(self) -> int:
        """
        Number of GPU units (full GPUs or MIG slices) that share the same VRAM
        as the best unit.  Used for within-node tensor-parallel sharding.

        NOTE: MIG slices are hardware-isolated; you cannot shard a single model
        across MIG slices on one physical GPU.  When MIG is active, this returns
        the slice count for display purposes, but the caller should treat
        within-node sharding capacity as 1 unit (one slice per job).
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
        if self.has_mig:
            names = ", ".join(g.name for g in self.gpus)
            return f"CUDA/MIG ({names})"
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
            if self.has_mig:
                lines.append(
                    "MIG  : Active — slices are hardware-isolated; "
                    "cross-node tensor-parallel is still available"
                )
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

def _parse_mig_from_list(output: str) -> List[GPUInfo]:
    """
    Parse MIG entries from ``nvidia-smi -L``.

    On MIG-enabled systems the output looks like::

        GPU 0: NVIDIA A30 (UUID: GPU-...)
          MIG 2g.12gb     Device  0: (UUID: MIG-...)
          MIG 2g.12gb     Device  1: (UUID: MIG-...)

    VRAM is derived from the profile name (e.g. ``2g.12gb`` → 12 × 1024 MiB).
    This is the nominal value; actual framebuffer may differ by a few hundred
    MiB (e.g. A30 2g.12gb = 12032 MiB vs 12 × 1024 = 12288 MiB).
    """
    gpus: List[GPUInfo] = []
    current_gpu = ""
    for line in output.splitlines():
        gm = re.match(r"^GPU \d+: (.+?) \(UUID:", line)
        if gm:
            current_gpu = gm.group(1).strip()
            continue
        mm = re.match(r"\s+MIG\s+(\S+)\s+Device\s+\d+", line)
        if mm:
            profile = mm.group(1)                   # e.g. "2g.12gb"
            pm = re.search(r"(\d+(?:\.\d+)?)gb", profile, re.IGNORECASE)
            if pm:
                vram_mb = int(float(pm.group(1)) * 1024)
                name = f"{current_gpu} MIG {profile}" if current_gpu else f"MIG {profile}"
                gpus.append(GPUInfo(name=name, vram_mb=vram_mb, is_mig_slice=True))
    return gpus


def _parse_mig_from_text(output: str) -> List[GPUInfo]:
    """
    Parse MIG device rows from the plain ``nvidia-smi`` ASCII table.

    Matches data rows like::

        |  0    1   0   0  |              72MiB / 12032MiB    | ...

    and reads the total shared-memory column (12032 in the example).
    Parent GPU names are fetched via a separate ``--query-gpu`` call for labels.
    """
    # Quick check: is there a MIG section at all?
    if "MIG devices:" not in output:
        return []

    # Collect parent GPU names for richer labels.
    parent_names: dict = {}
    name_raw = _run(["nvidia-smi", "--query-gpu=index,name",
                     "--format=csv,noheader"])
    if name_raw:
        for ln in name_raw.splitlines():
            pr = [p.strip() for p in ln.split(",", 1)]
            if len(pr) == 2:
                try:
                    parent_names[int(pr[0])] = pr[1]
                except ValueError:
                    pass

    gpus: List[GPUInfo] = []
    in_mig = False
    for line in output.splitlines():
        if "MIG devices:" in line:
            in_mig = True
            continue
        if not in_mig:
            continue
        # Match: | GPU_IDX  GI  CI  MIG_DEV |  used MiB / total MiB  |
        m = re.search(
            r"^\|\s+(\d+)\s+\d+\s+\d+\s+\d+\s+\|\s+\d+MiB\s*/\s*(\d+)MiB",
            line,
        )
        if m:
            gpu_idx = int(m.group(1))
            vram_mb = int(m.group(2))
            parent = parent_names.get(gpu_idx, f"GPU {gpu_idx}")
            gpus.append(GPUInfo(
                name=f"{parent} MIG slice",
                vram_mb=vram_mb,
                is_mig_slice=True,
            ))
    return gpus


def _detect_nvidia_mig() -> List[GPUInfo]:
    """
    Detect NVIDIA MIG slices visible to the current process.

    Four-tier fallback chain (attempts each in order):

    1. ``--query-mig-device=index,name,memory.total``
       Structured query — exact framebuffer.  May fail on some driver versions
       (e.g. when the user lacks NVML MIG enumeration privileges, or on newer
       CUDA toolkits where the field list changed).

    2. ``nvidia-smi -L``
       Lists MIG device entries with profile names (e.g. ``2g.12gb``).
       VRAM is derived from the profile name — accurate to ±256 MiB.

    3. Plain ``nvidia-smi`` text output
       Parses the ASCII MIG device table to extract exact framebuffer values.
       Works even when structured queries fail.

    4. Parent-GPU fallback
       If all slice-enumeration methods fail, the parent GPU is reported with
       ``is_mig_slice=True`` so the MIG flag is still communicated to the user.

    Returns ``[]`` when MIG is not enabled.
    """
    # ── Is MIG mode enabled? ──────────────────────────────────────────────
    mig_check = _run([
        "nvidia-smi",
        "--query-gpu=mig.mode.current",
        "--format=csv,noheader",
    ])
    if not mig_check or "Enabled" not in mig_check:
        return []

    # ── Tier 1: structured query (tries two field sets) ──────────────────
    for fields in ("index,name,memory.total", "name,memory.total"):
        raw = _run([
            "nvidia-smi",
            f"--query-mig-device={fields}",
            "--format=csv,noheader,nounits",
        ])
        if raw:
            gpus: List[GPUInfo] = []
            for line in raw.splitlines():
                parts = [p.strip() for p in line.split(",")]
                try:
                    # Last field is always memory_mb; second-to-last is name.
                    vram = int(parts[-1])
                    name = parts[-2]
                    gpus.append(GPUInfo(name=name, vram_mb=vram, is_mig_slice=True))
                except (ValueError, IndexError):
                    pass
            if gpus:
                return gpus

    # ── Tier 2: nvidia-smi -L (profile name → nominal VRAM) ──────────────
    list_out = _run(["nvidia-smi", "-L"])
    if list_out:
        gpus = _parse_mig_from_list(list_out)
        if gpus:
            return gpus

    # ── Tier 3: parse plain nvidia-smi ASCII table (exact framebuffer) ───
    text_out = _run(["nvidia-smi"])
    if text_out:
        gpus = _parse_mig_from_text(text_out)
        if gpus:
            return gpus

    # ── Tier 4: last resort — mark parent GPU(s) as MIG-active ───────────
    # All slice-enumeration methods failed.  Report the parent GPU with the
    # MIG flag set so downstream code still shows the MIG isolation warning.
    full_out = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if full_out:
        gpus = []
        for line in full_out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    gpus.append(GPUInfo(
                        name=f"{parts[0]} (MIG enabled, slice sizes unknown)",
                        vram_mb=int(parts[1]),
                        is_mig_slice=True,
                    ))
                except ValueError:
                    pass
        if gpus:
            return gpus

    return []


def _detect_nvidia() -> List[GPUInfo]:
    # MIG takes priority: if MIG instances are accessible, report those.
    mig = _detect_nvidia_mig()
    if mig:
        return mig

    # Standard whole-GPU detection.
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
