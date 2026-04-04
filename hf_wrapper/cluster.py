"""
cluster.py — Multi-node and cross-queue GPU sharding analysis.

Supports:
  - Cross-node tensor-parallel sharding for single-GPU-per-node HPC clusters
  - Comparison against a catalog of well-known advanced node configurations

Terminology
-----------
  node_count      : number of identical nodes allocated from the Slurm queue
  gpu_count       : GPUs per node (e.g. 8 for an H100 DGX node)
  vram_per_gpu_mb : VRAM per GPU in MiB
  total_vram_mb   : node_count × gpu_count × vram_per_gpu_mb

For vLLM the mapping to parallelism flags is:
  tensor-parallel  (TP) = gpu_count        (GPUs within a single node)
  pipeline-parallel (PP) = node_count      (nodes in the job)
  → vllm serve ... --tensor-parallel-size <TP> --pipeline-parallel-size <PP>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from hf_wrapper.quantization import (
    best_quantization,
    native_vram_mb,
    QuantOption,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_mb(mb: int) -> str:
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb} MB"


_NODE_STEPS = (1, 2, 4, 8, 16)


# ---------------------------------------------------------------------------
# NodeSpec — hardware description for one compute node
# ---------------------------------------------------------------------------

@dataclass
class NodeSpec:
    """Hardware specification for a single compute node."""
    label: str             # short key, e.g. "H100-SXM-80GB-8GPU"
    gpu_model: str         # human-readable GPU name
    gpu_count: int         # GPUs per node
    vram_per_gpu_mb: int   # VRAM per GPU in MiB

    @property
    def total_vram_mb(self) -> int:
        return self.gpu_count * self.vram_per_gpu_mb

    def __str__(self) -> str:
        per = _fmt_mb(self.vram_per_gpu_mb)
        total = _fmt_mb(self.total_vram_mb)
        if self.gpu_count == 1:
            return f"{self.gpu_model} ({per})"
        return f"{self.gpu_count}× {self.gpu_model} ({per} each, {total}/node)"


# ---------------------------------------------------------------------------
# Catalog of well-known HPC / cloud node configurations
# Ordered roughly by typical availability in research HPC environments.
# ---------------------------------------------------------------------------

KNOWN_NODE_CONFIGS: List[NodeSpec] = [
    # --- NVIDIA flagship ---
    NodeSpec("H100-NVL-94GB-8GPU",   "NVIDIA H100 NVL 94GB",      8,  96_256),
    NodeSpec("H100-SXM-80GB-8GPU",   "NVIDIA H100 SXM5 80GB",     8,  81_920),
    NodeSpec("H100-PCIe-80GB-8GPU",  "NVIDIA H100 PCIe 80GB",     8,  81_920),
    # --- AMD MI series ---
    NodeSpec("MI300X-192GB-8GPU",    "AMD MI300X 192GB",           8, 196_608),
    NodeSpec("MI250X-128GB-8GPU",    "AMD MI250X 128GB",           8, 131_072),
    # --- NVIDIA A100 ---
    NodeSpec("A100-SXM4-80GB-8GPU",  "NVIDIA A100 SXM4 80GB",     8,  81_920),
    NodeSpec("A100-SXM4-40GB-8GPU",  "NVIDIA A100 SXM4 40GB",     8,  40_960),
    NodeSpec("A100-PCIe-40GB-4GPU",  "NVIDIA A100 PCIe 40GB",     4,  40_960),
    # --- NVIDIA V100 ---
    NodeSpec("V100-SXM2-32GB-8GPU",  "NVIDIA V100 SXM2 32GB",     8,  32_768),
    NodeSpec("V100-PCIe-16GB-4GPU",  "NVIDIA V100 PCIe 16GB",     4,  16_384),
    # --- Consumer / Workstation ---
    NodeSpec("4090-24GB-2GPU",       "NVIDIA RTX 4090 24GB",      2,  24_576),
    NodeSpec("4090-24GB-8GPU",       "NVIDIA RTX 4090 24GB",      8,  24_576),
]


# ---------------------------------------------------------------------------
# NodeScalingOption — result for one (node_spec, node_count) combination
# ---------------------------------------------------------------------------

@dataclass
class NodeScalingOption:
    """Describes model fit for a given node type at a specific node count."""
    node_spec: NodeSpec
    node_count: int
    native_fits: bool
    best_quant: Optional[QuantOption]

    @property
    def total_vram_mb(self) -> int:
        return self.node_count * self.node_spec.total_vram_mb

    @property
    def total_gpu_count(self) -> int:
        return self.node_count * self.node_spec.gpu_count

    @property
    def is_viable(self) -> bool:
        return self.native_fits or self.best_quant is not None

    @property
    def tp_pp_flags(self) -> str:
        """vLLM flags for this configuration.

        For single-GPU-per-node clusters, pipeline-parallel maps directly to
        node count.  For multi-GPU nodes, TP handles within-node, PP handles
        across nodes.  When node_count == 1, PP is omitted.
        """
        tp = self.node_spec.gpu_count
        pp = self.node_count
        if pp == 1 and tp == 1:
            return "(single GPU — no flags needed)"
        if pp == 1:
            return f"--tensor-parallel-size {tp}"
        if tp == 1:
            # Single-GPU nodes: pure tensor-parallel across nodes
            return f"--tensor-parallel-size {pp}"
        return f"--tensor-parallel-size {tp} --pipeline-parallel-size {pp}"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def suggest_node_scaling(
    param_count: int,
    node_spec: NodeSpec,
    max_nodes: int = 16,
) -> List[NodeScalingOption]:
    """
    Return scaling options for 1, 2, 4, … nodes of *node_spec*, stopping once
    the model fits natively.  Always includes at least one entry even when the
    model can't fit within *max_nodes*.

    Example::

        opts = suggest_node_scaling(70_000_000_000, H100_SXM_8GPU, max_nodes=4)
    """
    options: List[NodeScalingOption] = []
    for n in _NODE_STEPS:
        if n > max_nodes:
            break
        total_mb = n * node_spec.total_vram_mb
        nat = native_vram_mb(param_count) <= total_mb
        best = None if nat else best_quantization(param_count, total_mb)
        options.append(NodeScalingOption(node_spec, n, nat, best))
        if nat:
            break
    # If the loop never ran (max_nodes < 1), return a 1-node option anyway
    if not options:
        total_mb = node_spec.total_vram_mb
        nat = native_vram_mb(param_count) <= total_mb
        best = None if nat else best_quantization(param_count, total_mb)
        options.append(NodeScalingOption(node_spec, 1, nat, best))
    return options


def compare_with_catalog(
    param_count: int,
    catalog: Optional[List[NodeSpec]] = None,
    max_nodes: int = 4,
) -> List[NodeScalingOption]:
    """
    For each node type in *catalog*, find the minimum node count (up to
    *max_nodes*) at which the model is viable (fits natively or with
    quantization).  Returns one representative option per node type.

    If no viable config exists within *max_nodes*, the *max_nodes* option
    is returned so the table is always fully populated.
    """
    if catalog is None:
        catalog = KNOWN_NODE_CONFIGS

    results: List[NodeScalingOption] = []
    for spec in catalog:
        best_viable: Optional[NodeScalingOption] = None
        last: Optional[NodeScalingOption] = None
        for n in _NODE_STEPS:
            if n > max_nodes:
                break
            total_mb = n * spec.total_vram_mb
            nat = native_vram_mb(param_count) <= total_mb
            best = None if nat else best_quantization(param_count, total_mb)
            opt = NodeScalingOption(spec, n, nat, best)
            last = opt
            if opt.is_viable and best_viable is None:
                best_viable = opt
            if nat:
                break

        results.append(best_viable if best_viable is not None else (last or NodeScalingOption(spec, max_nodes, False, None)))
    return results


def node_spec_from_gpu(gpu_name: str, vram_mb: int, gpu_count: int = 1) -> NodeSpec:
    """Build a NodeSpec from detected hardware values."""
    label = f"current-{gpu_count}gpu"
    return NodeSpec(
        label=label,
        gpu_model=gpu_name,
        gpu_count=gpu_count,
        vram_per_gpu_mb=vram_mb,
    )
