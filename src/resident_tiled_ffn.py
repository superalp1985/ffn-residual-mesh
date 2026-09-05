from __future__ import annotations

import time

import numpy as np
import torch

from resident_residual_cuda import (
    launch_fused_gate_up_base_residual,
    launch_fused_gate_up_residual_tile,
    launch_merge_swiglu,
)
from resident_residual_format import ResidentArtifact
from resident_tile_cache import ResidentTileCache
from resident_tile_plan import TilePlan


class TiledResidentGateUp:
    """Exact gate/up + SwiGLU path using row-tiled resident residuals."""

    def __init__(
        self,
        artifact: ResidentArtifact,
        *,
        tile_rows: int = 1024,
        persistent: bool = False,
        base_on_gpu: bool = False,
        use_cuda_graph: bool = False,
        device: str | torch.device = "cuda",
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("TiledResidentGateUp requires CUDA")
        self.artifact = artifact
        self.device = torch.device(device)
        self.base_on_gpu = bool(base_on_gpu)
        self.use_cuda_graph = bool(use_cuda_graph)
        self.rows = int(artifact.projections["gate"]["rows"])
        self.cols = int(artifact.projections["gate"]["cols"])
        self.plan = TilePlan(
            rows=self.rows,
            tile_rows=tile_rows,
            projections=("gate", "up"),
        )
        arrays = {
            name: {
                "residual": artifact.arrays[name]["residual"],
                "alpha": artifact.arrays[name]["alpha"],
            }
            for name in ("gate", "up")
        }
        tile_bytes = self.plan.tile_bytes(
            cols=self.cols,
            alpha_cols=self.cols // 32,
        )
        self.cache = ResidentTileCache(
            arrays,
            plan=self.plan,
            vram_budget_bytes=(
                self.plan.total_bytes(cols=self.cols, alpha_cols=self.cols // 32)
                if persistent else tile_bytes * 2
            ),
            device=self.device,
        )
        if persistent:
            self.cache.cache.initialize_persistent(
                list(range(len(self.plan.tile_slices())))
            )
        self.stream = torch.cuda.Stream(device=self.device)
        self.base_stream = torch.cuda.Stream(device=self.device) if self.base_on_gpu else None
        self.host_x = torch.empty(self.cols, dtype=torch.float32, pin_memory=True)
        self.host_base = {
            name: torch.empty(self.rows, dtype=torch.float32, pin_memory=True)
            for name in ("gate", "up")
        }
        self.device_x = torch.empty(self.cols, device=self.device, dtype=torch.float32)
        self.device_base = {
            name: torch.empty(self.rows, device=self.device, dtype=torch.float32)
            for name in ("gate", "up")
        }
        self.base_resident = {}
        self.base_accum = {}
        self.device_group_sums = None
        if self.base_on_gpu:
            # Coefficients are static and paid for once at cold start. FP32 is
            # sufficient for the split residual tolerance and avoids slow FP64
            # GEMV on consumer GPUs.
            for name in ("gate", "up"):
                coefficient = np.asarray(artifact.arrays[name]["coefficient"], dtype=np.float32)
                self.base_resident[name] = torch.from_numpy(
                    np.array(coefficient, copy=True)
                ).to(self.device)
                self.base_accum[name] = torch.empty(
                    self.rows, device=self.device, dtype=torch.float32
                )
            self.device_group_sums = torch.empty(
                self.cols // 32, device=self.device, dtype=torch.float32
            )
        self.output = {
            name: torch.empty(self.rows, device=self.device, dtype=torch.float32)
            for name in ("gate", "up", "swiglu")
        }
        self.residual = {
            name: torch.empty(self.rows, device=self.device, dtype=torch.float32)
            for name in ("gate", "up")
        }
        self._group_sums = np.empty(self.cols // 32, dtype=np.float64)
        self.host_group_sums = torch.empty(
            self.cols // 32, dtype=torch.float32, pin_memory=True
        ) if self.base_on_gpu else None
        self._cuda_graph: torch.cuda.CUDAGraph | None = None
        self._graph_package: dict[str, torch.Tensor] | None = None
        self._graph_down = None
        self._graph_capture_ms = 0.0

    def _capture_full_cuda_graph(self, down=None) -> None:
        """Capture the fixed-shape resident super-tile path once."""
        if self._cuda_graph is not None:
            return
        if not self.base_on_gpu or len(self.plan.tile_slices()) != 1:
            raise RuntimeError("CUDA graph requires a resident full-layer GPU-base path")
        if self.host_group_sums is None or self.device_group_sums is None:
            raise RuntimeError("CUDA graph requires resident group-sum buffers")
        self.cache.acquire(0)
        self._graph_package = self.cache.package(0)
        package = self._graph_package
        assert package is not None

        # Force Triton compilation and allocator activity before capture.
        with torch.cuda.stream(self.stream):
            self.device_x.copy_(self.host_x, non_blocking=True)
            self.device_group_sums.copy_(self.host_group_sums, non_blocking=True)
            launch_fused_gate_up_base_residual(
                package["gate.residual"],
                package["gate.alpha"],
                package["up.residual"],
                package["up.alpha"],
                self.base_resident["gate"],
                self.base_resident["up"],
                self.device_group_sums,
                self.device_x,
                self.output["gate"],
                self.output["up"],
                self.output["swiglu"],
                rows=self.rows,
                cols=self.cols,
            )
            if down is not None:
                down.launch(self.output["swiglu"])
        self.stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        capture_begin = time.perf_counter()
        with torch.cuda.graph(graph, stream=self.stream):
            self.device_x.copy_(self.host_x, non_blocking=True)
            self.device_group_sums.copy_(self.host_group_sums, non_blocking=True)
            launch_fused_gate_up_base_residual(
                package["gate.residual"],
                package["gate.alpha"],
                package["up.residual"],
                package["up.alpha"],
                self.base_resident["gate"],
                self.base_resident["up"],
                self.device_group_sums,
                self.device_x,
                self.output["gate"],
                self.output["up"],
                self.output["swiglu"],
                rows=self.rows,
                cols=self.cols,
            )
            if down is not None:
                down.launch(self.output["swiglu"])
        self.stream.synchronize()
        self._cuda_graph = graph
        self._graph_down = down
        self._graph_capture_ms = (time.perf_counter() - capture_begin) * 1000

    def run(
        self,
        activation: np.ndarray,
        *,
        down=None,
        return_outputs: bool = True,
    ) -> dict[str, object]:
        x = np.asarray(activation, dtype=np.float32)
        if x.shape != (self.cols,) or not np.isfinite(x).all():
            raise ValueError("finite one-token activation required")
        if down is not None and (down.cols != self.rows or down.rows != self.cols):
            raise ValueError("down projection dimensions must reverse gate/up dimensions")
        graph_down_compatible = (
            self._cuda_graph is None or down is self._graph_down
        )
        if (
            self.use_cuda_graph
            and self.base_on_gpu
            and len(self.plan.tile_slices()) == 1
            and graph_down_compatible
        ):
            return self._run_full_cuda_graph(
                x, down=down, return_outputs=return_outputs
            )
        begin = time.perf_counter()
        self.host_x.numpy()[:] = x
        activation_begin = torch.cuda.Event(enable_timing=True)
        activation_end = torch.cuda.Event(enable_timing=True)
        residual_end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.stream):
            activation_begin.record()
            self.device_x.copy_(self.host_x, non_blocking=True)
            activation_end.record()
        tile_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        down_begin = torch.cuda.Event(enable_timing=True) if down is not None else None
        down_end = torch.cuda.Event(enable_timing=True) if down is not None else None
        grouped = x.astype(np.float64).reshape(-1, 32)
        base_begin = torch.cuda.Event(enable_timing=True)
        base_end = torch.cuda.Event(enable_timing=True)
        merge_end = torch.cuda.Event(enable_timing=True)
        cpu_base_ms = 0.0
        full_gpu_fused = self.base_on_gpu and len(self.plan.tile_slices()) == 1
        if self.base_on_gpu:
            self._group_sums[:] = grouped.sum(axis=1)
            self.host_group_sums.numpy()[:] = self._group_sums.astype(np.float32)
            assert self.base_stream is not None and self.device_group_sums is not None
            with torch.cuda.stream(self.base_stream):
                base_begin.record()
                self.device_group_sums.copy_(self.host_group_sums, non_blocking=True)
                if not full_gpu_fused:
                    for name in ("gate", "up"):
                        torch.mv(
                            self.base_resident[name],
                            self.device_group_sums,
                            out=self.base_accum[name],
                        )
                        self.device_base[name].copy_(
                            self.base_accum[name], non_blocking=True
                        )
                base_end.record()
            self.stream.wait_event(base_end)
        if full_gpu_fused:
            tile = 0
            start, stop = self.plan.tile_slices()[0]
            self.cache.acquire(tile)
            package = self.cache.package(tile)
            begin_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(self.stream):
                begin_event.record()
                launch_fused_gate_up_base_residual(
                    package["gate.residual"],
                    package["gate.alpha"],
                    package["up.residual"],
                    package["up.alpha"],
                    self.base_resident["gate"],
                    self.base_resident["up"],
                    self.device_group_sums,
                    self.device_x,
                    self.output["gate"],
                    self.output["up"],
                    self.output["swiglu"],
                    rows=stop - start,
                    cols=self.cols,
                )
                end_event.record()
                residual_end.record()
                merge_end.record()
            tile_events.append((begin_event, end_event))
            self.cache.release(tile)
        else:
            for tile, (start, stop) in enumerate(self.plan.tile_slices()):
                self.cache.acquire(tile)
                package = self.cache.package(tile)
                begin_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(self.stream):
                    begin_event.record()
                    launch_fused_gate_up_residual_tile(
                        package["gate.residual"],
                        package["gate.alpha"],
                        package["up.residual"],
                        package["up.alpha"],
                        self.device_x,
                        self.residual["gate"][start:stop],
                        self.residual["up"][start:stop],
                        rows=stop - start,
                        cols=self.cols,
                    )
                    end_event.record()
                tile_events.append((begin_event, end_event))
                self.cache.release(tile)
            with torch.cuda.stream(self.stream):
                residual_end.record()

        if not self.base_on_gpu:
            cpu_begin = time.perf_counter()
            self._group_sums[:] = grouped.sum(axis=1)
            for name in ("gate", "up"):
                self.host_base[name].numpy()[:] = (
                    self.artifact.arrays[name]["coefficient"] @ self._group_sums
                )
            cpu_base_ms = (time.perf_counter() - cpu_begin) * 1000
            with torch.cuda.stream(self.stream):
                base_begin.record()
                for name in ("gate", "up"):
                    self.device_base[name].copy_(self.host_base[name], non_blocking=True)
                base_end.record()
        if not full_gpu_fused:
            with torch.cuda.stream(self.stream):
                launch_merge_swiglu(
                    self.residual["gate"],
                    self.residual["up"],
                    self.device_base["gate"],
                    self.device_base["up"],
                    self.output["gate"],
                    self.output["up"],
                    self.output["swiglu"],
                    rows=self.rows,
                )
                merge_end.record()
        if down is not None:
            with torch.cuda.stream(self.stream):
                down_begin.record()
                down.launch(self.output["swiglu"])
                down_end.record()
        # The optional down projection may own its launch stream (Triton uses
        # the current stream at dispatch time). Synchronize the device so its
        # completion event is valid before reading timings or outputs.
        torch.cuda.synchronize(self.device)
        wall_ms = (time.perf_counter() - begin) * 1000
        tile_kernel_ms = float(sum(start.elapsed_time(end) for start, end in tile_events))
        result = {
            "cpu_base_ms": cpu_base_ms,
            "activation_h2d_ms": float(activation_begin.elapsed_time(activation_end)),
            "tile_kernel_ms": tile_kernel_ms,
            "exposed_cpu_submission_gap_ms": float(residual_end.elapsed_time(base_begin)),
            "base_h2d_ms": float(base_begin.elapsed_time(base_end)),
            "merge_swiglu_ms": (
                0.0 if full_gpu_fused else float(base_end.elapsed_time(merge_end))
            ),
            "fused_base_residual_swiglu_ms": (
                float(begin_event.elapsed_time(end_event))
                if full_gpu_fused else 0.0
            ),
            "wall_ms": wall_ms,
            "activation_h2d_bytes": self.cols * 4,
            "base_h2d_bytes": (
                (self.cols // 32) * 4 if self.base_on_gpu else self.rows * 2 * 4
            ),
            "base_resident_bytes": (
                sum(value.numel() * value.element_size() for value in self.base_resident.values())
                if self.base_on_gpu else 0
            ),
            "weight_h2d_bytes": self.cache.traffic["weight_h2d_bytes"],
            "resident_weight_h2d_bytes": 0,
            "tile_count": len(self.plan.tile_slices()),
            "base_compute_device": "cuda" if self.base_on_gpu else "cpu",
            "kernel_mode": (
                "fused_base_residual_swiglu_super_tile"
                if full_gpu_fused
                else (
                    "fused_residual_gpu_base_overlap_then_merge"
                    if self.base_on_gpu
                    else "fused_residual_cpu_overlap_then_merge"
                )
            ),
        }
        if down is not None:
            result["down_stream_ms"] = float(down_begin.elapsed_time(down_end))
        if return_outputs:
            result.update({
                name: output.cpu().numpy()
                for name, output in self.output.items()
            })
            if down is not None:
                result["down"] = down.output.cpu().numpy()
        return result

    def _run_full_cuda_graph(
        self,
        x: np.ndarray,
        *,
        down=None,
        return_outputs: bool,
    ) -> dict[str, object]:
        if self.host_group_sums is None:
            raise RuntimeError("CUDA graph path requires GPU base buffers")
        begin = time.perf_counter()
        self.host_x.numpy()[:] = x
        self._group_sums[:] = x.astype(np.float64).reshape(-1, 32).sum(axis=1)
        self.host_group_sums.numpy()[:] = self._group_sums.astype(np.float32)
        if self._cuda_graph is None:
            self._capture_full_cuda_graph(down=down)
        assert self._cuda_graph is not None
        begin_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.stream):
            begin_event.record()
            self._cuda_graph.replay()
            end_event.record()
        self.stream.synchronize()
        wall_ms = (time.perf_counter() - begin) * 1000
        graph_ms = float(begin_event.elapsed_time(end_event))
        result: dict[str, object] = {
            "cpu_base_ms": 0.0,
            "activation_h2d_ms": 0.0,
            "tile_kernel_ms": graph_ms,
            "exposed_cpu_submission_gap_ms": 0.0,
            "base_h2d_ms": 0.0,
            "merge_swiglu_ms": 0.0,
            "fused_base_residual_swiglu_ms": graph_ms,
            "cuda_graph_replay_ms": graph_ms,
            "cuda_graph_capture_ms": self._graph_capture_ms,
            "wall_ms": wall_ms,
            "activation_h2d_bytes": self.cols * 4,
            "base_h2d_bytes": (self.cols // 32) * 4,
            "base_resident_bytes": sum(
                value.numel() * value.element_size()
                for value in self.base_resident.values()
            ),
            "weight_h2d_bytes": self.cache.traffic["weight_h2d_bytes"],
            "resident_weight_h2d_bytes": 0,
            "tile_count": 1,
            "base_compute_device": "cuda",
            "kernel_mode": (
                "cuda_graph_fused_base_residual_swiglu_down"
                if down is not None
                else "cuda_graph_fused_base_residual_swiglu"
            ),
            "graph_includes_down": down is not None,
        }
        if return_outputs:
            result.update({
                name: output.cpu().numpy()
                for name, output in self.output.items()
            })
            if down is not None:
                result["down"] = down.output.cpu().numpy()
        return result

    def close(self) -> None:
        self.cache.close()

    def __enter__(self) -> TiledResidentGateUp:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
