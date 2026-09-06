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
        block_rows: int = 2,
        num_warps: int = 2,
        pipeline_depth: int = 3,
        device: str | torch.device = "cuda",
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("TiledResidentGateUp requires CUDA")
        self.artifact = artifact
        self.device = torch.device(device)
        self.base_on_gpu = bool(base_on_gpu)
        self.use_cuda_graph = bool(use_cuda_graph)
        self._persistent = bool(persistent)
        if block_rows not in (1, 2, 4, 8):
            raise ValueError("block_rows must be one of 1, 2, 4, 8")
        if num_warps not in (2, 4, 8):
            raise ValueError("num_warps must be 2, 4, or 8")
        if pipeline_depth < 1:
            raise ValueError("pipeline_depth must be positive")
        self.block_rows = int(block_rows)
        self.num_warps = int(num_warps)
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
        # Keep three transient tiles available so copy and compute can run as
        # a producer/consumer ring. Persistent layers do not need staging.
        self._pipeline_depth = int(pipeline_depth) if not self._persistent else 0
        staging_slots = max(2, self._pipeline_depth)
        self.cache = ResidentTileCache(
            arrays,
            plan=self.plan,
            vram_budget_bytes=(
                self.plan.total_bytes(cols=self.cols, alpha_cols=self.cols // 32)
                if self._persistent else tile_bytes * staging_slots
            ),
            device=self.device,
        )
        if self._persistent:
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
                block_rows=self.block_rows,
                num_warps=self.num_warps,
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
                block_rows=self.block_rows,
                num_warps=self.num_warps,
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
                    block_rows=self.block_rows,
                    num_warps=self.num_warps,
                )
                end_event.record()
                residual_end.record()
                merge_end.record()
            tile_events.append((begin_event, end_event))
            self.cache.release(tile)
        else:
            tile_ranges = self.plan.tile_slices()
            pipelined = len(tile_ranges) > 1 and self._pipeline_depth > 0
            pending_done: dict[int, torch.cuda.Event] = {}

            def retire_tile(tile_index: int) -> None:
                """Release a tile only after its compute event is complete."""
                done = pending_done.pop(tile_index, None)
                if done is not None and not done.query():
                    done.synchronize()
                # Finalize the copy ticket before dropping the device tensor.
                # This also clears a published async ticket; otherwise a
                # later run could observe a stale ticket without its payload.
                try:
                    self.cache.wait_prefetch(tile_index)
                except RuntimeError:
                    pass
                if tile_index in self.cache.device_layers():
                    self.cache.release(tile_index)
                    self.cache.evict(tile_index)

            if pipelined:
                for tile_index in range(min(self._pipeline_depth, len(tile_ranges))):
                    self.cache.prefetch_async([tile_index])

            for tile, (start, stop) in enumerate(tile_ranges):
                if pipelined:
                    if tile not in self.cache.device_layers():
                        if tile not in self.cache.pending_layers():
                            self.cache.prefetch_async([tile])
                        self.cache.wait_prefetch(tile, stream=self.stream)
                else:
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
                        block_rows=self.block_rows,
                        num_warps=self.num_warps,
                    )
                    end_event.record()
                tile_events.append((begin_event, end_event))
                if pipelined:
                    pending_done[tile] = end_event
                    retire_index = tile - self._pipeline_depth + 1
                    if retire_index >= 0:
                        retire_tile(retire_index)
                    next_tile = tile + 1
                    if (
                        tile >= self._pipeline_depth - 1
                        and next_tile < len(tile_ranges)
                    ):
                        self.cache.prefetch_async([next_tile])
                else:
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
        if not full_gpu_fused and "pipelined" in locals() and pipelined:
            for tile_index in list(pending_done):
                retire_tile(tile_index)
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

    def run_device(
        self,
        activation: torch.Tensor,
        *,
        down=None,
        stream: torch.cuda.Stream | None = None,
        return_outputs: bool = True,
        synchronize: bool = True,
        measure_events: bool = True,
    ) -> dict[str, object]:
        """Run a full resident layer from a CUDA activation without host staging.

        This path is intended for chaining FFN layers or adjacent GPU-resident
        operators. It computes group sums on CUDA, so no activation or base
        vector crosses H2D. A caller may pass one shared stream to impose
        ordering across several layer runners. Set ``synchronize=False`` and
        ``return_outputs=False`` to enqueue several layers and synchronize the
        shared stream once at the end; the returned completion event marks the
        device output boundary.
        """
        if not synchronize and return_outputs:
            raise ValueError("asynchronous device runs require return_outputs=False")
        expected_device = self.device.index
        if expected_device is None:
            expected_device = torch.cuda.current_device()
        if (
            activation.device.type != "cuda"
            or activation.device.index != expected_device
            or activation.dtype != torch.float32
        ):
            raise ValueError("device activation must be a CUDA fp32 tensor on the runner device")
        if activation.ndim != 1 or activation.numel() != self.cols:
            raise ValueError("device activation shape does not match gate/up input width")
        if not activation.is_contiguous():
            raise ValueError("device activation must be contiguous")
        if not self.base_on_gpu or len(self.plan.tile_slices()) != 1:
            raise RuntimeError(
                "run_device requires base_on_gpu=True and a full resident super-tile"
            )
        if down is not None and (down.cols != self.rows or down.rows != self.cols):
            raise ValueError("down projection dimensions must reverse gate/up dimensions")

        # With no explicit stream, use the caller's current stream. This makes
        # a tensor produced by the preceding GPU layer safe to consume without
        # an implicit host round-trip or an unknown cross-stream dependency.
        run_stream = stream or torch.cuda.current_stream(self.device)
        begin = time.perf_counter()
        base_begin = torch.cuda.Event(enable_timing=True) if measure_events else None
        base_end = torch.cuda.Event(enable_timing=True) if measure_events else None
        fused_begin = torch.cuda.Event(enable_timing=True) if measure_events else None
        fused_end = torch.cuda.Event(enable_timing=True) if measure_events else None
        down_begin = (
            torch.cuda.Event(enable_timing=True)
            if down is not None and measure_events else None
        )
        down_end = (
            torch.cuda.Event(enable_timing=True)
            if down is not None and measure_events else None
        )

        assert self.device_group_sums is not None
        # The device path already receives activation from the caller's CUDA
        # stream. Keep the reduction on that same stream: a second stream would
        # add an event dependency without hiding any H2D work.
        with torch.cuda.stream(run_stream):
            if base_begin is not None:
                base_begin.record()
            torch.sum(
                activation.view(-1, 32),
                dim=1,
                dtype=torch.float32,
                out=self.device_group_sums,
            )
            if base_end is not None:
                base_end.record()

        with torch.cuda.stream(run_stream):
            # Operations submitted to one CUDA stream are already ordered.
            # Waiting on an event recorded by this same stream adds no DMA
            # overlap and only creates another dependency node.
            self.cache.acquire(0)
            package = self.cache.package(0)
            if fused_begin is not None:
                fused_begin.record()
            launch_fused_gate_up_base_residual(
                package["gate.residual"],
                package["gate.alpha"],
                package["up.residual"],
                package["up.alpha"],
                self.base_resident["gate"],
                self.base_resident["up"],
                self.device_group_sums,
                activation,
                self.output["gate"],
                self.output["up"],
                self.output["swiglu"],
                rows=self.rows,
                cols=self.cols,
                block_rows=self.block_rows,
                num_warps=self.num_warps,
            )
            if fused_end is not None:
                fused_end.record()
            if down is not None:
                if down_begin is not None:
                    down_begin.record()
                down.launch(self.output["swiglu"])
                if down_end is not None:
                    down_end.record()
            completion_event = torch.cuda.Event(enable_timing=True)
            completion_event.record()
            self.cache.release(0)

        enqueue_wall_ms = (time.perf_counter() - begin) * 1000
        if not synchronize:
            result: dict[str, object] = {
                "cpu_base_ms": 0.0,
                "activation_h2d_ms": 0.0,
                "activation_d2d_ms": 0.0,
                "base_compute_ms": None,
                "tile_kernel_ms": None,
                "fused_base_residual_swiglu_ms": None,
                "base_h2d_ms": 0.0,
                "merge_swiglu_ms": 0.0,
                "wall_ms": None,
                "enqueue_wall_ms": enqueue_wall_ms,
                "activation_h2d_bytes": 0,
                "activation_d2d_bytes": 0,
                "base_h2d_bytes": 0,
                "base_resident_bytes": sum(
                    value.numel() * value.element_size()
                    for value in self.base_resident.values()
                ),
                "weight_h2d_bytes": self.cache.traffic["weight_h2d_bytes"],
                "resident_weight_h2d_bytes": 0,
                "tile_count": 1,
                "base_compute_device": "cuda",
                "kernel_mode": "device_activation_fused_base_residual_swiglu",
                "activation_source": "caller_gpu_tensor",
                "synchronized": False,
                "timing_events": measure_events,
                "completion_event": completion_event,
            }
            if down is not None:
                result["down_stream_ms"] = None
            return result

        completion_event.synchronize()
        wall_ms = (time.perf_counter() - begin) * 1000
        result: dict[str, object] = {
            "cpu_base_ms": 0.0,
            "activation_h2d_ms": 0.0,
            "activation_d2d_ms": 0.0,
            "base_compute_ms": (
                float(base_begin.elapsed_time(base_end))
                if base_begin is not None and base_end is not None else None
            ),
            "tile_kernel_ms": (
                float(fused_begin.elapsed_time(fused_end))
                if fused_begin is not None and fused_end is not None else None
            ),
            "fused_base_residual_swiglu_ms": (
                float(fused_begin.elapsed_time(fused_end))
                if fused_begin is not None and fused_end is not None else None
            ),
            "base_h2d_ms": 0.0,
            "merge_swiglu_ms": 0.0,
            "wall_ms": wall_ms,
            "activation_h2d_bytes": 0,
            "activation_d2d_bytes": 0,
            "base_h2d_bytes": 0,
            "base_resident_bytes": sum(
                value.numel() * value.element_size()
                for value in self.base_resident.values()
            ),
            "weight_h2d_bytes": self.cache.traffic["weight_h2d_bytes"],
            "resident_weight_h2d_bytes": 0,
            "tile_count": 1,
            "base_compute_device": "cuda",
            "kernel_mode": "device_activation_fused_base_residual_swiglu",
            "activation_source": "caller_gpu_tensor",
            "synchronized": True,
            "timing_events": measure_events,
        }
        if down is not None:
            result["down_stream_ms"] = (
                float(down_begin.elapsed_time(down_end))
                if down_begin is not None and down_end is not None else None
            )
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
