from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("triton"),
                     "tiled FFN test requires the cu130 venv")
class TiledResidentFfnTests(unittest.TestCase):
    def test_tiled_gate_up_swiglu_matches_artifact_reference(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from tests.gguf_fixture import write_fixture
        from compile_resident_residual_artifact import compile_layer
        from resident_residual_format import ResidentArtifact
        from resident_tiled_ffn import TiledResidentGateUp

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "fixture.gguf")
            compile_layer(root / "fixture.gguf", 0, 4, root / "artifact")
            with ResidentArtifact.open(root / "artifact") as artifact:
                runner = TiledResidentGateUp(artifact, tile_rows=64)
                x = np.random.default_rng(560).standard_normal(runner.cols).astype(np.float32)
                result = runner.run(x)
                gate = artifact.reconstruct_weights("gate").astype(np.float64) @ x
                up = artifact.reconstruct_weights("up").astype(np.float64) @ x
                swiglu = gate / (1 + np.exp(-gate)) * up
                np.testing.assert_allclose(result["gate"], gate, rtol=1e-4, atol=1e-4)
                np.testing.assert_allclose(result["up"], up, rtol=1e-4, atol=1e-4)
                np.testing.assert_allclose(result["swiglu"], swiglu, rtol=2e-4, atol=2e-4)
                self.assertEqual(result["kernel_mode"], "fused_residual_cpu_overlap_then_merge")
                self.assertEqual(result["weight_h2d_bytes"], runner.cache.traffic["weight_h2d_bytes"])
                self.assertEqual(result["resident_weight_h2d_bytes"], 0)

    def test_tiled_runner_reports_cpu_base_and_tile_traffic_separately(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from tests.gguf_fixture import write_fixture
        from compile_resident_residual_artifact import compile_layer
        from resident_residual_format import ResidentArtifact
        from resident_tiled_ffn import TiledResidentGateUp

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "fixture.gguf")
            compile_layer(root / "fixture.gguf", 0, 4, root / "artifact")
            with ResidentArtifact.open(root / "artifact") as artifact:
                runner = TiledResidentGateUp(artifact, tile_rows=64)
                result = runner.run(np.zeros(runner.cols, dtype=np.float32))
                self.assertIn("cpu_base_ms", result)
                self.assertIn("tile_kernel_ms", result)
                self.assertEqual(result["resident_weight_h2d_bytes"], 0)
                self.assertGreater(result["weight_h2d_bytes"], 0)

    def test_nonresident_pipeline_depths_preserve_output_and_reap_async_tiles(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from tests.gguf_fixture import write_fixture
        from compile_resident_residual_artifact import compile_layer
        from resident_residual_format import ResidentArtifact
        from resident_tiled_ffn import TiledResidentGateUp

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "fixture.gguf")
            compile_layer(root / "fixture.gguf", 0, 4, root / "artifact")
            with ResidentArtifact.open(root / "artifact") as artifact:
                for depth in (1, 2, 3):
                    with self.subTest(pipeline_depth=depth):
                        runner = TiledResidentGateUp(
                            artifact,
                            tile_rows=64,
                            pipeline_depth=depth,
                        )
                        x = np.random.default_rng(565 + depth).standard_normal(
                            runner.cols
                        ).astype(np.float32)
                        expected_gate = (
                            artifact.reconstruct_weights("gate").astype(np.float64) @ x
                        )
                        expected_up = (
                            artifact.reconstruct_weights("up").astype(np.float64) @ x
                        )
                        first = runner.run(x)
                        second = runner.run(x)
                        np.testing.assert_allclose(
                            second["gate"], expected_gate, rtol=1e-4, atol=1e-4
                        )
                        np.testing.assert_allclose(
                            second["up"], expected_up, rtol=1e-4, atol=1e-4
                        )
                        self.assertGreater(second["weight_h2d_bytes"],
                                           first["weight_h2d_bytes"])
                        self.assertEqual(runner.cache.pending_layers(), [])
                        self.assertEqual(runner.cache.device_layers(), [])
                        runner.close()

    def test_persistent_tiles_have_zero_weight_h2d_after_cold_start(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from tests.gguf_fixture import write_fixture
        from compile_resident_residual_artifact import compile_layer
        from resident_residual_format import ResidentArtifact
        from resident_tiled_ffn import TiledResidentGateUp

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "fixture.gguf")
            compile_layer(root / "fixture.gguf", 0, 4, root / "artifact")
            with ResidentArtifact.open(root / "artifact") as artifact:
                runner = TiledResidentGateUp(artifact, tile_rows=64, persistent=True)
                x = np.random.default_rng(562).standard_normal(runner.cols).astype(np.float32)
                cold = runner.run(x)
                before = runner.cache.traffic["weight_h2d_bytes"]
                warm = runner.run(x)
                self.assertGreater(cold["weight_h2d_bytes"], 0)
                self.assertEqual(warm["weight_h2d_bytes"], before)
                self.assertEqual(
                    runner.cache.traffic["weight_h2d_bytes"],
                    before,
                )

    def test_tiled_runner_can_feed_original_down_projection(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from tests.gguf_fixture import write_fixture
        from compile_resident_residual_artifact import compile_layer
        from resident_residual_cuda import DirectIQ4NLProjection
        from resident_residual_format import ResidentArtifact
        from resident_tiled_ffn import TiledResidentGateUp
        from gguf import GGUFReader
        from gguf.quants import dequantize

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "fixture.gguf", quantized_down=True)
            compile_layer(root / "fixture.gguf", 0, 4, root / "artifact")
            with ResidentArtifact.open(root / "artifact") as artifact:
                reader = GGUFReader(root / "fixture.gguf")
                tensor = next(item for item in reader.tensors if item.name == "blk.0.ffn_down.weight")
                down = DirectIQ4NLProjection(tensor.data, int(tensor.shape[0]))
                x = np.random.default_rng(564).standard_normal(256).astype(np.float32)
                with TiledResidentGateUp(artifact, tile_rows=64, persistent=True) as runner:
                    result = runner.run(x, down=down)
                    expected = dequantize(tensor.data, tensor.tensor_type).astype(np.float64) @ result["swiglu"]
                    np.testing.assert_allclose(result["down"], expected, rtol=2e-4, atol=2e-4)
                    self.assertIn("down_stream_ms", result)
                reader.data._mmap.close()

    def test_gpu_base_mode_keeps_coefficients_resident_and_reduces_base_upload(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from tests.gguf_fixture import write_fixture
        from compile_resident_residual_artifact import compile_layer
        from resident_residual_format import ResidentArtifact
        from resident_tiled_ffn import TiledResidentGateUp

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "fixture.gguf")
            compile_layer(root / "fixture.gguf", 0, 4, root / "artifact")
            with ResidentArtifact.open(root / "artifact") as artifact:
                cpu_runner = TiledResidentGateUp(artifact, tile_rows=64, persistent=True)
                gpu_runner = TiledResidentGateUp(
                    artifact,
                    tile_rows=256,
                    persistent=True,
                    base_on_gpu=True,
                )
                x = np.random.default_rng(569).standard_normal(256).astype(np.float32)
                cpu = cpu_runner.run(x)
                gpu = gpu_runner.run(x)
                np.testing.assert_allclose(gpu["gate"], cpu["gate"], rtol=2e-4, atol=2e-4)
                np.testing.assert_allclose(gpu["up"], cpu["up"], rtol=2e-4, atol=2e-4)
                np.testing.assert_allclose(gpu["swiglu"], cpu["swiglu"], rtol=3e-4, atol=3e-4)
                self.assertEqual(gpu["base_compute_device"], "cuda")
                self.assertEqual(gpu["kernel_mode"], "fused_base_residual_swiglu_super_tile")
                self.assertLess(gpu["base_h2d_bytes"], cpu["base_h2d_bytes"])
                self.assertGreater(gpu["base_resident_bytes"], 0)
                cpu_runner.close()
                gpu_runner.close()

    def test_gpu_base_fp16_coefficients_reduce_resident_bytes(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from tests.gguf_fixture import write_fixture
        from compile_resident_residual_artifact import compile_layer
        from resident_residual_format import ResidentArtifact
        from resident_tiled_ffn import TiledResidentGateUp

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "fixture.gguf")
            compile_layer(root / "fixture.gguf", 0, 4, root / "artifact")
            with ResidentArtifact.open(root / "artifact") as artifact:
                fp32 = TiledResidentGateUp(
                    artifact, tile_rows=256, persistent=True, base_on_gpu=True
                )
                fp16 = TiledResidentGateUp(
                    artifact, tile_rows=256, persistent=True, base_on_gpu=True,
                    base_dtype="float16",
                )
                x = np.random.default_rng(581).standard_normal(256).astype(np.float32)
                out32 = fp32.run(x)
                out16 = fp16.run(x)
                self.assertEqual(fp32.base_resident["gate"].dtype, torch.float32)
                self.assertEqual(fp16.base_resident["gate"].dtype, torch.float16)
                self.assertEqual(
                    out16["base_resident_bytes"],
                    out32["base_resident_bytes"] // 2,
                )
                for name, tolerance in (("gate", 3e-3), ("up", 3e-3), ("swiglu", 4e-3)):
                    error = np.linalg.norm(out16[name] - out32[name]) / max(
                        np.linalg.norm(out32[name]), 1e-8
                    )
                    self.assertLess(float(error), tolerance)
                fp32.close()
                fp16.close()

    def test_fp16_base_rejects_multitile_and_nonfinite_storage(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from tests.gguf_fixture import write_fixture
        from compile_resident_residual_artifact import compile_layer
        from resident_residual_format import ResidentArtifact
        from resident_tiled_ffn import TiledResidentGateUp

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "fixture.gguf")
            compile_layer(root / "fixture.gguf", 0, 4, root / "artifact")
            with ResidentArtifact.open(root / "artifact") as artifact:
                for options, message in (
                    ({"tile_rows": 64}, "full-layer"),
                    ({"base_on_gpu": False}, "base_on_gpu"),
                    ({"base_dtype": "int8"}, "base_dtype"),
                ):
                    arguments = dict(
                        tile_rows=256, persistent=True, base_on_gpu=True,
                        base_dtype="float16",
                    )
                    arguments.update(options)
                    with self.subTest(options=options), self.assertRaisesRegex(
                        ValueError, message
                    ):
                        with TiledResidentGateUp(artifact, **arguments):
                            pass
                original = artifact.arrays["gate"]["coefficient"]
                try:
                    for value in (70000.0, float("inf"), float("nan")):
                        coefficient = np.array(original, copy=True)
                        coefficient[0, 0] = value
                        artifact.arrays["gate"]["coefficient"] = coefficient
                        with self.subTest(value=value), self.assertRaisesRegex(
                            ValueError, "finite FP16"
                        ):
                            with TiledResidentGateUp(
                                artifact, tile_rows=256, persistent=True,
                                base_on_gpu=True, base_dtype="float16",
                            ):
                                pass
                finally:
                    artifact.arrays["gate"]["coefficient"] = original

    def test_fp16_base_matches_rounding_correction_reference(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from gguf import GGUFReader
        from gguf.quants import dequantize
        from tests.gguf_fixture import write_fixture
        from compile_resident_residual_artifact import compile_layer
        from resident_residual_format import ResidentArtifact
        from resident_tiled_ffn import TiledResidentGateUp

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for types in (("Q4_K", "Q4_K"), ("Q5_K", "Q5_K"), ("Q5_K", "Q4_K")):
                with self.subTest(types=types):
                    model = root / f"{types[0]}_{types[1]}.gguf"
                    compiled = root / f"{types[0]}_{types[1]}"
                    write_fixture(model, gate_up_types=types)
                    compile_layer(model, 0, None, compiled, format_version=2)
                    reader = GGUFReader(model)
                    try:
                        tensors = {t.name: t for t in reader.tensors}
                        weights = {
                            name: dequantize(
                                tensors[f"blk.0.ffn_{name}.weight"].data,
                                tensors[f"blk.0.ffn_{name}.weight"].tensor_type,
                            ).astype(np.float64)
                            for name in ("gate", "up")
                        }
                    finally:
                        reader.data._mmap.close()
                    with ResidentArtifact.open(compiled) as artifact, TiledResidentGateUp(
                        artifact, tile_rows=256, persistent=True, base_on_gpu=True,
                        base_dtype="float16", use_cuda_graph=True,
                    ) as runner:
                        before = runner.cache.traffic["weight_h2d_bytes"]
                        for seed in (581, 582, 583):
                            x = np.random.default_rng(seed).standard_normal(256).astype(np.float32)
                            expected = {}
                            sums = x.astype(np.float64).reshape(-1, 32).sum(axis=1)
                            for name in ("gate", "up"):
                                coefficient = np.asarray(artifact.arrays[name]["coefficient"])
                                rounding = coefficient.astype(np.float16).astype(np.float64) - coefficient
                                expected[name] = weights[name] @ x + rounding @ sums
                            gate, up = expected["gate"], expected["up"]
                            expected["swiglu"] = gate * np.exp(-np.logaddexp(0, -gate)) * up
                            outputs = (runner.run(x), runner.run_device(torch.from_numpy(x).cuda()))
                            for output in outputs:
                                for name in ("gate", "up", "swiglu"):
                                    np.testing.assert_allclose(
                                        output[name], expected[name], rtol=2e-4, atol=2e-4
                                    )
                            self.assertEqual(runner.cache.traffic["weight_h2d_bytes"], before)

    def test_cuda_graph_replay_accepts_new_activation_and_preserves_output(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from tests.gguf_fixture import write_fixture
        from compile_resident_residual_artifact import compile_layer
        from resident_residual_format import ResidentArtifact
        from resident_tiled_ffn import TiledResidentGateUp

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "fixture.gguf")
            compile_layer(root / "fixture.gguf", 0, 4, root / "artifact")
            with ResidentArtifact.open(root / "artifact") as artifact:
                runner = TiledResidentGateUp(
                    artifact,
                    tile_rows=256,
                    persistent=True,
                    base_on_gpu=True,
                    use_cuda_graph=True,
                )
                x1 = np.random.default_rng(570).standard_normal(256).astype(np.float32)
                x2 = np.random.default_rng(571).standard_normal(256).astype(np.float32)
                first = runner.run(x1)
                second = runner.run(x2)
                expected_gate = artifact.reconstruct_weights("gate").astype(np.float64) @ x2
                expected_up = artifact.reconstruct_weights("up").astype(np.float64) @ x2
                np.testing.assert_allclose(second["gate"], expected_gate, rtol=2e-4, atol=2e-4)
                np.testing.assert_allclose(second["up"], expected_up, rtol=2e-4, atol=2e-4)
                self.assertEqual(first["kernel_mode"], "cuda_graph_fused_base_residual_swiglu")
                self.assertEqual(second["base_h2d_bytes"], 32)
                self.assertGreater(second["cuda_graph_replay_ms"], 0.0)
                runner.close()

    def test_cuda_graph_can_capture_resident_down_projection(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from tests.gguf_fixture import write_fixture
        from compile_resident_residual_artifact import compile_layer
        from resident_residual_cuda import DirectIQ4NLProjection
        from resident_residual_format import ResidentArtifact
        from resident_tiled_ffn import TiledResidentGateUp
        from gguf import GGUFReader
        from gguf.quants import dequantize

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "fixture.gguf", quantized_down=True)
            compile_layer(root / "fixture.gguf", 0, 4, root / "artifact")
            with ResidentArtifact.open(root / "artifact") as artifact:
                reader = GGUFReader(root / "fixture.gguf")
                tensor = next(
                    item for item in reader.tensors
                    if item.name == "blk.0.ffn_down.weight"
                )
                down = DirectIQ4NLProjection(tensor.data, int(tensor.shape[0]))
                x = np.random.default_rng(572).standard_normal(256).astype(np.float32)
                with TiledResidentGateUp(
                    artifact,
                    tile_rows=256,
                    persistent=True,
                    base_on_gpu=True,
                    use_cuda_graph=True,
                ) as runner:
                    result = runner.run(x, down=down)
                    expected = (
                        dequantize(tensor.data, tensor.tensor_type).astype(np.float64)
                        @ result["swiglu"]
                    )
                    np.testing.assert_allclose(
                        result["down"], expected, rtol=2e-4, atol=2e-4
                    )
                    self.assertTrue(result["graph_includes_down"])
                reader.data._mmap.close()

    def test_run_device_consumes_gpu_activation_without_copy(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from tests.gguf_fixture import write_fixture
        from compile_resident_residual_artifact import compile_layer
        from resident_residual_format import ResidentArtifact
        from resident_tiled_ffn import TiledResidentGateUp

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "fixture.gguf")
            compile_layer(root / "fixture.gguf", 0, 4, root / "artifact")
            with ResidentArtifact.open(root / "artifact") as artifact:
                with TiledResidentGateUp(
                    artifact,
                    tile_rows=256,
                    persistent=True,
                    base_on_gpu=True,
                ) as runner:
                    x = torch.randn(runner.cols, device="cuda", dtype=torch.float32)
                    result = runner.run_device(x)
                    expected_gate = artifact.reconstruct_weights("gate").astype(np.float64) @ x.cpu().numpy()
                    expected_up = artifact.reconstruct_weights("up").astype(np.float64) @ x.cpu().numpy()
                    np.testing.assert_allclose(
                        result["gate"], expected_gate, rtol=2e-4, atol=2e-4
                    )
                    np.testing.assert_allclose(
                        result["up"], expected_up, rtol=2e-4, atol=2e-4
                    )
                    self.assertEqual(result["activation_h2d_bytes"], 0)
                    self.assertEqual(result["activation_d2d_bytes"], 0)
                    self.assertEqual(result["base_h2d_bytes"], 0)
                    self.assertEqual(result["activation_source"], "caller_gpu_tensor")

    def test_run_device_can_chain_two_gpu_resident_layers(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from tests.gguf_fixture import write_fixture
        from compile_resident_residual_artifact import compile_layer
        from resident_residual_format import ResidentArtifact
        from resident_tiled_ffn import TiledResidentGateUp

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "fixture.gguf")
            compile_layer(root / "fixture.gguf", 0, 4, root / "artifact")
            with ResidentArtifact.open(root / "artifact") as artifact:
                with TiledResidentGateUp(
                    artifact,
                    tile_rows=256,
                    persistent=True,
                    base_on_gpu=True,
                ) as first, TiledResidentGateUp(
                    artifact,
                    tile_rows=256,
                    persistent=True,
                    base_on_gpu=True,
                ) as second:
                    stream = torch.cuda.Stream()
                    x = torch.randn(first.cols, device="cuda", dtype=torch.float32)
                    with torch.cuda.stream(stream):
                        first_result = first.run_device(
                            x, stream=stream, return_outputs=False
                        )
                        hidden = first.output["swiglu"]
                        second_result = second.run_device(
                            hidden, stream=stream, return_outputs=False
                        )
                    stream.synchronize()
                    hidden_host = hidden.cpu().numpy()
                    expected_gate = artifact.reconstruct_weights("gate").astype(np.float64) @ hidden_host
                    expected_up = artifact.reconstruct_weights("up").astype(np.float64) @ hidden_host
                    np.testing.assert_allclose(
                        second.output["gate"].cpu().numpy(),
                        expected_gate,
                        rtol=2e-4,
                        atol=2e-4,
                    )
                    np.testing.assert_allclose(
                        second.output["up"].cpu().numpy(),
                        expected_up,
                        rtol=2e-4,
                        atol=2e-4,
                    )
                    self.assertEqual(first_result["activation_d2d_bytes"], 0)
                    self.assertEqual(second_result["activation_d2d_bytes"], 0)
                    self.assertEqual(second_result["activation_h2d_bytes"], 0)

    def test_run_device_async_chain_synchronizes_once_at_end(self) -> None:
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from tests.gguf_fixture import write_fixture
        from compile_resident_residual_artifact import compile_layer
        from resident_residual_format import ResidentArtifact
        from resident_tiled_ffn import TiledResidentGateUp

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "fixture.gguf")
            compile_layer(root / "fixture.gguf", 0, 4, root / "artifact")
            with ResidentArtifact.open(root / "artifact") as artifact:
                with TiledResidentGateUp(
                    artifact,
                    tile_rows=256,
                    persistent=True,
                    base_on_gpu=True,
                ) as first, TiledResidentGateUp(
                    artifact,
                    tile_rows=256,
                    persistent=True,
                    base_on_gpu=True,
                ) as second:
                    stream = torch.cuda.Stream()
                    x = torch.randn(first.cols, device="cuda", dtype=torch.float32)
                    with torch.cuda.stream(stream):
                        first_result = first.run_device(
                            x,
                            stream=stream,
                            return_outputs=False,
                            synchronize=False,
                        )
                        second_result = second.run_device(
                            first.output["swiglu"],
                            stream=stream,
                            return_outputs=False,
                            synchronize=False,
                        )
                    self.assertFalse(first_result["synchronized"])
                    self.assertFalse(second_result["synchronized"])
                    self.assertIsNotNone(first_result["completion_event"])
                    self.assertIsNotNone(second_result["completion_event"])
                    stream.synchronize()
                    hidden = first.output["swiglu"].cpu().numpy()
                    expected_gate = artifact.reconstruct_weights("gate").astype(np.float64) @ hidden
                    expected_up = artifact.reconstruct_weights("up").astype(np.float64) @ hidden
                    np.testing.assert_allclose(
                        second.output["gate"].cpu().numpy(),
                        expected_gate,
                        rtol=2e-4,
                        atol=2e-4,
                    )
                    np.testing.assert_allclose(
                        second.output["up"].cpu().numpy(),
                        expected_up,
                        rtol=2e-4,
                        atol=2e-4,
                    )
                    self.assertEqual(second_result["activation_h2d_bytes"], 0)
                    self.assertEqual(second_result["activation_d2d_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
