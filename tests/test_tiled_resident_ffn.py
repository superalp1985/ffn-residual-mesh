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
