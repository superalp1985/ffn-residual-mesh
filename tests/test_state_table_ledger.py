import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from estimate_exact_state_table_ledger import exact_residual_bytes, state_table_ledger


def test_four_state_blocks_have_expected_table_and_runtime_sizes() -> None:
    ledger = state_table_ledger(
        input_dim=2048,
        output_rows=6144,
        block_size=4,
        states_per_value=4,
        table_entry_bytes=2,
    )

    assert ledger["input_blocks"] == 512
    assert ledger["states_per_block"] == 256
    assert ledger["table_bytes"] == 1_610_612_736
    assert ledger["runtime_table_read_bytes_per_token"] == 6_291_456
    assert ledger["runtime_read_reduction_vs_fp16_dense"] == 0.75


def test_exact_two_bit_residual_is_lossless_and_packed() -> None:
    residual = exact_residual_bytes(rows=6144, input_dim=2048, residual_bits=2, metadata_bytes=1_572_864)

    assert residual["lossless"] is True
    assert residual["packed_residual_bytes"] == 3_145_728
    assert residual["total_gpu_payload_bytes"] == 4_718_592


def test_single_value_fp16_state_table_is_already_terabyte_scale() -> None:
    ledger = state_table_ledger(
        input_dim=2048,
        output_rows=6144,
        block_size=1,
        states_per_value=65536,
        table_entry_bytes=2,
    )

    assert ledger["table_bytes"] == 1_649_267_441_664
    assert ledger["runtime_table_read_bytes_per_token"] == 25_165_824
