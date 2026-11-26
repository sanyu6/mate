"""FP4 symmetric layout and collective IPC probe, runnable under torchrun."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from mate._fp4_moe_buffer import FP4SymmetricLayout


def test_layout_alignment():
    offsets = FP4SymmetricLayout(8192, 4096, 6).offsets
    assert list(offsets.values()) == sorted(offsets.values())
    assert all(value % 256 == 0 for value in offsets.values())
    assert offsets["x"] == 512
    assert offsets["x_sf"] - offsets["x"] == 8192 * 4096 // 2


@pytest.mark.parametrize("ranks", [1, 8])
def test_ipc_probe(ranks):
    pytest.importorskip("torch_musa")
    if not torch.musa.is_available() or torch.musa.device_count() < ranks:
        pytest.skip("requires MP32 GPUs")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={ranks}",
            str(Path(__file__).resolve()),
        ],
        check=True,
        timeout=300,
    )


def main():
    import torch_musa  # noqa:F401
    import torch.distributed as dist
    from mate._fp4_moe_buffer import FP4FP4SymmBuffer

    rank, world, local = (
        int(os.environ[key]) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    )
    torch.musa.set_device(local)
    dist.init_process_group("mccl", device_id=torch.device("musa", local))
    sym = FP4FP4SymmBuffer(dist.group.WORLD, world * 4, 256, 2, 256, 256)
    for phase in range(3):
        sym.x[0, 0] = rank + 1 + phase * 16
        observed = sym.ipc_probe().cpu()
        assert observed.tolist() == [r + 1 + phase * 16 for r in range(world)], (
            rank,
            observed,
        )
    if rank == 0:
        print("FP4_IPC_PROBE_PASS", world, sym.layout.offsets, flush=True)
    sym.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
