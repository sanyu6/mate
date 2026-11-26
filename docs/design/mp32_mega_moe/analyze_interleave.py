"""Validate gate/up ownership against coordinates emitted by MUTE TiledMma."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "coordinates",
    nargs="?",
    type=Path,
    default=Path(__file__).with_name("mma_coordinates.txt"),
)
args = parser.parse_args()
coords = defaultdict(dict)
for line in args.coordinates.read_text().splitlines():
    if line.startswith("TYPE"):
        print(line)
        continue
    _, name, thread, reg, row, col = line.split()
    key = int(row), int(col)
    assert key not in coords[name], (name, key)
    coords[name][key] = int(thread), int(reg)

for name, layout in coords.items():
    assert len(layout) == 256 * 256
    for gran in (1, 2, 4, 8, 16, 32, 64):
        owners = Counter()
        offsets = Counter()
        for row in range(256):
            for logical_col in range(128):
                gate_col = 2 * gran * (logical_col // gran) + logical_col % gran
                gate_thread, gate_reg = layout[row, gate_col]
                up_thread, up_reg = layout[row, gate_col + gran]
                if gate_thread == up_thread:
                    owner = "same_thread"
                    offsets[up_reg - gate_reg] += 1
                elif gate_thread // 32 == up_thread // 32:
                    owner = "same_warp"
                elif gate_thread // 128 == up_thread // 128:
                    owner = "same_squad"
                else:
                    owner = "cross_squad"
                owners[owner] += 1
        print(name, f"gran={gran}", dict(owners), "reg_delta", dict(offsets))
        if (name, gran) == ("SS", 4):
            assert owners == {"same_thread": 32768}
            assert offsets == {1: 32768}
        if (name, gran) == ("SS", 16):
            assert owners == {"same_thread": 32768}
            assert offsets == {4: 32768}
        if (name, gran) == ("RS_W4A8", 32):
            assert owners == {"same_thread": 32768}
            assert offsets == {32: 32768}

    # Check which threads jointly own a row's 32 activation columns.
    gran = 16 if name == "SS" else 32
    widths = Counter()
    groups = Counter()
    for row in range(256):
        for quant_group in range(4):
            ownership = defaultdict(list)
            for logical_col in range(quant_group * 32, (quant_group + 1) * 32):
                gate_col = 2 * gran * (logical_col // gran) + logical_col % gran
                thread, reg = layout[row, gate_col]
                ownership[thread].append((logical_col, reg))
            widths[len(ownership)] += 1
            if name == "SS":
                assert len(ownership) == 4
                assert len({thread // 4 for thread in ownership}) == 1
                assert all(len(values) == 8 for values in ownership.values())
                groups["4 adjacent lanes, 8 values each"] += 1
            else:
                assert len(ownership) == 32
                assert len({thread // 128 for thread in ownership}) == 1
                lane_columns = {thread % 4 for thread in ownership}
                assert len(lane_columns) == 1
                assert len({thread // 32 for thread in ownership}) == 4
                warp_owners = defaultdict(list)
                for thread in ownership:
                    warp_owners[thread // 32].append(thread % 32)
                lane_col = next(iter(lane_columns))
                for lanes in warp_owners.values():
                    assert sorted(lanes) == [lane_col + 4 * i for i in range(8)]
                groups["32 lanes spread across 4 warps in one squad"] += 1
    print(name, "K32 quant ownership", dict(widths), dict(groups))

    if name == "SS":
        for row in range(256):
            for block in range(8):
                for lane_col in range(4):
                    gate = [
                        layout[row, block * 32 + lane_col + 4 * i] for i in range(4)
                    ]
                    up = [
                        layout[row, block * 32 + 16 + lane_col + 4 * i]
                        for i in range(4)
                    ]
                    assert len({t for t, _ in gate + up}) == 1
                    gate_regs = [r for _, r in gate]
                    assert gate_regs == list(range(gate_regs[0], gate_regs[0] + 4))
                    assert [r for _, r in up] == [r + 4 for r in gate_regs]
        print(
            "SS gran16: four consecutive gate fragment elements followed by four up elements"
        )

    if name == "RS_W4A8":
        for logical_col in range(128):
            physical_col = 64 * (logical_col // 32) + logical_col % 32
            for token_block in range(0, 256, 16):
                for lane_col in range(4):
                    rows = [token_block + lane_col + 4 * i for i in range(4)]
                    gate = [layout[row, physical_col] for row in rows]
                    up = [layout[row, physical_col + 32] for row in rows]
                    assert len({t for t, _ in gate + up}) == 1
                    gate_regs = [r for _, r in gate]
                    assert gate_regs == list(range(gate_regs[0], gate_regs[0] + 4))
                    assert [r for _, r in up] == [r + 32 for r in gate_regs]
        print(
            "RS gran32: contiguous float4 gate/up fragments over four tokens, pair delta=32"
        )
        print("RS K32 warp lanes have stride 4: shuffle XOR 4/8/16 grouping verified")

print("PASS: tile coverage, all pairing permutations, and K32 quant ownership")
