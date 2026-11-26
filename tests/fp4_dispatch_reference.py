"""Independent CPU routing oracle; does not use the device scheduler/layout."""

import struct


def dispatch_reference(peer_inputs, owner, experts):
    groups = experts // len(peer_inputs)
    rows = [[] for _ in range(groups)]
    invalid = False
    for source, peer in enumerate(peer_inputs):
        for token, ids in enumerate(peer["ids"]):
            enabled = [e for e in ids if e != -1]
            if len(enabled) != len(set(enabled)):
                raise ValueError("expert IDs must be unique per token")
            for slot, expert in enumerate(ids):
                if not 0 <= expert < experts:
                    invalid |= expert != -1
                    continue
                if expert // groups != owner:
                    continue
                bits = struct.unpack(
                    "<i", struct.pack("<f", peer["weights"][token][slot])
                )[0]
                rows[expert % groups].append(
                    (
                        (source, token, slot, bits),
                        peer["payload"][token],
                        peer["scales"][token],
                    )
                )
    offsets = [0]
    for r in rows:
        offsets.append(offsets[-1] + ((len(r) + 31) // 32) * 32)
    # Invalid IDs are reported on the rank providing that input, not globally.
    return rows, offsets, invalid
