#!/usr/bin/env python3
"""Check actual retained endpoint phases and their LIF holonomy readback.

This reader creates no phases, convergence records, or events. The native
runner owns source/build/dispatch provenance. Absence of a synchronized
population is reported as unexercised, unless --require-synchronized is used.
"""
import argparse
import math
import struct
from pathlib import Path

LEAVES = 8
TILE_BYTES = 225408
NUMERICAL_BYTES = 38656
PHASE_OFFSET = 24576
MAP_OFFSET = 30976
EDGE_OFFSET = 29696
COUNT_OFFSET = 225284
CONVERGENCE_BYTES = 128
EVENT_BYTES = 96


def require(value, message):
    if not value:
        raise ValueError(message)


def read_exact(path, size):
    data = Path(path).read_bytes()
    require(len(data) == size, f"wrong extent: {path}")
    return data


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def endpoint_epoch(prefix, epoch):
    stem = f"{prefix}.epoch{epoch}"
    conv = read_exact(stem + ".conv.bin", LEAVES * CONVERGENCE_BYTES)
    tiles = read_exact(stem + ".tiles.bin", LEAVES * TILE_BYTES)
    numerical = read_exact(stem + ".numerical.bin", LEAVES * NUMERICAL_BYTES)
    atoms, edges, phases, maps, traces = [], [], [], [], []
    for leaf in range(LEAVES):
        c = leaf * CONVERGENCE_BYTES
        header = struct.unpack_from("<6I", conv, c)
        require(header == (0x42505631, leaf, 0, 0, 0, 4 * epoch),
                f"unaccepted convergence epoch={epoch} leaf={leaf}")
        lam, tau = struct.unpack_from("<2f", conv, c + 48)
        require(all(math.isfinite(v) and -2.0 <= v <= 2.0 for v in (lam, tau)),
                f"trace outside [-2,2] epoch={epoch} leaf={leaf}")
        t, n = leaf * TILE_BYTES, leaf * NUMERICAL_BYTES
        count = struct.unpack_from("<I", tiles, t + COUNT_OFFSET)[0]
        require(0 < count <= 1536, "empty or oversized physical edge population")
        ids = tuple(v & 0xFFFFFF for v in struct.unpack_from("<256I", tiles, t))
        require(len(set(ids)) == 256, "duplicate physical atom IDs")
        pairs = []
        for e in range(count):
            u, v = struct.unpack_from("<2B", tiles, t + EDGE_OFFSET + 2 * e)
            require(u != v, "self edge")
            pairs.append((ids[u], ids[v]))
        require(len(set(pairs)) == count, "duplicate physical oriented edge")
        phase = struct.unpack_from(f"<{count}f", numerical, n + PHASE_OFFSET)
        require(all(math.isfinite(v) for v in phase), "nonfinite accepted phase")
        atoms.append(ids)
        edges.append(pairs)
        phases.append(phase)
        maps.append(struct.unpack_from(f"<{count}H", numerical, n + MAP_OFFSET))
        traces.append((lam, tau))
    synchronized = []
    for leaf in range(LEAVES):
        peer = (leaf + 1) % LEAVES
        require(len(edges[leaf]) == len(edges[peer]), "unequal peer population")
        require(set(maps[leaf]) == set(range(len(edges[peer]))), "incomplete peer edge bijection")
        same = True
        for e, j in enumerate(maps[leaf]):
            require(edges[leaf][e] == edges[peer][j], "peer map changes physical oriented edge")
            same = same and phases[leaf][e] == phases[peer][j]
        if same:
            require(traces[leaf][0] == 2.0, "synchronized actual phases did not produce lambda trace 2")
        if all(v == 0.0 for v in phases[leaf]):
            require(traces[leaf][1] == 2.0, "actual zero phases did not produce tau trace 2")
        synchronized.append(same)
    return atoms, traces, synchronized


def check(prefix, epochs, require_packets=False, require_synchronized=False):
    require(1 <= epochs <= 500, "invalid epoch extent")
    endpoints = {epoch: endpoint_epoch(prefix, epoch) for epoch in range(1, epochs + 1)}
    queue_path = Path(str(prefix) + ".lif_work.queue.96b")
    count_path = Path(str(prefix) + ".lif_work.queue_count.u32le")
    require(queue_path.exists() == count_path.exists(), "partial queue readback")
    covered = set()
    total = 0
    if queue_path.exists():
        raw = queue_path.read_bytes()
        require(len(raw) % EVENT_BYTES == 0, "partial packet")
        total = struct.unpack("<I", read_exact(count_path, 4))[0]
        require(0 < total <= len(raw) // EVENT_BYTES, "empty or overflowing emitted population")
        seen = set()
        for ordinal in range(total):
            o = ordinal * EVENT_BYTES
            epoch, atom = struct.unpack_from("<2I", raw, o + 12)
            holonomy = struct.unpack_from("<f", raw, o + 24)[0]
            leaf, vertex = struct.unpack_from("<2H", raw, o + 64)
            magic = struct.unpack_from("<I", raw, o + 72)[0]
            reserved = struct.unpack_from("<I", raw, o + 92)[0]
            require(epoch in endpoints and leaf < LEAVES and vertex < 192,
                    "packet outside actual endpoint population")
            require(magic == 0x56313053 and reserved == 0, "wrong packet schema")
            key = epoch, leaf, vertex
            require(key not in seen, "duplicate packet for one observation vertex")
            seen.add(key)
            atoms, traces, synchronized = endpoints[epoch]
            require(atom == atoms[leaf][vertex], "packet atom does not match actual endpoint")
            expected = max(f32(2.0 - traces[leaf][0]), 0.0)
            require(math.isfinite(holonomy) and struct.pack("<f", holonomy) == struct.pack("<f", expected),
                    "packet holonomy disagrees with actual current endpoint trace")
            if synchronized[leaf]:
                require(holonomy == 0.0, "synchronized actual observation has nonzero holonomy drive")
                covered.add((epoch, leaf))
    elif require_packets:
        raise ValueError("actual LIF packet population was not retained")
    synchronized_epochs = [e for e, (_, _, s) in endpoints.items() if all(s)]
    complete_zero_controls = [e for e in synchronized_epochs if all((e, leaf) in covered for leaf in range(LEAVES))]
    if require_synchronized:
        require(bool(complete_zero_controls), "all-eight synchronized zero-drive negative control unexercised")
    print("actual_endpoint_trace_range_pass=True", "epochs=", epochs,
          "actual_packets=", total, "synchronized_epochs=", synchronized_epochs,
          "all8_zero_drive_controls=", complete_zero_controls,
          "negative_control_exercised=", bool(complete_zero_controls),
          "full_trajectory_acceptance=False")
    return complete_zero_controls


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--require-packets", action="store_true")
    parser.add_argument("--require-synchronized", action="store_true")
    args = parser.parse_args()
    check(args.prefix, args.epochs, args.require_packets, args.require_synchronized)
