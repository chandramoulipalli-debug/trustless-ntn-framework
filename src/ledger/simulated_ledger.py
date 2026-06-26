"""
Simulated permissioned distributed ledger.
Models: immutable trust evidence records, PBFT-like consensus delay,
network partition (split into components), and post-reconnection reconciliation.
No real blockchain — pure Python simulation for reproducible benchmarks.
"""

import time
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrustRecord:
    """One immutable ledger entry (trust evidence record)."""
    record_id: int
    from_node: int
    to_node: int
    quality: float
    timestamp_ms: float          # UNIX-epoch milliseconds (simulated)
    delay_ms: float              # propagation delay that applied
    pseudonym_from: str
    pseudonym_to: str
    signature: str               # simulated — "sig_{from}_{to}_{ts}"
    is_valid: bool = True

    def age_ms(self, current_time_ms: float) -> float:
        return current_time_ms - self.timestamp_ms


@dataclass
class Block:
    block_id: int
    records: list[TrustRecord]
    timestamp_ms: float
    consensus_delay_ms: float    # time taken for consensus round


class SimulatedLedger:
    """
    Permissioned ledger with:
    - Immutable record storage
    - Simulated PBFT consensus delay per block
    - Network partition support (split into two independent ledgers)
    - Reconciliation on reconnection with trust aging applied
    """

    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.blocks: list[Block] = []
        self.pending: list[TrustRecord] = []
        self.record_counter = 0
        self.current_time_ms = 0.0
        self.batch_size = cfg["ledger"]["batch_size"]
        self.consensus_delay_range = cfg["ledger"]["consensus_delay_ms"]
        self._partitioned = False
        self._partition_ledger: Optional["SimulatedLedger"] = None
        self.throughput_log: list[dict] = []  # for Exp 3

    def submit_record(self, from_node: int, to_node: int,
                      quality: float, delay_ms: float) -> TrustRecord:
        """Add a trust evidence record to the pending pool."""
        record = TrustRecord(
            record_id=self.record_counter,
            from_node=from_node,
            to_node=to_node,
            quality=quality,
            timestamp_ms=self.current_time_ms,
            delay_ms=delay_ms,
            pseudonym_from=f"pseu_{from_node}_{int(self.current_time_ms) % 1000}",
            pseudonym_to=f"pseu_{to_node}_{int(self.current_time_ms) % 1000}",
            signature=f"sig_{from_node}_{to_node}_{self.current_time_ms:.0f}",
        )
        self.record_counter += 1
        self.pending.append(record)
        # Auto-commit when batch is full
        if len(self.pending) >= self.batch_size:
            self._commit_block()
        return record

    def _commit_block(self):
        """Commit pending records as a new block after consensus delay."""
        if not self.pending:
            return
        consensus_delay = float(self.rng.uniform(*self.consensus_delay_range))
        block = Block(
            block_id=len(self.blocks),
            records=list(self.pending),
            timestamp_ms=self.current_time_ms,
            consensus_delay_ms=consensus_delay,
        )
        self.blocks.append(block)
        n = len(self.pending)
        self.pending.clear()
        self.throughput_log.append({
            "block_id": block.block_id,
            "n_records": n,
            "consensus_delay_ms": consensus_delay,
            "tps": n / (consensus_delay / 1000 + 1e-9),
        })

    def tick(self, delta_ms: float):
        """Advance simulated time."""
        self.current_time_ms += delta_ms

    def partition(self) -> "SimulatedLedger":
        """
        Simulate network partition: return a second ledger instance
        that operates independently from this point.
        Both carry on with local views only.
        """
        self._partitioned = True
        fork = SimulatedLedger(self.cfg, self.rng)
        fork.blocks = list(self.blocks)           # shared history up to split
        fork.current_time_ms = self.current_time_ms
        fork.record_counter = self.record_counter
        self._partition_ledger = fork
        return fork

    def reconcile(self, fork: "SimulatedLedger",
                  trust_engine) -> int:
        """
        Merge divergent ledger histories after reconnection.
        Applies trust aging to records from the isolated partition
        so stale trust does not carry over unchecked.
        Returns number of records reconciled.
        """
        existing_ids = {r.record_id
                        for b in self.blocks for r in b.records}
        new_records = [r for b in fork.blocks for r in b.records
                       if r.record_id not in existing_ids]

        staleness_count = 0
        for rec in new_records:
            age = rec.age_ms(self.current_time_ms)
            # Only accept records; aging is applied via trust_engine.update
            # with full elapsed time (age = staleness) as delta_t
            trust_engine.update(
                rec.from_node, rec.to_node, rec.quality, age
            )
            staleness_count += 1

        # Merge all new blocks into main ledger
        for blk in fork.blocks:
            if blk.block_id >= len(self.blocks):
                self.blocks.append(blk)

        self._partitioned = False
        return staleness_count

    def get_records_for_node(self, node_id: int) -> list[TrustRecord]:
        return [r for b in self.blocks for r in b.records
                if r.to_node == node_id]

    def total_records(self) -> int:
        return sum(len(b.records) for b in self.blocks)

    def ledger_size_kb(self) -> float:
        # Approx: each record is ~200 bytes
        return self.total_records() * 200 / 1024

    def avg_tps(self) -> float:
        if not self.throughput_log:
            return 0.0
        return float(np.mean([x["tps"] for x in self.throughput_log]))
