# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Lazy runtime checker for the WAR fine-grained fast path.

The fast path records a read-done event right after replay_prepare, assuming
the captured decode graph reads only its static metadata snapshot, never the
live shared pool (req_to_token). This verifies that behaviorally on an early
real decode -- where the KV cache already holds varied content: rebuild the
snapshot from a collapsed page table (control) and, separately, mutate the live
pool after the snapshot is built; the output stays unchanged iff the graph
reads only the snapshot.

Capture-time / warmup batches use a trivial prompt -> uniform KV -> the probe
has no observable signal there; the first real, long-enough decode does.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )

logger = logging.getLogger(__name__)

# Batches shorter than this carry too little KV signal to probe (warmup); skip
# them without consuming an attempt and wait for a real, longer decode.
_MIN_PROBE_LEN = 32
# Give up after this many inconclusive probes so a pathological (uniform-KV)
# workload cannot retry -- and pay the probe overhead -- forever.
_MAX_ATTEMPTS = 8


def maybe_run_war_fastpath_check(
    runner: DecodeCudaGraphRunner, forward_batch: ForwardBatch
) -> None:
    """One-shot (per server) check gating model_runner.shared_buf_read_done_safe.

    Runs on the first real, long-enough non-spec decode replays until it reaches
    a conclusive verdict. SAFE -> enable; UNSAFE / give-up -> stay disabled (the
    conservative whole-forward wait_stream).
    """
    mr = runner.model_runner
    if mr._war_fastpath_checked:
        return
    # The fast path is only taken on the non-spec decode cuda-graph path; skip
    # configs this probe does not model.
    if (
        not mr.spec_algorithm.is_none()
        or runner.pp_size > 1
        or runner.enable_pdmux
        or runner.enable_two_batch_overlap
    ):
        mr._war_fastpath_checked = True
        return
    bs = forward_batch.batch_size
    if forward_batch.seq_lens_cpu is None or bs < 1:
        return
    if int(forward_batch.seq_lens_cpu[:bs].max()) < _MIN_PROBE_LEN:
        return  # too short to carry signal; wait for a larger real batch

    try:
        verdict = _run_check(runner, forward_batch)
    except Exception as e:  # never let the checker break serving
        logger.warning(
            "WAR fast-path isolation check errored (%r); keeping whole-forward barrier.",
            e,
        )
        verdict = "error"
    finally:
        mr.shared_buf_read_done_fresh = False  # probe replays toggled it

    if verdict == "safe":
        mr.shared_buf_read_done_safe = True
        mr._war_fastpath_checked = True
        logger.info("WAR fast-path enabled: decode graph reads static snapshot only.")
    elif verdict == "unsafe":
        mr._war_fastpath_checked = True
        logger.warning(
            "WAR fast-path disabled: decode graph reads the live pool in-graph; "
            "using whole-forward barrier."
        )
    else:  # inconclusive / error -> retry on a later decode
        mr._war_fastpath_check_attempts += 1
        if mr._war_fastpath_check_attempts >= _MAX_ATTEMPTS:
            mr._war_fastpath_checked = True
            logger.info(
                "WAR fast-path disabled (%s after %d attempts); using whole-forward "
                "barrier.",
                verdict,
                mr._war_fastpath_check_attempts,
            )


def _run_check(runner: DecodeCudaGraphRunner, forward_batch: ForwardBatch) -> str:
    mr = runner.model_runner
    device = runner.device
    req_to_token = mr.req_to_token_pool.req_to_token

    bs = forward_batch.batch_size
    pool_idx = forward_batch.req_pool_indices[:bs]
    # Cover every used column; the graph reads only up to each request's seq_len.
    max_len = int(forward_batch.seq_lens_cpu[:bs].max())
    cols = torch.arange(max_len, device=device)

    saved = req_to_token[pool_idx[:, None], cols[None, :]].clone()
    # Collapse every position to the request's first KV slot. This changes the
    # ATTENDED SET, not just its order -- a permutation (e.g. roll) would leave
    # the output unchanged because attention is order-invariant over the key
    # set. The collapsed entry is still a real, in-range slot.
    scribble = saved[:, :1].repeat(1, max_len)

    def set_rows(values: torch.Tensor) -> None:
        req_to_token[pool_idx[:, None], cols[None, :]] = values

    def logits_of(output) -> Optional[torch.Tensor]:
        ntl = getattr(output, "next_token_logits", None)
        return None if ntl is None else ntl[:bs].float().clone()

    def replay_with(snapshot_rows: torch.Tensor, live_rows: torch.Tensor):
        # Build the snapshot from snapshot_rows, then set the live pool to
        # live_rows before the captured graph runs. forward_metadata_ready=False
        # forces replay_prepare to rebuild the snapshot.
        forward_batch.forward_metadata_ready = False
        set_rows(snapshot_rows)
        runner.replay_prepare(forward_batch)
        set_rows(live_rows)
        return logits_of(runner.backend.replay(runner._replay_graph_key, forward_batch))

    # Isolate the live-pool mutation from any concurrent overlap forward.
    runner.device_module.synchronize()
    try:
        with runner.backend.replay_session():
            out_clean = replay_with(saved, saved)
            # control: snapshot from the collapsed table -> output must change,
            # else the probe is blind on this batch (degenerate KV).
            out_ctrl = replay_with(scribble, scribble)
            # test: clean snapshot, live pool collapsed afterwards -> output
            # stays == clean iff the graph reads only the snapshot.
            out_live = replay_with(saved, scribble)
    finally:
        set_rows(saved)
        # Make the subsequent real replay_prepare rebuild from the restored pool.
        forward_batch.forward_metadata_ready = False
        runner.device_module.synchronize()

    if out_clean is None or out_ctrl is None or out_live is None:
        return "inconclusive"
    if torch.allclose(out_ctrl, out_clean):
        return "inconclusive"
    return "safe" if torch.allclose(out_live, out_clean) else "unsafe"
