# Copyright 2023-2024 SGLang Team
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
"""CUDA: the check_weights fan-out catches a REAL anomaly on a draft runner.

Companion to test/registered/unit/utils/test_check_weights_draft_fanout.py, which mocks each runner's
check_weights and runs on CPU (it tests selector routing / merge / labels, not real
detection). Detection MUST run on CUDA: WeightChecker._snapshot stores
`param.data.detach().cpu()`, and on a CPU model `.cpu()` aliases the live storage, so
a later mutation would not diverge from the snapshot and the mismatch would be
invisible. With CUDA params the snapshot is an independent CPU copy, so a mutated
draft weight is genuinely caught.

A realistic target ("tp_worker") + draft worker are each backed by a real
WeightChecker over real CUDA Parameters and driven through the real scheduler
fan-out: snapshot(all) -> mutate one private weight -> compare. The failure must be
reported and carry the role label of the runner that diverged ([draft] / [target]).
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.io_struct import CheckWeightsReqInput
from sglang.srt.managers.scheduler_update_weights_mixin import (
    SchedulerUpdateWeightsMixin,
)
from sglang.srt.utils.weight_checker import WeightChecker
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small")


class _TestScheduler(SchedulerUpdateWeightsMixin):
    pass


class _NamedParamModel:
    """The model surface WeightChecker reads: named_parameters() / named_buffers().
    Sharing a Parameter object between two models models tied embed/head storage."""

    def __init__(self, named_params):
        self._named_params = list(named_params)

    def named_parameters(self):
        return list(self._named_params)

    def named_buffers(self):
        return []


class _CheckerRunner:
    """A model_runner backed by a real WeightChecker, so snapshot / compare / checksum
    run their actual logic over the given CUDA Parameters (no mocking)."""

    # Single-process parallelism stand-in the checksum action's ParallelismInfo reads.
    tp_rank = 0
    tp_size = 1
    dp_rank = 0
    dp_size = 1
    pp_rank = 0
    pp_size = 1

    def __init__(self, named_params):
        self.model = _NamedParamModel(named_params)
        self._checker = WeightChecker(self)

    def check_weights(self, action):
        return self._checker.handle(action)


def _draft_worker(*pairs):
    # Fake draft worker exposing iter_draft_runners() (discovery is covered by
    # test/registered/unit/spec/test_draft_runner_discovery.py); here each runner is a real checker.
    return SimpleNamespace(iter_draft_runners=lambda: list(pairs))


def _scheduler(tp_worker, draft_worker):
    scheduler = _TestScheduler()
    scheduler.tp_worker = tp_worker
    scheduler.draft_worker = draft_worker
    return scheduler


def _call(scheduler, action, selector):
    return SchedulerUpdateWeightsMixin.check_weights(
        scheduler, CheckWeightsReqInput(action=action, selector=selector)
    )


def _param(*vals):
    return torch.nn.Parameter(
        torch.tensor(vals, device="cuda", dtype=torch.float32), requires_grad=False
    )


class TestDraftFanoutDetectionCUDA(CustomTestCase):
    """Target + draft, each a real WeightChecker over CUDA params, driven through the
    real scheduler fan-out. `embed` is shared (tied to the target); each side also
    holds a private weight so a divergence can be attributed to one runner."""

    def _build(self):
        shared = _param(1.0, 2.0, 3.0, 4.0)  # tied embed/head storage
        t_priv = _param(5.0, 6.0, 7.0, 8.0)  # target-only
        d_priv = _param(9.0, 10.0, 11.0, 12.0)  # draft-only
        target = _CheckerRunner([("embed", shared), ("t_priv", t_priv)])
        draft = _CheckerRunner([("embed", shared), ("d_priv", d_priv)])
        scheduler = _scheduler(
            SimpleNamespace(model_runner=target),
            _draft_worker(("draft", draft)),
        )
        return scheduler, shared, t_priv, d_priv

    def test_unchanged_compare_passes(self):
        # Positive control: snapshot then compare with no mutation must pass, so a
        # later failure is attributable to the mutation, not to setup noise.
        scheduler, *_ = self._build()
        self.assertTrue(_call(scheduler, "snapshot", "all").success)
        out = _call(scheduler, "compare", "all")
        self.assertTrue(out.success, out.message)

    def test_mutated_draft_weight_caught_with_draft_label(self):
        # A draft-private weight diverges from its snapshot; the real checker on the
        # draft runner must catch it and the scheduler must label it [draft].
        scheduler, _shared, _t_priv, d_priv = self._build()
        self.assertTrue(_call(scheduler, "snapshot", "all").success)
        with torch.no_grad():
            d_priv.add_(1.0)
        out = _call(scheduler, "compare", "all")
        self.assertFalse(out.success)
        self.assertIn("[draft]", out.message)
        self.assertIn("max_abs_err", out.message)  # from the real _check_tensors

    def test_mutated_target_weight_caught_with_target_label(self):
        # Symmetric sanity: a target-only divergence is attributed to [target].
        scheduler, _shared, t_priv, _d_priv = self._build()
        self.assertTrue(_call(scheduler, "snapshot", "all").success)
        with torch.no_grad():
            t_priv.add_(1.0)
        out = _call(scheduler, "compare", "all")
        self.assertFalse(out.success)
        self.assertIn("[target]", out.message)

    def test_draft_selector_catches_draft_without_touching_target(self):
        # selector="draft": a draft divergence is caught even though the target is
        # untouched and is never compared on this selection.
        scheduler, _shared, _t_priv, d_priv = self._build()
        self.assertTrue(_call(scheduler, "snapshot", "all").success)
        with torch.no_grad():
            d_priv.add_(1.0)
        out = _call(scheduler, "compare", "draft")
        self.assertFalse(out.success)
        self.assertIn("[draft]", out.message)

    def test_checksum_catches_draft_diverged_from_expected(self):
        # compare() is blind to "should have changed but didn't" (a stale runner
        # matches its own snapshot). checksum closes that gap: it hashes the live
        # state per runner, so a draft that doesn't match the EXPECTED reference
        # (held by the caller/trainer here) is caught and pinpointed. This is the
        # signal an RL trainer compares against; the checker itself only reports it.
        scheduler, _shared, _t_priv, d_priv = self._build()
        expected = _call(scheduler, "checksum", "all").payload["checksums"]
        self.assertIn("draft.d_priv", expected)  # draft hashed independently

        # The draft's weight no longer matches what the trainer pushed.
        with torch.no_grad():
            d_priv.add_(1.0)
        actual = _call(scheduler, "checksum", "all").payload["checksums"]

        # The draft-private key diverges from expected; every other key (the
        # target's and the shared embed) is unchanged, so the anomaly is the draft.
        self.assertNotEqual(actual["draft.d_priv"], expected["draft.d_priv"])
        for k in expected:
            if k != "draft.d_priv":
                self.assertEqual(actual[k], expected[k], k)


if __name__ == "__main__":
    unittest.main()
