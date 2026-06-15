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
"""BaseRunner — the surface shared by every phase runner.

Two kinds of runner subclass this:

  - ``BaseCudaGraphRunner`` (and its ``Decode``/``PrefillCudaGraphRunner``
    subclasses) — capture a ``torch.cuda.CUDAGraph`` per shape and replay it.
  - ``EagerRunner`` (upcoming) — no capture; runs ``model.forward`` live each
    iteration over one static batch.

``BaseRunner`` holds only what both share: the shared ``__init__`` and the
abstract per-iteration entry points (``can_run_graph`` dispatch gate,
``load_batch`` copy-into-static, ``execute`` run). All capture/shape machinery
(``capture`` / ``capture_prepare`` / ``capture_one_shape`` / bucket padding / the
graph-only ``ExecutionBackend``) lives on ``BaseCudaGraphRunner``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.batch_overlap.two_batch_overlap import TboCudaGraphRunnerPlugin
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class BaseRunner(ABC):
    """Abstract base shared by the cuda-graph runners and the eager runner.

    Methods:
      - can_run_graph(forward_batch) — should forward_batch go through cuda
        graph replay (vs eager fallback)? Dispatch gate; the eager runner
        always returns True.
      - load_batch(forward_batch, ...) — copy the live fb into the runner's
        static buffers and refresh dynamic attention metadata.
      - execute(forward_batch, ...) — run one batch (graph replay for the
        cuda-graph runners; model.forward for eager) and slice to raw size.
    """

    def __init__(self, model_runner: ModelRunner) -> None:
        self.model_runner = model_runner
        self.device = model_runner.device
        self.device_module = torch.get_device_module(self.device)
        self.tp_size = model_runner.server_args.tp_size
        self.dp_size = model_runner.server_args.dp_size
        self.pp_size = model_runner.server_args.pp_size
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.tbo_plugin = TboCudaGraphRunnerPlugin()

    @abstractmethod
    def can_run_graph(self, forward_batch: ForwardBatch) -> bool: ...

    @abstractmethod
    def load_batch(
        self,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any: ...

    @abstractmethod
    def execute(
        self,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any: ...
