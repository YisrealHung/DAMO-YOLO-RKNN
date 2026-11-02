#!/usr/bin/env python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
# torch>=2.6: torch.load defaults to weights_only=True
import torch as _torch
_torch_load = _torch.load


def _compat_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _torch_load(*args, **kwargs)


_torch.load = _compat_torch_load
from .apis import *
from .base_models import *
from .config import *
from .detectors import *
from .utils import *

__version__ = '0.1.0'
