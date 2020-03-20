from abc import ABC
from collections import deque

from typing import Deque


class GlobalMemory(ABC):
    _memory: Deque = deque(maxlen=100000)

    def append(self, x):
        self._memory.append(x)

    def __getitem__(self, item):
        return self._memory[item]

    def __setitem__(self, key, value):
        self._memory[key] = value