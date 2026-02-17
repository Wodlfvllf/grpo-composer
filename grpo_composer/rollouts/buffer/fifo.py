
from ...interfaces import Buffer, BufferEntry
import torch
import torch.nn as nn
from collections import deque

class FIFOBuffer:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.queue = deque()

    def insert(self, entry):
        if len(self.queue) >= self.max_size:
            self.queue.popleft()  # maintain fixed size
        self.queue.append(entry)

    def get_all(self):
        items = list(self.queue)
        self.queue.clear()
        return items

    def clear(self):
        self.queue.clear()

    def current_size(self):
        return len(self.queue)

