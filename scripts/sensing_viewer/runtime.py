from __future__ import annotations

import logging
from dataclasses import dataclass
from queue import Empty, Full


LOGGER = logging.getLogger(__name__)


@dataclass
class FrameBuffer:
    frame_id: int = 0
    total_chunks: int = 0
    received_chunks: int = 0

    def __post_init__(self) -> None:
        self.buffer = [None] * 1024

    def init(self, frame_id: int, total_chunks: int) -> None:
        self.frame_id = int(frame_id)
        self.total_chunks = int(total_chunks)
        self.buffer = [None] * self.total_chunks
        self.received_chunks = 0

    def add_chunk(self, chunk_id: int, data: bytes) -> bool:
        if chunk_id < self.total_chunks and self.buffer[chunk_id] is None:
            self.buffer[chunk_id] = data
            self.received_chunks += 1
            return self.received_chunks == self.total_chunks
        return False

    def assemble_payload(self) -> tuple[int, bytes]:
        return self.frame_id, b"".join(self.buffer[:self.total_chunks])


def drop_oldest_then_put(queue_obj, item, *, queue_name: str = "viewer queue") -> bool:
    if queue_obj.full():
        try:
            queue_obj.get_nowait()
            LOGGER.warning("%s full; dropping oldest queued item", queue_name)
        except Empty:
            pass
    try:
        queue_obj.put_nowait(item)
        return True
    except Full:
        LOGGER.warning("%s remained full; dropping newest item", queue_name)
        return False
