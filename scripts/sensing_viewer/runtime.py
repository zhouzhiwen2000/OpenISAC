from __future__ import annotations

from dataclasses import dataclass


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


def drop_oldest_then_put(queue_obj, item) -> None:
    if queue_obj.full():
        try:
            queue_obj.get_nowait()
        except Exception:
            pass
    queue_obj.put(item)
