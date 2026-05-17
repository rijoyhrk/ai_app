from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CodeChunk:
    chunk_id: str
    file_path: str
    file_type: str          # sql | python | shell
    chunk_type: str         # statement | function | class | block | command_group
    raw_text: str
    line_start: int
    line_end: int
    metadata: dict = field(default_factory=dict)

    def to_embedding_text(self) -> str:
        return self.raw_text

    def summary(self) -> str:
        return (
            f"[{self.file_type.upper()}] {self.file_path} "
            f"lines {self.line_start}-{self.line_end} ({self.chunk_type})"
        )
