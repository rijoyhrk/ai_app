from dataclasses import dataclass, field
from typing import List, Dict, Any
from src.chunkers.base import CodeChunk
from src.extractors.entity_extractor import ChunkEntities


@dataclass
class EnrichedDocument:
    chunk_id: str
    file_path: str
    file_type: str
    chunk_type: str
    line_start: int
    line_end: int
    raw_text: str
    # extracted entities
    canonical_tables: List[str]
    aliases_used: List[str]
    operations: List[str]      # READ, WRITE, CREATE, etc.
    # text sent to embedding model
    embedding_text: str
    # full metadata for ChromaDB
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_enriched_document(
    chunk: CodeChunk,
    entities: ChunkEntities,
) -> EnrichedDocument:
    """
    Combine a code chunk with its extracted entities into an enriched document.
    The embedding_text weaves together raw code + canonical entity metadata so
    semantic search bridges aliases (cust_tab) to canonical names (customer).
    """
    canonical_tables = []
    aliases_used = []
    operations = []

    for ref in entities.refs:
        if ref.canonical is None or ref.ref_type == "COLUMN":
            continue
        canonical = ref.canonical.lower().strip()
        raw = ref.raw.lower().strip()
        if canonical and canonical not in canonical_tables:
            canonical_tables.append(canonical)
        if raw and raw != canonical and raw not in aliases_used:
            aliases_used.append(raw)
        if ref.ref_type and ref.ref_type not in operations:
            operations.append(ref.ref_type)

    # Build enriched embedding text: code + entity context
    entity_section_parts = []
    if canonical_tables:
        entity_section_parts.append(f"Tables referenced: {', '.join(canonical_tables)}")
    if aliases_used:
        entity_section_parts.append(f"Aliases used in code: {', '.join(aliases_used)}")
    if operations:
        entity_section_parts.append(f"Operations: {', '.join(operations)}")
    entity_section_parts.append(f"File: {chunk.file_path}, lines {chunk.line_start}-{chunk.line_end}")
    entity_section_parts.append(f"Language: {chunk.file_type}, block type: {chunk.chunk_type}")

    embedding_text = (
        "[CODE]\n"
        + chunk.raw_text[:3000]
        + "\n\n[ENTITIES]\n"
        + "\n".join(entity_section_parts)
    )

    metadata = {
        "file_path": chunk.file_path,
        "file_type": chunk.file_type,
        "chunk_type": chunk.chunk_type,
        "line_start": chunk.line_start,
        "line_end": chunk.line_end,
        # pipe-delimited for ChromaDB (it doesn't support list values natively)
        "canonical_tables": "|".join(canonical_tables),
        "aliases_used": "|".join(aliases_used),
        "operations": "|".join(operations),
    }

    return EnrichedDocument(
        chunk_id=chunk.chunk_id,
        file_path=chunk.file_path,
        file_type=chunk.file_type,
        chunk_type=chunk.chunk_type,
        line_start=chunk.line_start,
        line_end=chunk.line_end,
        raw_text=chunk.raw_text,
        canonical_tables=canonical_tables,
        aliases_used=aliases_used,
        operations=operations,
        embedding_text=embedding_text,
        metadata=metadata,
    )


def build_enriched_documents(
    chunks: List[CodeChunk],
    all_entities: List[ChunkEntities],
) -> List[EnrichedDocument]:
    entity_map = {ce.chunk_id: ce for ce in all_entities}
    docs = []
    for chunk in chunks:
        entities = entity_map.get(chunk.chunk_id)
        if entities is None:
            from src.extractors.entity_extractor import ChunkEntities
            entities = ChunkEntities(chunk_id=chunk.chunk_id, file_path=chunk.file_path)
        docs.append(build_enriched_document(chunk, entities))
    return docs
