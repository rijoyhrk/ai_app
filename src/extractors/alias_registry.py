import json
import os
from pathlib import Path
from typing import Dict, List, Set
from dataclasses import dataclass, field
from .entity_extractor import ChunkEntities, EntityRef


@dataclass
class AliasEntry:
    canonical: str
    aliases: Set[str] = field(default_factory=set)
    files: Set[str] = field(default_factory=set)


class AliasRegistry:
    """
    Global registry mapping alias names → canonical table names.
    Built from LLM entity extraction results and persisted to JSON.
    """

    def __init__(self):
        # canonical → AliasEntry
        self._by_canonical: Dict[str, AliasEntry] = {}
        # raw alias → canonical
        self._alias_to_canonical: Dict[str, str] = {}

    def ingest(self, chunk_entities: List[ChunkEntities]) -> None:
        for ce in chunk_entities:
            for ref in ce.refs:
                if ref.canonical is None or ref.ref_type == "COLUMN":
                    continue
                canonical = ref.canonical.lower().strip()
                raw = ref.raw.lower().strip()
                if not canonical or not raw:
                    continue

                if canonical not in self._by_canonical:
                    self._by_canonical[canonical] = AliasEntry(canonical=canonical)

                entry = self._by_canonical[canonical]
                entry.aliases.add(raw)
                entry.files.add(ce.file_path)
                # also track the canonical itself as an alias
                entry.aliases.add(canonical)

                # update reverse map
                self._alias_to_canonical[raw] = canonical
                self._alias_to_canonical[canonical] = canonical

    def resolve(self, name: str) -> str:
        """Return the canonical name for a given alias or raw name."""
        return self._alias_to_canonical.get(name.lower().strip(), name.lower().strip())

    def get_all_aliases(self, canonical: str) -> List[str]:
        """Return all known aliases for a canonical table name."""
        canonical = canonical.lower().strip()
        entry = self._by_canonical.get(canonical)
        if entry:
            return sorted(entry.aliases)
        # if not found as canonical, try resolving first
        resolved = self.resolve(canonical)
        entry = self._by_canonical.get(resolved)
        return sorted(entry.aliases) if entry else [canonical]

    def get_files_for(self, canonical: str) -> List[str]:
        """Return all files where a canonical table is referenced."""
        canonical = canonical.lower().strip()
        resolved = self.resolve(canonical)
        entry = self._by_canonical.get(resolved)
        return sorted(entry.files) if entry else []

    def all_canonicals(self) -> List[str]:
        return sorted(self._by_canonical.keys())

    def save(self, path: str) -> None:
        """Atomically write the registry to disk (temp file + os.replace)."""
        data = {}
        for canonical, entry in self._by_canonical.items():
            data[canonical] = {
                "aliases": sorted(entry.aliases),
                "files": sorted(entry.files),
            }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = p.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, indent=2))
        os.replace(str(tmp_path), str(p))

    @classmethod
    def load(cls, path: str) -> "AliasRegistry":
        registry = cls()
        if not Path(path).exists():
            return registry
        data = json.loads(Path(path).read_text())
        for canonical, info in data.items():
            entry = AliasEntry(
                canonical=canonical,
                aliases=set(info.get("aliases", [])),
                files=set(info.get("files", [])),
            )
            registry._by_canonical[canonical] = entry
            for alias in entry.aliases:
                registry._alias_to_canonical[alias] = canonical
        return registry

    def summary(self) -> str:
        lines = [f"AliasRegistry: {len(self._by_canonical)} canonical tables"]
        for canonical in sorted(self._by_canonical):
            entry = self._by_canonical[canonical]
            aliases = sorted(entry.aliases - {canonical})
            alias_str = f" (aliases: {', '.join(aliases)})" if aliases else ""
            lines.append(f"  {canonical}{alias_str} — {len(entry.files)} file(s)")
        return "\n".join(lines)
