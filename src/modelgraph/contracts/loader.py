"""YAML/JSON loading with duplicate-key detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from modelgraph.errors import ModelGraphError


class DuplicateKeyError(ModelGraphError):
    code = "MG-1001"


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=False)
        if key in mapping:
            raise DuplicateKeyError(
                f"duplicate key {key!r}",
                path=f"line {key_node.start_mark.line + 1}",
            )
        mapping[key] = loader.construct_object(value_node, deep=False)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def load_document(path: str | Path) -> dict[str, Any]:
    """Load one YAML or JSON document, rejecting duplicate mapping keys."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModelGraphError(f"cannot read {source}: {exc}", path=str(source)) from exc

    try:
        if source.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            data = yaml.load(text, Loader=_UniqueKeyLoader)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ModelGraphError(f"invalid document: {exc}", path=str(source)) from exc

    if not isinstance(data, dict):
        raise ModelGraphError("document root must be a mapping", path=str(source))
    return data
