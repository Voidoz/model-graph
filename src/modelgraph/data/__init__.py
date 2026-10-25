"""Trace-to-dataset export primitives.

The public entry point is :func:`modelgraph.data.export.export_dataset`; the CLI exposes it as
``mg data export``. See ``docs/training.md`` §3.4 for the format and the consent/redaction
policy.
"""

from modelgraph.data.dedupe import deduplicate, hamming_distance, normalize_text, simhash
from modelgraph.data.export import (
    DATASET_SCHEMA_FILE,
    EXAMPLE_FILE,
    MANIFEST_FILE,
    RECORD_SCHEMA,
    REDACTION_FILE,
    SPLITS_FILE,
    DatasetBundle,
    DatasetExportError,
    ExportResult,
    dataset_schema,
    export_dataset,
    load_dataset,
    validate_dataset_example,
    validate_dataset_manifest,
)
from modelgraph.data.redaction import RedactionError, Redactor, build_redactor

__all__ = [
    "DATASET_SCHEMA_FILE",
    "EXAMPLE_FILE",
    "MANIFEST_FILE",
    "RECORD_SCHEMA",
    "REDACTION_FILE",
    "SPLITS_FILE",
    "DatasetBundle",
    "DatasetExportError",
    "ExportResult",
    "RedactionError",
    "Redactor",
    "build_redactor",
    "dataset_schema",
    "deduplicate",
    "export_dataset",
    "hamming_distance",
    "load_dataset",
    "normalize_text",
    "simhash",
    "validate_dataset_example",
    "validate_dataset_manifest",
]
