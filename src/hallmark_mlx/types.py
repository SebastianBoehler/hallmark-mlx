"""Shared enumerations and small type aliases."""

from __future__ import annotations

from enum import Enum
from typing import Any

JSONDict = dict[str, Any]


class InputType(str, Enum):
    """Supported verification task shapes."""

    RAW_CITATION_STRING = "raw_citation_string"
    BIBTEX_ENTRY = "bibtex_entry"
    PARAGRAPH_WITH_CITATION = "paragraph_with_citation"
    CLAIM_FOR_SUPPORTING_REFS = "claim_for_supporting_refs"


class SuspectedIssueCode(str, Enum):
    """Frequent citation verification failure modes."""

    MISSING_DOI = "missing_doi"
    TITLE_AMBIGUITY = "title_ambiguity"
    AUTHOR_MISMATCH = "author_mismatch"
    YEAR_MISMATCH = "year_mismatch"
    VENUE_MISMATCH = "venue_mismatch"
    PREPRINT_AS_PUBLISHED = "preprint_as_published"
    POSSIBLE_HALLUCINATION = "possible_hallucination"
    INSUFFICIENT_METADATA = "insufficient_metadata"


class VerificationAction(str, Enum):
    """High-level next actions emitted by the policy model."""

    PARSE_INPUT = "parse_input"
    QUERY_BIBTEX_UPDATER = "query_bibtex_updater"
    QUERY_CROSSREF = "query_crossref"
    QUERY_OPENALEX = "query_openalex"
    QUERY_DBLP = "query_dblp"
    QUERY_ACL_ANTHOLOGY = "query_acl_anthology"
    QUERY_SEMANTIC_SCHOLAR = "query_semantic_scholar"
    RANK_CANDIDATES = "rank_candidates"
    FINALIZE = "finalize"
    ABSTAIN = "abstain"


class ToolName(str, Enum):
    """Supported external verification tools."""

    BIBTEX_UPDATER = "bibtex_updater"
    CROSSREF = "crossref"
    OPENALEX = "openalex"
    DBLP = "dblp"
    ACL_ANTHOLOGY = "acl_anthology"
    SEMANTIC_SCHOLAR = "semantic_scholar"


class VerificationVerdict(str, Enum):
    """Final citation verification verdict."""

    VERIFIED = "verified"
    CORRECTED = "corrected"
    HALLUCINATED = "hallucinated"
    UNSUPPORTED = "unsupported"
    ABSTAIN = "abstain"


class HallmarkLabel(str, Enum):
    """Label set accepted by HALLMARK evaluation."""

    VALID = "VALID"
    HALLUCINATED = "HALLUCINATED"
    UNCERTAIN = "UNCERTAIN"


class ModelBackend(str, Enum):
    """Inference or training backend identifiers."""

    MLX = "mlx"
    WARM_START = "warm_start"
    EXTERNAL = "external"


class FinalizationMode(str, Enum):
    """How the system should derive the final verification verdict."""

    GENERATIVE = "generative"
    DETERMINISTIC = "deterministic"


class TrainingExampleFormat(str, Enum):
    """Supported supervised fine-tuning example layouts."""

    TRACE_JSON = "trace_json"
    TOOL_TRANSCRIPT = "tool_transcript"
    TOOL_TRANSCRIPT_STEPS = "tool_transcript_steps"


class EvidenceStrength(str, Enum):
    """Coarse evidence buckets for result summaries."""

    NONE = "none"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
