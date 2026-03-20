"""Lightweight query classification for hybrid retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Pattern


class QueryIntent(str, Enum):
    """Intent buckets used to adapt retrieval weights."""

    CODE_EXPLANATION = "code_explanation"
    TROUBLESHOOTING = "troubleshooting"
    HOW_TO = "how_to"
    BEST_PRACTICES = "best_practices"
    COMPARISON = "comparison"
    TECHNICAL_DOCS = "technical_docs"
    CONCEPTUAL = "conceptual"
    GENERAL_QA = "general_qa"


@dataclass(slots=True)
class QueryClassification:
    """Classification result for a query."""

    primary_intent: QueryIntent
    confidence: float
    keywords: List[str] = field(default_factory=list)
    is_technical: bool = True
    all_intents: Dict[QueryIntent, float] = field(default_factory=dict)


class QueryClassifier:
    """Rule-based classifier focused on retrieval routing."""

    _INTENT_PATTERNS: dict[QueryIntent, dict[str, object]] = {
        QueryIntent.CODE_EXPLANATION: {
            "patterns": [
                r"\bexplain\s+(this\s+)?code\b",
                r"\bwhat\s+does\s+this\s+code\b",
                r"\bhow\s+does\s+(this\s+|the\s+)?\w+\s+work\b",
                r"\bwalk\s+through\b",
                r"\bbreak\s+down\b",
            ],
            "keywords": ["explain", "code", "function", "method", "class", "work"],
            "weight": 1.1,
        },
        QueryIntent.TROUBLESHOOTING: {
            "patterns": [
                r"\berror\b",
                r"\bexception\b",
                r"\bfailed\b",
                r"\bnot\s+working\b",
                r"\bbug\b",
                r"\bproblem\b",
                r"\bfix\b",
                r"\bdebug\b",
            ],
            "keywords": ["error", "exception", "failed", "bug", "fix", "debug", "problem"],
            "weight": 1.3,
        },
        QueryIntent.HOW_TO: {
            "patterns": [
                r"\bhow\s+to\b",
                r"\bhow\s+do\s+i\b",
                r"\bhow\s+can\s+i\b",
                r"\bsteps\s+to\b",
                r"\bguide\s+to\b",
                r"\btutorial\b",
            ],
            "keywords": ["how to", "guide", "tutorial", "steps", "build", "create", "implement"],
            "weight": 1.0,
        },
        QueryIntent.BEST_PRACTICES: {
            "patterns": [
                r"\bbest\s+practice",
                r"\brecommended\s+(way|approach|method|practices?)\b",
                r"\bshould\s+i\b",
                r"\bidiomatic\b",
                r"\bconvention\b",
                r"\bavoid\b",
            ],
            "keywords": ["best practice", "recommended", "should", "idiomatic", "avoid"],
            "weight": 1.2,
        },
        QueryIntent.COMPARISON: {
            "patterns": [
                r"\bvs\.?\b",
                r"\bversus\b",
                r"\bcompare\b",
                r"\bdifference\s+between\b",
                r"\bwhich\s+is\s+better\b",
            ],
            "keywords": ["vs", "versus", "compare", "difference", "better"],
            "weight": 1.0,
        },
        QueryIntent.TECHNICAL_DOCS: {
            "patterns": [
                r"\bapi\b",
                r"\bparameter\b",
                r"\bargument\b",
                r"\bconfiguration\b",
                r"\bsyntax\b",
                r"\breference\b",
                r"\bdocumentation\b",
            ],
            "keywords": ["api", "parameter", "argument", "config", "reference", "docs"],
            "weight": 1.0,
        },
        QueryIntent.CONCEPTUAL: {
            "patterns": [
                r"\bwhat\s+is\b",
                r"\bwhat\s+are\b",
                r"\bwhy\s+(does|is|do)\b",
                r"\bconcept\b",
                r"\barchitecture\b",
                r"\bdesign\s+pattern\b",
            ],
            "keywords": ["what is", "what are", "why", "concept", "architecture", "design"],
            "weight": 1.0,
        },
    }

    _ENTITY_PATTERNS: dict[str, dict[str, str]] = {
        "language": {
            "python": r"\b(python|py)\b",
            "javascript": r"\b(javascript|js|node\.?js)\b",
            "typescript": r"\b(typescript|ts)\b",
            "go": r"\b(go|golang)\b",
            "rust": r"\brust\b",
        },
        "framework": {
            "fastapi": r"\bfastapi\b",
            "django": r"\bdjango\b",
            "flask": r"\bflask\b",
            "react": r"\breact\b",
        },
        "library": {
            "numpy": r"\bnumpy\b",
            "pandas": r"\bpandas\b",
            "faiss": r"\bfaiss\b",
        },
    }

    _NON_TECHNICAL_PATTERNS = [
        r"\bhello\b",
        r"\bhi\b",
        r"\bthank(s| you)\b",
        r"\bplease\b",
        r"\bsorry\b",
        r"\bhow\s+are\s+you\b",
    ]

    _STOP_WORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "you",
    }

    def __init__(self, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        self._compiled_intent_patterns: dict[QueryIntent, list[Pattern[str]]] = {}
        for intent, config in self._INTENT_PATTERNS.items():
            self._compiled_intent_patterns[intent] = [
                re.compile(pattern, re.IGNORECASE) for pattern in config["patterns"]  # type: ignore[index]
            ]

        self._compiled_entity_patterns: dict[str, dict[str, Pattern[str]]] = {}
        for entity_type, patterns in self._ENTITY_PATTERNS.items():
            self._compiled_entity_patterns[entity_type] = {
                name: re.compile(pattern, re.IGNORECASE) for name, pattern in patterns.items()
            }

        self._compiled_non_technical_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self._NON_TECHNICAL_PATTERNS
        ]

    def classify(self, query: str) -> QueryClassification:
        normalized = query.lower().strip()
        intent_scores = self._detect_intents(normalized)
        primary_intent = max(intent_scores, key=intent_scores.get) if intent_scores else QueryIntent.GENERAL_QA
        keywords = self._extract_keywords(normalized)

        classification = QueryClassification(
            primary_intent=primary_intent,
            confidence=float(intent_scores.get(primary_intent, 0.5 if intent_scores else 0.5)),
            keywords=keywords,
            is_technical=self._is_technical_query(normalized),
            all_intents=intent_scores,
        )
        return classification

    def _is_technical_query(self, query: str) -> bool:
        if any(pattern.search(query) for pattern in self._compiled_non_technical_patterns):
            return False

        if any(
            pattern.search(query)
            for pattern_group in self._compiled_entity_patterns.values()
            for pattern in pattern_group.values()
        ):
            return True

        return True

    def _detect_intents(self, query: str) -> Dict[QueryIntent, float]:
        scores: Dict[QueryIntent, float] = {}

        for intent, config in self._INTENT_PATTERNS.items():
            score = 0.0
            pattern_matches = sum(
                1 for pattern in self._compiled_intent_patterns[intent] if pattern.search(query)
            )
            if pattern_matches:
                score += min(pattern_matches * 0.3, 0.9)

            keyword_matches = sum(1 for keyword in config["keywords"] if keyword in query)  # type: ignore[index]
            if keyword_matches:
                score += min(keyword_matches * 0.1, 0.4)

            if score > 0:
                score = min(score * float(config["weight"]), 1.0)  # type: ignore[index]
                if score >= self.confidence_threshold:
                    scores[intent] = score

        if not scores:
            scores[QueryIntent.GENERAL_QA] = 0.5

        return scores

    def _extract_keywords(self, query: str) -> List[str]:
        tokens = re.findall(r"\b\w+\b", query)
        keywords = [token for token in tokens if len(token) > 2 and token not in self._STOP_WORDS]
        return keywords[:10]

