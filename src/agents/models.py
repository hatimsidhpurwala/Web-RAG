"""
Pydantic models used across agents for structured LLM outputs.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Intent Classification
# ---------------------------------------------------------------------------

class IntentClassification(BaseModel):
    """Structured output from the intent-classification agent."""

    intent: Literal[
        "greeting",
        "gratitude",
        "general_knowledge",
        "retrieval_needed",
        "clarification",
        "farewell",
    ] = Field(..., description="The classified user intent")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for the classification"
    )
    reasoning: str = Field(..., description="Short explanation for the decision")
    needs_retrieval: bool = Field(
        ..., description="Whether the question requires a knowledge-base lookup"
    )


# ---------------------------------------------------------------------------
# Query Generation
# ---------------------------------------------------------------------------

class QueryGeneration(BaseModel):
    """Optimised search queries generated from the user's question."""

    queries: List[str] = Field(
        ..., min_length=1, max_length=4,
        description="2-4 optimised search queries"
    )
    primary_entities: List[str] = Field(
        ..., description="Key entities extracted from the question"
    )
    query_type: Literal[
        "factual",
        "comparative",
        "explanatory",
        "procedural",
        "commercial",    # buy / price / where-to-buy / dealer
        "contact",       # contact details / location / address
        "document",      # questions about an uploaded file
    ] = Field(..., description="The type of query")


# ---------------------------------------------------------------------------
# Source Reference
# ---------------------------------------------------------------------------

class SourceReference(BaseModel):
    """A reference to a source used in the answer."""

    source_url: str = Field(..., description="URL or identifier of the source")
    site_name: str = Field(default="", description="Human-readable source name")
    relevance_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Relevance score of this source"
    )


# ---------------------------------------------------------------------------
# Response Generation
# ---------------------------------------------------------------------------

class ResponseGeneration(BaseModel):
    """Structured output from the response-generation agent."""

    answer: str = Field(..., description="The generated answer")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the answer"
    )
    sources_used: List[SourceReference] = Field(
        default_factory=list, description="Sources used to build the answer"
    )
    follow_up_suggestions: Optional[List[str]] = Field(
        default=None, description="Suggested follow-up questions"
    )
