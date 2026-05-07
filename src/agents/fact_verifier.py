"""
Automated Fact Verification – extracts factual claims from an answer,
cross-references them against sources, and marks each as verified,
unverified, or contradicted.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from groq import Groq

from config.settings import GROQ_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)


# ======================================================================
# Models
# ======================================================================

class ClaimVerification(BaseModel):
    claim: str
    status: str = "unverified"  # verified | unverified | contradicted
    confidence: float = 0.0
    evidence: str = ""
    source: str = ""


class VerificationReport(BaseModel):
    verified_answer: str = ""
    claims: List[ClaimVerification] = Field(default_factory=list)
    overall_reliability: float = 0.0
    contradictions: List[str] = Field(default_factory=list)
    verification_summary: str = ""


# ======================================================================
# Prompts
# ======================================================================

_EXTRACT_CLAIMS_PROMPT = """\
Extract ALL factual claims from the following answer. A factual claim
is a statement that can be verified as true or false.

Return JSON:
{
  "claims": ["claim1", "claim2", ...]
}

Only include verifiable facts (numbers, names, specifications, dates,
prices). Do NOT include opinions or subjective statements.
"""

_VERIFY_CLAIMS_PROMPT = """\
You are a fact-checking agent. For each claim, check it against the
provided source evidence and determine if it is verified, unverified,
or contradicted.

Return JSON:
{
  "verifications": [
    {
      "claim": "the original claim",
      "status": "verified|unverified|contradicted",
      "confidence": 0.9,
      "evidence": "supporting or contradicting evidence text",
      "source": "which source/chunk supported this"
    }
  ],
  "contradictions": ["description of any contradictions found"],
  "overall_reliability": 0.85
}

Rules:
- verified: Claim is directly supported by source evidence (confidence > 0.8)
- unverified: No evidence found to confirm or deny (confidence 0.3-0.6)
- contradicted: Evidence directly contradicts the claim (flag clearly)
"""


# ======================================================================
# Fact Verifier
# ======================================================================

class FactVerifier:
    """Extracts claims from an answer and verifies each one."""

    def verify_answer(
        self,
        answer: str,
        source_chunks: List[dict],
        question: str = "",
    ) -> VerificationReport:
        """Verify all factual claims in *answer* against *source_chunks*.

        Returns a VerificationReport with per-claim status and
        the answer annotated with verification markers.
        """
        # Step 1: Extract claims
        claims = self._extract_claims(answer)
        if not claims:
            return VerificationReport(
                verified_answer=answer,
                overall_reliability=0.7,
                verification_summary="No specific factual claims to verify.",
            )

        # Step 2: Verify claims against sources
        report = self._verify_claims(claims, source_chunks)

        # Step 3: Annotate the answer
        report.verified_answer = self._annotate_answer(answer, report.claims)
        report.verification_summary = self._build_summary(report)

        logger.info(
            "Fact verification: %d claims, reliability=%.2f",
            len(report.claims),
            report.overall_reliability,
        )
        return report

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_claims(self, answer: str) -> List[str]:
        """Extract verifiable factual claims from *answer*."""
        client = Groq(api_key=GROQ_API_KEY)
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": _EXTRACT_CLAIMS_PROMPT},
                    {"role": "user", "content": answer},
                ],
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            claims = data.get("claims", [])
            logger.info("Extracted %d claims", len(claims))
            return claims
        except Exception as exc:
            logger.error("Claim extraction failed: %s", exc)
            return []

    def _verify_claims(
        self,
        claims: List[str],
        source_chunks: List[dict],
    ) -> VerificationReport:
        """Verify *claims* against *source_chunks*."""
        client = Groq(api_key=GROQ_API_KEY)

        # Build evidence block — trim each chunk to 400 chars to avoid overflow
        evidence_parts = []
        for i, chunk in enumerate(source_chunks[:8], 1):
            text = chunk.get("text", "")[:400]
            evidence_parts.append(
                f"[Source {i}: {chunk.get('source_url', 'unknown')}]\n"
                f"{text}\n"
            )
        evidence_block = "\n".join(evidence_parts) if evidence_parts else "(no sources)"

        user_content = (
            f"Claims to verify:\n"
            + "\n".join(f"- {c}" for c in claims)
            + f"\n\nSource evidence:\n{evidence_block}"
        )

        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": _VERIFY_CLAIMS_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)

            verifications = [
                ClaimVerification(**v)
                for v in data.get("verifications", [])
            ]

            return VerificationReport(
                claims=verifications,
                contradictions=data.get("contradictions", []),
                overall_reliability=data.get("overall_reliability", 0.5),
            )
        except Exception as exc:
            logger.error("Claim verification failed: %s", exc)
            return VerificationReport(
                claims=[
                    ClaimVerification(claim=c, status="unverified", confidence=0.5)
                    for c in claims
                ],
                overall_reliability=0.5,
            )

    def _annotate_answer(
        self, answer: str, claims: List[ClaimVerification]
    ) -> str:
        """Add verification markers to the answer text."""
        annotated = answer
        for cv in claims:
            if cv.status == "verified" and cv.confidence >= 0.8:
                marker = " ✅"
            elif cv.status == "contradicted":
                marker = " ⚠️"
            elif cv.status == "unverified":
                marker = " ❓"
            else:
                marker = ""

            # Try to find and annotate the claim in the answer
            if cv.claim in annotated:
                annotated = annotated.replace(
                    cv.claim, f"{cv.claim}{marker}", 1
                )
        return annotated

    def _build_summary(self, report: VerificationReport) -> str:
        verified = sum(1 for c in report.claims if c.status == "verified")
        unverified = sum(1 for c in report.claims if c.status == "unverified")
        contradicted = sum(1 for c in report.claims if c.status == "contradicted")
        total = len(report.claims)

        parts = [f"Verified {verified}/{total} claims"]
        if unverified:
            parts.append(f"{unverified} unverified")
        if contradicted:
            parts.append(f"⚠️ {contradicted} contradicted")
        parts.append(f"Reliability: {report.overall_reliability:.0%}")

        return " | ".join(parts)
