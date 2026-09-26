from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.evaluation.contracts import (
    EvaluationCheck,
    EvaluationCheckType,
    EvaluationIssue,
    EvaluationRequest,
    EvaluationResult,
    EvaluationSeverity,
    EvaluationStatus,
)
from app.evaluation.port import EvaluationPort


class QualityEvaluator(EvaluationPort):
    """Evaluate deterministic quality signals for generated artifacts."""

    MIN_CONTENT_LENGTH = 20

    @property
    def check_type(self) -> EvaluationCheckType:
        return EvaluationCheckType.QUALITY

    async def evaluate(
        self,
        request: EvaluationRequest,
    ) -> EvaluationResult:
        if not isinstance(request, EvaluationRequest):
            raise TypeError("request must be an EvaluationRequest.")

        issues: list[EvaluationIssue] = []
        scores: list[float] = []

        objective = request.transformation_request.objective
        audience = request.transformation_request.audience
        tone = request.transformation_request.tone

        for index, artifact in enumerate(
            request.transformation_result.artifacts
        ):
            artifact_score, artifact_issues = self._evaluate_artifact(
                artifact.content,
                index=index,
                objective=objective,
                audience=audience,
                tone=tone,
            )

            scores.append(artifact_score)
            issues.extend(artifact_issues)

        if not scores:
            issues.append(
                self._issue(
                    code="NO_ARTIFACTS",
                    message="Cannot evaluate quality without artifacts.",
                )
            )
            score = 0.0
        else:
            score = round(sum(scores) / len(scores), 4)

        if any(
            issue.severity == EvaluationSeverity.ERROR
            for issue in issues
        ):
            status = EvaluationStatus.FAILED
        elif score < 0.5:
            status = EvaluationStatus.NEEDS_REVIEW
        else:
            status = EvaluationStatus.PASSED

        check = EvaluationCheck(
            check_type=EvaluationCheckType.QUALITY,
            status=status,
            score=score,
            issues=issues,
            metadata={
                "artifact_count": len(scores),
                "minimum_content_length": self.MIN_CONTENT_LENGTH,
            },
        )

        return EvaluationResult(
            status=status,
            score=score,
            checks=[check],
            issues=issues,
            metadata={
                "evaluator": self.__class__.__name__,
                "artifact_count": len(scores),
            },
        )

    def _evaluate_artifact(
        self,
        content: Any,
        *,
        index: int,
        objective: str | None,
        audience: str | None,
        tone: str | None,
    ) -> tuple[float, list[EvaluationIssue]]:
        issues: list[EvaluationIssue] = []

        if content is None:
            return (
                0.0,
                [
                    self._issue(
                        code=f"ARTIFACT_{index}_CONTENT_MISSING",
                        message="Artifact content is missing.",
                    )
                ],
            )

        text = self._extract_text(content)

        if not text.strip():
            return (
                0.0,
                [
                    self._issue(
                        code=f"ARTIFACT_{index}_CONTENT_EMPTY",
                        message="Artifact contains no meaningful content.",
                    )
                ],
            )

        score = 0.5

        if len(text.strip()) >= self.MIN_CONTENT_LENGTH:
            score += 0.2
        else:
            issues.append(
                self._issue(
                    code=f"ARTIFACT_{index}_CONTENT_TOO_SHORT",
                    message=(
                        "Artifact content is shorter than the minimum "
                        "meaningful-content threshold."
                    ),
                    severity=EvaluationSeverity.WARNING,
                )
            )

        if objective:
            if self._contains_signal(text, objective):
                score += 0.1
            else:
                issues.append(
                    self._issue(
                        code=f"ARTIFACT_{index}_OBJECTIVE_ALIGNMENT_WEAK",
                        message=(
                            "The artifact does not contain an obvious "
                            "textual signal matching the requested objective."
                        ),
                        severity=EvaluationSeverity.WARNING,
                    )
                )

        if audience:
            if self._contains_signal(text, audience):
                score += 0.05
            else:
                issues.append(
                    self._issue(
                        code=f"ARTIFACT_{index}_AUDIENCE_SIGNAL_WEAK",
                        message=(
                            "The artifact does not contain an obvious "
                            "textual audience signal."
                        ),
                        severity=EvaluationSeverity.WARNING,
                    )
                )

        if tone:
            if self._contains_signal(text, tone):
                score += 0.05
            else:
                issues.append(
                    self._issue(
                        code=f"ARTIFACT_{index}_TONE_SIGNAL_WEAK",
                        message=(
                            "The artifact does not contain an obvious "
                            "textual tone signal."
                        ),
                        severity=EvaluationSeverity.WARNING,
                    )
                )

        return round(min(score, 1.0), 4), issues

    @staticmethod
    def _extract_text(content: Any) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, Mapping):
            parts: list[str] = []

            for value in content.values():
                parts.append(
                    QualityEvaluator._extract_text(value)
                )

            return " ".join(
                part for part in parts if part.strip()
            )

        if isinstance(content, (list, tuple, set)):
            return " ".join(
                QualityEvaluator._extract_text(value)
                for value in content
            )

        return str(content)

    @staticmethod
    def _contains_signal(
        text: str,
        signal: str,
    ) -> bool:
        normalized_text = text.casefold()
        normalized_signal = signal.strip().casefold()

        if not normalized_signal:
            return True

        return normalized_signal in normalized_text

    @staticmethod
    def _issue(
        *,
        code: str,
        message: str,
        severity: EvaluationSeverity = EvaluationSeverity.ERROR,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationIssue:
        return EvaluationIssue(
            code=code,
            message=message,
            severity=severity,
            check_type=EvaluationCheckType.QUALITY,
            metadata=metadata or {},
        )