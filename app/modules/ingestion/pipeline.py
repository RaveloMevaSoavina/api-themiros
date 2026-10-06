"""Pipeline d'ingestion (spec §2.1) : extraction → détection → pertinence →
décision → segmentation → indexation, chaque étape journalisée.

Les étapes repartent de ce qui est conservé en base : requalifier ou indexer
ne refait jamais l'OCR, et les vecteurs calculés pour le score servent aussi
à l'indexation.
"""

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from app.core.config import Settings
from app.core.errors import ApiError
from app.modules.ingestion.analysis.country import CountryDetector
from app.modules.ingestion.analysis.embeddings import (
    Embedder,
    sample_indices,
    semantic_score,
)
from app.modules.ingestion.analysis.language import detect_language
from app.modules.ingestion.analysis.segmenter import (
    PageText,
    Segment,
    count_tokens,
    segment_pages,
)
from app.modules.ingestion.analysis.signals import detect_signals
from app.modules.ingestion.domain import (
    Decision,
    Detection,
    IngestionRules,
    RelevanceScore,
    WorkspaceProfile,
    decide,
    fingerprint_text,
    is_eligible_for_index,
    score_relevance,
)
from app.modules.ingestion.errors import IngestionError, no_exploitable_content
from app.modules.ingestion.extraction import (
    ExtractedDocument,
    OcrOptions,
    extract_document,
)
from app.modules.ingestion.repository import (
    ClaimedJob,
    DocumentRecord,
    IngestionRepository,
    StoredPages,
)

StageCallback = Callable[[str], Awaitable[None]]
SEGMENT_BATCH_SIZE = 32
OCR_LANGUAGES = {"fr": "fra", "en": "eng", "pt": "por", "es": "spa"}


@dataclass(frozen=True, slots=True)
class ExtractionOutcome:
    document: ExtractedDocument
    stored: StoredPages
    language: Detection


@dataclass(frozen=True, slots=True)
class QualificationOutcome:
    decision: Decision
    score: RelevanceScore
    language: Detection
    country: Detection
    status: str
    segments: list[Segment]
    # Vecteurs déjà calculés pour le score, par indice de segment : ils sont
    # réutilisés à l'indexation.
    sampled_vectors: dict[int, list[float]]


@dataclass(frozen=True, slots=True)
class IndexOutcome:
    segments_count: int
    embeddings_count: int


class IngestionPipeline:
    def __init__(
        self,
        repository: IngestionRepository,
        embedder: Embedder,
        country_detector: CountryDetector,
        settings: Settings,
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._countries = country_detector
        self._settings = settings

    async def run(self, job: ClaimedJob, on_stage: StageCallback) -> dict[str, Any]:
        """Exécute une tâche de la file selon son type."""
        document = await self._repository.load_document(job.document_id)
        profile, financier_terms = await self._repository.load_profile(job.workspace_id)
        rules = await self._repository.load_rules(job.workspace_id)
        result: dict[str, Any] = {"kind": job.kind}

        if job.kind in {"full", "extract"}:
            extraction = await self.extract(job, document, profile, rules, on_stage)
            stored = extraction.stored
            result["extraction"] = {
                "pdf_type": extraction.document.pdf_type,
                "total_pages": stored.total_pages,
                "exploitable_pages": stored.exploitable_pages,
            }
            if job.kind == "extract":
                return result
        else:
            stored = await self._repository.load_pages(job.document_id)

        if job.kind in {"full", "requalify", "qualify"}:
            qualification = await self.qualify(
                job, profile, financier_terms, rules, stored, on_stage
            )
            result["qualification"] = {
                "status": qualification.status,
                "relevance_score": qualification.score.final,
                "reason_code": qualification.decision.reason_code,
                "language": qualification.language.value,
                "country": qualification.country.value,
                "message": qualification.decision.message,
            }
            # Spec §9.4 : indexation à la fin de l'ingestion, si conforme.
            if job.kind != "qualify" and is_eligible_for_index(qualification.status):
                indexed = await self.index(
                    job,
                    qualification.segments,
                    qualification.language.value,
                    on_stage,
                    known_vectors=qualification.sampled_vectors,
                )
                result["indexation"] = {"segments_count": indexed.segments_count}
            return result

        if job.kind == "index":
            if not is_eligible_for_index(document.status):
                raise IngestionError(
                    status_code=409,
                    code="DOCUMENT_NOT_ELIGIBLE_FOR_INDEXING",
                    detail=(
                        "Seuls les documents conformes ou intégrés sur décision "
                        "humaine sont indexés"
                    ),
                )
            segments = await asyncio.to_thread(segment_pages, stored.pages)
            indexed = await self.index(
                job, segments, document.detected_language, on_stage
            )
            result["indexation"] = {"segments_count": indexed.segments_count}
            return result

        raise IngestionError(
            code="INVALID_JOB_KIND", detail=f"Type de tâche inconnu : {job.kind}"
        )

    async def extract(
        self,
        job: ClaimedJob,
        document: DocumentRecord,
        profile: WorkspaceProfile,
        rules: IngestionRules,
        on_stage: StageCallback,
        content: bytes | None = None,
    ) -> ExtractionOutcome:
        await on_stage("extraction")
        if content is None:
            content = await self._repository.download(document.storage_path)
        if hashlib.sha256(content).hexdigest() != document.file_hash:
            raise IngestionError(
                status_code=409,
                code="FILE_INTEGRITY_ERROR",
                detail="Le fichier ne correspond pas au document enregistré",
            )

        extracted = await asyncio.to_thread(
            extract_document, content, document.original_filename, self._ocr(profile)
        )
        min_chars = rules.exploitable_page_min_chars
        pages_payload = [
            {
                "page_number": page.page_number,
                "text": page.text,
                "section_title": page.section_title,
                "is_ocr": page.is_ocr,
                "ocr_failed": page.ocr_failed,
                "is_exploitable": page.is_exploitable(min_chars),
                "char_start": page.char_start,
            }
            for page in extracted.pages
        ]
        exploitable = extracted.exploitable_pages(min_chars)
        await self._repository.commit_extraction(
            job,
            pages=pages_payload,
            pdf_type=extracted.pdf_type,
            metadata={
                "format": extracted.format,
                "tokens": await asyncio.to_thread(count_tokens, extracted.text),
                "ocr_pages": sum(page.is_ocr for page in extracted.pages),
                "ocr_failed_pages": sum(page.ocr_failed for page in extracted.pages),
            },
        )
        # Arbitrage n° 8 : aucun contenu exploitable = échec d'extraction,
        # document non intégré (le texte extrait reste conservé).
        if exploitable == 0:
            raise no_exploitable_content()

        stored = StoredPages(
            pages=[
                PageText(
                    page.page_number, page.text, page.char_start, page.section_title
                )
                for page in extracted.pages
            ],
            total_pages=extracted.total_pages,
            exploitable_pages=exploitable,
        )
        language = await asyncio.to_thread(
            detect_language, extracted.text, rules.language_confidence_threshold
        )
        return ExtractionOutcome(extracted, stored, language)

    async def qualify(
        self,
        job: ClaimedJob,
        profile: WorkspaceProfile,
        financier_terms: Sequence[str],
        rules: IngestionRules,
        stored: StoredPages,
        on_stage: StageCallback,
    ) -> QualificationOutcome:
        await on_stage("detection")
        text = "\n\n".join(page.text for page in stored.pages)
        overrides = await self._repository.load_overrides(job.document_id)
        language = (
            Detection(overrides["language"], 1.0, "user")
            if "language" in overrides
            else await asyncio.to_thread(
                detect_language, text, rules.language_confidence_threshold
            )
        )
        country = (
            Detection(overrides["country"].upper(), 1.0, "user")
            if "country" in overrides
            else await asyncio.to_thread(self._countries.detect, text)
        )

        await on_stage("pertinence")
        segments = await asyncio.to_thread(segment_pages, stored.pages)
        if not segments:
            raise no_exploitable_content()
        fingerprint = await self._fingerprint(profile)
        # Le score est calculé sur un échantillon de segments répartis dans
        # tout le document : la vectorisation complète n'a lieu qu'à
        # l'indexation, donc jamais pour un document rejeté ou à vérifier.
        sample = sample_indices(
            len(segments), self._settings.ingestion_relevance_sample_segments
        )
        sample_vectors = await self._embedder.embed(
            [segments[index].text for index in sample]
        )
        semantic = semantic_score(
            sample_vectors,
            fingerprint,
            [len(segments[index].text) for index in sample],
        )
        signals = await asyncio.to_thread(
            detect_signals,
            text,
            financier_terms=financier_terms,
            themes=profile.themes,
            off_topic_keywords=rules.off_topic_keywords,
        )
        score = score_relevance(
            semantic=semantic,
            profile=profile,
            language=language,
            country=country,
            signals=signals,
            rules=rules,
        )
        decision = decide(
            score=score.final,
            profile=profile,
            country=country,
            exploitable_pages=stored.exploitable_pages,
            total_pages=stored.total_pages,
            rules=rules,
        )
        updated = await self._repository.commit_qualification(
            job,
            {
                "language": language.value,
                "language_confidence": language.confidence,
                "language_source": language.source,
                # Spec §5.6 : sous 0,7 de confiance, la langue est « à confirmer ».
                "language_to_confirm": language.source == "auto"
                and language.confidence < rules.language_confidence_threshold,
                # RG-4.5 : une langue non attendue est signalée, pas bloquante.
                "language_expected": language.value in profile.expected_languages
                if language.value
                else None,
                "country": country.value,
                "country_confidence": country.confidence,
                "country_source": country.source,
                "relevance_score": score.final,
                "relevance_details": {
                    "semantic_score": score.semantic,
                    "adjustments": score.adjustments,
                    "signals": {
                        "financiers": list(signals.financiers_found),
                        "themes": list(signals.themes_found),
                        "off_topic_keywords": list(signals.off_topic_keywords_found),
                    },
                    "thresholds": {
                        "conformity": rules.conformity_threshold,
                        "ambiguous": rules.ambiguous_threshold,
                    },
                    "segments_count": len(segments),
                    "segments_sampled": len(sample),
                    "embedding_model": self._embedder.model,
                },
                "status": decision.status,
                "status_reason": decision.message,
                "status_reason_code": decision.reason_code,
                "status_reason_params": decision.reason_params,
                "settings_version": rules.version,
            },
        )
        status = (updated or {}).get("status", decision.status)
        return QualificationOutcome(
            decision,
            score,
            language,
            country,
            status,
            segments,
            dict(zip(sample, sample_vectors, strict=True)),
        )

    async def index(
        self,
        job: ClaimedJob,
        segments: Sequence[Segment],
        language: str | None,
        on_stage: StageCallback,
        known_vectors: Mapping[int, Sequence[float]] | None = None,
    ) -> IndexOutcome:
        await on_stage("indexation")
        known = dict(known_vectors or {})
        missing = [index for index in range(len(segments)) if index not in known]
        if missing:
            computed = await self._embedder.embed(
                [segments[index].text for index in missing]
            )
            known.update(zip(missing, computed, strict=True))
        vectors = [known[index] for index in range(len(segments))]
        generation = uuid4()
        for start in range(0, len(segments), SEGMENT_BATCH_SIZE):
            rows = [
                {
                    "page_number": segment.page_number,
                    "section_title": segment.section_title,
                    "paragraph_index": segment.paragraph_index,
                    "text": segment.text,
                    "language": language,
                    "char_start": segment.char_start,
                    "char_end": segment.char_end,
                    "token_count": segment.token_count,
                    "embedding": [round(value, 8) for value in vector],
                    "model": self._embedder.model,
                }
                for segment, vector in zip(
                    segments[start : start + SEGMENT_BATCH_SIZE],
                    vectors[start : start + SEGMENT_BATCH_SIZE],
                    strict=True,
                )
            ]
            await self._repository.append_segments(job, generation, rows)
        published = await self._repository.publish_index(
            job, generation, self._embedder.model
        )
        return IndexOutcome(segments_count=published, embeddings_count=published)

    async def _fingerprint(self, profile: WorkspaceProfile) -> list[float]:
        """L'embedding de l'empreinte n'est recalculé que si elle change."""
        source = fingerprint_text(profile)
        source_hash = hashlib.sha256(
            f"{self._embedder.model}\n{source}".encode()
        ).hexdigest()
        cached = await self._repository.load_fingerprint(UUID(profile.id))
        if cached and cached[0] == source_hash:
            return cached[1]
        vector = (await self._embedder.embed([source]))[0]
        await self._repository.save_fingerprint(
            profile,
            source_text=source,
            source_hash=source_hash,
            vector=vector,
            model=self._embedder.model,
        )
        return vector

    def _ocr(self, profile: WorkspaceProfile) -> OcrOptions:
        languages = self._settings.ingestion_ocr_base_languages.split("+")
        for language in profile.expected_languages:
            code = OCR_LANGUAGES.get(language)
            if code and code not in languages:
                languages.append(code)
        return OcrOptions(
            languages="+".join(languages),
            dpi=self._settings.ingestion_ocr_dpi,
            page_timeout_seconds=self._settings.ingestion_ocr_page_timeout_seconds,
            parallelism=self._settings.ingestion_ocr_parallelism,
        )


def lease_lost() -> ApiError:
    return ApiError(
        status_code=409,
        code="LEASE_LOST",
        detail="Le verrou de la tâche d'ingestion a été perdu",
    )
