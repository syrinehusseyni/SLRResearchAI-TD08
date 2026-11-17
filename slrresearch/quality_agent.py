import re
from collections.abc import Callable

from .planning_models import EvidenceItem
from .quality_models import Claim, ClaimFinding, QualityJudgeDecision, QualityReport


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zàâçéèêëîïôûùüÿœæ0-9]+", text.lower()) if len(token) >= 4}


def parse_claims(report: dict) -> list[Claim]:
    # TODO 1 — Transformer les dictionnaires en Claim validés.
    return [
        Claim.model_validate(item)
        for item in report.get("claims", [])
    ]

def evaluate_rules(
    report: dict,
    evidence: list[EvidenceItem],
    required_rqs: set[str],
    threshold: float = 0.80,
) -> QualityReport:
    """Contrôle déterministe : structure, citations puis ancrage lexical."""
    evidence_by_id = {item.id: item for item in evidence}
    claims = parse_claims(report)
    sections = {section.get("rq_id") for section in report.get("sections", [])}
    # TODO 2 — Remplacer None par la formule de couverture.
    rq_coverage = len(sections & required_rqs) / max(1, len(required_rqs))
    findings: list[ClaimFinding] = []
    valid_citations = 0
    for claim in claims:
        if not claim.evidence_ids:
            findings.append(ClaimFinding(claim_id=claim.id, status="unsupported", reason="Aucune preuve annoncée."))
            continue
        # TODO 3a — Récupérer les preuves citées.
        selected = [evidence_by_id.get(identifier) for identifier in claim.evidence_ids]
        if any(item is None for item in selected):
            findings.append(ClaimFinding(claim_id=claim.id, status="unsupported", reason="Une preuve citée est inconnue."))
            continue
        valid_citations += 1
        source_text = " ".join((item.finding or item.text) for item in selected if item is not None)
        # TODO 3b — Calculer le recouvrement lexical.
        overlap = _tokens(claim.text) & _tokens(source_text)
        if len(overlap) < 2:
            findings.append(ClaimFinding(claim_id=claim.id, status="unsupported", reason="La citation ne soutient pas clairement l'affirmation."))
        else:
            findings.append(ClaimFinding(claim_id=claim.id, status="supported", reason="Mots-clés de l'affirmation retrouvés dans la preuve."))
    citation_validity = valid_citations / max(1, len(claims))
    supported = sum(item.status == "supported" for item in findings)
    faithfulness = supported / max(1, len(claims))
    score = (rq_coverage + citation_validity + faithfulness) / 3
    feedback: list[str] = []
    if rq_coverage < 1:
        feedback.append("Ajouter une section pour chaque question de recherche.")
    for finding in findings:
        if finding.status != "supported":
            feedback.append(f"Réviser {finding.claim_id} : {finding.reason}")
    if not claims:
        feedback.append("Le rapport ne contient aucune affirmation atomique.")
    # TODO 3c — Choisir ACCEPT ou REVISE.
    verdict = "ACCEPT" if score >= threshold and not feedback else "REVISE"
    # Un problème de preuve doit retourner vers RAG ; une formulation faible
    # peut rester du ressort du WritingAgent.
    # TODO 3d — Choisir rag ou writing.
    repair_target = (
    "rag"
    if any(f.status != "supported" for f in findings)
    else "writing"
    )
    return QualityReport(
        verdict=verdict,
        score=score,
        rq_coverage=rq_coverage,
        citation_validity=citation_validity,
        faithfulness=faithfulness,
        findings=findings,
        feedback=feedback,
        repair_target=repair_target,
    )

def build_llm_judge(model):
    """Construit un juge LLM ; les règles déterministes restent prioritaires."""
    structured = model.with_structured_output(QualityJudgeDecision)

    def judge(report: dict, rule_report: QualityReport) -> QualityReport:
        # TODO 4a — Construire le prompt.
        prompt = f"Rapport : {report}\nContrôle : {rule_report.model_dump()}"

        # TODO 4b — Appeler le modèle structuré.
        result = structured.invoke(prompt)

        # TODO 4c — Valider la sortie Pydantic.
        decision = (
            result
            if isinstance(result, QualityJudgeDecision)
            else QualityJudgeDecision.model_validate(result)
        )

        # Le juge LLM enrichit le contrôle, mais ne peut pas annuler une
        # alerte déterministe : une preuve inconnue ou non ancrée reste REVISE.
        feedback = list(dict.fromkeys(rule_report.feedback + decision.feedback))
        if "ESCALATE" in {rule_report.verdict, decision.verdict}:
            verdict = "ESCALATE"
        elif "REVISE" in {rule_report.verdict, decision.verdict}:
            verdict = "REVISE"
        else:
            verdict = "ACCEPT"
        score = min(rule_report.score, decision.score)
        return rule_report.model_copy(update={
            "verdict": verdict,
            "score": score,
            "feedback": feedback,
            "llm_calls": rule_report.llm_calls + 1,
        })

    return judge
def run_quality_loop(
    draft: dict,
    evidence: list[EvidenceItem],
    required_rqs: set[str],
    writer: Callable[[dict, list[str]], dict],
    max_revisions: int = 1,
    judge: Callable[[dict, QualityReport], QualityReport] | None = None,
) -> dict:
    """Évalue, renvoie un feedback borné, puis accepte ou escalade."""
    current = draft
    reports: list[QualityReport] = []
    for revision in range(max_revisions + 1):
        quality = evaluate_rules(current, evidence, required_rqs)
        if judge is not None:
            quality = judge(current, quality)
        quality = quality.model_copy(update={"revision": revision})
        reports.append(quality)
        # TODO 5a — Arrêt aimmédiat si la qualité est acceptable.
        if quality.verdict == "ACCEPT":
            return {"report": current, "quality": quality, "history": reports}
        # TODO 5b — Budget épuisé : escalade sans nouvel appel Writer.
        if quality.verdict == "ESCALATE" or revision == max_revisions:
            final = quality.model_copy(update={"verdict": "ESCALATE" if quality.verdict != "ACCEPT" else quality.verdict})
            return {"report": current, "quality": final, "history": reports}
        # TODO 5c — Révision contrôlée par le feedback.
        current = writer(current, quality.feedback)
    raise RuntimeError("Boucle de qualité impossible")
