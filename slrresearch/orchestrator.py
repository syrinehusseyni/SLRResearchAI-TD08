from collections.abc import Callable

from .quality_models import Handoff, OrchestrationResult, QualityReport, SharedState


ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "search": {"screening"},
    "screening": {"rag"},
    "rag": {"writing"},
    "writing": {"quality"},
    "quality": {"writing", "finish", "escalate"},
}

FULL_ORDER = ["search", "screening", "rag", "writing", "quality"]
FULL_TRANSITIONS = {
    "search": {"screening"},
    "screening": {"rag"},
    "rag": {"writing"},
    "writing": {"quality"},
    "quality": {"search", "screening", "rag", "writing", "finish", "escalate"},
}


def validate_handoff(
    handoff: Handoff,
    allowed: dict[str, set[str]],
    available_evidence: set[str],
) -> None:
    # TODO 6 — Compléter le contrôle de permission.
    allowed_target = handoff.target in allowed.get(handoff.source, set())
    if not allowed_target:
        raise PermissionError("transition interdite")
    # Contrôles des preuves et du budget déjà fournis.
    if not set(handoff.evidence_ids) <= available_evidence:
        raise ValueError("Preuve absente de l'état partagé")
    if handoff.budget_left < 0:
        raise ValueError("Budget négatif")


def make_handoff(source: str, target: str, state: SharedState, status: str = "ok", artifact: dict | None = None) -> Handoff:
    return Handoff(
        source=source,
        target=target,
        objective=state.objective,
        artifact=artifact if artifact is not None else state.report,
        evidence_ids=state.evidence_ids,
        status=status,
        budget_left=state.budget_left,
    )


def _run_worker(worker_name: str, state: SharedState, workers: dict[str, Callable[[SharedState], dict]]) -> None:
    output = workers[worker_name](state)
    if worker_name == "writing":
        if "writing" not in state.artifacts:
            state.artifacts["writing_initial"] = output
        else:
            revision_number = sum(1 for key in state.artifacts if key.startswith("writing_revision_")) + 2
            state.artifacts[f"writing_revision_{revision_number}"] = output
    state.artifacts[worker_name] = output
    state.current_agent = worker_name
    state.history.append(worker_name)
    if worker_name == "writing":
        state.report = output


def full_pipeline_run(
    state: SharedState,
    workers: dict[str, Callable[[SharedState], dict]],
    evaluate: Callable[[dict], QualityReport],
) -> OrchestrationResult:
    """Pipeline complet : Search → Screening → RAG → Writing → Quality."""
    state = state.model_copy(deep=True)
    handoffs: list[Handoff] = []
    transitions: list[str] = []
    actions = 0
    for source, target in zip(FULL_ORDER, FULL_ORDER[1:]):
        if state.budget_left <= 0:
            return OrchestrationResult(pattern="pipeline", status="BUDGET_EXHAUSTED", state=state, handoffs=handoffs, transitions=transitions, actions=actions)
        _run_worker(source, state, workers)
        # TODO 7 — Construire le handoff de l'étape courante.
        handoff = make_handoff(
            source,
            target,
            state,
            artifact=state.artifacts[source],
        )
        validate_handoff(handoff, FULL_TRANSITIONS, set(state.evidence_ids))
        handoffs.append(handoff)
        transitions.append(f"{source}→{target}")
        state.budget_left -= 1
        actions += 1
    _run_worker("quality", state, workers)
    state.quality = evaluate(state.report)
    actions += 1
    if state.quality.verdict == "ACCEPT":
        state.next_agent = "finish"
        transitions.append("quality→finish")
        status = "FINISH"
    else:
        state.next_agent = "escalate"
        transitions.append("quality→escalate")
        status = "ESCALATE"
    return OrchestrationResult(pattern="pipeline", status=status, state=state, handoffs=handoffs, transitions=transitions, actions=actions)


def full_supervisor_run(
    state: SharedState,
    workers: dict[str, Callable[[SharedState], dict]],
    evaluate: Callable[[dict], QualityReport],
    max_handoffs: int = 7,
) -> OrchestrationResult:
    """Superviseur complet : il peut renvoyer WritingAgent vers QualityAgent."""
    state = state.model_copy(deep=True)
    handoffs: list[Handoff] = []
    transitions: list[str] = []
    actions = 0
    for source, target in zip(FULL_ORDER, FULL_ORDER[1:]):
        if actions >= max_handoffs or state.budget_left <= 0:
            return OrchestrationResult(pattern="supervisor", status="BUDGET_EXHAUSTED", state=state, handoffs=handoffs, transitions=transitions, actions=actions)
        _run_worker(source, state, workers)
        handoff = make_handoff(source, target, state, artifact=state.artifacts[source])
        validate_handoff(handoff, FULL_TRANSITIONS, set(state.evidence_ids))
        handoffs.append(handoff)
        transitions.append(f"{source}→{target}")
        state.budget_left -= 1
        actions += 1
    _run_worker("quality", state, workers)
    state.quality = evaluate(state.report)
    actions += 1
    if state.quality.verdict == "ACCEPT":
        state.next_agent = "finish"
        transitions.append("quality→finish")
        return OrchestrationResult(pattern="supervisor", status="FINISH", state=state, handoffs=handoffs, transitions=transitions, actions=actions)
    while state.quality.verdict == "REVISE" and actions < max_handoffs and state.budget_left > 0:
        # La cible est fournie par QualityAgent.
        target = state.quality.repair_target
        if target == "human":
            break
        # TODO 8 — Construire le handoff Quality vers la cible.
        # TODO 8 — Construire le handoff Quality vers la cible.
        handoff = make_handoff(
            "quality",
            target,
            state,
            status="partial",
            artifact={
                "feedback": state.quality.feedback,
                "repair_target": target,
            },
        )
        validate_handoff(handoff, FULL_TRANSITIONS, set(state.evidence_ids))
        handoffs.append(handoff)
        transitions.append(f"quality→{target}")
        state.budget_left -= 1
        actions += 1
        if target != "writing":
            # Exécuter la réparation ciblée (RAG ou Writing).
            _run_worker(target, state, workers)
            if actions >= max_handoffs or state.budget_left <= 0:
                break
            handoff = make_handoff(target, "writing", state, artifact=state.artifacts[target])
            validate_handoff(handoff, FULL_TRANSITIONS, set(state.evidence_ids))
            handoffs.append(handoff)
            transitions.append(f"{target}→writing")
            state.budget_left -= 1
            actions += 1
        # Repasser par Writing avant Quality.
        _run_worker("writing", state, workers)
        if actions >= max_handoffs or state.budget_left <= 0:
            break
        handoff = make_handoff("writing", "quality", state, artifact=state.artifacts["writing"])
        validate_handoff(handoff, FULL_TRANSITIONS, set(state.evidence_ids))
        handoffs.append(handoff)
        transitions.append("writing→quality")
        state.budget_left -= 1
        actions += 1
        # Réévaluer le rapport après réparation.
        _run_worker("quality", state, workers)
        state.quality = evaluate(state.report)
    if state.quality.verdict == "ACCEPT":
        state.next_agent = "finish"
        transitions.append("quality→finish")
        status = "FINISH"
    else:
        state.next_agent = "escalate"
        transitions.append("quality→escalate")
        status = "BUDGET_EXHAUSTED" if state.budget_left == 0 else "ESCALATE"
    return OrchestrationResult(pattern="supervisor", status=status, state=state, handoffs=handoffs, transitions=transitions, actions=actions)


def pipeline_run(
    state: SharedState,
    evaluate: Callable[[dict], QualityReport],
) -> OrchestrationResult:
    """Pipeline fixe : writing → quality, sans retour automatique."""
    handoffs: list[Handoff] = []
    transitions: list[str] = []
    state = state.model_copy(deep=True)
    state.next_agent = "quality"
    handoff = make_handoff("writing", "quality", state)
    validate_handoff(handoff, ALLOWED_TRANSITIONS, set(state.evidence_ids))
    handoffs.append(handoff)
    transitions.append("writing→quality")
    state.budget_left -= 1
    state.quality = evaluate(state.report)
    state.history.append("quality")
    if state.quality.verdict == "ACCEPT":
        state.next_agent = "finish"
        transitions.append("quality→finish")
        return OrchestrationResult(pattern="pipeline", status="FINISH", state=state, handoffs=handoffs, transitions=transitions, actions=1)
    state.next_agent = "escalate"
    transitions.append("quality→escalate")
    return OrchestrationResult(pattern="pipeline", status="ESCALATE", state=state, handoffs=handoffs, transitions=transitions, actions=1)


def supervisor_run(
    state: SharedState,
    evaluate: Callable[[dict], QualityReport],
    writer: Callable[[dict, list[str]], dict],
    max_handoffs: int = 3,
) -> OrchestrationResult:
    """Superviseur : évalue, renvoie au rédacteur si nécessaire, puis arrête."""
    state = state.model_copy(deep=True)
    handoffs: list[Handoff] = []
    transitions: list[str] = []
    actions = 0
    current_source = "writing"
    for _ in range(max_handoffs):
        if state.budget_left <= 0:
            break
        handoff = make_handoff(current_source, "quality", state, "partial" if current_source == "quality" else "ok")
        validate_handoff(handoff, ALLOWED_TRANSITIONS, set(state.evidence_ids))
        handoffs.append(handoff)
        transitions.append(f"{current_source}→quality")
        state.budget_left -= 1
        actions += 1
        state.quality = evaluate(state.report)
        state.history.append("quality")
        if state.quality.verdict == "ACCEPT":
            state.next_agent = "finish"
            transitions.append("quality→finish")
            return OrchestrationResult(pattern="supervisor", status="FINISH", state=state, handoffs=handoffs, transitions=transitions, actions=actions)
        if actions >= max_handoffs or state.budget_left <= 0:
            break
        state.report = writer(state.report, state.quality.feedback)
        state.history.append("writing")
        transitions.append("quality→writing")
        current_source = "writing"
    state.next_agent = "escalate"
    transitions.append("quality→escalate")
    status = "BUDGET_EXHAUSTED" if state.budget_left == 0 else "ESCALATE"
    return OrchestrationResult(pattern="supervisor", status=status, state=state, handoffs=handoffs, transitions=transitions, actions=actions)
