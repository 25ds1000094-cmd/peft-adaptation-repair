from __future__ import annotations

import math
import re
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="PEFT Adaptation and Repair API")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INTERVENTIONS = [
    "prompt_only",
    "retrieval",
    "lora",
    "qlora",
]

CHOOSE_CODES = [
    "INVALID_INPUT",
    "UNAVAILABLE",
    "QUALITY_FLOOR",
    "FRESHNESS_REQUIRED",
    "LATENCY_LIMIT",
    "MEMORY_LIMIT",
    "DATA_LIMIT",
    "COST_LIMIT",
]

REPAIR_CODES = [
    "INVALID_TOKEN",
    "INVALID_PARAMETER",
    "CHAT_TEMPLATE_COUNT",
    "INFERENCE_MODE",
    "FULL_MODEL_ARTIFACT",
    "ADAPTER_FILE_SET",
    "INCOMPLETE_CHECKPOINT",
    "MUTABLE_BASE_REVISION",
    "LINEAGE_MISMATCH",
    "EFFECTIVE_BATCH_MISMATCH",
    "EVAL_LEAKAGE",
    "EVAL_DROPOUT_ACTIVE",
    "RESUME_DIVERGENCE",
]

SAFE_INT_MAX = 2**53 - 1

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

ROLES = {"system", "user", "assistant"}

REQUIRED_CHECKPOINT_KEYS = {
    "model",
    "optimizer",
    "scheduler",
    "step",
    "rng",
    "dataPosition",
}

EXPECTED_ADAPTER_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def is_finite_number(value: Any) -> bool:
    # bool is technically an int in Python, but must not count as a number.
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def is_safe_int(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= SAFE_INT_MAX
    )


def is_positive_safe_int(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= SAFE_INT_MAX
    )


def utf8_key(value: str) -> bytes:
    return value.encode("utf-8")


def sorted_unique_strings(values: list[str]) -> list[str]:
    return sorted(set(values), key=utf8_key)


def add_reason(reasons: set[str], code: str) -> None:
    reasons.add(code)


def empty_reason_map() -> dict[str, list[str]]:
    return {name: [] for name in INTERVENTIONS}


def all_repair_flags_false() -> dict[str, Any]:
    return {
        "labels": [],
        "templatePass": False,
        "trainableParams": [],
        "trainableCount": 0,
        "peftConfigPass": False,
        "adapterFiles": [],
        "checkpointComplete": False,
        "lineagePass": False,
        "evalIsolated": False,
        "evaluationDeterministic": False,
        "resumePass": False,
        "reasonCodes": [],
    }


# ---------------------------------------------------------------------------
# Strict-ish JSON structure validation
# ---------------------------------------------------------------------------

def is_dict(value: Any) -> bool:
    return isinstance(value, dict)


def is_list(value: Any) -> bool:
    return isinstance(value, list)


# ---------------------------------------------------------------------------
# CHOOSE
# ---------------------------------------------------------------------------

def choose_operation(payload: dict[str, Any]) -> dict[str, Any]:
    policy = payload.get("policy")

    # The output always contains all four intervention names.
    total_costs: dict[str, Any] = {name: None for name in INTERVENTIONS}
    reason_map: dict[str, list[str]] = empty_reason_map()

    # Policy must be a dictionary.
    policy_valid = is_dict(policy)

    if not policy_valid:
        for name in INTERVENTIONS:
            reason_map[name] = ["INVALID_INPUT"]

        return {
            "selected": None,
            "eligible": [],
            "totalCosts": total_costs,
            "reasonCodes": reason_map,
        }

    # Validate policy fields.
    min_quality = policy.get("minQuality")
    freshness_required = policy.get("freshnessRequired")
    max_latency = policy.get("maxLatencyMs")
    max_memory = policy.get("maxMemoryMb")
    max_labeled = policy.get("maxLabeledExamples")
    max_total_cost = policy.get("maxTotalCost")
    horizon = policy.get("horizonRequests")

    policy_valid = (
        is_finite_number(min_quality)
        and 0 <= float(min_quality) <= 1
        and is_bool(freshness_required)
        and is_finite_number(max_latency)
        and float(max_latency) >= 0
        and is_finite_number(max_memory)
        and float(max_memory) >= 0
        and is_safe_int(max_labeled)
        and is_finite_number(max_total_cost)
        and float(max_total_cost) >= 0
        and is_safe_int(horizon)
    )

    candidates = payload.get("candidates")

    if not is_list(candidates):
        policy_valid = False
        candidates = []

    # There must be exactly one candidate for each intervention.
    by_name: dict[str, dict[str, Any]] = {}

    for candidate in candidates:
        if not is_dict(candidate):
            policy_valid = False
            continue

        name = candidate.get("name")

        if not isinstance(name, str) or name not in INTERVENTIONS:
            policy_valid = False
            continue

        if name in by_name:
            policy_valid = False
            continue

        by_name[name] = candidate

    if set(by_name.keys()) != set(INTERVENTIONS):
        policy_valid = False

    if not policy_valid:
        for name in INTERVENTIONS:
            reason_map[name] = ["INVALID_INPUT"]

        return {
            "selected": None,
            "eligible": [],
            "totalCosts": total_costs,
            "reasonCodes": reason_map,
        }

    eligible: list[str] = []

    for name in INTERVENTIONS:
        candidate = by_name[name]
        reasons: set[str] = set()

        available = candidate.get("available")
        quality = candidate.get("quality")
        freshness = candidate.get("freshness")
        latency = candidate.get("latencyMs")
        memory = candidate.get("memoryMb")
        labeled = candidate.get("labeledExamples")
        one_time = candidate.get("oneTimeCost")
        recurring = candidate.get("recurringCost")

        candidate_valid = (
            is_bool(available)
            and is_finite_number(quality)
            and 0 <= float(quality) <= 1
            and is_bool(freshness)
            and is_finite_number(latency)
            and float(latency) >= 0
            and is_finite_number(memory)
            and float(memory) >= 0
            and is_safe_int(labeled)
            and is_finite_number(one_time)
            and float(one_time) >= 0
            and is_finite_number(recurring)
            and float(recurring) >= 0
        )

        if not candidate_valid:
            reasons.add("INVALID_INPUT")
            reason_map[name] = sorted(reasons, key=utf8_key)
            continue

        # Compute exactly as requested, rounded to 12 decimal places.
        total_cost = round(
            float(one_time) + float(horizon) * float(recurring),
            12,
        )
        total_costs[name] = total_cost

        if not available:
            reasons.add("UNAVAILABLE")

        if float(quality) < float(min_quality):
            reasons.add("QUALITY_FLOOR")

        if bool(freshness_required) and not freshness:
            reasons.add("FRESHNESS_REQUIRED")

        if float(latency) > float(max_latency):
            reasons.add("LATENCY_LIMIT")

        if float(memory) > float(max_memory):
            reasons.add("MEMORY_LIMIT")

        if labeled > max_labeled:
            reasons.add("DATA_LIMIT")

        if total_cost > float(max_total_cost):
            reasons.add("COST_LIMIT")

        reason_map[name] = sorted(reasons, key=utf8_key)

        if not reasons:
            eligible.append(name)

    selected = eligible[0] if eligible else None

    return {
        "selected": selected,
        "eligible": eligible,
        "totalCosts": total_costs,
        "reasonCodes": reason_map,
    }


# ---------------------------------------------------------------------------
# REPAIR
# ---------------------------------------------------------------------------

def repair_operation(payload: dict[str, Any]) -> dict[str, Any]:
    reasons: set[str] = set()

    # Defaults for all output fields.
    labels: list[int] = []
    template_pass = False
    trainable_params: list[str] = []
    trainable_count = 0
    peft_config_pass = False
    adapter_files: list[str] = []
    checkpoint_complete = False
    lineage_pass = False
    eval_isolated = False
    evaluation_deterministic = False
    resume_pass = False

    # -----------------------------------------------------------------------
    # Tokens / assistant-only labels
    # -----------------------------------------------------------------------

    tokens = payload.get("tokens")

    tokens_valid = (
        is_list(tokens)
        and len(tokens) > 0
    )

    if tokens_valid:
        for token in tokens:
            if not is_dict(token):
                tokens_valid = False
                break

            token_id = token.get("id")
            role = token.get("role")
            padding = token.get("padding")
            text = token.get("text")

            if not is_safe_int(token_id):
                tokens_valid = False
                break

            if role not in ROLES:
                tokens_valid = False
                break

            if not is_bool(padding):
                tokens_valid = False
                break

            if not isinstance(text, str):
                tokens_valid = False
                break

    if not tokens_valid:
        labels = [-100] * len(tokens) if is_list(tokens) else []
        reasons.add("INVALID_TOKEN")
    else:
        labels = [
            token["id"]
            if token["role"] == "assistant" and token["padding"] is False
            else -100
            for token in tokens
        ]

    # -----------------------------------------------------------------------
    # Chat template
    # -----------------------------------------------------------------------

    template_applications = payload.get("templateApplications")

    if template_applications == 1:
        template_pass = True
    else:
        reasons.add("CHAT_TEMPLATE_COUNT")

    # -----------------------------------------------------------------------
    # Parameters / LoRA targets
    # -----------------------------------------------------------------------

    parameters = payload.get("parameters")
    allowed_targets = payload.get("allowedTargets")

    parameters_valid = is_list(parameters)
    allowed_targets_valid = (
        is_list(allowed_targets)
        and len(allowed_targets) > 0
        and all(isinstance(x, str) and x != "" for x in allowed_targets)
        and len(set(allowed_targets)) == len(allowed_targets)
    )

    parameter_names: set[str] = set()

    if parameters_valid:
        for parameter in parameters:
            if not is_dict(parameter):
                parameters_valid = False
                break

            name = parameter.get("name")
            target = parameter.get("target")
            numel = parameter.get("numel")

            if not isinstance(name, str) or not name:
                parameters_valid = False
                break

            if name in parameter_names:
                parameters_valid = False
                break

            parameter_names.add(name)

            if not isinstance(target, str) or not target:
                parameters_valid = False
                break

            if not is_positive_safe_int(numel):
                parameters_valid = False
                break

    selected_parameter_objects: list[dict[str, Any]] = []

    if parameters_valid and allowed_targets_valid:
        allowed = set(allowed_targets)

        for parameter in parameters:
            name = parameter["name"]
            target = parameter["target"]

            # Only LoRA A/B parameters attached to an allowed target are
            # trainable.
            if (
                target in allowed
                and (
                    name.endswith(".lora_A.weight")
                    or name.endswith(".lora_B.weight")
                )
            ):
                selected_parameter_objects.append(parameter)

        if not selected_parameter_objects:
            parameters_valid = False

    if not parameters_valid or not allowed_targets_valid:
        reasons.add("INVALID_PARAMETER")
    else:
        trainable_params = sorted(
            [p["name"] for p in selected_parameter_objects],
            key=utf8_key,
        )

        # Safe integer sum.
        trainable_count = sum(
            p["numel"] for p in selected_parameter_objects
        )

        # Since every numel is a safe positive integer, reject overflow
        # beyond safe integer range.
        if trainable_count > SAFE_INT_MAX:
            reasons.add("INVALID_PARAMETER")
            trainable_params = []
            trainable_count = 0
            parameters_valid = False

    peft_config_pass = parameters_valid and allowed_targets_valid

    # -----------------------------------------------------------------------
    # Inference mode
    # -----------------------------------------------------------------------

    inference_mode = payload.get("inferenceMode")

    if inference_mode is not False:
        reasons.add("INFERENCE_MODE")

    # -----------------------------------------------------------------------
    # Adapter files
    # -----------------------------------------------------------------------

    artifact_files = payload.get("artifactFiles")

    adapter_files_valid = (
        is_list(artifact_files)
        and len(artifact_files) == 2
        and all(isinstance(x, str) for x in artifact_files)
        and sorted(artifact_files, key=utf8_key)
        == sorted(
            EXPECTED_ADAPTER_FILES,
            key=utf8_key,
        )
    )

    if adapter_files_valid:
        adapter_files = sorted(
            artifact_files,
            key=utf8_key,
        )
    else:
        reasons.add("ADAPTER_FILE_SET")

    # Detect an explicitly supplied full-model artifact as its own failure.
    if is_list(artifact_files):
        if any(
            isinstance(x, str)
            and x not in EXPECTED_ADAPTER_FILES
            for x in artifact_files
        ):
            reasons.add("FULL_MODEL_ARTIFACT")

    # -----------------------------------------------------------------------
    # Train / eval split isolation
    # -----------------------------------------------------------------------

    train_ids = payload.get("trainRowIds")
    eval_ids = payload.get("evalRowIds")

    train_valid = (
        is_list(train_ids)
        and len(train_ids) > 0
        and all(isinstance(x, str) and x != "" for x in train_ids)
        and len(set(train_ids)) == len(train_ids)
    )

    eval_valid = (
        is_list(eval_ids)
        and len(eval_ids) > 0
        and all(isinstance(x, str) and x != "" for x in eval_ids)
        and len(set(eval_ids)) == len(eval_ids)
    )

    if train_valid and eval_valid:
        if set(train_ids).isdisjoint(set(eval_ids)):
            eval_isolated = True
        else:
            reasons.add("EVAL_LEAKAGE")
    else:
        reasons.add("EVAL_LEAKAGE")

    # -----------------------------------------------------------------------
    # Evaluation determinism
    # -----------------------------------------------------------------------

    dropout_active = payload.get("dropoutActiveDuringEval")

    if dropout_active is False:
        evaluation_deterministic = True
    else:
        reasons.add("EVAL_DROPOUT_ACTIVE")

    # -----------------------------------------------------------------------
    # Checkpoint completeness
    # -----------------------------------------------------------------------

    checkpoint = payload.get("checkpoint")

    if (
        is_dict(checkpoint)
        and REQUIRED_CHECKPOINT_KEYS.issubset(checkpoint.keys())
    ):
        checkpoint_complete = True
    else:
        reasons.add("INCOMPLETE_CHECKPOINT")

    # -----------------------------------------------------------------------
    # Lineage
    # -----------------------------------------------------------------------

    base_revision = payload.get("baseRevision")
    dataset_digest = payload.get("datasetDigest")
    code_digest = payload.get("codeDigest")
    config_digest = payload.get("configDigest")
    expected_digests = payload.get("expectedDigests")

    base_revision_valid = (
        isinstance(base_revision, str)
        and HEX40.fullmatch(base_revision) is not None
    )

    if not base_revision_valid:
        reasons.add("MUTABLE_BASE_REVISION")

    digest_fields = [dataset_digest, code_digest, config_digest]

    digests_valid = all(
        isinstance(x, str)
        and HEX64.fullmatch(x) is not None
        for x in digest_fields
    )

    expected_valid = (
        is_dict(expected_digests)
        and all(
            isinstance(expected_digests.get(key), str)
            and HEX64.fullmatch(expected_digests[key]) is not None
            for key in ("datasetDigest", "codeDigest", "configDigest")
        )
    )

    digest_matches = (
        digests_valid
        and expected_valid
        and dataset_digest == expected_digests["datasetDigest"]
        and code_digest == expected_digests["codeDigest"]
        and config_digest == expected_digests["configDigest"]
    )

    if not digest_matches:
        reasons.add("LINEAGE_MISMATCH")

    lineage_pass = base_revision_valid and digest_matches

    # -----------------------------------------------------------------------
    # Effective batch
    # -----------------------------------------------------------------------

    micro_batch = payload.get("microBatch")
    gradient_accumulation = payload.get("gradientAccumulation")
    replicas = payload.get("replicas")
    expected_effective_batch = payload.get("expectedEffectiveBatch")

    batch_valid = (
        is_positive_safe_int(micro_batch)
        and is_positive_safe_int(gradient_accumulation)
        and is_positive_safe_int(replicas)
        and is_positive_safe_int(expected_effective_batch)
    )

    if batch_valid:
        actual_effective_batch = (
            micro_batch
            * gradient_accumulation
            * replicas
        )

        if actual_effective_batch != expected_effective_batch:
            reasons.add("EFFECTIVE_BATCH_MISMATCH")
    else:
        reasons.add("EFFECTIVE_BATCH_MISMATCH")

    # -----------------------------------------------------------------------
    # Resume equality
    # -----------------------------------------------------------------------

    uninterrupted = payload.get("uninterruptedWeights")
    resumed = payload.get("resumedWeights")
    tolerance = payload.get("resumeTolerance")

    resume_shape_valid = (
        is_list(uninterrupted)
        and len(uninterrupted) > 0
        and is_list(resumed)
        and len(resumed) > 0
        and len(uninterrupted) == len(resumed)
        and all(is_finite_number(x) for x in uninterrupted)
        and all(is_finite_number(x) for x in resumed)
        and is_finite_number(tolerance)
        and float(tolerance) >= 0
    )

    if resume_shape_valid:
        resume_pass = all(
            abs(float(a) - float(b)) <= float(tolerance)
            for a, b in zip(uninterrupted, resumed)
        )

        if not resume_pass:
            reasons.add("RESUME_DIVERGENCE")
    else:
        reasons.add("RESUME_DIVERGENCE")

    # -----------------------------------------------------------------------
    # Final exact output shape
    # -----------------------------------------------------------------------

    return {
        "labels": labels,
        "templatePass": template_pass,
        "trainableParams": trainable_params,
        "trainableCount": trainable_count,
        "peftConfigPass": peft_config_pass,
        "adapterFiles": adapter_files,
        "checkpointComplete": checkpoint_complete,
        "lineagePass": lineage_pass,
        "evalIsolated": eval_isolated,
        "evaluationDeterministic": evaluation_deterministic,
        "resumePass": resume_pass,
        "reasonCodes": sorted(reasons, key=utf8_key),
    }


# ---------------------------------------------------------------------------
# HTTP endpoint
# ---------------------------------------------------------------------------

@app.post("/adapt")
async def adapt(request: Request):
    """
    Single deterministic endpoint.

    Unknown or missing operation must return exactly:
        HTTP 400
        {"error":"INVALID_INPUT"}
    """

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    if not isinstance(payload, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    operation = payload.get("operation")

    if operation not in {"choose", "repair"}:
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    if operation == "choose":
        result = choose_operation(payload)
    else:
        result = repair_operation(payload)

    return JSONResponse(
        status_code=200,
        content=result,
    )


@app.get("/")
async def root():
    return {"service": "peft-adaptation-repair", "status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}
