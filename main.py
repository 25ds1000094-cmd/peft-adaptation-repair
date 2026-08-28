from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import math
import re

app = FastAPI()

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

MAX_SAFE_INT = 9007199254740991

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

ROLES = {"system", "user", "assistant"}

CHECKPOINT_KEYS = {
    "model",
    "optimizer",
    "scheduler",
    "step",
    "rng",
    "dataPosition",
}

ADAPTER_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
}


def finite_number(x):
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
    )


def safe_int(x):
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and 0 <= x <= MAX_SAFE_INT
    )


def positive_safe_int(x):
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and 1 <= x <= MAX_SAFE_INT
    )


def utf8_sort(values):
    return sorted(set(values), key=lambda x: x.encode("utf-8"))


def choose(payload):
    total_costs = {name: None for name in INTERVENTIONS}
    reasons = {name: [] for name in INTERVENTIONS}

    policy = payload.get("policy")
    candidates = payload.get("candidates")

    if not isinstance(policy, dict) or not isinstance(candidates, list):
        for name in INTERVENTIONS:
            reasons[name] = ["INVALID_INPUT"]

        return {
            "selected": None,
            "eligible": [],
            "totalCosts": total_costs,
            "reasonCodes": reasons,
        }

    min_quality = policy.get("minQuality")
    freshness_required = policy.get("freshnessRequired")
    max_latency = policy.get("maxLatencyMs")
    max_memory = policy.get("maxMemoryMb")
    max_labeled = policy.get("maxLabeledExamples")
    max_cost = policy.get("maxTotalCost")
    horizon = policy.get("horizonRequests")

    valid_policy = (
        finite_number(min_quality)
        and 0 <= float(min_quality) <= 1
        and isinstance(freshness_required, bool)
        and finite_number(max_latency)
        and float(max_latency) >= 0
        and finite_number(max_memory)
        and float(max_memory) >= 0
        and safe_int(max_labeled)
        and finite_number(max_cost)
        and float(max_cost) >= 0
        and safe_int(horizon)
    )

    by_name = {}

    if valid_policy:
        for candidate in candidates:
            if not isinstance(candidate, dict):
                valid_policy = False
                break

            name = candidate.get("name")

            if name not in INTERVENTIONS or name in by_name:
                valid_policy = False
                break

            by_name[name] = candidate

        if set(by_name) != set(INTERVENTIONS):
            valid_policy = False

    if not valid_policy:
        for name in INTERVENTIONS:
            reasons[name] = ["INVALID_INPUT"]

        return {
            "selected": None,
            "eligible": [],
            "totalCosts": total_costs,
            "reasonCodes": reasons,
        }

    eligible = []

    for name in INTERVENTIONS:
        c = by_name[name]
        r = set()

        valid_candidate = (
            isinstance(c.get("available"), bool)
            and finite_number(c.get("quality"))
            and 0 <= float(c["quality"]) <= 1
            and isinstance(c.get("freshness"), bool)
            and finite_number(c.get("latencyMs"))
            and float(c["latencyMs"]) >= 0
            and finite_number(c.get("memoryMb"))
            and float(c["memoryMb"]) >= 0
            and safe_int(c.get("labeledExamples"))
            and finite_number(c.get("oneTimeCost"))
            and float(c["oneTimeCost"]) >= 0
            and finite_number(c.get("recurringCost"))
            and float(c["recurringCost"]) >= 0
        )

        if not valid_candidate:
            reasons[name] = ["INVALID_INPUT"]
            continue

        cost = round(
            float(c["oneTimeCost"])
            + float(horizon) * float(c["recurringCost"]),
            12,
        )

        total_costs[name] = cost

        if not c["available"]:
            r.add("UNAVAILABLE")

        if float(c["quality"]) < float(min_quality):
            r.add("QUALITY_FLOOR")

        if freshness_required and not c["freshness"]:
            r.add("FRESHNESS_REQUIRED")

        if float(c["latencyMs"]) > float(max_latency):
            r.add("LATENCY_LIMIT")

        if float(c["memoryMb"]) > float(max_memory):
            r.add("MEMORY_LIMIT")

        if c["labeledExamples"] > max_labeled:
            r.add("DATA_LIMIT")

        if cost > float(max_cost):
            r.add("COST_LIMIT")

        reasons[name] = sorted(r, key=lambda x: x.encode("utf-8"))

        if not r:
            eligible.append(name)

    return {
        "selected": eligible[0] if eligible else None,
        "eligible": eligible,
        "totalCosts": total_costs,
        "reasonCodes": reasons,
    }


def repair(payload):
    reason = set()

    # ---------------- Tokens ----------------

    tokens = payload.get("tokens")
    labels = []

    tokens_valid = isinstance(tokens, list) and len(tokens) > 0

    if tokens_valid:
        for token in tokens:
            if not isinstance(token, dict):
                tokens_valid = False
                break

            if not safe_int(token.get("id")):
                tokens_valid = False
                break

            if token.get("role") not in ROLES:
                tokens_valid = False
                break

            if not isinstance(token.get("padding"), bool):
                tokens_valid = False
                break

            if not isinstance(token.get("text"), str):
                tokens_valid = False
                break

    if tokens_valid:
        labels = [
            t["id"]
            if t["role"] == "assistant" and t["padding"] is False
            else -100
            for t in tokens
        ]
    else:
        labels = [-100] * len(tokens) if isinstance(tokens, list) else []
        reason.add("INVALID_TOKEN")

    # ---------------- Template ----------------

    template_pass = payload.get("templateApplications") == 1

    if not template_pass:
        reason.add("CHAT_TEMPLATE_COUNT")

    # ---------------- Parameters ----------------

    parameters = payload.get("parameters")
    allowed = payload.get("allowedTargets")

    parameters_valid = isinstance(parameters, list)
    allowed_valid = (
        isinstance(allowed, list)
        and len(allowed) > 0
        and all(isinstance(x, str) and x for x in allowed)
        and len(set(allowed)) == len(allowed)
    )

    names = set()

    if parameters_valid:
        for p in parameters:
            if not isinstance(p, dict):
                parameters_valid = False
                break

            name = p.get("name")
            target = p.get("target")
            numel = p.get("numel")

            if not isinstance(name, str) or not name:
                parameters_valid = False
                break

            if name in names:
                parameters_valid = False
                break

            names.add(name)

            if not isinstance(target, str) or not target:
                parameters_valid = False
                break

            if not positive_safe_int(numel):
                parameters_valid = False
                break

    selected = []

    if parameters_valid and allowed_valid:
        allowed_set = set(allowed)

        for p in parameters:
            if (
                p["target"] in allowed_set
                and (
                    p["name"].endswith(".lora_A.weight")
                    or p["name"].endswith(".lora_B.weight")
                )
            ):
                selected.append(p)

        if not selected:
            parameters_valid = False

    if not parameters_valid or not allowed_valid:
        reason.add("INVALID_PARAMETER")
        trainable_params = []
        trainable_count = 0
        peft_pass = False
    else:
        trainable_params = sorted(
            [p["name"] for p in selected],
            key=lambda x: x.encode("utf-8"),
        )

        trainable_count = sum(p["numel"] for p in selected)

        if trainable_count > MAX_SAFE_INT:
            reason.add("INVALID_PARAMETER")
            trainable_params = []
            trainable_count = 0
            peft_pass = False
        else:
            peft_pass = True

    # ---------------- Inference ----------------

    if payload.get("inferenceMode") is not False:
        reason.add("INFERENCE_MODE")

    # ---------------- Adapter files ----------------

    files = payload.get("artifactFiles")

    adapter_valid = (
        isinstance(files, list)
        and len(files) == 2
        and all(isinstance(x, str) for x in files)
        and set(files) == ADAPTER_FILES
    )

    if adapter_valid:
        adapter_files = sorted(
            files,
            key=lambda x: x.encode("utf-8"),
        )
    else:
        adapter_files = []
        reason.add("ADAPTER_FILE_SET")

    if isinstance(files, list):
        for f in files:
            if isinstance(f, str) and f not in ADAPTER_FILES:
                reason.add("FULL_MODEL_ARTIFACT")

    # ---------------- Evaluation isolation ----------------

    train_ids = payload.get("trainRowIds")
    eval_ids = payload.get("evalRowIds")

    train_valid = (
        isinstance(train_ids, list)
        and len(train_ids) > 0
        and all(isinstance(x, str) and x for x in train_ids)
        and len(set(train_ids)) == len(train_ids)
    )

    eval_valid = (
        isinstance(eval_ids, list)
        and len(eval_ids) > 0
        and all(isinstance(x, str) and x for x in eval_ids)
        and len(set(eval_ids)) == len(eval_ids)
    )

    eval_isolated = False

    if train_valid and eval_valid:
        if set(train_ids).isdisjoint(set(eval_ids)):
            eval_isolated = True
        else:
            reason.add("EVAL_LEAKAGE")
    else:
        reason.add("EVAL_LEAKAGE")

    # ---------------- Evaluation determinism ----------------

    evaluation_deterministic = payload.get("dropoutActiveDuringEval") is False

    if not evaluation_deterministic:
        reason.add("EVAL_DROPOUT_ACTIVE")

    # ---------------- Checkpoint ----------------

    checkpoint = payload.get("checkpoint")

    checkpoint_complete = (
        isinstance(checkpoint, dict)
        and CHECKPOINT_KEYS.issubset(checkpoint.keys())
    )

    if not checkpoint_complete:
        reason.add("INCOMPLETE_CHECKPOINT")

    # ---------------- Lineage ----------------

    base = payload.get("baseRevision")
    dataset = payload.get("datasetDigest")
    code = payload.get("codeDigest")
    config = payload.get("configDigest")
    expected = payload.get("expectedDigests")

    base_valid = (
        isinstance(base, str)
        and HEX40.fullmatch(base) is not None
    )

    if not base_valid:
        reason.add("MUTABLE_BASE_REVISION")

    digest_valid = (
        isinstance(dataset, str)
        and HEX64.fullmatch(dataset) is not None
        and isinstance(code, str)
        and HEX64.fullmatch(code) is not None
        and isinstance(config, str)
        and HEX64.fullmatch(config) is not None
    )

    expected_valid = (
        isinstance(expected, dict)
        and isinstance(expected.get("datasetDigest"), str)
        and isinstance(expected.get("codeDigest"), str)
        and isinstance(expected.get("configDigest"), str)
        and HEX64.fullmatch(expected["datasetDigest"]) is not None
        and HEX64.fullmatch(expected["codeDigest"]) is not None
        and HEX64.fullmatch(expected["configDigest"]) is not None
    )

    lineage_match = (
        digest_valid
        and expected_valid
        and dataset == expected["datasetDigest"]
        and code == expected["codeDigest"]
        and config == expected["configDigest"]
    )

    if not lineage_match:
        reason.add("LINEAGE_MISMATCH")

    lineage_pass = base_valid and lineage_match

    # ---------------- Effective batch ----------------

    mb = payload.get("microBatch")
    ga = payload.get("gradientAccumulation")
    replicas = payload.get("replicas")
    expected_batch = payload.get("expectedEffectiveBatch")

    batch_valid = (
        positive_safe_int(mb)
        and positive_safe_int(ga)
        and positive_safe_int(replicas)
        and positive_safe_int(expected_batch)
    )

    batch_pass = False

    if batch_valid:
        batch_pass = mb * ga * replicas == expected_batch

    if not batch_pass:
        reason.add("EFFECTIVE_BATCH_MISMATCH")

    # ---------------- Resume ----------------

    uninterrupted = payload.get("uninterruptedWeights")
    resumed = payload.get("resumedWeights")
    tolerance = payload.get("resumeTolerance")

    resume_valid = (
        isinstance(uninterrupted, list)
        and len(uninterrupted) > 0
        and isinstance(resumed, list)
        and len(resumed) == len(uninterrupted)
        and len(resumed) > 0
        and all(finite_number(x) for x in uninterrupted)
        and all(finite_number(x) for x in resumed)
        and finite_number(tolerance)
        and float(tolerance) >= 0
    )

    resume_pass = False

    if resume_valid:
        resume_pass = all(
            abs(float(a) - float(b)) <= float(tolerance)
            for a, b in zip(uninterrupted, resumed)
        )

    if not resume_pass:
        reason.add("RESUME_DIVERGENCE")

    # ---------------- Final response ----------------

    return {
        "labels": labels,
        "templatePass": template_pass,
        "trainableParams": trainable_params,
        "trainableCount": trainable_count,
        "peftConfigPass": peft_pass,
        "adapterFiles": adapter_files,
        "checkpointComplete": checkpoint_complete,
        "lineagePass": lineage_pass,
        "evalIsolated": eval_isolated,
        "evaluationDeterministic": evaluation_deterministic,
        "resumePass": resume_pass,
        "reasonCodes": sorted(
            reason,
            key=lambda x: x.encode("utf-8"),
        ),
    }


@app.post("/adapt")
async def adapt(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    operation = body.get("operation")

    if operation == "choose":
        return choose(body)

    if operation == "repair":
        return repair(body)

    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"},
    )


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}
