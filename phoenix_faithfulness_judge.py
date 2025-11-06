#!/usr/bin/env python3
"""
PhoenixFaithfulnessJudgeV1
- Structured, deterministic "LLM-as-judge" for summary faithfulness (hallucination check).
- Multi-judge (two models) + simple adjudication (median score).
- Strict JSON output with Pydantic validation.
- Emits OpenTelemetry traces to Arize Phoenix.

CONFIG VIA ENV:
  PHOENIX_OTLP_ENDPOINT=http://localhost:6006/v1/traces     # Phoenix OTLP HTTP
  AZURE_OPENAI_ENDPOINT=https://<your-azure-openai>.openai.azure.com/
  AZURE_OPENAI_API_KEY=<key>
  AZURE_OPENAI_API_VERSION=2024-05-01-preview               # or your version
  JUDGE_A_DEPLOYMENT=<azure-deployment-name-1>              # e.g. gpt-4o-mini-judge
  JUDGE_B_DEPLOYMENT=<azure-deployment-name-2>              # a different model/setting/vendor if possible
  # Optional:
  OTEL_SERVICE_NAME=phoenix-faithfulness-judge
"""

import os
import json
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

# -------- Validation --------
from pydantic import BaseModel, Field, ValidationError

# -------- Phoenix / OpenTelemetry wiring --------
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# -------- Azure OpenAI client (chat completions) --------
try:
    # Newer SDK (openai >= 1.0)
    from openai import AzureOpenAI  # pip install openai
    _HAS_AZURE = True
except Exception:
    _HAS_AZURE = False


# ======================
# 1) Models / Schemas
# ======================
class Highlight(BaseModel):
    claim: str = Field(..., max_length=240)
    status: str = Field(..., pattern="^(supported|unsupported|contradicted)$")


class JudgeVerdict(BaseModel):
    score: int = Field(..., ge=1, le=5)
    issues: List[str] = Field(default_factory=list)  # e.g., ["unsupported_entity","wrong_quantity"]
    justification: str = Field(..., max_length=120)
    highlights: Optional[List[Highlight]] = None


# ======================
# 2) Prompts
# ======================
JUDGE_SYSTEM = (
    "You are a strict evaluator of summary FAITHFULNESS to the provided DATA. "
    "Return ONLY valid JSON, no prose."
)

JUDGE_USER_TEMPLATE = """\
TASK: Score how faithful the SUMMARY is to the DATA (1=lowest, 5=highest).

Rules:
- Penalize any unsupported claims, fabricated entities/quantities, or contradictions.
- A single severe hallucination caps score at 2 or below.
- Be concise. Output JSON only.

DATA:
{data}

SUMMARY:
{summary}

Return JSON matching exactly this schema:
{{
  "score": <int 1-5>,
  "issues": [<string>],
  "justification": "<<=120 chars>",
  "highlights": [{{"claim":"...", "status":"supported|unsupported|contradicted"}}]
}}
"""


def build_prompt(data: str, summary: str) -> List[Dict[str, str]]:
    """Chat-style messages."""
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": JUDGE_USER_TEMPLATE.format(data=data, summary=summary),
        },
    ]


# ======================
# 3) Azure client helper
# ======================
@dataclass
class AzureCfg:
    endpoint: str
    key: str
    api_version: str

    @classmethod
    def from_env(cls) -> "AzureCfg":
        return cls(
            endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip(),
            key=os.environ.get("AZURE_OPENAI_API_KEY", "").strip(),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-05-01-preview").strip(),
        )

    def ok(self) -> bool:
        return all([self.endpoint, self.key, self.api_version])


class AzureChatClient:
    """Thin wrapper around Azure OpenAI Chat Completions."""

    def __init__(self, cfg: AzureCfg):
        if not _HAS_AZURE:
            raise RuntimeError("openai package not found. pip install openai")
        if not cfg.ok():
            raise RuntimeError("Azure OpenAI env not configured.")
        self.client = AzureOpenAI(
            api_key=cfg.key,
            api_version=cfg.api_version,
            azure_endpoint=cfg.endpoint,
        )

    def chat_completion(self, deployment: str, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 512) -> str:
        resp = self.client.chat.completions.create(
            model=deployment,  # deployment name
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},  # ask SDK to enforce JSON if supported
        )
        return resp.choices[0].message.content or ""


# ======================
# 4) Judge core
# ======================
def sanitize_raw_json(raw: str) -> str:
    """Remove accidental code fences etc."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if "\n" in s:
            s = s.split("\n", 1)[-1]
    return s.strip()


def call_judge_with_model(
    chat_client: AzureChatClient,
    deployment: str,
    data: str,
    summary: str,
    tracer: trace.Tracer,
    max_retries: int = 2,
) -> JudgeVerdict:
    messages = build_prompt(data=data, summary=summary)
    last_err: Optional[Exception] = None
    with tracer.start_as_current_span("judge.call") as span:
        span.set_attribute("judge.deployment", deployment)
        t0 = time.time()
        for attempt in range(max_retries + 1):
            try:
                raw = chat_client.chat_completion(deployment=deployment, messages=messages, temperature=0.0)
                raw = sanitize_raw_json(raw)
                obj = json.loads(raw)
                verdict = JudgeVerdict(**obj)
                latency_ms = int((time.time() - t0) * 1000)
                span.set_attribute("judge.latency_ms", latency_ms)
                span.set_attribute("judge.score", verdict.score)
                span.set_attribute("judge.issues", json.dumps(verdict.issues))
                # Do not log raw text from data/summary here to avoid leaking PII; log lengths only
                span.set_attribute("judge.summary_len", len(summary))
                span.set_attribute("judge.data_len", len(data))
                return verdict
            except (json.JSONDecodeError, ValidationError) as e:
                last_err = e
                # Minimal repair: ask again, reminding it to output strict JSON only.
                messages = build_prompt(data=data, summary=summary)
                messages.insert(
                    1,
                    {
                        "role": "system",
                        "content": "Your previous output was invalid. Return STRICT JSON only. No prose.",
                    },
                )
                time.sleep(0.25)
            except Exception as e:  # transport/transient
                last_err = e
                time.sleep(0.25)

        span.set_attribute("judge.error", str(last_err)[:240])
        raise ValueError(f"Judge output invalid after retries: {last_err}")


def adjudicate(verdicts: List[JudgeVerdict]) -> JudgeVerdict:
    scores = sorted(v.score for v in verdicts)
    median = scores[len(scores) // 2]
    all_issues = sorted({i for v in verdicts for i in v.issues})
    justification = min((v.justification for v in verdicts), key=len, default="Consensus")
    # merge a small subset of highlights if present
    merged_high: List[Highlight] = []
    for v in verdicts:
        if v.highlights:
            merged_high.extend(v.highlights[:2])
    if merged_high:
        return JudgeVerdict(score=median, issues=all_issues, justification=justification, highlights=merged_high[:4])
    return JudgeVerdict(score=median, issues=all_issues, justification=justification)


# ======================
# 5) Phoenix / OTEL wiring
# ======================
def setup_tracing() -> trace.Tracer:
    endpoint = os.environ.get("PHOENIX_OTLP_ENDPOINT", "http://localhost:6006/v1/traces")
    service_name = os.environ.get("OTEL_SERVICE_NAME", "phoenix-faithfulness-judge")
    provider = TracerProvider(resource=None)  # resource can be extended with service.name via OTEL env
    exporter = OTLPSpanExporter(endpoint=endpoint)
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer(service_name)
    return tracer


# ======================
# 6) Public API
# ======================
class PhoenixFaithfulnessJudgeV1:
    """Main entrypoint usable from your app."""
    def __init__(self):
        self.tracer = setup_tracing()
        if not _HAS_AZURE:
            raise RuntimeError("Missing dependency: pip install openai")
        self.azure = AzureChatClient(AzureCfg.from_env())
        # Load judge deployments
        self.judge_a = os.environ.get("JUDGE_A_DEPLOYMENT", "").strip()
        self.judge_b = os.environ.get("JUDGE_B_DEPLOYMENT", "").strip() or self.judge_a
        if not self.judge_a:
            raise RuntimeError("Set JUDGE_A_DEPLOYMENT env var to an Azure deployment name.")

    def evaluate(self, data: str, summary: str) -> JudgeVerdict:
        with self.tracer.start_as_current_span("judge.evaluate") as span:
            span.set_attribute("eval.version", "PhoenixFaithfulnessJudgeV1")
            span.set_attribute("eval.summary_len", len(summary))
            span.set_attribute("eval.data_len", len(data))
            # Optionally add your dataset/run identifiers here
            # span.set_attribute("eval.dataset_id", "ds-123")
            # span.set_attribute("eval.run_id", "run-456")

            v1 = call_judge_with_model(self.azure, self.judge_a, data, summary, tracer=self.tracer)
            v2 = call_judge_with_model(self.azure, self.judge_b, data, summary, tracer=self.tracer)

            # If large disagreement, we could add a third adjudicator model here.
            if abs(v1.score - v2.score) >= 2:
                span.set_attribute("adjudication.triggered", True)
            verdict = adjudicate([v1, v2])

            span.set_attribute("final.score", verdict.score)
            span.set_attribute("final.issues", json.dumps(verdict.issues))
            return verdict


# ======================
# 7) Quick CLI demo
# ======================
def _demo():
    """Run: python phoenix_faithfulness_judge.py"""
    data = (
        "ACME Q2 report: revenue $12.4M (up 8% YoY), net income $1.2M, "
        "launched Beta of Compass product on May 9; no layoffs announced."
    )
    summary_good = "ACME’s Q2 revenue was $12.4M (up 8% YoY). Net income reached $1.2M. " \
                   "They launched a Compass beta on May 9. No layoffs were announced."
    summary_bad = "Revenue hit $20M with 20% YoY growth, profits doubled, and ACME laid off 5%."

    judge = PhoenixFaithfulnessJudgeV1()

    print("=== GOOD SUMMARY ===")
    print(judge.evaluate(data, summary_good).model_dump())

    print("\n=== BAD SUMMARY ===")
    print(judge.evaluate(data, summary_bad).model_dump())


if __name__ == "__main__":
    _demo()
