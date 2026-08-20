"""
evaluation.py - Automated RAG Evaluation & LLM-as-a-Judge Suite
---------------------------------------------------------------
This module systematically evaluates the accuracy and safety of the RAG pipeline:
1. Retrieval Metrics:
   - Hit@1, Hit@3, Hit@4: Measures if the ground-truth chapter section is in top-k.
   - Mean Reciprocal Rank (MRR): Measures ranking precision (1/rank).
   - Top-1 and Average Cosine similarity scores.
2. Citation Integrity:
   - Validates that source markers [1]..[k] exist and only reference valid chunk indices.
3. LLM-as-a-Judge Scoring (Groq):
   - Faithfulness / Groundedness: 1-5 score measuring whether claims are supported by context.
   - Answer Relevance: 1-5 score measuring if the question is directly addressed.
4. Out-of-Domain Safety & Refusal:
   - Evaluates a negative test case (Type 2 diabetes) to ensure the system refuses
     to hallucinate when context is insufficient.
5. Exports complete structured results into evaluation_report.json.
"""

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from main import get_env_value
from rag_chat import GROQ_API_URL, GROQ_MODEL, build_context, generate_rag_answer
from retrieval import retrieve


# -----------------------------------------------------------------------------
# Configuration & Benchmark Cases
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
REPORT_PATH = PROJECT_DIR / "evaluation_report.json"
RETRIEVAL_LIMIT = 4

# Test cases representing key clinical & scientific sections of the epilepsy document.
EVALUATION_CASES = [
    {
        "id": "CASE-1",
        "question": "What are the main causes of epilepsy?",
        "expected_section": "2.2 Etiological Classification and Risk Factors",
    },
    {
        "id": "CASE-2",
        "question": "How common is epilepsy worldwide?",
        "expected_section": "2.1 Global Incidence and Prevalence",
    },
    {
        "id": "CASE-3",
        "question": "How do ion-channel problems contribute to epilepsy?",
        "expected_section": "3.2 Ion-Channel Dysfunction",
    },
    {
        "id": "CASE-4",
        "question": "What is the role of EEG in epilepsy diagnosis?",
        "expected_section": "4.1.2 Electrophysiological Assessment",
    },
    {
        "id": "CASE-5",
        "question": "How is drug-resistant epilepsy managed?",
        "expected_section": "4.2.3 Management of Drug-Resistant Epilepsy",
    },
]

# Negative test case to verify that the RAG model refrains from hallucinating
# when the document lacks relevant information.
OUT_OF_DOMAIN_CASE = {
    "id": "NEG-1",
    "question": "What are the first-line medication regimens and dietary management for Type 2 diabetes mellitus?",
    "expected_behavior": "Should state that context is insufficient and avoid hallucinating medical advice.",
}

# Evaluator LLM model on Groq
JUDGE_MODEL = "openai/gpt-oss-20b"



# -----------------------------------------------------------------------------
# LLM Judge Output Parser (Multi-strategy JSON parsing)
# -----------------------------------------------------------------------------
def parse_judge_json(content: str) -> dict[str, Any]:
    """Robustly parse JSON response from the evaluator LLM with fallbacks.

    Handles:
    1. Pure JSON strings.
    2. Markdown code fences (```json ... ```).
    3. Partial/embedded JSON objects.
    4. Regex field extraction fallback.
    """
    try:
        return json.loads(content.strip())
    except Exception:
        pass

    # Match JSON inside markdown code fence
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except Exception:
            pass

    # Match any balanced outer JSON object
    match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    # Regex fallback for score and reasoning
    score_match = re.search(r'"score"\s*:\s*(\d+)', content)
    reason_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', content)
    faithful_match = re.search(r'"is_faithful"\s*:\s*(true|false)', content, re.IGNORECASE)
    relevant_match = re.search(r'"is_relevant"\s*:\s*(true|false)', content, re.IGNORECASE)
    refused_match = re.search(r'"correctly_refused_or_flagged_insufficient"\s*:\s*(true|false)', content, re.IGNORECASE)

    res: dict[str, Any] = {}
    if score_match:
        res["score"] = int(score_match.group(1))
    if faithful_match:
        res["is_faithful"] = faithful_match.group(1).lower() == "true"
    if relevant_match:
        res["is_relevant"] = relevant_match.group(1).lower() == "true"
    if refused_match:
        res["correctly_refused_or_flagged_insufficient"] = refused_match.group(1).lower() == "true"
    if reason_match:
        res["reasoning"] = reason_match.group(1)

    if res:
        return res

    return {"raw_response": content, "score": None, "explanation": "Failed to parse structured JSON"}


# -----------------------------------------------------------------------------
# LLM Judge Invocation (with HTTP 429 backoff)
# -----------------------------------------------------------------------------
def call_groq_judge(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    """Query Groq to act as an impartial evaluator and return parsed JSON.

    Args:
        system_prompt: Impartial judging guidelines.
        user_prompt: Context/Question and Answer to evaluate.

    Returns:
        Structured evaluation dict with scores and reasoning.
    """
    api_key = get_env_value("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")

    payload = json.dumps(
        {
            "model": JUDGE_MODEL,
            "temperature": 0.0,             # 0.0 temperature for deterministic evaluation
            "max_completion_tokens": 700,
            "messages": [
                {"role": "system", "content": system_prompt + "\nYou must reply strictly with a JSON object."},
                {"role": "user", "content": user_prompt},
            ],
        }
    ).encode("utf-8")

    request = Request(
        GROQ_API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "API-RAG-Eval/1.0",
        },
        method="POST",
    )

    max_retries = 4
    data = None

    for attempt in range(max_retries):
        try:
            with urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
                break
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            if error.code == 429 and attempt < max_retries - 1:
                wait_time = 2.0 * (2 ** attempt)
                match = re.search(r"try again in (\d+\.?\d*)s", details)
                if match:
                    wait_time = max(wait_time, float(match.group(1)) + 1.0)
                time.sleep(wait_time)
                continue
            raise RuntimeError(f"Groq Judge API call failed ({error.code}): {details}") from error
        except URLError as error:
            if attempt < max_retries - 1:
                time.sleep(2.0)
                continue
            raise RuntimeError(f"Could not reach Groq API: {error.reason}") from error

    if not data:
        raise RuntimeError("Failed to receive response from Groq Judge.")

    content = data["choices"][0]["message"]["content"].strip()
    return parse_judge_json(content)


def source_sections(results: list[dict[str, Any]]) -> list[str]:
    """Extract section labels from Qdrant result metadata."""
    return [
        str(result["metadata"].get("SubSection") or result["metadata"].get("Section") or "")
        for result in results
    ]


def evaluate_retrieval() -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Evaluate Hit@1, Hit@3, Hit@4, MRR, and similarity scores across test cases."""
    results = []
    hit_1_list = []
    hit_3_list = []
    hit_4_list = []
    reciprocal_ranks = []
    top_scores = []

    for case in EVALUATION_CASES:
        retrieved = retrieve(case["question"], RETRIEVAL_LIMIT)
        sections = source_sections(retrieved)
        expected = case["expected_section"]

        # Check hits at different cutoffs
        hit_1 = expected in sections[:1]
        hit_3 = expected in sections[:3]
        hit_4 = expected in sections[:4]

        # Calculate Reciprocal Rank (1/rank of first hit)
        rr = 0.0
        for rank, sec in enumerate(sections, start=1):
            if expected in sec or sec in expected:
                rr = 1.0 / rank
                break

        top_score = float(retrieved[0]["score"]) if retrieved else 0.0
        scores = [float(item["score"]) for item in retrieved]

        hit_1_list.append(1.0 if hit_1 else 0.0)
        hit_3_list.append(1.0 if hit_3 else 0.0)
        hit_4_list.append(1.0 if hit_4 else 0.0)
        reciprocal_ranks.append(rr)
        top_scores.append(top_score)

        results.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected_section": expected,
                "hit_at_1": hit_1,
                "hit_at_3": hit_3,
                "hit_at_4": hit_4,
                "reciprocal_rank": round(rr, 3),
                "top_score": round(top_score, 4),
                "scores": [round(s, 4) for s in scores],
                "retrieved_sections": sections,
            }
        )

    summary = {
        "hit_rate_at_1": sum(hit_1_list) / len(hit_1_list),
        "hit_rate_at_3": sum(hit_3_list) / len(hit_3_list),
        "hit_rate_at_4": sum(hit_4_list) / len(hit_4_list),
        "mean_reciprocal_rank": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "avg_top_score": sum(top_scores) / len(top_scores),
    }

    return results, summary


def evaluate_citation_integrity(answer: str, num_sources: int) -> dict[str, Any]:
    """Validate citation presence, index boundaries, and formatting."""
    citations = [int(m) for m in re.findall(r"\[(\d+)\]", answer)]
    has_citations = len(citations) > 0
    valid_citations = [c for c in citations if 1 <= c <= num_sources]
    all_valid = has_citations and (len(valid_citations) == len(citations))
    precision = (len(valid_citations) / len(citations)) if citations else 0.0

    return {
        "has_citations": has_citations,
        "citation_count": len(citations),
        "extracted_indices": citations,
        "all_citations_in_range": all_valid,
        "citation_precision": round(precision, 2),
    }


def judge_faithfulness(context: str, answer: str) -> dict[str, Any]:
    """Evaluate whether the answer is strictly grounded in the retrieved context."""
    system_prompt = (
        "You are an impartial evaluator for a medical RAG system. "
        "Evaluate whether the generated answer is completely faithful to and grounded in the retrieved context. "
        "An answer is faithful if all factual claims can be directly verified from the provided context. "
        "Output strictly a JSON object with keys: 'score' (integer 1-5, 5=fully grounded), "
        "'is_faithful' (boolean), and 'reasoning' (string)."
    )
    user_prompt = f"Retrieved Context:\n{context}\n\nGenerated Answer:\n{answer}"
    return call_groq_judge(system_prompt, user_prompt)


def judge_relevance(question: str, answer: str) -> dict[str, Any]:
    """Evaluate whether the answer directly and accurately answers the question."""
    system_prompt = (
        "You are an impartial evaluator for a question-answering system. "
        "Evaluate how directly, accurately, and clearly the answer addresses the user's question. "
        "Output strictly a JSON object with keys: 'score' (integer 1-5, 5=perfectly relevant), "
        "'is_relevant' (boolean), and 'reasoning' (string)."
    )
    user_prompt = f"User Question:\n{question}\n\nGenerated Answer:\n{answer}"
    return call_groq_judge(system_prompt, user_prompt)


def evaluate_generation_and_judges() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate answers for all test cases and compute citation + LLM judge metrics."""
    case_results = []
    faithfulness_scores = []
    relevance_scores = []
    citation_valid_count = 0

    for i, case in enumerate(EVALUATION_CASES, start=1):
        print(f"  -> Testing Case {i}/{len(EVALUATION_CASES)}: '{case['question'][:40]}...'")
        rag_output = generate_rag_answer(case["question"], limit=RETRIEVAL_LIMIT)
        answer = rag_output["answer"]
        sources = rag_output["sources"]
        context_str = build_context(sources)

        citation_eval = evaluate_citation_integrity(answer, len(sources))
        if citation_eval["all_citations_in_range"]:
            citation_valid_count += 1

        time.sleep(1.0)
        faith_judge = judge_faithfulness(context_str, answer)
        time.sleep(1.0)
        rel_judge = judge_relevance(case["question"], answer)
        time.sleep(1.0)

        if isinstance(faith_judge.get("score"), (int, float)):
            faithfulness_scores.append(float(faith_judge["score"]))
        if isinstance(rel_judge.get("score"), (int, float)):
            relevance_scores.append(float(rel_judge["score"]))

        case_results.append(
            {
                "id": case["id"],
                "question": case["question"],
                "answer_preview": answer[:180] + ("..." if len(answer) > 180 else ""),
                "citation_metrics": citation_eval,
                "faithfulness_judge": faith_judge,
                "relevance_judge": rel_judge,
            }
        )

    summary = {
        "cases_evaluated": len(case_results),
        "citation_validity_rate": citation_valid_count / len(case_results),
        "avg_faithfulness_score": (
            round(sum(faithfulness_scores) / len(faithfulness_scores), 2)
            if faithfulness_scores
            else None
        ),
        "avg_relevance_score": (
            round(sum(relevance_scores) / len(relevance_scores), 2)
            if relevance_scores
            else None
        ),
    }

    return case_results, summary


def evaluate_negative_case() -> dict[str, Any]:
    """Test out-of-domain handling to ensure the system does not hallucinate."""
    rag_output = generate_rag_answer(OUT_OF_DOMAIN_CASE["question"], limit=RETRIEVAL_LIMIT)
    answer = rag_output["answer"]

    system_prompt = (
        "You are evaluating if a medical RAG system properly recognized that the context is insufficient "
        "for an out-of-domain question (Type 2 diabetes asked to an epilepsy document). "
        "The model SHOULD state that the context is insufficient or that it cannot answer from the documents, "
        "and MUST NOT fabricate ungrounded diabetes guidelines as document facts. "
        "Output strictly a JSON object with keys: 'correctly_refused_or_flagged_insufficient' (boolean) "
        "and 'reasoning' (string)."
    )
    user_prompt = f"Question:\n{OUT_OF_DOMAIN_CASE['question']}\n\nGenerated Answer:\n{answer}"
    judge = call_groq_judge(system_prompt, user_prompt)

    return {
        "id": OUT_OF_DOMAIN_CASE["id"],
        "question": OUT_OF_DOMAIN_CASE["question"],
        "answer": answer,
        "judge_assessment": judge,
        "passed": bool(judge.get("correctly_refused_or_flagged_insufficient")),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 60)
    print("🚀 Running Comprehensive RAG Pipeline Evaluation...")
    print("=" * 60)

    # 1. Retrieval Evaluation
    print("\n[1/3] Evaluating Retrieval Performance (BAAI + Qdrant)...")
    retrieval_results, retrieval_summary = evaluate_retrieval()
    print(f"  ✓ Hit@1: {retrieval_summary['hit_rate_at_1']:.0%}")
    print(f"  ✓ Hit@3: {retrieval_summary['hit_rate_at_3']:.0%}")
    print(f"  ✓ Hit@4: {retrieval_summary['hit_rate_at_4']:.0%}")
    print(f"  ✓ MRR (Mean Reciprocal Rank): {retrieval_summary['mean_reciprocal_rank']:.3f}")
    print(f"  ✓ Avg Top-1 Score: {retrieval_summary['avg_top_score']:.4f}")

    # 2. Generation & LLM Judge Evaluation
    print("\n[2/3] Evaluating Generation, Citations & LLM-as-a-Judge (Groq)...")
    generation_results, generation_summary = evaluate_generation_and_judges()
    print(f"  ✓ Citation Validity Rate: {generation_summary['citation_validity_rate']:.0%}")
    print(f"  ✓ Avg Faithfulness Score: {generation_summary['avg_faithfulness_score']}/5.0")
    print(f"  ✓ Avg Relevance Score: {generation_summary['avg_relevance_score']}/5.0")

    # 3. Negative / Out-of-Domain Evaluation
    print("\n[3/3] Evaluating Out-of-Domain Rejection & Refusal...")
    negative_result = evaluate_negative_case()
    print(f"  ✓ Negative Case Passed: {negative_result['passed']}")
    print(f"  ✓ Model Reasoning: {negative_result['judge_assessment'].get('reasoning')}")

    # Compile Complete Report
    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model_config": {
            "embedding_model": "BAAI/bge-base-en-v1.5 (768 dim)",
            "vector_db": "Qdrant (Cosine Similarity)",
            "llm_model": GROQ_MODEL,
            "retrieval_limit": RETRIEVAL_LIMIT,
        },
        "summary": {
            "retrieval": retrieval_summary,
            "generation": generation_summary,
            "out_of_domain_handling": {
                "passed": negative_result["passed"],
            },
        },
        "retrieval_cases": retrieval_results,
        "generation_cases": generation_results,
        "out_of_domain_case": negative_result,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "=" * 60)
    print(f"🎉 Evaluation complete! Full detailed report saved to: {REPORT_PATH.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
