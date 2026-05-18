import os
import json
import time
import requests

from ingest import load_index

# ── Ollama config ─────────────────────────────────────────────────────────────
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama:latest")

# ── Load search index at startup ──────────────────────────────────────────────
index = load_index()

# ── Boosted search ────────────────────────────────────────────────────────────
def search(query):
    boost = {
        "exercise_name":        2.11,
        "type_of_activity":     1.46,
        "type_of_equipment":    0.65,
        "body_part":            2.65,
        "type":                 1.31,
        "muscle_groups_activated": 2.54,
        "instructions":         0.74,
    }
    return index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=5,
    )

# ── Prompt templates ──────────────────────────────────────────────────────────
prompt_template = """
You are a knowledgeable fitness assistant. Answer the QUESTION using ONLY the CONTEXT below.
Be concise and accurate. If the context doesn't contain enough information, say so honestly.

CONTEXT:
{context}

QUESTION: {question}
""".strip()

entry_template = """
exercise_name: {exercise_name}
type_of_activity: {type_of_activity}
type_of_equipment: {type_of_equipment}
body_part: {body_part}
type: {type}
muscle_groups_activated: {muscle_groups_activated}
instructions: {instructions}
""".strip()

evaluation_prompt_template = """
You are a strict RAG evaluator. Evaluate whether the GENERATED ANSWER correctly answers the QUESTION.
Return ONLY valid JSON with no markdown, no explanation outside the JSON.

QUESTION: {question}
GENERATED ANSWER: {answer_llm}

Return exactly this structure:
{{
  "Relevance": "RELEVANT" | "PARTLY_RELEVANT" | "NON_RELEVANT",
  "Explanation": "one short sentence"
}}
""".strip()


def build_prompt(query, search_results):
    context = "\n\n".join(entry_template.format(**doc) for doc in search_results)
    return prompt_template.format(question=query, context=context)


# ── Ollama LLM call ───────────────────────────────────────────────────────────
def llm(prompt, model=OLLAMA_MODEL):
    """Call local Ollama and return (response_text, token_stats)."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "").strip()

        # Ollama returns token counts in the response
        tokens = {
            "prompt_tokens":     data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens":      data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        }
        return text, tokens

    except requests.exceptions.ConnectionError:
        msg = "Cannot reach Ollama service. Is the container running?"
        return msg, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    except Exception as e:
        msg = f"LLM error: {e}"
        return msg, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def evaluate_relevance(question, answer):
    prompt = evaluation_prompt_template.format(question=question, answer_llm=answer)
    # Keyword-based relevance for small local models
    eval_tokens = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    if not answer or 'LLM error' in answer:
        return 'UNKNOWN', 'No answer generated', eval_tokens
    q_words = set(question.lower().split())
    a_words = set(answer.lower().split())
    overlap = len(q_words & a_words)
    if overlap >= 2:
        return 'RELEVANT', 'Keyword overlap', eval_tokens
    elif overlap == 1:
        return 'PARTLY_RELEVANT', 'Partial overlap', eval_tokens
    else:
        return 'NON_RELEVANT', 'No overlap', eval_tokens


# ── Main RAG pipeline ─────────────────────────────────────────────────────────
def rag(query, model=OLLAMA_MODEL):
    t0 = time.time()

    search_results = search(query)
    prompt         = build_prompt(query, search_results)
    answer, tokens = llm(prompt, model=model)

    relevance, explanation, eval_tokens = evaluate_relevance(query, answer)

    t1 = time.time()

    return {
        "answer":                  answer,
        "model_used":              model,
        "response_time":           t1 - t0,
        "relevance":               relevance,
        "relevance_explanation":   explanation,
        # answer tokens
        "prompt_tokens":           tokens["prompt_tokens"],
        "completion_tokens":       tokens["completion_tokens"],
        "total_tokens":            tokens["total_tokens"],
        # eval tokens
        "eval_prompt_tokens":      eval_tokens["prompt_tokens"],
        "eval_completion_tokens":  eval_tokens["completion_tokens"],
        "eval_total_tokens":       eval_tokens["total_tokens"],
        # Ollama is free — cost is always 0
        "openai_cost":             0.0,
    }
