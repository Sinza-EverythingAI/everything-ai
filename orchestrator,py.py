import os
from dataclasses import dataclass
from typing import List, Dict

# Optional providers — install only what you need:
# pip install openai google-generativeai anthropic
try:
    import openai
except Exception:
    openai = None

try:
    import google.generativeai as genai
except Exception:
    genai = None


@dataclass
class ModelAnswer:
    name: str
    text: str
    meta: Dict


# ---------- Local stub models (fallbacks) ----------
def call_model_structure_stub(question: str, mode: str) -> ModelAnswer:
    text = (
        f"[Structured Core · {mode}] I’ll break this down clearly:\n"
        f"- What it is\n- Why it matters\n- How it works\n- Examples\n- Caveats"
    )
    return ModelAnswer("StructuredCoreStub", text, {"type": "stub", "role": "structure"})


def call_model_facts_stub(question: str, mode: str) -> ModelAnswer:
    text = (
        f"[Factual Lens · {mode}] Definitions, key terms, and neutral background to ground the answer."
    )
    return ModelAnswer("FactsStub", text, {"type": "stub", "role": "facts"})


# ---------- OpenAI GPT ----------
def call_model_openai(question: str, mode: str) -> ModelAnswer:
    if openai is None:
        return ModelAnswer("OpenAI-GPT", "OpenAI SDK not installed.", {"type": "llm", "role": "primary reasoning"})

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ModelAnswer("OpenAI-GPT", "Missing OPENAI_API_KEY.", {"type": "llm", "role": "primary reasoning"})

    openai.api_key = api_key

    prompt = (
        f"You are a precise, helpful assistant.\nMode: {mode}\nQuestion: {question}\n"
        "Return a clear, accurate answer with concise structure."
    )

    # Using Chat Completions for broad compatibility
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = resp["choices"][0]["message"]["content"]
    except Exception as e:
        text = f"OpenAI error: {e}"

    return ModelAnswer("OpenAI-GPT", text, {"type": "llm", "role": "primary reasoning"})


# ---------- Google Gemini ----------
def call_model_gemini(question: str, mode: str) -> ModelAnswer:
    if genai is None:
        return ModelAnswer("Gemini", "Gemini SDK not installed.", {"type": "llm", "role": "breadth & creativity"})

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return ModelAnswer("Gemini", "Missing GEMINI_API_KEY.", {"type": "llm", "role": "breadth & creativity"})

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"Mode: {mode}\nQuestion: {question}\n"
            "Provide a clear answer with practical examples and multiple perspectives."
        )
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "") or "No text returned from Gemini."
    except Exception as e:
        text = f"Gemini error: {e}"

    return ModelAnswer("Gemini-1.5-Flash", text, {"type": "llm", "role": "breadth & creativity"})


# ---------- Scoring & synthesis ----------
def score_answer(answer: ModelAnswer) -> float:
    text = (answer.text or "").lower()
    length_score = min(len(text) / 600, 1.0)
    signals = ["because", "therefore", "for example", "in summary", "first", "second"]
    signal_score = min(sum(1 for s in signals if s in text) / 4, 1.0)
    return round(0.55 * length_score + 0.45 * signal_score, 3)


def synthesize(question: str, answers: List[ModelAnswer], mode: str) -> str:
    if not answers:
        return "No answers were generated."

    scored = [(score_answer(a), a) for a in answers]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best = scored[0]
    others = [a for _, a in scored[1:]]

    out = []
    out.append("Unified answer:\n")
    out.append(best.text.strip() + "\n")

    if others:
        out.append("\nCross‑checks and additional perspectives:\n")
        for a in others:
            out.append(f"- {a.name}: {a.text.strip()}\n")

    if mode == "fast":
        out.append("\nNote: Fast mode favors brevity over exhaustive detail.")
    elif mode == "factual":
        out.append("\nNote: Factual mode emphasizes grounded, neutral statements.")
    elif mode == "creative":
        out.append("\nNote: Creative mode uses analogies and broader connections.")

    return "\n".join(out)


def orchestrate_question(question: str, mode: str = "balanced") -> Dict:
    # Choose which models to call (you can add/remove freely)
    raw: List[ModelAnswer] = [
        call_model_openai(question, mode),
        call_model_gemini(question, mode),
        call_model_structure_stub(question, mode),
        call_model_facts_stub(question, mode),
    ]

    final_text = synthesize(question, raw, mode)

    return {
        "question": question,
        "final_answer": final_text,
        "models": [
            {"name": a.name, "text": a.text, "meta": a.meta, "score": score_answer(a)}
            for a in raw
        ],
    }