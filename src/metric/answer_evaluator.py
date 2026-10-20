"""Terminal task utility ``U(tau)`` (paper Section 3.3).

The final answer of an episode is compared with the ground truth either by normalised exact
match (``method="exact"``) or by an LLM judge that checks semantic equivalence
(``method="llm"``). The resulting binary reward is the sparse, verifiable training signal.
"""

import json
import re
from typing import Any, Dict, List

from src.logger import logger
from src.metric.exact_match import question_scorer
from src.models import ChatMessage, MessageRole, model_manager

LLM_JUDGE_PROMPT = '''You will be given a question and its ground truth answer list where each item can be a ground truth answer. Provided a pred_answer, you need to judge if the pred_answer correctly answers the question based on the ground truth answer list.
You should first give your rationale for the judgement, and then give your judgement result (i.e., correct or incorrect).

Here is the criteria for the judgement:
1. The pred_answer doesn't need to be exactly the same as any of the ground truth answers, but should be semantically same for the question.
2. Each item in the ground truth answer list can be viewed as a ground truth answer for the question, and the pred_answer should be semantically same to at least one of them.

question: {question}
ground truth answers: {gt_answer}
pred_answer: {pred_answer}

The output should in the following json format:

{{
  "rationale": "...",
  "judgement": "correct" | "incorrect"
}}
'''


def _ensure_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(item) for item in x]
    return [str(x)]


def _parse_llm_judgement(content: str) -> Dict[str, str]:
    """Parse the judge's JSON answer (tolerates Markdown code fences)."""
    content = re.sub(r"^```(?:json)?\s*", "", content.strip())
    content = re.sub(r"\s*```$", "", content)
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM judgement: {e}. Content: {content[:200]}")
        return {"judgement": "incorrect", "rationale": f"Unparsable judgement: {content[:200]}"}
    judgement = str(data.get("judgement", "incorrect")).lower().strip()
    return {
        "judgement": judgement if judgement in ("correct", "incorrect") else "incorrect",
        "rationale": str(data.get("rationale", "")),
    }


async def _llm_judge(question: str, ground_truth: Any, pred_answer: str, model_id: str) -> Dict[str, str]:
    model = model_manager.registed_models.get(model_id)
    if model is None:
        raise ValueError(f"Judge model '{model_id}' is not registered.")
    prompt = LLM_JUDGE_PROMPT.format(
        question=question,
        gt_answer=json.dumps(_ensure_list(ground_truth), ensure_ascii=False),
        pred_answer=pred_answer,
    )
    response = await model(messages=[ChatMessage.from_dict({"role": MessageRole.USER, "content": prompt})])
    return _parse_llm_judgement(response.content or "")


async def evaluate_answer(
    question: str,
    pred_answer: Any,
    ground_truth: Any,
    method: str = "llm",
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """Score a final answer against the ground truth.

    Args:
        question: Task question.
        pred_answer: Final answer produced by EvoMAS.
        ground_truth: Reference answer (string or list of acceptable answers).
        method: ``"exact"`` (normalised exact match) or ``"llm"`` (semantic LLM judge).
        model: Registered model id used by the LLM judge.

    Returns:
        ``{"reward": 0 | 1, "is_correct": bool, "rationale": str, "method": str}``
    """
    pred = "" if pred_answer is None else str(pred_answer)
    method = method.lower().strip()
    try:
        if method == "exact":
            is_correct = any(question_scorer(pred, gt) for gt in _ensure_list(ground_truth))
            rationale = "Normalised exact match"
        elif method == "llm":
            judgement = await _llm_judge(question, ground_truth, pred, model)
            is_correct = judgement["judgement"] == "correct"
            rationale = judgement["rationale"]
        else:
            raise ValueError(f"Unknown evaluation method: {method!r} (use 'exact' or 'llm').")
    except Exception as e:
        logger.warning(f"Answer evaluation failed: {e}")
        is_correct, rationale = False, f"Evaluation error: {e}"

    return {"reward": int(is_correct), "is_correct": is_correct, "rationale": rationale, "method": method}
