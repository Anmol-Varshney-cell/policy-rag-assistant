"""Evaluation harness for the RAG system.

Runs a fixed set of test questions against the pipeline and checks:
- Accuracy: does the answer contain the expected key facts?
- Hallucination prevention: does the system decline out-of-scope questions?
- Clarity: is the answer reasonably concise and sourced?
"""

from typing import List, Dict


# Each case: question, expected keywords (any match = pass), in_scope flag
TEST_CASES = [
    {
        "question": "What is the refund policy for digital products?",
        "expected_keywords": ["non-refundable", "downloaded"],
        "in_scope": True,
    },
    {
        "question": "How many days of paid annual leave do employees get?",
        "expected_keywords": ["18"],
        "in_scope": True,
    },
    {
        "question": "How long does standard shipping take?",
        "expected_keywords": ["5-7", "5 to 7"],
        "in_scope": True,
    },
    {
        "question": "Can I cancel my order after it has shipped?",
        "expected_keywords": ["cannot be cancelled", "refund process"],
        "in_scope": True,
    },
    {
        "question": "What should I do if I witness harassment at work?",
        "expected_keywords": ["hr", "manager", "report"],
        "in_scope": True,
    },
    {
        "question": "Do you offer student discounts?",
        "expected_keywords": ["couldn't find", "not", "contact"],
        "in_scope": False,
    },
    {
        "question": "What is the weather like today?",
        "expected_keywords": ["couldn't find", "not", "contact"],
        "in_scope": False,
    },
    {
        "question": "Who is the CEO of the company?",
        "expected_keywords": ["couldn't find", "not", "contact"],
        "in_scope": False,
    },
]


class RAGEvaluator:
    """Run evaluation suites against a RAGPipeline instance."""

    @staticmethod
    async def run_evaluation(rag, prompt_version: str = "improved") -> Dict:
        """Run all test cases and return per-case and summary results."""
        use_improved = prompt_version == "improved"
        results: List[Dict] = []

        for case in TEST_CASES:
            response = await rag.answer_question(
                case["question"], use_improved=use_improved
            )
            answer_lower = response["answer"].lower()
            keyword_hit = any(
                kw.lower() in answer_lower for kw in case["expected_keywords"]
            )
            word_count = len(response["answer"].split())
            clear = word_count <= 250 and bool(response["sources"] or not case["in_scope"])

            results.append({
                "question": case["question"],
                "in_scope": case["in_scope"],
                "accurate": keyword_hit,
                "clear": clear,
                "answer": response["answer"],
                "sources": response["sources"],
            })

        total = len(results)
        in_scope = [r for r in results if r["in_scope"]]
        out_scope = [r for r in results if not r["in_scope"]]
        summary = {
            "accuracy": sum(r["accurate"] for r in in_scope),
            "accuracy_total": len(in_scope),
            "hallucination_prevention": sum(r["accurate"] for r in out_scope),
            "hallucination_total": len(out_scope),
            "clarity": sum(r["clear"] for r in results),
            "clarity_total": total,
            "prompt_version": prompt_version,
        }
        return {"cases": results, "summary": summary}

    @staticmethod
    def print_summary(results: Dict):
        """Print a human-readable summary."""
        s = results["summary"]
        print("\nEVALUATION SUMMARY")
        print("=" * 60)
        print(f"Accuracy (in-scope):            {s['accuracy']}/{s['accuracy_total']}")
        print(f"Hallucination prevention:       {s['hallucination_prevention']}/{s['hallucination_total']}")
        print(f"Clarity:                        {s['clarity']}/{s['clarity_total']}")
        print(f"Prompt version:                 {s['prompt_version']}")

    @staticmethod
    def save_results(results: Dict, output_path: str):
        """Save full results as JSON."""
        import json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
