import os
import json
import csv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.orchestrator import KarpathyTwinOrchestrator
from agent.persona_validator import PersonaValidator
from eval.llm_judge import judge_response

def run_evaluation(golden_path="eval/golden_qa.jsonl", output_csv="eval/results_baseline.csv"):
    os.makedirs("eval", exist_ok=True)
    validator = PersonaValidator()
    orch = KarpathyTwinOrchestrator(session_id="eval_session_v1")
    
    rows = []
    if not os.path.exists(golden_path):
        print(f"Golden dataset not found at {golden_path}")
        return
        
    with open(golden_path, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    print(f"Starting evaluation of {len(examples)} examples...")
    for ex in examples:
        print(f"\nEvaluating Q: {ex['question']}")
        
        result = orch.chat(user_message=ex["question"], thread_id=f"eval_{ex['id']}")
        response = result.get("response", "")
        
        validation = validator.validate(response)
        
        retrieved_ids = [s.get("title") for s in result.get("sources", [])]
        retrieval_hit = ex.get("source_doc_id") in retrieved_ids
        
        judge_score = judge_response(ex["question"], response, ex.get("expected_topics", []))
        
        row = {
            "id": ex["id"], 
            "category": ex.get("category", "unknown"),
            "persona_valid": validation["is_valid"],
            "severity": validation["severity"],
            "retrieval_hit": retrieval_hit,
            "judge_score": judge_score if judge_score is not None else "",
        }
        rows.append(row)
        print(f"Result: Persona={row['persona_valid']}, Hit={row['retrieval_hit']}, Score={row['judge_score']}")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    pass_rate = sum(r['persona_valid'] for r in rows) / n if n else 0
    hit_rate = sum(r['retrieval_hit'] for r in rows) / n if n else 0
    valid_scores = [r['judge_score'] for r in rows if r['judge_score']]
    avg_judge = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    
    print("\n--- EVALUATION SUMMARY ---")
    print(f"Persona pass rate: {pass_rate:.2%}")
    print(f"Retrieval hit@5: {hit_rate:.2%}")
    print(f"Avg judge score: {avg_judge:.2f}/5")

if __name__ == "__main__":
    run_evaluation()
