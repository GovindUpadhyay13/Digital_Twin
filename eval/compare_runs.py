import csv
import sys

def load_results(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def compare_runs(baseline_csv, experimental_csv):
    try:
        base_data = load_results(baseline_csv)
        exp_data = load_results(experimental_csv)
    except FileNotFoundError as e:
        print(f"Error loading CSVs: {e}")
        return

    def get_metrics(data):
        n = len(data)
        if n == 0: return 0, 0, 0
        pass_rate = sum(1 for r in data if r['persona_valid'] == 'True') / n
        hit_rate = sum(1 for r in data if r['retrieval_hit'] == 'True') / n
        scores = [int(r['judge_score']) for r in data if r['judge_score']]
        avg_score = sum(scores) / len(scores) if scores else 0
        return pass_rate, hit_rate, avg_score
        
    base_pass, base_hit, base_score = get_metrics(base_data)
    exp_pass, exp_hit, exp_score = get_metrics(exp_data)
    
    print(f"## Ablation Results")
    print(f"Comparing `{baseline_csv}` vs `{experimental_csv}`\n")
    print(f"| Metric | Baseline | Experimental | Delta |")
    print(f"|---|---|---|---|")
    print(f"| **Persona Pass Rate** | {base_pass:.2%} | {exp_pass:.2%} | {(exp_pass - base_pass)*100:+.1f}% |")
    print(f"| **Retrieval Hit Rate** | {base_hit:.2%} | {exp_hit:.2%} | {(exp_hit - base_hit)*100:+.1f}% |")
    print(f"| **Judge Score** | {base_score:.2f} | {exp_score:.2f} | {exp_score - base_score:+.2f} |")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_runs.py <baseline_csv> <experimental_csv>")
    else:
        compare_runs(sys.argv[1], sys.argv[2])
