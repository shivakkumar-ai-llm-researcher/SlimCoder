import json
import os

def main():
    results_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    if not os.path.exists(results_path):
        print("No evaluation results found. Run evaluate.py first.")
        return
        
    with open(results_path, "r") as f:
        results = json.load(f)
        
    total_tasks = len(results)
    if total_tasks == 0:
        return
        
    routing_passes = sum(1 for r in results if r["routing_passed"])
    ast_passes = sum(1 for r in results if r["ast_passed"])
    avg_keyword_score = sum(r["keyword_score"] for r in results) / total_tasks
    avg_latency = sum(r["latency_sec"] for r in results) / total_tasks
    
    report = f"""# SLM Harness Evaluation Report

## Executive Summary
- **Total Tasks Evaluated**: {total_tasks}
- **Average Latency**: {avg_latency:.2f} seconds
- **Orchestrator Routing Accuracy**: {(routing_passes/total_tasks)*100:.0f}%
- **Code Validity (AST Pass Rate)**: {(ast_passes/total_tasks)*100:.0f}%
- **Keyword Match Accuracy**: {avg_keyword_score*100:.0f}%

## Task Breakdown

| Task ID | Latency | Routing | AST Check | Keyword Score | Code Length |
|---------|---------|---------|-----------|---------------|-------------|
"""

    for r in results:
        routing_icon = "✅" if r['routing_passed'] else "❌"
        ast_icon = "✅" if r['ast_passed'] else "❌"
        kw_score_str = f"{r['keyword_score']*100:.0f}%"
        
        report += f"| {r['id']} | {r['latency_sec']:.2f}s | {routing_icon} | {ast_icon} | {kw_score_str} | {r['generated_code_length']} chars |\n"

    out_path = os.path.join(os.path.dirname(__file__), "evaluation_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"Report generated successfully at {out_path}")

if __name__ == "__main__":
    main()
