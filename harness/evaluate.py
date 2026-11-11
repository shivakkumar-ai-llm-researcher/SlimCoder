import os
import sys
import json
import time
import ast

# Add parent directory to path so we can import pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import OrgSLMPipeline

def evaluate_ast(code: str, domain: str) -> bool:
    if "python" not in domain:
        return True # Skip non-python for AST checks
        
    clean_code = code
    if clean_code.startswith("```"):
        lines = clean_code.split('\n')
        if len(lines) > 2:
            clean_code = '\n'.join(lines[1:-1])
            
    try:
        ast.parse(clean_code)
        return True
    except SyntaxError:
        return False

def evaluate_keywords(code: str, keywords: list) -> float:
    if not keywords:
        return 1.0
    matched = sum(1 for kw in keywords if kw in code)
    return matched / len(keywords)

def main():
    print("Initializing SLM Evaluation Harness...")
    pipeline = None
    try:
        pipeline = OrgSLMPipeline("checkpoints/base_pretrained.pt")
    except Exception as e:
        print(f"Failed to load pipeline. Ensure weights are present. Error: {e}")
        return

    benchmarks_path = os.path.join(os.path.dirname(__file__), "benchmarks.json")
    with open(benchmarks_path, "r") as f:
        benchmarks = json.load(f)

    results = []
    print(f"\nLoaded {len(benchmarks)} benchmark tasks.")
    
    for idx, b in enumerate(benchmarks):
        print(f"\n--- Running Task {idx+1}/{len(benchmarks)}: {b['id']} ---")
        prompt = b['prompt']
        
        start_time = time.time()
        result_payload = pipeline.run(prompt, verbose=False)
        latency = time.time() - start_time
        
        domains_used = result_payload.get("domains_used", [])
        merged_code = result_payload.get("merged_output", {}).get("code", "")
        
        routing_passed = b['expected_domain'] in domains_used
        ast_passed = evaluate_ast(merged_code, b['expected_domain'])
        kw_score = evaluate_keywords(merged_code, b['required_keywords'])
        
        print(f"Latency: {latency:.2f}s")
        print(f"Routing Correct: {routing_passed} (Expected: {b['expected_domain']}, Got: {domains_used})")
        print(f"AST Valid: {ast_passed}")
        print(f"Keyword Score: {kw_score*100:.0f}%")
        
        results.append({
            "id": b['id'],
            "prompt": prompt,
            "latency_sec": latency,
            "routing_passed": routing_passed,
            "ast_passed": ast_passed,
            "keyword_score": kw_score,
            "generated_code_length": len(merged_code)
        })

    out_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Done] Raw evaluation results saved to {out_path}")

if __name__ == "__main__":
    main()
