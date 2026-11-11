# generate_datasets.py
import json
import os
from pipeline import CloudOrchestrator
from logger_config import setup_system_logger

logger = setup_system_logger()

def build_training_data():
    kimi = CloudOrchestrator()
    logger.info("DATASET: Launching dataset building sequence using Kimi-K2.6...")
    
    # Define topics to generate training pairs for
    domains = {
        "frontend": "Write 5 complex instruction-response pairs for creating scalable cross-platform components in React Native CLI with Material 3.",
        "backend": "Write 5 complex instruction-response pairs for highly efficient FastAPI endpoints using PyDantic V2 and PostgreSQL transactions."
    }
    
    for domain, prompt in domains.items():
        logger.info(f"PROMPTING: Requesting structured synthetic samples for '{domain}' from NVIDIA Cloud...")
        raw_output = kimi.call_kimi_orchestrator(
            f"{prompt} Return the result strictly as a valid minified JSON array containing objects with 'instruction' and 'output' fields. Do not write any markdown code blocks wrapping it."
        )
        
        try:
            clean_output = raw_output.strip()
            if clean_output.startswith("```json"):
                clean_output = clean_output[7:]
            elif clean_output.startswith("```"):
                clean_output = clean_output[3:]
            if clean_output.endswith("```"):
                clean_output = clean_output[:-3]
            clean_output = clean_output.strip()
            
            if clean_output.startswith("Fallback Triggered"):
                logger.error(f"FAILED: API Error - {clean_output}")
                continue
                
            try:
                parsed_data = json.loads(clean_output)
            except Exception as e:
                # Save raw output to file for debugging
                with open(f"{domain}_debug.txt", "w", encoding="utf-8") as f:
                    f.write(raw_output)
                raise e
                
            os.makedirs("data/train", exist_ok=True)
            with open(f"data/train/{domain}.json", "w") as f:
                json.dump(parsed_data, f, indent=4)
            logger.info(f"SUCCESS: Successfully saved data/train/{domain}.json data blocks!")
        except Exception as e:
            logger.error(f"FAILED: Syntax validation failed for synthesized JSON framework blocks. Error: {str(e)}")

if __name__ == "__main__":
    build_training_data()
