# logger_config.py
import logging
import json
import time

class StructuralJsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage()
        }
        # Add tracking data if available
        if hasattr(record, 'vram_mb'):
            log_entry["vram_mb"] = record.vram_mb
        return json.dumps(log_entry)

def setup_system_logger():
    logger = logging.getLogger("OrgSLM_Pruner")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to prevent double logging
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # File Handler for Desktop App polling
    fh = logging.FileHandler("orchestrator_system.log")
    fh.setFormatter(StructuralJsonFormatter())
    logger.addHandler(fh)
    
    return logger
