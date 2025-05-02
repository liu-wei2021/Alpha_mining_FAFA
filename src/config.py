import os
import yaml
from types import SimpleNamespace
from pathlib import Path

class Settings:
    def __init__(self, path: str = "configs/config.yaml"):
        # Load YAML configuration
        cfg_path = Path(path)
        cfg = yaml.safe_load(cfg_path.read_text())

        # Data settings
        self.input_parquet = cfg["data"]["input_parquet"]
        # Optional outputs directory for all JSON/CSV outputs
        self.outputs_dir = cfg["data"].get("outputs_dir", "data/outputs_dir")

        # Pipeline settings
        pipeline_cfg = cfg.get("pipeline", {})
        self.n_clusters     = pipeline_cfg.get("n_clusters")
        self.random_state   = pipeline_cfg.get("random_state")
        # Any extra pipeline entries (like function definitions for LLM prompts)
        self.pipeline = SimpleNamespace(
            function_definition = pipeline_cfg.get("function_definition", "")
        )
        

        # DeepSeek / LLM settings
        deep_cfg = cfg.get("deepseek", {})
        self.deepseek = SimpleNamespace(
            api_key     = os.getenv(
                "DEEPSEEK_API_KEY", 
                deep_cfg.get("api_key", "")
            ),
            api_base    = deep_cfg.get("api_base", ""),
            model       = deep_cfg.get("model", ""),
            temperature = deep_cfg.get("temperature", 0)
        )

        # Store path for reference
        self._path = str(cfg_path)

# Singleton instance
settings = Settings()
