import os
import yaml
from types import SimpleNamespace

class Settings:
    def __init__(self, path: str = "configs/config.yaml"):
        # Load YAML configuration
        with open(path) as f:
            cfg = yaml.safe_load(f)

        # Data settings
        self.input_path    = cfg["data"]["input_parquet"]

        # Pipeline settings
        self.n_clusters    = cfg["pipeline"]["n_clusters"]
        self.random_state  = cfg["pipeline"]["random_state"]

        # Deepseek settings grouped in a namespace
        self.deepseek = SimpleNamespace(
            api_key     = os.getenv(
                "DEEPSEEK_API_KEY",
                cfg.get("deepseek", {}).get("api_key", "")
            ),
            api_base    = cfg["deepseek"]["api_base"],
            model       = cfg["deepseek"]["model"],
            temperature = cfg["deepseek"]["temperature"]
        )

        # Keep track of config file path
        self._path = path

# Instantiate a singleton
settings = Settings()
