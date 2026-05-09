import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path("data/document_registry.json")

class MetadataRegistry:
    def __init__(self):
        self.registry_path = REGISTRY_PATH
        self._ensure_file()

    def _ensure_file(self):
        if not self.registry_path.parent.exists():
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            with open(self.registry_path, "w") as f:
                json.dump({}, f)

    def _load(self):
        try:
            with open(self.registry_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self, data):
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_profile(self, hash_id: str):
        return self._load().get(hash_id)

    def save_profile(self, hash_id: str, profile: dict):
        data = self._load()
        data[hash_id] = profile
        self._save(data)

    def get_all_profiles(self):
        return self._load()
