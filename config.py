import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """从YAML文件加载配置。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def merge_config_args(config: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    """合并配置和命令行参数，命令行参数优先。"""
    merged = config.copy() if config else {}
    for k, v in args.items():
        if v is not None:
            merged[k] = v
    return merged 