import json
import os
from typing import Dict, Any

class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return config
    
    def get_yolo_config(self) -> Dict[str, Any]:
        """获取YOLO配置"""
        return self.config.get("yolo", {})
    
    def get_vlm_config(self) -> Dict[str, Any]:
        """获取VLM配置"""
        return self.config.get("vlm", {})
    
    def get_flask_config(self) -> Dict[str, Any]:
        """获取Flask配置"""
        return self.config.get("flask", {})
    
    def get_prompt(self) -> str:
        """获取提示词"""
        return self.config.get("prompt", "")
    
    def get(self, key: str, default=None):
        """获取配置项"""
        return self.config.get(key, default)
    
    def reload(self):
        """重新加载配置"""
        self.config = self._load_config()

# 全局配置实例
config = ConfigLoader() 