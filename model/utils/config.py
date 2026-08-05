#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置文件管理
"""

import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """配置管理类"""

    def __init__(self, config_path: str):
        """
        初始化配置

        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项

        Args:
            key: 配置键，支持点号分隔，如"data.image_size"
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def update(self, key: str, value: Any):
        """
        更新配置项

        Args:
            key: 配置键
            value: 新值
        """
        keys = key.split('.')
        config = self._config

        # 遍历到倒数第二个键
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # 设置最后一个键的值
        config[keys[-1]] = value

    def save(self, save_path: str = None):
        """
        保存配置到文件

        Args:
            save_path: 保存路径，默认为原路径
        """
        if save_path is None:
            save_path = self.config_path

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)

    def __getattr__(self, name: str) -> Any:
        """通过属性访问配置"""
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return ConfigDict(value)
            return value
        raise AttributeError(f"配置中无属性: {name}")


class ConfigDict:
    """配置字典包装类，支持点号访问"""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return ConfigDict(value)
            return value
        raise AttributeError(f"配置字典中无属性: {name}")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        """安全获取值，支持点号分隔的嵌套键，如 'generator.latent_dim'"""
        keys = key.split('.')
        value = self._data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value