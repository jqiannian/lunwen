"""
日志配置模块

统一管理项目日志，支持文件和控制台输出。

设计依据：
- README.md §7（Python代码规范）
- 禁止使用print，统一使用logger
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import structlog


def setup_logger(
    name: str = "traffic_rules",
    log_dir: Optional[Path] = None,
    level: str = "INFO",
    enable_file: bool = True,
    enable_console: bool = True,
) -> logging.Logger:
    """
    配置并返回logger实例
    
    Args:
        name: logger名称
        log_dir: 日志文件目录（默认：项目根目录/logs）
        level: 日志级别（DEBUG/INFO/WARNING/ERROR）
        enable_file: 是否启用文件日志
        enable_console: 是否启用控制台日志
    
    Returns:
        logger: 配置好的logger实例
    
    Example:
        >>> logger = setup_logger("data_loader")
        >>> logger.info("数据加载完成", num_samples=100)
    """
    # 确定日志目录
    if log_dir is None:
        # 默认：项目根目录/logs
        project_root = Path(__file__).parent.parent.parent.parent
        log_dir = project_root / "logs"
    else:
        log_dir = Path(log_dir)
    
    # 创建日志目录
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if not enable_console else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # 获取标准库logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除已有handlers（避免重复）
    logger.handlers.clear()
    
    # 文件handler
    if enable_file:
        log_file = log_dir / f"{name}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, level.upper()))
        
        # 文件格式：JSON（便于解析）
        file_formatter = logging.Formatter(
            fmt='{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # 控制台handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        
        # 控制台格式：人类可读
        console_formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = "traffic_rules") -> structlog.BoundLogger:
    """
    获取structlog包装的logger（推荐使用）
    
    Args:
        name: logger名称（通常使用模块名）
    
    Returns:
        logger: structlog.BoundLogger实例
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("处理完成", scene_id="scene_001", num_entities=5)
    """
    # 如果logger未配置，使用默认配置
    if not logging.getLogger(name).handlers:
        setup_logger(name)
    
    return structlog.get_logger(name)





