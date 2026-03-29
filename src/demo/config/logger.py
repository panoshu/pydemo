import logging
import logging.config

from demo.config import PROJECT_ROOT, get_settings


def setup_logging():
    """
    企业级全局日志配置初始化 (完美压制 Uvicorn 覆盖问题)
    """
    config = get_settings()
    log_file_path = PROJECT_ROOT / config.log.file
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # 确保日志级别字符串是大写 (如 "DEBUG", "INFO")
    log_level = str(config.log.level).upper()

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": config.log.format,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "level": log_level,  # Handler 也要加上 Level
                "formatter": "standard",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "level": log_level,  # Handler 也要加上 Level
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "standard",
                "filename": str(log_file_path),
                "maxBytes": config.log.max_bytes,
                "backupCount": config.log.backup_count,
                "encoding": "utf8",
            },
        },
        "loggers": {
            # 1. 业务日志：精确捕获以 demo 开头的所有模块
            "demo": {
                "handlers": ["console", "file"],
                "level": log_level,
                "propagate": False,
            },
            # 2. 强制接管 Uvicorn 日志，使其和业务日志格式/输出统一
            "uvicorn": {
                "handlers": ["console", "file"],
                "level": "INFO",  # 框架启动日志 INFO 即可，避免太吵
                "propagate": False,
            },
            "uvicorn.error": {
                "level": "INFO",
                "propagate": True,  # 向上抛给 uvicorn 处理
            },
            "uvicorn.access": {
                "handlers": ["console", "file"],
                "level": "INFO",  # HTTP 请求日志
                "propagate": False,
            },
        },
        "root": {
            # 3. Root 兜底：必须是 WARNING 或 INFO，千万别写 DEBUG！
            # 否则 httpx, milvus 等第三方包的底层通讯报文会刷爆你的控制台
            "handlers": ["console", "file"],
            "level": "WARNING",
        },
    }

    # 1. 应用字典配置
    logging.config.dictConfig(logging_config)

    # 2. 🌟 核心杀招：强行拔掉 Uvicorn 默认自带的 Handlers！
    # 因为 Uvicorn 启动时可能会再次强插它自己的 Handler，导致出现双重打印
    for logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(logger_name)
        # 清除多余的 handler，强制走我们 dictConfig 里配置的逻辑
        uvicorn_logger.handlers.clear()

    # 3. 初始化完成提示 (强行指定一个 demo 命名空间的 logger 来打印)
    startup_logger = logging.getLogger("demo.system")
    startup_logger.info(
        f"✅ 全局日志系统挂载成功 | 级别: {log_level} | 路径: {log_file_path}"
    )
