import logging
import logging.config
from pathlib import Path

from demo.config import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def setup_logging():
    """
    企业级全局日志配置初始化
    """
    config = get_settings()
    log_file_path = PROJECT_ROOT / config.log.file
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    logging_config = {
        "version": 1,
        # ⚠️ 关键：千万不要禁用已存在的 Logger，否则 Uvicorn/LlamaIndex 日志会丢失
        "disable_existing_loggers": False,
        "formatters": {
            # 标准的控制台与文件输出格式
            "standard": {
                "format": config.log.format,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            # 控制台输出
            "console": {
                "formatter": "standard",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            # 文件输出：使用大小轮转（最大 10MB，保留 5 个备份）
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "standard",
                "filename": str(log_file_path),
                "maxBytes": config.log.max_bytes,
                "backupCount": config.log.backup_count,
                "encoding": "utf8",
            },
        },
        "loggers": {
            # 1. 捕获业务代码的 logger (模块名都是以 demo. 开头)
            "demo": {
                "handlers": ["console", "file"],
                "level": config.log.level,
                "propagate": False,  # 防止向上传递给 root 重复打印
            },
            # 2. 捕获 FastAPI/Uvicorn 的框架和请求日志，统统吸入咱们的日志文件中
            "uvicorn": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
        },
        "root": {
            # 3. 其他所有第三方包（如 httpx, sqlalchemy 等）走 Root，默认设为 WARNING 防止刷屏
            "handlers": ["console", "file"],
            "level": "WARNING",
        },
    }

    # 应用配置
    logging.config.dictConfig(logging_config)

    # 初始化完成后测试打印一条
    logger = logging.getLogger("demo.logger")
    logger.debug(f"日志系统初始化完成，输出至: {log_file_path}")
