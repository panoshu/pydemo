from functools import lru_cache

from demo.config.base import PROJECT_ROOT
from demo.config.loader import AppSettings


@lru_cache()
def get_settings() -> AppSettings:
    """
    全局配置获取函数。
    使用 lru_cache 确保整个应用生命周期内只读取一次文件并实例化一次。
    延迟加载：只有在业务代码第一次调用时，才会触发文件的读取和校验。
    """
    return AppSettings.load()


# 导出获取函数和类
__all__ = ["AppSettings", "get_settings", "PROJECT_ROOT"]
