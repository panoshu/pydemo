from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from .models import (
    ApiConfig,
    DatabaseConfig,
    FeatureConfig,
    LlmConfig,
    LogConfig,
    MilvusConfig,
    RagConfig,
)


class BootstrapConfig(BaseSettings):
    """仅用于在启动时探测当前环境"""

    app_env: str = "dev"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class AppSettings(BaseSettings):
    """主配置加载器：组装所有模块"""

    app_env: str = "dev"

    db: DatabaseConfig
    api: ApiConfig
    llm: LlmConfig
    log: LogConfig
    rag: RagConfig
    milvus: MilvusConfig
    feature: FeatureConfig

    model_config = SettingsConfigDict(env_nested_delimiter="__", extra="ignore")

    @classmethod
    def load(cls) -> "AppSettings":
        kwargs: dict = {}
        return cls(**kwargs)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:

        env = BootstrapConfig().app_env
        _ = dotenv_settings

        base_env = DotEnvSettingsSource(settings_cls, env_file=".env")
        env_specific = DotEnvSettingsSource(settings_cls, env_file=f".env.{env}")
        yaml_source = YamlConfigSettingsSource(
            settings_cls, yaml_file=f"config.{env}.yml"
        )

        return (
            init_settings,
            env_settings,
            env_specific,
            base_env,
            yaml_source,
            file_secret_settings,
        )
