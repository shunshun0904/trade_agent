"""Secret access via SSM Parameter Store (spec 12).

Standard SecureString parameters are free; Secrets Manager charges per secret
per month, which is why it is not used. Values are cached per process so a warm
Lambda does not re-fetch on every invocation.

The `mcp` function has no IAM permission to read the bitbank parameters
(spec 12/16.3); a call there raises rather than silently returning None, so a
mis-wired deployment fails loudly instead of trading with no credentials.
"""

from __future__ import annotations

import functools
import logging
import os

from ..errors import ConfigError

log = logging.getLogger(__name__)


class SecretProvider:
    def __init__(self, client=None):
        self._client = client
        self._cache: dict[str, str] = {}

    @property
    def client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "ssm", region_name=os.environ.get("AWS_REGION", "ap-northeast-1"))
        return self._client

    def get(self, parameter_name: str) -> str:
        """Environment override first — that is how local runs and the test
        suite supply credentials without touching AWS."""
        env_name = _env_var_for(parameter_name)
        if env_name in os.environ:
            return os.environ[env_name]
        if parameter_name in self._cache:
            return self._cache[parameter_name]
        try:
            response = self.client.get_parameter(Name=parameter_name,
                                                 WithDecryption=True)
        except Exception as exc:
            raise ConfigError(
                f"could not read SSM parameter {parameter_name}: {exc}") from exc
        value = response["Parameter"]["Value"]
        self._cache[parameter_name] = value
        return value

    def get_optional(self, parameter_name: str) -> str | None:
        try:
            return self.get(parameter_name)
        except ConfigError:
            return None


def _env_var_for(parameter_name: str) -> str:
    """`/trade-agent/bitbank/api-key` -> `TA_SECRET_BITBANK_API_KEY`."""
    slug = parameter_name.strip("/").replace("trade-agent/", "", 1)
    return "TA_SECRET_" + slug.replace("/", "_").replace("-", "_").upper()


@functools.lru_cache(maxsize=1)
def default_provider() -> SecretProvider:
    return SecretProvider()
