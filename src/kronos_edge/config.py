"""Schema for the user's config.yaml and the supported-environment registry."""

from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError

VERSION = Annotated[StrictStr, Field(pattern=r"^\d+(\.\d+)+$")]


class EnvironmentConfig(BaseModel):
    """What the user asks for. Every field is required and must be a quoted string."""

    model_config = ConfigDict(extra="forbid")

    gpu: Annotated[StrictStr, Field(min_length=1)]
    driver: VERSION
    cuda: VERSION
    tensorrt: VERSION
    torch: VERSION


CONFIG_FIELDS = tuple(EnvironmentConfig.model_fields)

PRECISIONS = ("fp32", "fp16")
METRICS = ("cosine",)


class Config(EnvironmentConfig):
    """The user's config.yaml: the environment fields plus the check settings.

    Settings are not part of environment matching; changing one re-runs the
    checks in the same environment and overwrites the artifacts.
    """

    precision: Literal["fp32", "fp16"]
    metric: Literal["cosine"]
    threshold: Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
    # Routing key of the evaluator registered from this machine (`yasp-eval register --tag routing=...`),
    # which pins kernelgen's evaluations to it. Required: without it any evaluator advertising the
    # config's GPU could take the job, and the evaluation must run on this machine.
    eval_routing: Annotated[StrictStr, Field(min_length=1)]


class Environment(EnvironmentConfig):
    """One entry of the registry: a config plus the image that provides it."""

    name: StrictStr
    image: StrictStr
    dockerfile: StrictStr

    def matches(self, config: EnvironmentConfig) -> bool:
        return self.model_dump(include=set(CONFIG_FIELDS)) == config.model_dump(include=set(CONFIG_FIELDS))


class Registry(BaseModel):
    environments: list[Environment]


class InvalidConfig(ValueError):
    """config.yaml is readable but does not match the schema."""


def read_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_config(data) -> Config:
    if not isinstance(data, dict):
        raise InvalidConfig("top level must be a mapping of field: value")
    try:
        return Config.model_validate(data)
    except ValidationError as exc:
        lines = [f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()]
        if any(e["type"] == "string_type" for e in exc.errors()):
            lines.append('version numbers must be quoted strings, for example cuda: "13.0"')
        raise InvalidConfig("\n".join(lines)) from None


def load_environments(path) -> list[Environment]:
    return Registry.model_validate(read_yaml(path)).environments
