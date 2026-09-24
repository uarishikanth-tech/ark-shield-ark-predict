"""
All ORM columns and Python fields use snake_case (idiomatic Python),
but the REST API speaks camelCase JSON, matching the Node/Express
version of this backend so a frontend can talk to either one
unmodified. CamelModel is the shared base every request/response
schema subclasses to get that translation automatically:
  - Input: accepts either camelCase or snake_case keys.
  - Output (FastAPI response_model): serializes using camelCase,
    since FastAPI's default response_model_by_alias=True already
    serializes by alias when a model defines one.
  - from_attributes=True lets response schemas be built directly
    from SQLAlchemy ORM instances (schema(**{}) is never needed).
"""
from pydantic import BaseModel, ConfigDict


def to_camel(snake: str) -> str:
    first, *rest = snake.split("_")
    return first + "".join(word.capitalize() for word in rest)


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, from_attributes=True)
