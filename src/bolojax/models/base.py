"""Shared base model for all bolojax pydantic models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class BolojaxModel(BaseModel):
    """Base model with common configuration for all bolojax models.

    Notes
    -----
    - ``frozen=True`` is deliberately omitted because many models
      mutate after init (e.g., Camera sets ``_instrument``,
      Channel sets ``_bandwidth``).
    - ``extra="forbid"`` catches typos in YAML configs early.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_default=True,
        extra="forbid",
    )
