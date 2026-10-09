"""The stable request contract. Image IDs never change after selection."""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImageReference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    image_id: str = Field(min_length=1, max_length=100)
    source: str = Field(min_length=1, description="HTTP URL, data URI, or local image path")
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssessmentField(BaseModel):
    """A deliberately small declarative schema; never executable Python or JSON Schema."""
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)
    description: str = Field(min_length=1, max_length=2000)
    type: Literal["boolean", "string", "integer", "number", "enum"] = "boolean"
    values: list[str] | None = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def valid_definition(self):
        if self.name.startswith("model_") or hasattr(BaseModel, self.name):
            raise ValueError("Field name conflicts with Pydantic")
        if not self.description.strip():
            raise ValueError("Field description cannot be blank")
        if self.type == "enum":
            if not self.values or any(not value.strip() for value in self.values) or len(set(self.values)) != len(self.values):
                raise ValueError("Enum requires unique nonempty values")
        elif self.values is not None:
            raise ValueError("Only enum fields accept values")
        return self


class AssessmentDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    fields: list[AssessmentField] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def unique_fields(self):
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("Assessment field names must be unique")
        return self


class ImagePickerInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    product_id: str = Field(min_length=1)
    business_unit: str = Field(min_length=1)
    cms_vertical: str = Field(min_length=1)
    title: str = Field(min_length=1)
    attributes: dict[str, Any] = Field(default_factory=dict)
    images: list[ImageReference] = Field(min_length=1, max_length=32)
    assessment_definition: AssessmentDefinition
    assessment_instructions: str | None = Field(
        default=None, min_length=1, max_length=16000, pattern=r"\S",
        description="Optional vertical-specific assessment guidance, loaded by the caller",
    )
    selection_brief: str | None = None
    max_references: int | None = Field(default=None, ge=1, le=32)

    @model_validator(mode="after")
    def unique_image_ids(self):
        ids = [image.image_id for image in self.images]
        if any(not value.strip() for value in ids) or len(ids) != len(set(ids)):
            raise ValueError("image_id must be nonblank and unique within a request")
        return self
