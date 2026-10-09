"""Compile a per-request structured output contract from request definitions."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from .inputs import ImagePickerInput


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Variant(Contract):
    variant_id: str
    description: str
    image_ids: list[str]


def build_output_model(inp: ImagePickerInput):
    """The request defines observations; identity and selection stay stable."""
    fields = inp.assessment_definition.fields
    tag_type = Literal.__getitem__(tuple(field.name for field in fields))
    types = {"boolean": bool, "string": str, "integer": int, "number": float}
    definitions = {}
    for field in fields:
        field_type = Literal.__getitem__(tuple(field.values)) if field.type == "enum" else types[field.type]
        definitions[field.name] = (field_type | None, Field(..., description=field.description + " Use null when uncertain."))
    tags = create_model("ProductTags", __base__=Contract, **definitions)
    evidence = create_model(
        "Evidence", __base__=Contract,
        tag=(tag_type, ...),
        detail=(str, Field(..., description="Brief visible support or exact uncertainty; no chain of thought")),
        basis=(Literal["actual_product", "actual_contents", "packaging_depiction",
                       "application_result", "human_contact", "scene", "unclear"], ...),
        region=(str | None, Field(..., description="Optional location within the image, e.g. left panel")),
    )
    assessment = create_model(
        "ImageAssessment", __base__=Contract,
        image_id=(str, ...),
        variant_ids=(list[str], Field(..., description="All visibly supported listing variants; [] if none")),
        product_match=(Literal["matched", "uncertain", "mismatch", "no_product"], ...),
        quality=(Literal["usable", "limited", "unusable"], ...),
        duplicate_of=(str | None, Field(..., description="Earlier image ID only for a redundant view; null otherwise")),
        tags=(tags, ...),
        evidence=(list[evidence], Field(..., description="Evidence for every positive/value observation; false needs none, null may explain uncertainty")),
        limitations=(list[str], ...),
    )
    reference = create_model(
        "SelectedReference", __base__=Contract,
        image_id=(str, ...),
        roles=(list[tag_type], Field(..., description="Supported non-null observations assigned to this reference")),
        reason=(str, ...),
    )
    return create_model(
        "ImagePickerResult", __base__=Contract,
        __validators__={"consistent": model_validator(mode="after")(lambda result: validate_selection(inp, result))},
        assessments=(list[assessment], ...),
        variants=(list[Variant], ...),
        selected_variant_id=(str | None, ...),
        selected_references=(list[reference], ...),
        unresolved=(list[str], ...),
    )


def validate_selection(inp: ImagePickerInput, result):
    """Validate IDs and evidence relationships without changing model judgments."""
    expected = [image.image_id for image in inp.images]
    if [row.image_id for row in result.assessments] != expected:
        raise ValueError("Assessments must cover each original image exactly once, in input order")
    ids = set(expected)
    rows = {row.image_id: row for row in result.assessments}
    variants = {variant.variant_id: variant for variant in result.variants}
    if len(variants) != len(result.variants) or any(not key.strip() for key in variants):
        raise ValueError("Variant IDs must be unique and nonempty")
    positions = {key: index for index, key in enumerate(expected)}
    for row in result.assessments:
        if len(row.variant_ids) != len(set(row.variant_ids)) or not set(row.variant_ids) <= variants.keys():
            raise ValueError("Assessment refers to duplicate or undefined variants")
        if row.duplicate_of is not None and (row.duplicate_of not in ids or positions[row.duplicate_of] >= positions[row.image_id]):
            raise ValueError("duplicate_of must refer to an earlier input image")
        observed = {name for name, value in row.tags.model_dump().items() if value is not False and value is not None}
        evidence_tags = [item.tag for item in row.evidence]
        if not observed <= set(evidence_tags) or len(evidence_tags) != len(set(evidence_tags)):
            raise ValueError("Every positive/value observation needs evidence; evidence names must be unique")
        if any(not item.detail.strip() for item in row.evidence):
            raise ValueError("Evidence details must be nonempty")
    for variant in result.variants:
        members = {row.image_id for row in result.assessments if variant.variant_id in row.variant_ids}
        if not members or set(variant.image_ids) != members or len(variant.image_ids) != len(members):
            raise ValueError("Variant membership must agree exactly with image assessments")
    refs = result.selected_references
    selected_ids = [ref.image_id for ref in refs]
    if len(selected_ids) != len(set(selected_ids)) or not set(selected_ids) <= ids:
        raise ValueError("Selected references must be unique original image IDs")
    if not refs:
        if result.selected_variant_id is not None or not any(item.strip() for item in result.unresolved):
            raise ValueError("No references requires a null selected variant and an explanation")
    else:
        if result.selected_variant_id not in variants:
            raise ValueError("Selected variant must exist")
        if inp.max_references is not None and len(refs) > inp.max_references:
            raise ValueError("Selection exceeds max_references")
        for ref in refs:
            row = rows[ref.image_id]
            tags = row.tags.model_dump()
            if (result.selected_variant_id not in row.variant_ids or row.product_match != "matched"
                    or row.quality == "unusable" or row.duplicate_of is not None):
                raise ValueError("Selected reference lacks eligible evidence for the selected variant")
            supported = {item.tag for item in row.evidence if tags[item.tag] is not None and tags[item.tag] is not False}
            if not ref.roles or len(ref.roles) != len(set(ref.roles)) or not set(ref.roles) <= supported:
                raise ValueError("Selected roles must be unique supported observations with evidence")
            if not ref.reason.strip():
                raise ValueError("Selection reason must be nonempty")
    return result
