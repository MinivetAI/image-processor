"""Model response contract and deterministic mapping to stable API IDs."""
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from .inputs import ImagePickerInput


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


def _tag_models(inp: ImagePickerInput):
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
    return tag_type, tags, evidence


def build_output_model(inp: ImagePickerInput):
    """Build the LLM-facing schema. It contains positions and labels, never API IDs."""
    tag_type, tags, evidence = _tag_models(inp)
    variant = create_model("VariantLabel", __base__=Contract,
        label=(str, Field(..., description="Short label such as A or B, assigned by first appearance")),
        description=(str, ...))
    assessment = create_model(
        "ImageAssessment", __base__=Contract,
        image_number=(int, Field(..., ge=1, description="1-based position in the supplied image sequence")),
        variant_labels=(list[str], Field(..., description="Labels for every visible listing variant in this image; [] if none")),
        product_match=(Literal["matched", "uncertain", "mismatch", "no_product"], ...),
        quality=(Literal["usable", "limited", "unusable"], ...),
        duplicate_of=(int | None, Field(..., description="Earlier image number only for a redundant view; null otherwise")),
        tags=(tags, ...),
        evidence=(list[evidence], Field(..., description="Evidence for every positive/value observation; false needs none, null may explain uncertainty")),
        limitations=(list[str], ...),
    )
    reference = create_model(
        "SelectedReference", __base__=Contract,
        image_number=(int, Field(..., ge=1)),
        roles=(list[tag_type], Field(..., description="Supported non-null observations assigned to this reference")),
        reason=(str, ...),
    )
    return create_model(
        "ModelImagePickerResponse", __base__=Contract,
        __validators__={"consistent": model_validator(mode="after")(lambda result: validate_model_response(inp, result))},
        assessments=(list[assessment], ...),
        variants=(list[variant], ...),
        selected_variant_label=(str | None, ...),
        selected_references=(list[reference], ...),
        unresolved=(list[str], ...),
    )


def build_public_output_model(inp: ImagePickerInput):
    """Build the unchanged API result contract and its cross-field validation."""
    tag_type, tags, evidence = _tag_models(inp)
    variant = create_model("Variant", __base__=Contract,
        variant_id=(str, ...), description=(str, ...), image_ids=(list[str], ...))
    assessment = create_model(
        "PublicImageAssessment", __base__=Contract,
        image_id=(str, ...), variant_ids=(list[str], ...),
        product_match=(Literal["matched", "uncertain", "mismatch", "no_product"], ...),
        quality=(Literal["usable", "limited", "unusable"], ...), duplicate_of=(str | None, ...),
        tags=(tags, ...), evidence=(list[evidence], ...), limitations=(list[str], ...))
    reference = create_model("PublicSelectedReference", __base__=Contract,
        image_id=(str, ...), roles=(list[tag_type], ...), reason=(str, ...))
    return create_model(
        "ImagePickerResult", __base__=Contract,
        __validators__={"consistent": model_validator(mode="after")(lambda result: validate_selection(inp, result))},
        assessments=(list[assessment], ...), variants=(list[variant], ...),
        selected_variant_id=(str | None, ...), selected_references=(list[reference], ...), unresolved=(list[str], ...))


def validate_model_response(inp: ImagePickerInput, result):
    """Check model-authored labels once; Python will derive reciprocal memberships."""
    if [row.image_number for row in result.assessments] != list(range(1, len(inp.images) + 1)):
        raise ValueError("Assessments must cover image numbers 1..N exactly once, in input order")
    definitions = {}
    for item in result.variants:
        if not item.label.strip() or not re.fullmatch(r"[A-Z][A-Z0-9_]*", item.label):
            raise ValueError("Variant labels must be short uppercase labels such as A or B")
        if item.label in definitions:
            raise ValueError("Variant labels must be unique")
        if not item.description.strip():
            raise ValueError("Variant descriptions must be nonempty")
        definitions[item.label] = item.description
    labels_in_images = set()
    rows = {row.image_number: row for row in result.assessments}
    for row in result.assessments:
        if len(row.variant_labels) != len(set(row.variant_labels)):
            raise ValueError("An image cannot repeat a variant label")
        if not set(row.variant_labels) <= definitions.keys():
            raise ValueError("Image refers to an undefined variant label")
        labels_in_images.update(row.variant_labels)
        if row.duplicate_of is not None and not 1 <= row.duplicate_of < row.image_number:
            raise ValueError("duplicate_of must refer to an earlier image number")
        _validate_evidence(row)
    if labels_in_images != definitions.keys():
        raise ValueError("Every declared variant label must be used by an image")
    references = result.selected_references
    selected_numbers = [ref.image_number for ref in references]
    if len(selected_numbers) != len(set(selected_numbers)) or any(number not in rows for number in selected_numbers):
        raise ValueError("Selected references must use unique valid image numbers")
    if not references:
        if result.selected_variant_label is not None or not any(item.strip() for item in result.unresolved):
            raise ValueError("No references requires a null selected variant and an explanation")
        has_candidate = any(
            row.product_match == "matched" and row.quality != "unusable"
            and row.duplicate_of is None and row.variant_labels
            and any(row.tags.model_dump()[item.tag] is not None and row.tags.model_dump()[item.tag] is not False
                    for item in row.evidence)
            for row in result.assessments
        )
        if has_candidate:
            raise ValueError("At least one eligible image needs to be selected")
    else:
        label = result.selected_variant_label
        if label not in definitions:
            raise ValueError("Selected variant label must be declared")
        if inp.max_references is not None and len(references) > inp.max_references:
            raise ValueError("Selection exceeds max_references")
        for ref in references:
            row = rows[ref.image_number]
            tags = row.tags.model_dump()
            if (label not in row.variant_labels or row.product_match != "matched"
                    or row.quality == "unusable" or row.duplicate_of is not None):
                raise ValueError("Selected reference lacks eligible evidence for the selected variant")
            supported = {item.tag for item in row.evidence if tags[item.tag] is not None and tags[item.tag] is not False}
            if not ref.roles or len(ref.roles) != len(set(ref.roles)) or not set(ref.roles) <= supported:
                raise ValueError("Selected roles must be unique supported observations with evidence")
            if not ref.reason.strip():
                raise ValueError("Selection reason must be nonempty")
    return result


def _validate_evidence(row):
    observed = {name for name, value in row.tags.model_dump().items() if value is not False and value is not None}
    evidence_tags = [item.tag for item in row.evidence]
    if not observed <= set(evidence_tags) or len(evidence_tags) != len(set(evidence_tags)):
        raise ValueError("Every positive/value observation needs evidence; evidence names must be unique")
    if any(not item.detail.strip() for item in row.evidence):
        raise ValueError("Evidence details must be nonempty")


def validate_selection(inp: ImagePickerInput, result):
    """Validate the public response after stable image IDs and memberships are derived."""
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
        _validate_evidence(row)
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


def to_public_result(inp: ImagePickerInput, raw):
    """Validate the model payload and deterministically map labels to API variant IDs."""
    response_model = build_output_model(inp)
    source = raw.model_dump() if isinstance(raw, BaseModel) else raw
    model = response_model.model_validate(source)
    labels = []
    for row in model.assessments:
        for label in row.variant_labels:
            if label not in labels:
                labels.append(label)
    descriptions = {variant.label: variant.description for variant in model.variants}
    label_to_id = {label: f"V{index}" for index, label in enumerate(labels, 1)}
    image_ids = {number: image.image_id for number, image in enumerate(inp.images, 1)}
    assessments = []
    for row in model.assessments:
        assessments.append({
            "image_id": image_ids[row.image_number],
            "variant_ids": [label_to_id[label] for label in row.variant_labels],
            "product_match": row.product_match,
            "quality": row.quality,
            "duplicate_of": image_ids[row.duplicate_of] if row.duplicate_of is not None else None,
            "tags": row.tags.model_dump(),
            "evidence": [item.model_dump() for item in row.evidence],
            "limitations": row.limitations,
        })
    variants = [{
        "variant_id": label_to_id[label],
        "description": descriptions[label],
        "image_ids": [image_ids[row.image_number] for row in model.assessments if label in row.variant_labels],
    } for label in labels]
    result = {
        "assessments": assessments,
        "variants": variants,
        "selected_variant_id": label_to_id[model.selected_variant_label] if model.selected_variant_label else None,
        "selected_references": [{
            "image_id": image_ids[ref.image_number], "roles": ref.roles, "reason": ref.reason,
        } for ref in model.selected_references],
        "unresolved": model.unresolved,
    }
    return build_public_output_model(inp).model_validate(result)
