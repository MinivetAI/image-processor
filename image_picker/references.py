"""Resolve archived tag definitions when a request does not supply one."""
import json
from pathlib import Path


REFERENCE_ROOT = Path(__file__).resolve().parent.parent / "reference" / "image_tags"


def _match_child(directory: Path, name: str, *, kind: str) -> Path | None:
    """Return a case-insensitive direct child without trusting caller path text."""
    normalized = name.casefold()
    for child in directory.iterdir():
        if child.is_dir() == (kind == "directory") and child.name.casefold() == normalized:
            return child
    return None


def load_reference_definition(business_unit: str, cms_vertical: str):
    """Load a boolean assessment definition for a known business-unit/vertical pair.

    Importing ``AssessmentDefinition`` lazily avoids a model/resolver import cycle.
    """
    from .inputs import AssessmentDefinition

    if not REFERENCE_ROOT.is_dir():
        raise ValueError("Reference definitions are unavailable; supply assessment_definition explicitly")
    business_unit_directory = _match_child(REFERENCE_ROOT, business_unit, kind="directory")
    if business_unit_directory is None:
        raise ValueError(
            f"No reference definition for business_unit={business_unit!r}; "
            "supply assessment_definition explicitly"
        )
    reference_file = _match_child(business_unit_directory, f"{cms_vertical}.json", kind="file")
    if reference_file is None:
        raise ValueError(
            f"No reference definition for business_unit={business_unit!r}, cms_vertical={cms_vertical!r}; "
            "supply assessment_definition explicitly"
        )
    try:
        data = json.loads(reference_file.read_text(encoding="utf-8"))
        fields = [
            {"name": tag["tag"], "type": "boolean", "description": tag["description"]}
            for tag in data["tags"]
        ]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid reference definition: {reference_file.relative_to(REFERENCE_ROOT)}") from error
    return AssessmentDefinition.model_validate({"fields": fields})
