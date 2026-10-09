"""Generate clearly labelled illustrative fixtures; never contact an LLM."""
import json
from pathlib import Path

from image_processor.image_picker import ImagePickerInput, build_output_model
from image_processor.image_picker.instructions import build_instruction
from image_processor.image_picker.references import REFERENCE_ROOT


ROOT = Path(__file__).resolve().parents[1]


def make_row(fields, image_id, present, *, uncertain=(), duplicate_of=None,
             match="matched", variants=("V1",), limitations=()):
    states = {field["name"]: False for field in fields}
    evidence = []
    for state, items in ((True, present), (None, uncertain)):
        for name, detail, basis in items:
            states[name] = state
            evidence.append({"tag": name, "detail": detail, "basis": basis, "region": None})
    return {"image_id": image_id, "variant_ids": list(variants), "product_match": match,
            "quality": "usable", "duplicate_of": duplicate_of, "tags": states,
            "evidence": evidence, "limitations": list(limitations)}


def add_definition(payload, example_file):
    """Example authoring only: runtime never reads this archive."""
    data = json.loads((REFERENCE_ROOT / example_file).read_text())
    fields = [{"name": item["tag"], "type": "boolean", "description": item["description"]} for item in data["tags"]]
    payload["assessment_definition"] = {"fields": fields}
    return fields


def examples():
    backpack = {"product_id": "ILLUSTRATIVE-BACKPACK", "business_unit": "Lifestyle",
        "cms_vertical": "backpack", "title": "Blue backpack", "attributes": {"colour": "blue"},
        "images": [{"image_id": f"img-{i:02}", "source": f"replace-with-real-images/backpack-{i}.jpg"}
                   for i in range(1, 5)]}
    fields = add_definition(backpack, "LifeStyle/backpack.json")
    rows = [
        make_row(fields, "img-01", [("front_view", "Complete blue bag front is visible.", "actual_product"),
                 ("logo", "Brand mark is visible on the bag panel.", "actual_product")]),
        make_row(fields, "img-02", [("side_view", "Side profile of the same bag is visible.", "actual_product"),
                 ("closure_detail", "Visible zipper and pull at the side pocket.", "actual_product")]),
        make_row(fields, "img-03", [("back_view", "Back panel and both straps are visible.", "actual_product"),
                 ("human_interaction", "Both straps rest on the wearer's shoulders; bag already worn.", "human_contact"),
                 ("carried_or_worn", "The same blue bag is already worn on the back.", "human_contact")]),
        make_row(fields, "img-04", [("front_view", "Repeated front packshot of the same bag.", "actual_product")],
                 duplicate_of="img-01"),
    ]
    response = {"assessments": rows, "variants": [{"variant_id": "V1", "description": "Blue backpack with pictured front logo and side zipper",
        "image_ids": [row["image_id"] for row in rows]}], "selected_variant_id": "V1",
        "selected_references": [
            {"image_id": "img-01", "roles": ["front_view", "logo"], "reason": "Clear overall design and branding."},
            {"image_id": "img-02", "roles": ["side_view", "closure_detail"], "reason": "Complementary profile and zipper evidence."},
            {"image_id": "img-03", "roles": ["back_view", "human_interaction", "carried_or_worn"], "reason": "Back design and exact wearing relationship."}],
        "unresolved": []}
    yield "backpack", backpack, response

    lipstick = {"product_id": "ILLUSTRATIVE-LIPSTICK", "business_unit": "BPC",
        "cms_vertical": "lipstick", "title": "Red lipstick", "attributes": {"shade": "red"},
        "images": [{"image_id": f"img-{i:02}", "source": f"replace-with-real-images/lipstick-{i}.jpg"}
                   for i in range(1, 4)]}
    fields = add_definition(lipstick, "BPC/lipstick.json")
    rows = [
        make_row(fields, "img-01", [("front_view", "Actual tube front and branding are visible.", "actual_product"),
                 ("logo", "Printed mark is on the tube, not an overlay.", "actual_product")]),
        make_row(fields, "img-02", [("contents_visible", "Actual red lipstick bullet is exposed.", "actual_contents"),
                 ("open_lid_cap", "Cap is removed beside the matching tube.", "actual_product"),
                 ("texture_detail", "Close view shows the pictured bullet colour and surface.", "actual_contents")]),
        make_row(fields, "img-03", [], uncertain=[("applied_on_human", "Lips have red colour, but the pictured result cannot be attributed to this exact lipstick.", "application_result")],
                 match="uncertain", variants=(), limitations=["No matching tube or other identity evidence is pictured."]),
    ]
    response = {"assessments": rows, "variants": [{"variant_id": "V1", "description": "Pictured red lipstick tube and matching exposed bullet",
        "image_ids": ["img-01", "img-02"]}], "selected_variant_id": "V1",
        "selected_references": [
            {"image_id": "img-01", "roles": ["front_view", "logo"], "reason": "Identifiable product and branding."},
            {"image_id": "img-02", "roles": ["contents_visible", "open_lid_cap", "texture_detail"], "reason": "Actual contents and ready-state exposure."}],
        "unresolved": ["img-03 cannot establish an application result for the selected lipstick variant."]}
    yield "lipstick", lipstick, response

    bottle = {"product_id": "ILLUSTRATIVE-CUSTOM-BOTTLE", "business_unit": "Custom",
        "cms_vertical": "my_water_bottle", "title": "Printed 500 ml bottle",
        "assessment_instructions": "Prioritize visible lid design and readable capacity markings. Do not infer thermal insulation or leak resistance from appearance.",
        "images": [{"image_id": "bottle-01", "source": "replace-with-real-images/bottle.jpg"}],
        "assessment_definition": {"fields": [
            {"name": "lid_design", "type": "enum", "values": ["screw", "flip", "straw"], "description": "Visible lid mechanism; null if unclear."},
            {"name": "printed_capacity_ml", "type": "integer", "description": "Capacity printed visibly in ml; never infer from size."},
            {"name": "height_width_ratio", "type": "number", "description": "Approximate visible bottle height divided by width."},
            {"name": "visible_text", "type": "string", "description": "Exact readable text on the bottle."},
            {"name": "mouth_visible", "type": "boolean", "description": "Actual drinking opening is visibly exposed."}]}}
    values = {"lid_design": "screw", "printed_capacity_ml": 500, "height_width_ratio": 2.8,
              "visible_text": "500 ml", "mouth_visible": None}
    details = {"lid_design": "Threaded cap is beside the bottle.", "printed_capacity_ml": "500 ml is printed on the bottle.",
               "height_width_ratio": "Visible silhouette is approximately 2.8 times as tall as wide.",
               "visible_text": "The readable text is 500 ml."}
    row = {"image_id": "bottle-01", "variant_ids": ["V1"], "product_match": "matched", "quality": "limited",
           "duplicate_of": None, "tags": values,
           "evidence": [{"tag": name, "detail": detail, "basis": "actual_product", "region": None} for name, detail in details.items()],
           "limitations": ["Angle obscures the drinking opening."]}
    yield "custom-bottle", bottle, {"assessments": [row],
        "variants": [{"variant_id": "V1", "description": "Pictured printed 500 ml bottle", "image_ids": ["bottle-01"]}],
        "selected_variant_id": "V1", "selected_references": [{"image_id": "bottle-01", "roles": ["lid_design", "visible_text"],
        "reason": "Visible cap mechanism and label; opening remains uncertain."}], "unresolved": ["Drinking opening is obscured."]}


def main():
    target = ROOT / "examples"
    target.mkdir(exist_ok=True)
    for name, payload, raw in examples():
        output = build_output_model(ImagePickerInput.model_validate(payload))
        result = output.model_validate(raw)
        artifacts = {"input": payload, "response": raw, "schema": output.model_json_schema(),
            "output": {"example_kind": "illustrative_fixture_not_model_output", "llm_calls": 0,
                       "product_id": payload["product_id"],
                       "result": result.model_dump()}}
        for kind, value in artifacts.items():
            (target / f"{name}.{kind}.json").write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        (target / f"{name}.instruction.txt").write_text(build_instruction(ImagePickerInput.model_validate(payload).assessment_instructions))
        print(f"Validated illustrative {name}: {len(result.selected_references)} references; zero LLM calls")


if __name__ == "__main__":
    main()
