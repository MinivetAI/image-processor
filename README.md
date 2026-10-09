# Experimental image picker

Read [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) for the original-branch comparison,
reasons for this design, and decisions future changes should preserve.

One vision-model call assesses the supplied images and chooses complementary references
for one product variant. The assessment definition comes in the request. No category
lookup or category-specific Python is required. Recipe generation can consume the stable
image IDs, variant grouping, reference roles, evidence and limitations in a separate call.

The four task files have distinct jobs:

| File | Responsibility |
| --- | --- |
| `image_picker/inputs.py` | Validate product, images, requested fields and optional instructions |
| `image_picker/outputs.py` | Build the requested output model and validate selection relationships |
| `image_picker/instructions.py` | Shared policy plus caller-supplied assessment instructions |
| `image_picker/tasks.py` | Assemble the images, call Alfred once and validate the response |

There is no specifications file, custom LLM client, registry, or prepared-task wrapper.
The caller supplies and closes Alfred's `LLMServer`; configuration stays outside the task.
Existing production tasks and their runner are untouched.

## Request definitions

Add `assessment_definition` to the normal product/image input:

```json
{
  "assessment_definition": {
    "fields": [
      {"name": "mouth_visible", "type": "boolean", "description": "Actual drinking opening visibly exposed."},
      {"name": "lid_design", "type": "enum", "values": ["screw", "flip", "straw"], "description": "Visible lid mechanism."},
      {"name": "printed_capacity_ml", "type": "integer", "description": "Capacity visibly printed in ml; never infer from size."}
    ]
  }
}
```

Supported field types are `boolean` (default), `string`, `integer`, `number`, and string
`enum`. Every output field is required and accepts `null` for uncertainty. Boolean false
means visibly absent; true means visibly present. Definitions support 1–64 unique fields.
This is a small flat definition format, not arbitrary JSON Schema or executable code.
BU/category names are context strings and can be new categories. Fields and descriptions
may change on each request. The 319 archived definitions are example material only;
`generate_examples.py` converts two of them into request fields. Runtime reads none.

### Loadable instructions

Optionally include `assessment_instructions` as text in the request. The application can
load it from a file, database or configuration service; the task performs no lookup.
Shared instructions stay in place, and this guidance refines vertical-specific assessment
and selection without changing the supplied schema or Python validation. Existing requests
that omit it keep the shared instruction. Text must be nonblank and at most 16,000 characters.

```python
from pathlib import Path

payload["assessment_instructions"] = Path("bottle-instructions.txt").read_text()
# Pass payload to run_image_picker as usual: still one Alfred call.
```

The custom bottle fixture includes this field. The CLI saves the exact composed instruction
in `instruction.txt`, so you can review what a run sent to the model.

The stable output retains every original image assessment. Validation enforces image IDs
and order, exact variant membership, backward duplicate links, supported reference roles,
unique selected images and the optional `max_references` cap. Unknown identities, unusable
images and duplicates cannot be selected. No selection requires a null variant and an
explanation. True boolean and known scalar observations require brief visual evidence;
uncertain fields can carry explanatory evidence but cannot become reference roles.

## Setup and examples

Use Python 3.11 or newer. From this directory:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest test_picker -v
.venv/bin/python generate_examples.py
.venv/bin/python run.py prepare examples/custom-bottle.input.json --output runs/bottle-prepared
.venv/bin/python run.py validate examples/custom-bottle.input.json --response examples/custom-bottle.response.json --output runs/bottle-validation
```

Backpack, lipstick and a custom bottle have input, schema, response and output examples.
They are **illustrative fixtures, not model predictions**, and contain placeholder image
paths. The bottle demonstrates all five field types without using an archived category.

## Run with Alfred

Replace example product facts and image sources with actual listing images. Relative
paths resolve against the input file's directory. Alfred sends local files as data URIs;
HTTP URLs must be accessible to the backend. Supply endpoint and model explicitly:

```sh
export IMAGE_PICKER_LLM_URL='http://YOUR-ENDPOINT/v1'
export IMAGE_PICKER_LLM_MODEL='YOUR-MODEL'
# Set IMAGE_PICKER_LLM_API_KEY if needed.
.venv/bin/python run.py run real-input.json --output runs/first-run --disable-thinking
```

`--disable-thinking` is specific to Qwen/vLLM; omit it for other backends.
`endpoint.example.env` contains the internal deployment configuration previously checked
with its model-listing API. No inference is performed at import time.

```python
from alfred import LLMServer
from image_picker import run_image_picker

server = LLMServer(base_url, model, retries=0)
try:
    result = await run_image_picker(payload, server)
finally:
    await server.close()
```

The task calls `LLMServer.respond` directly with the generated Pydantic guide. Retries,
redirects and repair calls are disabled. Truncated, malformed or inconsistent output
fails instead of triggering another inference. Provider extras cannot override the
messages, schema or response mode. Token budget grows with image/field count and can be
overridden with `--max-tokens`; backend context/output limits still apply. Start with
3–6 images before trying the 32-image maximum.

The CLI saves input, schema and shared instructions, plus validated output, raw response,
model, usage and timing on success. Failed inference runs record the error and an unknown
call count; raw envelopes from failed validation are not retained. Use a fresh output
directory per run. Artifacts are ignored by Git. No API keys are saved.

Tests cover definition validation, all archived field sets, custom types, selection
consistency, real local HTTP requests, CLI artifacts and the one-call failure boundary.
Visual quality still needs a live run with real listing images and human review. Check
variant identity, duplicate judgments, useful role coverage, visual attribution, latency
and token usage before connecting recipe generation.
