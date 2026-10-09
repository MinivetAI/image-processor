# Image picker design decisions

The experimental image picker keeps a stable product/image request and a stable selection
result, while letting the caller define the observations and vertical-specific guidance
for each request. It uses one Alfred inference call. The aim is to support fine-grained
product differences without multiplying Python implementations or inference calls.

This document records the reasons for those decisions so future changes preserve the
user's intent. Read it before extending or integrating this experiment.

## Compared branches and scope

- Original branch: [`codex/prompt-packages-lifestyle`](https://github.com/MinivetAI/minivet-architecture/tree/codex/prompt-packages-lifestyle), as analyzed at commit `ca24e45`.
- Experimental branch: [`experimental/image-picking-and-recipe-generation`](https://github.com/MinivetAI/minivet-architecture/tree/experimental/image-picking-and-recipe-generation), implementation snapshot `4e12341`.
- New implementation: `experiments/image-picking/`.

The comparison is against the pinned original commit, not a promise about later changes
on that branch. The production image picker and shared task registry/runner remain in
place. The experiment is independently callable and is not a replacement production
endpoint. The only existing library changed is Alfred, described below.

Recipe generation is a downstream task envisioned to use one additional LLM call. Its
experimental directory is a scaffold; this branch does not implement recipe generation
or video generation. Local recipe reference documents are not required runtime assets.

## Why change the design?

A useful image of a water bottle may show its drinking opening, lid mechanism and readable
capacity. A useful image of a thermos may require different evidence. Lipstick may require
a distinction between exposed contents, packaging pictures and an application result.
Broad front/back/side labels cannot express all of those needs.

The user expects roughly 3,000–4,000 verticals and wants to try definitions that are not
already committed in the repository. Maintaining a Python output class, selector and
prompt registration for every vertical would make that experimentation cumbersome.

The user also wants GPUs available for generating videos. The intended budget is one
image-picking LLM call followed by one recipe-generation LLM call, rather than separate
calls for tagging, grouping, ranking, repairing and recipe planning.

The requirement is simplicity with useful capabilities: keep the familiar roles of
inputs, outputs, instructions and tasks; make the assessment content data supplied by
the caller; keep application configuration outside LLM orchestration.

## File-by-file comparison

A useful analogy is a form: inputs supply the photos and checklist, outputs define the
answer sheet, instructions explain how to fill it, and tasks connect the pieces.

| Responsibility | Original general task pattern / original image picker | Experiment | Reason |
| --- | --- | --- | --- |
| Inputs | Family-level [inputs.py](../../tasks/LLM/inputs.py) declares fixed request models. The original picker accepts image source strings, an enumerated `analytical_business_unit`, optional product context and aligned metadata. | [inputs.py](image_picker/inputs.py) accepts stable image IDs, sources, product context, an optional request definition and optional assessment instructions. | A custom category or field stays a request change; known archived categories can use their maintained reference definition. |
| Outputs | Family-level [outputs.py](../../tasks/LLM/outputs.py) contains fixed output classes. The original picker switches among Lifestyle, BPC, NonApparel and Home/BGM guides. | [outputs.py](image_picker/outputs.py) builds per-image observation fields from the request and validates the common identity/selection structure. | Category-specific observations vary; downstream identity and evidence relationships stay consistent. |
| Instructions | [instructions.py](../../tasks/LLM/ImagePickers/instructions.py) loads named prompt packages from repository files. The picker selects a package by business unit. | [instructions.py](image_picker/instructions.py) combines shared visual rules with optional caller-loaded vertical guidance. Field descriptions are embedded in the response schema. | Keep instructions loadable at scale without making the task own a configuration store or category lookup. |
| Tasks | [tasks.py](../../tasks/LLM/ImagePickers/tasks.py) registers specs in `TASKS`. The registry and shared runner create Alfred tasks, select the BU guide and invoke post-processing. | [tasks.py](image_picker/tasks.py) validates input, builds the guide, assembles labeled images, calls `LLMServer.respond` once and validates the response. | Keep the experiment directly callable without another client abstraction or changes to the production runner. |
| Selection | [post.py](../../tasks/LLM/ImagePickers/post.py) dispatches to BU-specific Python selectors that normalize grouping and choose final references. | The model proposes grouping and references in the same response; Python checks their consistency and eligibility without rewriting judgments. | Fine-grained selection should follow the requested evidence rather than a growing collection of category-specific Python policies. |

The experimental `tasks.py` is an execution function, not the same registration dictionary
used by the old pattern. That difference is deliberate and limited to this experiment.
If production integration becomes necessary, reuse the existing runner with the smallest
needed extension; do not introduce a second registry or transport stack merely to imitate
its structure.

## The request defines observations

`assessment_definition.fields` is required for custom categories. When the definition is
omitted, the runtime loads the matching boolean tag set under `reference/image_tags/` from
the request's business unit and CMS vertical. A supplied definition always takes priority.
Each field has a name, description and type.
Supported types are `boolean` (default), `string`, `integer`, `number` and string `enum`.
Enums also supply their allowed `values`.

```json
{
  "assessment_definition": {
    "fields": [
      {"name": "mouth_visible", "type": "boolean", "description": "Actual drinking opening is visibly exposed."},
      {"name": "lid_design", "type": "enum", "values": ["screw", "flip", "straw"], "description": "Visible lid mechanism; do not infer hidden construction."},
      {"name": "printed_capacity_ml", "type": "integer", "description": "Capacity visibly printed in ml; never infer from apparent size."}
    ]
  },
  "assessment_instructions": "Prioritize visible lid design and readable capacity markings."
}
```

This is a fragment of a product request; complete examples are in [examples/](examples/).
The definition is a deliberately small flat format, not support for arbitrary JSON
Schema, nested structures or executable expressions. There are 1–64 unique fields, with
validated names and types. Every output observation is required and nullable: `null`
means uncertain, not false. Boolean true means visibly present and false means visibly
absent. Other types must match their declared type; values are not silently coerced.

Business-unit and category names select an archived reference file only when the caller
omits `assessment_definition`; matching is case-insensitive and there is no fallback to a
different category. The 319 archived JSONs under `reference/image_tags/` remain optional:
new categories and any caller override use the supplied request definition instead.

## Instructions are also supplied by the caller

The application may load `assessment_instructions` from a file, database or configuration
service and include its text in the request. The task does not open paths or resolve
vertical names. Instructions are optional, nonblank when supplied, and limited to 16,000
characters. Omitting them retains the shared policy.

The shared policy establishes identity, uncertainty, evidence and selection rules.
Vertical guidance refines what to examine and prioritize. The output schema comes from
the assessment definition, not from prose. Python continues to enforce the schema and
selection relationships even when guidance changes. Prompt instructions alone do not
guarantee that all model judgments follow policy.

`build_instruction` is a small text-composition function, not a loader or prompt framework.
The task sends the supplied guidance once in the system instruction, excluding that field
from the product-context message. The CLI saves the composed text in `instruction.txt`.

## Stable evidence for downstream recipe generation

The common result contains `assessments`, `variants`, `selected_variant_id`,
`selected_references` and `unresolved`. Dynamic observations are inside each assessment's
`tags`. Selected reference roles use names from that request's definition.

Original image IDs survive grouping and selection. A reference can have several roles;
a collage or a useful human interaction should not be forced into a single angle label.
Evidence distinguishes actual product, actual contents, packaging depiction, application
result, human contact, scene and unclear support. Limitations retain uncertainty for the
next task. A recipe should not treat a packaging illustration as actual exposed contents
or attribute an unrelated application result to the selected product.

Python validation requires:

- Exactly one assessment per original image, in input order.
- Unique, nonempty variant IDs and matching memberships in variants and assessments.
- Duplicate links referring only to earlier input images.
- Evidence for every true boolean and known scalar observation, with unique evidence names and nonblank details.
- Unique selected original image IDs, all belonging to the selected variant, with matched identity, usable or limited quality and no duplicate flag.
- Nonempty, unique reference roles backed by evidence and values that are neither null nor boolean false. Numeric zero remains a valid known value.
- The explicit `max_references` limit, when supplied.
- A null selected variant and an explanation when no references are selected.

There is no reserved `M` variant, automatic multi-pack/combo rejection, fixed four-view
recipe, or universal category-tag rejection such as an implicit `ignore` field rule.
The shared policy asks for useful complementary evidence. Model judgments about identity,
quality and visual support still require evaluation; structural validation cannot inspect
pixels or establish that the chosen images are actually the best ones.

This is an intentional tradeoff: the original selector's deterministic ranking is replaced
by a more flexible model proposal with deterministic contract validation. The new approach
is not proven more visually accurate merely because it is more configurable.

## One call, using Alfred directly

The caller owns endpoint, model, authentication, timeout and session lifecycle. It supplies
an Alfred `LLMServer` configured with `retries=0` and closes it afterwards. The task uses
Alfred's media builder and `respond` with the generated Pydantic guide. There is no custom
HTTP client, client protocol, adapter, response-guide wrapper or synchronous bridge inside
the task. The CLI's `asyncio.run` is just its command-line entry point.

The original shared Alfred `Task` defaults to one schema-repair attempt. That can add an
inference call; the experiment deliberately bypasses that repair loop. Redirect following
is disabled too. A truncated response, malformed JSON, transport error or inconsistent
selection fails instead of causing a retry, repair or fallback to the old picker. Provider
extras cannot replace messages, response schema or response mode.

A successful invocation makes one inference request; rejected input makes none. Failed
CLI runs report an unknown call count rather than inventing a count. There is no recipe
planning inside the image-picking call.

The token budget grows with image and field count and can be overridden. This is a budget
estimate, not a guarantee that every 32-image/64-field request fits the backend's context
or generation limits. Start with small listings and measure output size and latency.

### Limited changes to Alfred

[alfie.py](alfred_src/alfred/alfie.py) adds opt-in `return_envelope` and
`allow_redirects` arguments to `LLMServer.respond`. Envelope access exposes finish reason,
model and usage; disabling redirects prevents an automatic resend. Existing callers keep
the default message return and redirect behavior. HTTP success is now explicitly 2xx;
non-followed 3xx responses are errors rather than successful completion payloads.

The experiment requires `finish_reason == "stop"`. Backend compatibility must be checked
before a live run. These additions reuse Alfred; they are not evidence of a problem that
required replacing Alfred with a custom client.

## Earlier experimental choices that were deliberately removed

These reversals occurred in the experiment, not in the original branch. Do not restore
them without a concrete requirement and discussion of the tradeoff:

| Removed choice | Why it was removed | Current replacement |
| --- | --- | --- |
| Custom vision HTTP client and later an Alfred adapter | Duplicated infrastructure and added abstraction over an existing abstraction. | Direct Alfred `LLMServer.respond`. |
| `specifications.py` with a category-file resolver | Constrained experimentation to committed category definitions and added an intermediate representation. | Validated definition in the request; `outputs.py` builds the guide directly. |
| `PreparedPicker` object and separate result-validation orchestration | Added another object to carry information already available from input and output. | Validated input plus generated output model, whose validator checks relationships. |
| Category-built instruction text | Coupled prompts to repository lookup and duplicated field definitions. | Shared policy plus caller-supplied guidance; field descriptions live in the schema. |
| Hard-coded present/absent/uncertain strings for every field | Limited observations to one state vocabulary. | Typed observations with null for uncertainty. |

The simplicity requested here is about reducing responsibilities and layers, not deleting
necessary validation. Keep application loading/configuration outside the task. Keep the
field definition format small until an actual use case requires more expressiveness.

## Original selector behaviors that motivated caution

The original code contains behaviors worth recognizing during integration. These are
examples from the pinned baseline, not a claim that all original tasks behave this way:

- Lifestyle reserves group `M` for composites and rewrites multi-product labels into it. Arbitrary physical variant IDs should not collide with a reserved letter in the experiment.
- Lifestyle normally chooses view slots but can fall back to all winning-group members. Do not import a fallback that bypasses an explicit reference cap or changes eligibility.
- NonApparel assigns a priority of 4 when a group has no supported view choices, but its winner condition accepts priorities below 5. That can yield a non-null winner with no selected references. The experiment explicitly rejects that inconsistent result.
- Lifestyle's role annotation returns the first supported role in a fixed order. A detail image with several signals can lose the detail role. Experimental references retain multiple supported roles.

The production implementations are not patched by this work. If fixing them separately,
use focused regression tests and preserve their public contracts; do not assume the
experiment's new request/result format can be dropped into those callers unchanged.

## Compatibility, verification and remaining work

This is a new experimental API, not backward compatibility with the original `image_picker`
payload. Original requests use source strings and `analytical_business_unit`; experimental
requests use image objects, `business_unit`, `cms_vertical`, a required `product_id`, title
and assessment definition. Results use stable IDs and evidence instead of the old picked
indices and BU-specific fields. Existing consumers need an explicit integration decision.

At implementation snapshot `4e12341`, 26 tests pass. They cover all 319 archived field sets,
custom types and nulls, malformed definitions, image/variant/evidence relationships,
reference limits, request-supplied instructions, local HTTP behavior, CLI artifacts and
one-call failure behavior. The legacy Alfred message return is also checked. Run them with:

```sh
cd experiments/image-picking
.venv/bin/python -m unittest test_picker -v
```

The backpack, lipstick and custom bottle outputs are labeled illustrative fixtures and
were generated without an LLM. Their paths are placeholders. No real-model image-picking
accuracy or latency has been established. A model-listing endpoint check is not a visual
evaluation. Successful runs save input, schema, composed instruction, response envelope,
validated result, usage and timing; failed validation currently does not retain the raw
envelope. Use fresh output directories to avoid mixing artifacts.

Before connecting a recipe task, evaluate real listings with human-reviewed identity,
variant grouping, duplicate decisions, selected roles and attribution. Measure failure
rate, latency and token usage. Keep recipe generation separate and preserve the one-call
budget for each task. Do not add an extra LLM router, tagging stage or repair loop to make
configuration dynamic: the caller already supplies the definition and instructions.
