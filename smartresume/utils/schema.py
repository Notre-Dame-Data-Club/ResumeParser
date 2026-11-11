"""
Extraction contract and CSV schema.

prompts.py shows the model a JSON TEMPLATE per section; this module names
those keys once and maps them onto CSV columns.  Consumers derive their
column list and their row flattening from CSV_SCHEMA instead of hardcoding
field names, so a prompt key and the CSV column reading it cannot drift apart.

Cell encoding (rows are scored by an LLM for a QLoRA dataset):
  * scalar column      -> plain string.
  * list_json column   -> JSON array, one element per resume entry (job, degree,
                          project).  Each element is a JSON object built from the
                          column's FieldSpecs, so one entry stays one coherent
                          record.  Blank entries are kept, never dropped.
Cells are written with json.dumps, never str(), so every JSON cell parses.

Dates are flat startDate/endDate keys in every section (no nested period
object): one shape for the small model to learn, one level less to close.

Kept separate from prompts.py so that file stays prompt text only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .prompts import (
    BASIC_INFO_EXAMPLE_OUTPUT,
    BASIC_INFO_TEMPLATE,
    EDUCATION_EXAMPLE_OUTPUT,
    EDUCATION_TEMPLATE,
    PROJECTS_EXAMPLE_OUTPUT,
    PROJECTS_TEMPLATE,
    WORK_EXPERIENCE_EXAMPLE_OUTPUT,
    WORK_EXPERIENCE_TEMPLATE,
)

# -- Sections (top-level JSON keys) -- #
SECTION_BASIC_INFO = "basicInfo"
SECTION_WORK_EXPERIENCE = "workExperience"
SECTION_EDUCATION = "education"
SECTION_PROJECTS = "projects"

# -- Field keys: must equal the keys in the prompts.py TEMPLATEs -- #
# basicInfo
FIELD_NAME = "name"
FIELD_EMAIL = "personalEmail"
FIELD_PHONE = "phoneNumber"
FIELD_LOCATION = "currentLocation"
FIELD_URLS = "URLs"

# shared flat date keys (workExperience, education, projects)
FIELD_START_DATE = "startDate"
FIELD_END_DATE = "endDate"

# workExperience
FIELD_COMPANY = "companyName"
FIELD_POSITION = "position"
FIELD_INTERNSHIP = "internship"
FIELD_LEADERSHIP = "isLeadership"
FIELD_JOB_TECHNOLOGIES = "technologies"
# INTERNAL: the model returns line indexes instead of re-typing the text.
# Never exported to CSV; data_processor resolves it into FIELD_JOB_DESCRIPTION.
INTERNAL_JOB_DESCRIPTION_RANGE_KEY = "jobDescription_refer_index_range"
# Not a prompt key: data_processor resolves the index range into this text.
FIELD_JOB_DESCRIPTION = "jobDescription"

# education
FIELD_SCHOOL = "school"
FIELD_DEGREE_LEVEL = "degreeLevel"
FIELD_DEPARTMENT = "department"
FIELD_MAJOR = "major"
FIELD_GPA = "GPA"
FIELD_EDUCATION_DESCRIPTION = "educationDescription"

# projects
FIELD_PROJECT_NAME = "name"
FIELD_PROJECT_TECHNOLOGIES = "technologies"
# INTERNAL: line-index range; data_processor resolves it into FIELD_PROJECT_DESCRIPTION.
INTERNAL_PROJECT_DESCRIPTION_RANGE_KEY = "description_refer_index_range"
# Not a prompt key (derived), like FIELD_JOB_DESCRIPTION.
FIELD_PROJECT_DESCRIPTION = "description"

LIST_SECTIONS = frozenset(
    {SECTION_WORK_EXPERIENCE, SECTION_EDUCATION, SECTION_PROJECTS}
)

# Section -> (template, example output) as shown to the model.
_SECTION_CONTRACTS: Dict[str, Tuple[str, str]] = {
    SECTION_BASIC_INFO: (BASIC_INFO_TEMPLATE, BASIC_INFO_EXAMPLE_OUTPUT),
    SECTION_WORK_EXPERIENCE: (WORK_EXPERIENCE_TEMPLATE, WORK_EXPERIENCE_EXAMPLE_OUTPUT),
    SECTION_EDUCATION: (EDUCATION_TEMPLATE, EDUCATION_EXAMPLE_OUTPUT),
    SECTION_PROJECTS: (PROJECTS_TEMPLATE, PROJECTS_EXAMPLE_OUTPUT),
}

# Template keys the model emits that no CSV column reads directly.
INTERNAL_KEYS: Dict[str, frozenset] = {
    SECTION_WORK_EXPERIENCE: frozenset({INTERNAL_JOB_DESCRIPTION_RANGE_KEY}),
    SECTION_PROJECTS: frozenset({INTERNAL_PROJECT_DESCRIPTION_RANGE_KEY}),
}

## Extraction type (prompt key) -> the one top-level section its answer may contain
EXTRACT_TYPE_SECTIONS: Dict[str, str] = {
    "basic_info": SECTION_BASIC_INFO,
    "work_experience": SECTION_WORK_EXPERIENCE,
    "education": SECTION_EDUCATION,
    "projects": SECTION_PROJECTS,
}


def _template_entry_keys(template: str, section: str) -> Tuple[str, ...]:
    """Keys of one entry in a TEMPLATE, in template order ("" -> no keys)."""
    try:
        body = json.loads(template)[section]
    except (json.JSONDecodeError, KeyError, TypeError):
        return ()
    entry = body[0] if isinstance(body, list) and body else body
    return tuple(entry) if isinstance(entry, dict) else ()


## Section -> the keys one entry may have, in template order (first key = entry anchor)
SECTION_ENTRY_KEYS: Dict[str, Tuple[str, ...]] = {
    section: _template_entry_keys(template, section)
    for section, (template, _example) in _SECTION_CONTRACTS.items()
}

KIND_SCALAR = "scalar"
KIND_LIST_JSON = "list_json"


# -- Schema -- #
@dataclass(frozen=True)
class FieldSpec:
    """
    One key inside a list_json entry object.

    out_key  key name in the CSV JSON object
    path     keys to walk inside the extracted entry, e.g. ("startDate",)
    derived  True when the value is produced by post-processing rather than
             emitted by the model
    source   for derived fields: the template key post-processing reads
             (validated to exist in the template)
    """

    out_key: str
    path: Tuple[str, ...]
    derived: bool = False
    source: str = ""


@dataclass(frozen=True)
class CsvColumn:
    """
    CSV Column definitions
    Column Name | Section Var | Kind | Field Var (scalar) | Fields (list_json)
    """

    name: str
    section: str = ""
    kind: str = KIND_SCALAR
    field: str = ""
    fields: Tuple[FieldSpec, ...] = ()


_WORK_FIELDS = (
    FieldSpec("company", (FIELD_COMPANY,)),
    FieldSpec("position", (FIELD_POSITION,)),
    FieldSpec("start", (FIELD_START_DATE,)),
    FieldSpec("end", (FIELD_END_DATE,)),
    FieldSpec("internship", (FIELD_INTERNSHIP,)),
    FieldSpec("leadership", (FIELD_LEADERSHIP,)),
    FieldSpec("technologies", (FIELD_JOB_TECHNOLOGIES,)),
    FieldSpec("description", (FIELD_JOB_DESCRIPTION,), derived=True,
              source=INTERNAL_JOB_DESCRIPTION_RANGE_KEY),
)

_EDUCATION_FIELDS = (
    FieldSpec("school", (FIELD_SCHOOL,)),
    FieldSpec("degree", (FIELD_DEGREE_LEVEL,)),
    FieldSpec("department", (FIELD_DEPARTMENT,)),
    FieldSpec("major", (FIELD_MAJOR,)),
    FieldSpec("start", (FIELD_START_DATE,)),
    FieldSpec("end", (FIELD_END_DATE,)),
    FieldSpec("gpa", (FIELD_GPA,)),
    FieldSpec("description", (FIELD_EDUCATION_DESCRIPTION,)),
)

_PROJECT_FIELDS = (
    FieldSpec("name", (FIELD_PROJECT_NAME,)),
    FieldSpec("description", (FIELD_PROJECT_DESCRIPTION,), derived=True,
              source=INTERNAL_PROJECT_DESCRIPTION_RANGE_KEY),
    FieldSpec("technologies", (FIELD_PROJECT_TECHNOLOGIES,)),
    FieldSpec("start", (FIELD_START_DATE,)),
    FieldSpec("end", (FIELD_END_DATE,)),
)

CSV_SCHEMA: Tuple[CsvColumn, ...] = (
    CsvColumn("filename"),
    CsvColumn("extraction_status"),
    CsvColumn("full_name", SECTION_BASIC_INFO, field=FIELD_NAME),
    CsvColumn("email",     SECTION_BASIC_INFO, field=FIELD_EMAIL),
    CsvColumn("phone",     SECTION_BASIC_INFO, field=FIELD_PHONE),
    CsvColumn("location",  SECTION_BASIC_INFO, field=FIELD_LOCATION),
    CsvColumn("urls",      SECTION_BASIC_INFO, field=FIELD_URLS),
    CsvColumn("work_experience", SECTION_WORK_EXPERIENCE, KIND_LIST_JSON, fields=_WORK_FIELDS),
    CsvColumn("education",       SECTION_EDUCATION,       KIND_LIST_JSON, fields=_EDUCATION_FIELDS),
    CsvColumn("projects",        SECTION_PROJECTS,        KIND_LIST_JSON, fields=_PROJECT_FIELDS),
    ## Filled by the runner: why a row is "partial" (cut-off output, dropped entries, ...)
    CsvColumn("warnings"),
    CsvColumn("error_message"),
)

CSV_COLUMNS: Tuple[str, ...] = tuple(col.name for col in CSV_SCHEMA)


# -- Validation -- #
def _section_shape(section: str, raw: str, label: str, problems: List[str]) -> Any:
    """Parse a template/example and return the per-entry object for `section`."""
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        problems.append(f"{section}: {label} is not valid JSON ({exc.msg})")
        return None
    if not isinstance(doc, dict) or list(doc) != [section]:
        problems.append(f"{section}: {label} must have exactly one top-level key '{section}'")
        return None
    body = doc[section]
    if section in LIST_SECTIONS:
        if not isinstance(body, list):
            problems.append(f"{section}: {label} must hold a JSON array")
            return None
        if not body:  # an empty example list is legal; nothing to check
            return {}
        if not all(isinstance(entry, dict) for entry in body):
            problems.append(f"{section}: {label} entries must be objects")
            return None
        return body
    if not isinstance(body, dict):
        problems.append(f"{section}: {label} must hold a JSON object")
        return None
    return body


def _has_path(obj: Any, path: Tuple[str, ...]) -> bool:
    for key in path:
        if not isinstance(obj, dict) or key not in obj:
            return False
        obj = obj[key]
    return True


def _keys_read(col: CsvColumn) -> set:
    """Top-level entry keys a column reads (derived fields read their source)."""
    if col.kind != KIND_LIST_JSON:
        return {col.field}
    return {spec.source if spec.derived else spec.path[0] for spec in col.fields}


def validate_csv_schema(schema: Tuple[CsvColumn, ...] = CSV_SCHEMA) -> List[str]:
    """Return one message per mismatch between CSV_SCHEMA and the prompts.

    Checks, per section:
      * the prompt TEMPLATE and EXAMPLE_OUTPUT parse as JSON with the right shape;
      * every column path (and every derived field's source key) exists in the
        template;
      * every template key is read by some column or listed in INTERNAL_KEYS
        (catches a key added to a prompt but never exported);
      * the example output uses exactly the template keys, so the one example
        the model imitates cannot teach it a stale or extra key.
    """
    problems: List[str] = []
    seen: set = set()
    read_by_section: Dict[str, set] = {}

    templates: Dict[str, Any] = {}
    for section, (template, example) in _SECTION_CONTRACTS.items():
        shape = _section_shape(section, template, "TEMPLATE", problems)
        if isinstance(shape, list):
            if len(shape) != 1:
                problems.append(f"{section}: TEMPLATE must show exactly one entry")
            shape = shape[0]
        templates[section] = shape
        example_shape = _section_shape(section, example, "EXAMPLE_OUTPUT", problems)
        if isinstance(shape, dict) and example_shape:
            for entry in example_shape if isinstance(example_shape, list) else [example_shape]:
                if set(entry) != set(shape):
                    extra = sorted(set(entry) - set(shape))
                    missing = sorted(set(shape) - set(entry))
                    problems.append(
                        f"{section}: EXAMPLE_OUTPUT keys differ from TEMPLATE "
                        f"(extra={extra}, missing={missing})"
                    )

    for col in schema:
        if col.name in seen:
            problems.append(f"{col.name}: duplicate column name")
        seen.add(col.name)

        if not col.section:
            continue
        if col.section not in _SECTION_CONTRACTS:
            problems.append(f"{col.name}: no prompt declares section '{col.section}'")
            continue
        template = templates.get(col.section)

        if col.kind == KIND_LIST_JSON:
            if col.section not in LIST_SECTIONS:
                problems.append(f"{col.name}: list_json but '{col.section}' is not a list section")
            if not col.fields:
                problems.append(f"{col.name}: list_json column has no fields")
            out_keys: set = set()
            paths = []
            for spec in col.fields:
                if spec.out_key in out_keys:
                    problems.append(f"{col.name}: duplicate out_key '{spec.out_key}'")
                out_keys.add(spec.out_key)
                if spec.derived:
                    if not spec.source:
                        problems.append(f"{col.name}: derived field '{spec.out_key}' has no source key")
                    else:
                        paths.append((spec.source,))
                else:
                    paths.append(spec.path)
        else:
            if col.section in LIST_SECTIONS:
                problems.append(f"{col.name}: scalar column on list section '{col.section}'")
            paths = [(col.field,)]

        read_by_section.setdefault(col.section, set()).update(_keys_read(col))
        if not isinstance(template, dict):
            continue  # template problem already reported
        for path in paths:
            if not _has_path(template, path):
                problems.append(
                    f"{col.name}: '{'.'.join(path)}' is not a key in the {col.section} TEMPLATE"
                )

    for section, template in templates.items():
        if not isinstance(template, dict):
            continue
        unread = set(template) - read_by_section.get(section, set()) - INTERNAL_KEYS.get(section, frozenset())
        for key in sorted(unread):
            problems.append(f"{section}: TEMPLATE key '{key}' is not read by any CSV column")
    return problems
