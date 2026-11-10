"""
Extraction contract and CSV schema.

prompts.py instructs the model to emit a specific set of JSON keys; this module
names those keys once and maps them onto CSV columns.  Consumers derive their
column list and their row flattening from CSV_SCHEMA instead of hardcoding
field names, so a prompt key and the CSV column reading it cannot drift apart.

Cell encoding (rows are scored by an LLM for a QLoRA dataset):
  * scalar column      -> plain string.
  * list_json column   -> JSON array, one element per resume entry (job, degree,
                          project).  Each element is a JSON object built from the
                          column's FieldSpecs, so one entry stays one coherent
                          record.  Blank entries are kept, never dropped.
Cells are written with json.dumps, never str(), so every JSON cell parses.

Kept separate from prompts.py so that file stays prompt text only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .prompts import (
    BASIC_INFO_PROMPT,
    EDUCATION_PROMPT,
    PROJECTS_PROMPT,
    WORK_EXPERIENCE_PROMPT,
)

# -- Sections (top-level JSON keys) -- #
SECTION_BASIC_INFO = "basicInfo"
SECTION_WORK_EXPERIENCE = "workExperience"
SECTION_EDUCATION = "education"
SECTION_PROJECTS = "projects"

# -- Field keys: must equal the keys in prompts.py -- #
# basicInfo
FIELD_NAME = "name"
FIELD_EMAIL = "personalEmail"
FIELD_PHONE = "phoneNumber"
FIELD_LOCATION = "currentLocation"
FIELD_URLS = "URLs"

# workExperience
FIELD_COMPANY = "companyName"
FIELD_POSITION = "position"
FIELD_EMPLOYMENT_PERIOD = "employmentPeriod"
FIELD_INTERNSHIP = "internship"
FIELD_LEADERSHIP = "isLeadership"
FIELD_JOB_TECHNOLOGIES = "technologies"
# INTERNAL: the model returns line indexes instead of re-typing the text.
# Never exported to CSV; data_processor resolves it into FIELD_JOB_DESCRIPTION.
INTERNAL_JOB_DESCRIPTION_RANGE_KEY = "jobDescription_refer_index_range"
# Not a prompt key: data_processor resolves the index range into this text.
FIELD_JOB_DESCRIPTION = "jobDescription"

# shared period keys (employmentPeriod / education.period)
FIELD_START_DATE = "startDate"
FIELD_END_DATE = "endDate"

# education
FIELD_SCHOOL = "school"
FIELD_DEGREE_LEVEL = "degreeLevel"
FIELD_PERIOD = "period"
FIELD_DEPARTMENT = "department"
FIELD_MAJOR = "major"
FIELD_GPA = "GPA"
FIELD_EDUCATION_DESCRIPTION = "educationDescription"

# projects
FIELD_PROJECT_NAME = "name"
# INTERNAL: line-index range; data_processor resolves it into FIELD_PROJECT_DESCRIPTION.
INTERNAL_PROJECT_DESCRIPTION_RANGE_KEY = "description_refer_index_range"
# Not a prompt key (derived), like FIELD_JOB_DESCRIPTION.
FIELD_PROJECT_DESCRIPTION = "description"
FIELD_PROJECT_TECHNOLOGIES = "technologies"
FIELD_PROJECT_START_DATE = "startDate"
FIELD_PROJECT_END_DATE = "endDate"

LIST_SECTIONS = frozenset(
    {SECTION_WORK_EXPERIENCE, SECTION_EDUCATION, SECTION_PROJECTS}
)

# Validate schema section check
_SECTION_PROMPTS: Dict[str, str] = {
    SECTION_BASIC_INFO: BASIC_INFO_PROMPT,
    SECTION_WORK_EXPERIENCE: WORK_EXPERIENCE_PROMPT,
    SECTION_EDUCATION: EDUCATION_PROMPT,
    SECTION_PROJECTS: PROJECTS_PROMPT,
}

KIND_SCALAR = "scalar"
KIND_LIST_JSON = "list_json"


# -- Schema -- #
@dataclass(frozen=True)
class FieldSpec:
    """
    One key inside a list_json entry object.

    out_key  key name in the CSV JSON object
    path     keys to walk inside the extracted entry, e.g.
             ("employmentPeriod", "startDate")
    derived  True when the value is produced by post-processing rather than
             emitted by the prompt (skipped by validate_csv_schema)
    """

    out_key: str
    path: Tuple[str, ...]
    derived: bool = False


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
    FieldSpec("start", (FIELD_EMPLOYMENT_PERIOD, FIELD_START_DATE)),
    FieldSpec("end", (FIELD_EMPLOYMENT_PERIOD, FIELD_END_DATE)),
    FieldSpec("internship", (FIELD_INTERNSHIP,)),
    FieldSpec("leadership", (FIELD_LEADERSHIP,)),
    FieldSpec("technologies", (FIELD_JOB_TECHNOLOGIES,)),
    FieldSpec("description", (FIELD_JOB_DESCRIPTION,), derived=True),
)

_EDUCATION_FIELDS = (
    FieldSpec("school", (FIELD_SCHOOL,)),
    FieldSpec("degree", (FIELD_DEGREE_LEVEL,)),
    FieldSpec("department", (FIELD_DEPARTMENT,)),
    FieldSpec("major", (FIELD_MAJOR,)),
    FieldSpec("start", (FIELD_PERIOD, FIELD_START_DATE)),
    FieldSpec("end", (FIELD_PERIOD, FIELD_END_DATE)),
    FieldSpec("gpa", (FIELD_GPA,)),
    FieldSpec("description", (FIELD_EDUCATION_DESCRIPTION,)),
)

_PROJECT_FIELDS = (
    FieldSpec("name", (FIELD_PROJECT_NAME,)),
    FieldSpec("description", (FIELD_PROJECT_DESCRIPTION,), derived=True),
    FieldSpec("technologies", (FIELD_PROJECT_TECHNOLOGIES,)),
    FieldSpec("start", (FIELD_PROJECT_START_DATE,)),
    FieldSpec("end", (FIELD_PROJECT_END_DATE,)),
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
    CsvColumn("error_message"),
)

CSV_COLUMNS: Tuple[str, ...] = tuple(col.name for col in CSV_SCHEMA)


def validate_csv_schema(schema: Tuple[CsvColumn, ...] = CSV_SCHEMA) -> List[str]:
    """Return one message per column whose field no longer matches a prompt.

    Catches the drift this module exists to prevent: renaming a JSON key in a
    prompt without updating CSV_SCHEMA (or the reverse) would otherwise just
    produce a silently empty CSV column.  Nested paths are checked key by key;
    derived fields are skipped.
    """
    problems: List[str] = []
    seen: set = set()
    for col in schema:
        if col.name in seen:
            problems.append(f"{col.name}: duplicate column name")
        seen.add(col.name)

        if not col.section:
            continue
        prompt = _SECTION_PROMPTS.get(col.section)
        if prompt is None:
            problems.append(f"{col.name}: no prompt declares section '{col.section}'")
            continue

        if col.kind == KIND_LIST_JSON:
            if col.section not in LIST_SECTIONS:
                problems.append(f"{col.name}: list_json but '{col.section}' is not a list section")
            if not col.fields:
                problems.append(f"{col.name}: list_json column has no fields")
            keys = []
            out_keys = set()
            for spec in col.fields:
                if spec.out_key in out_keys:
                    problems.append(f"{col.name}: duplicate out_key '{spec.out_key}'")
                out_keys.add(spec.out_key)
                if not spec.derived:
                    keys.extend(spec.path)
        else:
            if col.section in LIST_SECTIONS:
                problems.append(f"{col.name}: scalar column on list section '{col.section}'")
            keys = [col.field]

        for key in keys:
            if f'"{key}"' not in prompt:
                problems.append(
                    f"{col.name}: '{key}' is not a key in the {col.section} prompt"
                )
    return problems
