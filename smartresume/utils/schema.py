"""
Extraction contract and CSV schema.

prompts.py instructs the model to emit a specific set of JSON keys; this module
names those keys once and maps them onto CSV columns.  Consumers derive their
column list and their row flattening from CSV_SCHEMA instead of hardcoding
field names, so a prompt key and the CSV column reading it cannot drift apart.

Kept separate from prompts.py so that file stays prompt text only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .prompts import BASIC_INFO_PROMPT, EDUCATION_PROMPT, WORK_EXPERIENCE_PROMPT

# -- Fields included in CSV -- #
SECTION_BASIC_INFO = "basicInfo"
SECTION_WORK_EXPERIENCE = "workExperience"
SECTION_EDUCATION = "education"

# basicInfo
FIELD_NAME = "name"
FIELD_EMAIL = "personalEmail"
FIELD_PHONE = "phoneNumber"
FIELD_LOCATION = "currentLocation"

# workExperience
FIELD_COMPANY = "companyName"
FIELD_POSITION = "position"

# education
FIELD_SCHOOL = "school"
FIELD_DEGREE_LEVEL = "degreeLevel"
FIELD_MAJOR = "major"
FIELD_GPA = "GPA"
FIELD_EDUCATION_DESCRIPTION = "educationDescription"

LIST_SECTIONS = frozenset({SECTION_WORK_EXPERIENCE, SECTION_EDUCATION})

MULTI_VALUE_SEPARATOR = "; "

# Validate schema section check
_SECTION_PROMPTS: Dict[str, str] = {
    SECTION_BASIC_INFO: BASIC_INFO_PROMPT,
    SECTION_WORK_EXPERIENCE: WORK_EXPERIENCE_PROMPT,
    SECTION_EDUCATION: EDUCATION_PROMPT,
}



# -- Schema -- #
@dataclass(frozen=True)
class CsvColumn:
    """
    CSV Column definitions
    Column Name | Section Var | Field Var | 
    """

    name: str
    section: str = ""
    field: str = ""


CSV_SCHEMA: Tuple[CsvColumn, ...] = (
    CsvColumn("filename"),
    CsvColumn("extraction_status"),
    CsvColumn("full_name",         SECTION_BASIC_INFO,      FIELD_NAME),
    CsvColumn("email",             SECTION_BASIC_INFO,      FIELD_EMAIL),
    CsvColumn("phone",             SECTION_BASIC_INFO,      FIELD_PHONE),
    CsvColumn("location",          SECTION_BASIC_INFO,      FIELD_LOCATION),
    CsvColumn("work_companies",    SECTION_WORK_EXPERIENCE, FIELD_COMPANY),
    CsvColumn("positions",         SECTION_WORK_EXPERIENCE, FIELD_POSITION),
    CsvColumn("education_schools", SECTION_EDUCATION,       FIELD_SCHOOL),
    CsvColumn("education_degrees", SECTION_EDUCATION,       FIELD_DEGREE_LEVEL),
    CsvColumn("majors",            SECTION_EDUCATION,       FIELD_MAJOR),
    CsvColumn("gpa",               SECTION_EDUCATION,       FIELD_GPA),
    CsvColumn("honors",            SECTION_EDUCATION,       FIELD_EDUCATION_DESCRIPTION),
    CsvColumn("error_message"),
)

CSV_COLUMNS: Tuple[str, ...] = tuple(col.name for col in CSV_SCHEMA)


def validate_csv_schema(schema: Tuple[CsvColumn, ...] = CSV_SCHEMA) -> List[str]:
    """Return one message per column whose field no longer matches a prompt.

    Catches the drift this module exists to prevent: renaming a JSON key in a
    prompt without updating CSV_SCHEMA (or the reverse) would otherwise just
    produce a silently empty CSV column.
    """
    problems: List[str] = []
    for col in schema:
        if not col.section:
            continue
        prompt = _SECTION_PROMPTS.get(col.section)
        if prompt is None:
            problems.append(f"{col.name}: no prompt declares section '{col.section}'")
        elif f'"{col.field}"' not in prompt:
            problems.append(
                f"{col.name}: '{col.field}' is not a key in the {col.section} prompt"
            )
    return problems
