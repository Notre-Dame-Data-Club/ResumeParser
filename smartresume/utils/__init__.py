"""
Utilities module for common functionality
"""

from . import schema
from .config import config, Config
from .prompts import get_prompts
from .schema import (
    CSV_COLUMNS,
    CSV_SCHEMA,
    CsvColumn,
    FIELD_COMPANY,
    FIELD_DEGREE_LEVEL,
    FIELD_EDUCATION_DESCRIPTION,
    FIELD_EMAIL,
    FIELD_GPA,
    FIELD_LOCATION,
    FIELD_MAJOR,
    FIELD_NAME,
    FIELD_PHONE,
    FIELD_POSITION,
    FIELD_SCHOOL,
    LIST_SECTIONS,
    MULTI_VALUE_SEPARATOR,
    SECTION_BASIC_INFO,
    SECTION_EDUCATION,
    SECTION_WORK_EXPERIENCE,
    validate_csv_schema,
)

__all__ = [
    'config',
    'Config',
    'get_prompts',
    # Extraction contract / CSV schema (smartresume.utils.schema)
    'schema',
    'CsvColumn',
    'CSV_SCHEMA',
    'CSV_COLUMNS',
    'LIST_SECTIONS',
    'MULTI_VALUE_SEPARATOR',
    'validate_csv_schema',
    'SECTION_BASIC_INFO',
    'SECTION_WORK_EXPERIENCE',
    'SECTION_EDUCATION',
    'FIELD_NAME',
    'FIELD_EMAIL',
    'FIELD_PHONE',
    'FIELD_LOCATION',
    'FIELD_COMPANY',
    'FIELD_POSITION',
    'FIELD_SCHOOL',
    'FIELD_DEGREE_LEVEL',
    'FIELD_MAJOR',
    'FIELD_GPA',
    'FIELD_EDUCATION_DESCRIPTION',
]
