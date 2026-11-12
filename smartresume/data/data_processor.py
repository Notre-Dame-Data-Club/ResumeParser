#!/usr/bin/env python3
"""
Data processor
Responsible for processing and cleaning extracted resume data
"""
import logging
import math
import re

from datetime import date
from statistics import median
from typing import Dict, Any, List, Optional, Tuple
import tiktoken
import unicodedata
from smartresume.utils.config import config

logger = logging.getLogger(__name__)

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_PRESENT_WORDS = ("present", "current", "currently", "now", "ongoing", "today", "in progress",
                  "至今", "现在", "目前")
_MONTH_YEAR = re.compile(r"\b([A-Za-z]{3,9})\.?,?\s+'?(\d{4})\b")
_NUMERIC_MONTH_YEAR = re.compile(r"\b(\d{1,2})\s*[/.\-]\s*(\d{4})\b")
_YEAR_MONTH = re.compile(r"\b(\d{4})\s*[/.\-年]\s*(\d{1,2})\b")
_YEAR = re.compile(r"\b(19\d{2}|20\d{2}|21\d{2})\b")
## A whole range written into one date field: "May-June 2026", "2020-2025", "2022 - Present"
_MONTH_RANGE = re.compile(r"^\s*([A-Za-z]{3,9})\.?\s*(?:-|–|—|to)\s*([A-Za-z]{3,9})\.?,?\s+'?(\d{4})\s*$")
_YEAR_RANGE = re.compile(r"^\s*((?:19|20|21)\d{2})\s*(?:-|–|—|to)\s*((?:19|20|21)\d{2}|[A-Za-z ]+)\s*$")
## Dates later than this many years ahead are loop artefacts ("2040", "2041", ...)
_MAX_YEARS_AHEAD = 8
_MIN_YEAR = 1950
## Share of an entry name's words that must appear in the resume for it to count as real
_NAME_WORD_COVERAGE = 0.75
## Values a model writes instead of leaving a field empty
_PLACEHOLDERS = {"unknown", "n/a", "na", "none", "null", "nil", "not specified", "not provided",
                 "not available", "not mentioned", "unspecified", "-", "--", "?"}
_PHONE_SPLIT = re.compile(r"\s*(?:/|,|;|\||\bor\b)\s*", re.IGNORECASE)
## A string that can be a link (has a dot or slash, no spaces): "LinkedIn" is a label, not a link
_URL_LIKE = re.compile(r"^(?:https?://)?[^\s/]+\.[^\s]+$|^https?://\S+$", re.IGNORECASE)
## PDF link annotations above this share of page 1's height count as profile links
_HEADER_LINK_ZONE = 0.2


class DataProcessor:
    """Data processor class"""

    def __init__(self):
        """Initialize data processor"""
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.pattern = r'[a-zA-Z0-9\-~_]{40,}'

    def should_remove(self, match: re.Match) -> bool:
        """
        Decide whether a matched text should be removed.

        Args:
            match: Regex match object

        Returns:
            bool: Whether it should be removed
        """
        encoded = self.encoding.encode(match.group(0))
        return len(encoded) > len(match.group(0)) * 0.5

    def build_text_content(self, processed_data: List[Dict[str, Any]]) -> Tuple[List[str], str, str]:
        """
        Build text content from processed data.

        Args:
            processed_data: Processed data

        Returns:
            Tuple[List[str], str, str]: (lines, plain text, indexed text)
        """
        text_lines = []

        for page_data in processed_data:
            if 'text' in page_data:
                if isinstance(page_data['text'], list):
                    text_lines.extend(self._extract_text_lines(page_data['text']))
                elif isinstance(page_data['text'], str) and page_data['text'].strip():
                    text_lines.append(page_data['text'].strip())

        text_content = '\n'.join(text_lines)

        indexed_text_content = self._build_indexed_content(text_lines)

        return text_lines, text_content, indexed_text_content

    def _extract_text_lines(self, text_items: List[Any]) -> List[str]:
        """Build readable text lines from OCR/PDF items."""
        positioned_items = []
        loose_lines: List[str] = []

        for text_item in text_items:
            if isinstance(text_item, dict) and 'text' in text_item:
                text = self._clean_text_content(text_item['text'])
                if not text:
                    continue

                bbox = text_item.get('bbox')
                if isinstance(bbox, list) and len(bbox) == 4:
                    positioned_items.append({
                        "text": text,
                        "bbox": bbox,
                    })
                else:
                    loose_lines.extend(self._split_text_lines(text))
            elif isinstance(text_item, str) and text_item.strip():
                loose_lines.append(text_item.strip())

        if not positioned_items:
            return loose_lines

        return self._merge_positioned_text_lines(positioned_items) + loose_lines

    def _merge_positioned_text_lines(self, text_items: List[Dict[str, Any]]) -> List[str]:
        """Merge word-level OCR/PDF tokens back into line-like strings."""
        if not text_items:
            return []

        heights = [
            max(float(item['bbox'][3]) - float(item['bbox'][1]), 1.0)
            for item in text_items
        ]
        y_tolerance = max(4.0, median(heights) * 0.6)

        positioned = sorted(
            text_items,
            key=lambda item: (
                (float(item['bbox'][1]) + float(item['bbox'][3])) / 2,
                float(item['bbox'][0]),
            ),
        )

        merged_lines: List[str] = []
        current_line: List[Dict[str, Any]] = []
        current_center_y = 0.0

        for item in positioned:
            center_y = (float(item['bbox'][1]) + float(item['bbox'][3])) / 2
            if not current_line:
                current_line = [item]
                current_center_y = center_y
                continue

            if abs(center_y - current_center_y) <= y_tolerance:
                current_line.append(item)
                current_center_y = (current_center_y + center_y) / 2
            else:
                merged_line = self._join_line_tokens(current_line)
                if merged_line:
                    merged_lines.append(merged_line)
                current_line = [item]
                current_center_y = center_y

        if current_line:
            merged_line = self._join_line_tokens(current_line)
            if merged_line:
                merged_lines.append(merged_line)

        return merged_lines

    def _join_line_tokens(self, line_items: List[Dict[str, Any]]) -> str:
        """Join left-to-right tokens into a single line."""
        ordered_items = sorted(line_items, key=lambda item: float(item['bbox'][0]))
        tokens = [item['text'].strip() for item in ordered_items if item['text'].strip()]
        return ' '.join(tokens)

    def _clean_text_content(self, text: str) -> str:
        """
        Clean text content.

        Args:
            text: Raw text

        Returns:
            str: Cleaned text
        """
        if not text:
            return ""

        text = unicodedata.normalize('NFKC', text)

        text = re.sub(r'[\u0020\u00A0\u1680\u2000-\u200A\u2028\u2029\u202F\u205F\u3000\u00A7]', ' ', text)

        text = re.sub(r' {2,}', ' ', text)

        text = re.sub(self.pattern, lambda m: '' if self.should_remove(m) else m.group(0), text)

        return text.strip()

    def _split_text_lines(self, text: str) -> List[str]:
        """
        Split text into lines.

        Args:
            text: Text content

        Returns:
            List[str]: List of lines
        """
        if "\n" in text:
            return [line.strip() for line in text.split("\n") if line.strip()]
        else:
            return [text.strip()] if text.strip() else []

    def _build_indexed_content(self, text_lines: List[str]) -> str:
        """
        Build indexed text content for LLM input.

        Args:
            text_lines: List of text lines

        Returns:
            str: Indexed text content
        """
        trans_table = str.maketrans('', '', '""\'\\')
        indexed_text_lines = [
            f"[{i}]:{line.translate(trans_table) if isinstance(line, str) else ''}"
            for i, line in enumerate(text_lines)
        ]
        return '\n'.join(indexed_text_lines)

    def post_process(self, text_lines: List[str], structure_output: Dict[str, Any], processed_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Post-process structured output.

        Args:
            text_lines: Text line list
            structure_output: Structured output
            processed_data: Processed data

        Returns:
            Dict: Final processed result
        """
        try:
            issues: List[str] = []
            processed_result = self.process_resume_data(structure_output, text_lines, issues)

            pages_info = self._extract_pages_info(processed_data)
            if pages_info:
                processed_result['pages'] = pages_info

            processed_result['metadata'] = {
                'text_lines_count': len(text_lines),
                'pages_count': len(processed_data),
                ## What post-processing removed or blanked, for the CSV "warnings" column
                'issues': issues,
            }

            return processed_result

        except Exception as e:
            logger.error("Post-processing failed, keeping raw model output: %s", e, exc_info=True)
            return structure_output

    def process_resume_data(self, raw_data: Dict[str, Any], text_lines: List[str],
                            issues: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process raw resume data.

        Args:
            raw_data: Raw data
            text_lines: Text lines
            issues: Collects a note for every entry dropped or value blanked

        Returns:
            Dict: Processed data
        """
        if issues is None:
            issues = []
        try:
            processed_data = {}

            raw_data = self._blank_placeholders(raw_data)

            if 'basicInfo' in raw_data:
                processed_data['basicInfo'] = self._process_basic_info(raw_data['basicInfo'])

            if 'workExperience' in raw_data:
                processed_data['workExperience'] = self._process_work_experience(raw_data['workExperience'], text_lines)

            if 'education' in raw_data:
                processed_data['education'] = self._process_education(raw_data['education'], text_lines)

            if 'projects' in raw_data:
                processed_data['projects'] = self._process_projects(raw_data['projects'], text_lines)

            self._validate_fields_in_text(processed_data, text_lines, issues)
            self._dedupe_entries(processed_data, issues)
            self._resolve_work_project_overlap(processed_data, issues)
            self._normalize_entry_dates(processed_data, issues)

            return processed_data

        except Exception as e:
            logger.error("Processing extracted data failed, keeping raw model output: %s", e, exc_info=True)
            issues.append(f"post-processing failed: {e}")
            return raw_data

    def _process_basic_info(self, basic_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process basic info"""
        processed = {}

        if 'name' in basic_info:
            processed['name'] = self._clean_person_name(basic_info['name'])

        if 'phoneNumber' in basic_info:
            processed['phoneNumber'] = self._clean_phone(basic_info['phoneNumber'])

        if 'URLs' in basic_info:
            processed['URLs'] = self._clean_urls(basic_info['URLs'])

        if 'personalEmail' in basic_info:
            processed['personalEmail'] = self._clean_email(basic_info['personalEmail'])

        if 'age' in basic_info:
            processed['age'] = self._clean_text(basic_info['age'])
            processed['ageNum'] = self._extract_age_number(basic_info['age'])

        for key, value in basic_info.items():
            if key not in processed:
                processed[key] = self._clean_text(value) if isinstance(value, str) else value

        return processed

    def _process_work_experience(self, work_exp: List[Dict[str, Any]], text_lines: List[str]) -> List[Dict[str, Any]]:
        """Process work experience"""
        processed_list = []

        for exp in work_exp if isinstance(work_exp, list) else []:
            if not isinstance(exp, dict):
                continue
            processed_exp = {}

            if 'companyName' in exp:
                processed_exp['companyName'] = self._clean_company_name(exp['companyName'])

            if 'position' in exp:
                processed_exp['position'] = self._clean_text(exp['position'])

            if 'employmentPeriod' in exp:
                processed_exp['employmentPeriod'] = self._process_time_period(exp['employmentPeriod'])

            if 'jobDescription_refer_index_range' in exp:
                index_range = self._coerce_index_range(exp['jobDescription_refer_index_range'])
                processed_exp['jobDescription_refer_index_range'] = index_range
                processed_exp['jobDescription'] = self._extract_description_from_range(
                    index_range, text_lines,
                    processed_exp.get('companyName', ''), processed_exp.get('position', '')
                )
            elif 'jobDescription' in exp:
                processed_exp['jobDescription'] = self._clean_description(exp['jobDescription'])

            for key, value in exp.items():
                if key not in processed_exp:
                    processed_exp[key] = self._clean_text(value) if isinstance(value, str) else value

            processed_list.append(processed_exp)

        return processed_list

    def _process_education(self, education: List[Dict[str, Any]], text_lines: List[str]) -> List[Dict[str, Any]]:
        """Process education"""
        processed_list = []

        for edu in education if isinstance(education, list) else []:
            if not isinstance(edu, dict):
                continue
            processed_edu = {}

            if 'school' in edu:
                processed_edu['school'] = self._clean_school_name(edu['school'])

            if 'major' in edu:
                processed_edu['major'] = self._clean_text(edu['major'])

            if 'degreeLevel' in edu:
                processed_edu['degreeLevel'] = self._clean_text(edu['degreeLevel'])

            if 'period' in edu:
                processed_edu['period'] = self._process_time_period(edu['period'])

            # Prompt emits "GPA"; keep accepting legacy lowercase "gpa".
            gpa_key = 'GPA' if 'GPA' in edu else ('gpa' if 'gpa' in edu else None)
            if gpa_key:
                processed_edu['GPA'] = self._clean_text(edu[gpa_key])
                processed_edu['gpaNum'] = self._extract_gpa_number(edu[gpa_key])

            for key, value in edu.items():
                if key not in processed_edu and key != 'gpa':
                    processed_edu[key] = self._clean_text(value) if isinstance(value, str) else value

            processed_list.append(processed_edu)

        return processed_list

    def _process_projects(self, projects: List[Dict[str, Any]], text_lines: List[str]) -> List[Dict[str, Any]]:
        """Process projects: clean strings, resolve description from its index range."""
        if not isinstance(projects, list):
            return []
        processed_list = []
        for proj in projects:
            if not isinstance(proj, dict):
                continue
            processed = {
                key: self._clean_text(value) if isinstance(value, str) else value
                for key, value in proj.items()
            }
            if 'description_refer_index_range' in proj:
                index_range = self._coerce_index_range(proj['description_refer_index_range'])
                processed['description_refer_index_range'] = index_range
                valid = (
                    isinstance(index_range, list) and len(index_range) == 2
                    and all(isinstance(i, int) for i in index_range)
                )
                processed['description'] = self._extract_description_from_range(
                    index_range, text_lines, processed.get('name', ''), ''
                ) if valid else ''
            elif 'description' in proj:
                processed['description'] = self._clean_description(proj['description'])
            processed_list.append(processed)
        return processed_list

    def _clean_email(self, email: str) -> str:
        """Clean email address"""
        if not email:
            return ""

        email = str(email).strip().lower()

        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(email_pattern, email):
            return email

        return email

    def _clean_company_name(self, company: str) -> str:
        """Clean company name"""
        if not company:
            return ""

        company = str(company).strip()

        suffixes = ['有限公司', '股份有限公司', '科技有限公司', '网络科技有限公司']
        for suffix in suffixes:
            if company.count(suffix) > 1:
                company = company.replace(suffix, '', company.count(suffix) - 1)

        return company

    def _clean_school_name(self, school: str) -> str:
        """Clean school name"""
        if not school:
            return ""

        school = str(school).strip()

        school = re.sub(r'\([^)]*\)', '', school)
        school = re.sub(r'（[^）]*）', '', school)

        return school.strip()

    def _clean_text(self, text: str) -> str:
        """Clean general text"""
        if not text:
            return ""

        text = str(text).strip()

        text = re.sub(r'\s+', ' ', text)

        return text

    def _extract_age_number(self, age_text: str) -> int:
        """Extract age number from string"""
        if not age_text:
            return -1

        age_text = str(age_text).strip()

        age_pattern = r'(\d+)'
        match = re.search(age_pattern, age_text)

        if match:
            try:
                age = int(match.group(1))
                # Validate reasonable age range (16-99)
                if 16 <= age <= 99:
                    return age
            except ValueError:
                pass

        return -1

    def _extract_gpa_number(self, gpa_text: str) -> float:
        """Extract the smallest number from GPA text as GPA value"""
        if not gpa_text:
            return -1.0

        gpa_text = str(gpa_text).strip()

        gpa_pattern = r'(\d+\.?\d*)'
        matches = re.findall(gpa_pattern, gpa_text)

        if matches:
            try:
                # Convert to float and pick the minimum
                numbers = [float(match) for match in matches]
                min_gpa = min(numbers)
                # Validate reasonable GPA range (0.0-5.0)
                if 0.0 <= min_gpa <= 5.0:
                    return min_gpa
            except ValueError:
                pass

        return -1.0

    @staticmethod
    def _coerce_index_range(index_range: Any) -> Any:
        """
        [first, last] from what the model wrote for a line range.

        It sometimes lists every line ([41, 42, 43, 44, 45]) or a single one ([7]);
        both name a usable range. Anything else is returned unchanged and resolves
        to an empty description.
        """
        if not isinstance(index_range, (list, tuple)) or not index_range:
            return index_range
        if not all(isinstance(i, int) and not isinstance(i, bool) for i in index_range):
            return index_range
        if len(index_range) == 2:
            return list(index_range)
        if list(index_range) != sorted(index_range):
            return index_range
        return [index_range[0], index_range[-1]]

    @staticmethod
    def _split_date_range(text: str) -> Optional[Tuple[str, str]]:
        """("May 2026", "June 2026") for "May-June 2026"; ("2020", "2025") for "2020-2025"."""
        match = _MONTH_RANGE.match(text)
        if match and all(_MONTHS.get(m.lower()[:4] if m.lower().startswith("sept") else m.lower()[:3])
                         for m in match.group(1, 2)):
            return f"{match.group(1)} {match.group(3)}", f"{match.group(2)} {match.group(3)}"
        match = _YEAR_RANGE.match(text)
        if match:
            return match.group(1), match.group(2).strip()
        return None

    def _extract_description_from_range(self, index_range: List[int], text_lines: List[str], name: str, position: str) -> str:
        """
        Extract description from original text based on index range.

        Args:
            index_range: Index range [start, end]
            text_lines: List of text lines
            name: Name
            position: Position

        Returns:
            Extracted description text
        """
        if not isinstance(index_range, (list, tuple)) or len(index_range) != 2:
            return ""
        if not all(isinstance(i, int) and not isinstance(i, bool) for i in index_range):
            return ""

        start_idx, end_idx = index_range

        if start_idx < 0 or end_idx >= len(text_lines) or start_idx > end_idx:
            return ""

        try:
            extracted_lines = text_lines[start_idx:end_idx + 1]

            if config.processing.remove_position_and_company_line:
                normalized_name = self._normalize_unicode(name)
                normalized_position = self._normalize_unicode(position)
                extracted_lines = [line for line in extracted_lines if normalized_name != self._normalize_unicode(line) or normalized_position != self._normalize_unicode(line)]
                extracted_lines = [line for line in extracted_lines if not (normalized_name in self._normalize_unicode(line) and normalized_position in self._normalize_unicode(line))]

            if len(extracted_lines) == 0:
                return ""
            else:
                description = '\n'.join(line.strip() for line in extracted_lines if line.strip())

                return description

        except Exception:
            return ""

    def _clean_description(self, description: str, text_lines: List[str] = None) -> str:
        """Clean description text"""
        if not description:
            return ""

        description = str(description).strip()

        lines = description.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if line:
                line = re.sub(r'\s+', ' ', line)
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _process_time_period(self, period: Dict[str, Any]) -> Dict[str, Any]:
        """Process time period"""
        if not period:
            return {}

        processed_period = {}

        if 'startDate' in period:
            processed_period['startDate'] = self._normalize_date(period['startDate'])

        if 'endDate' in period:
            processed_period['endDate'] = self._normalize_date(period['endDate'])

        return processed_period

    def _normalize_date(self, date_str: Any) -> str:
        """
        Normalize one date to "YYYY-MM", "YYYY" or "Present".

        Handles "Jun 2024", "June, 2024", "Expected May 2029", "06/2024", "2024-06",
        "2024年6月", "Summer 2025" (year only) and present markers. Text with no
        recognisable date is returned unchanged.
        """
        if date_str is None:
            return ""
        text = str(date_str).strip()
        if not text:
            return ""

        lowered = text.lower()
        if any(re.search(rf"\b{re.escape(w)}\b", lowered) if w.isascii() else w in lowered
               for w in _PRESENT_WORDS) and not _YEAR.search(text):
            return "Present"

        for match in _MONTH_YEAR.finditer(text):
            word = match.group(1).lower()
            month = _MONTHS.get(word[:4]) if word.startswith("sept") else _MONTHS.get(word[:3])
            if month:
                return f"{match.group(2)}-{month:02d}"

        match = _YEAR_MONTH.search(text)
        if match and 1 <= int(match.group(2)) <= 12:
            return f"{match.group(1)}-{int(match.group(2)):02d}"

        match = _NUMERIC_MONTH_YEAR.search(text)
        if match and 1 <= int(match.group(1)) <= 12:
            return f"{match.group(2)}-{int(match.group(1)):02d}"

        match = _YEAR.search(text)
        if match:
            return match.group(1)

        return text

    @staticmethod
    def _date_year(value: str) -> Optional[int]:
        match = re.match(r"^(\d{4})", value or "")
        return int(match.group(1)) if match else None

    def _normalize_entry_dates(self, processed_data: Dict[str, Any], issues: List[str]) -> None:
        """
        Normalize startDate/endDate in every list section and blank impossible years.

        A range written into one field ("Jun 2024 - Aug 2024" as startDate) is split.
        """
        max_year = date.today().year + _MAX_YEARS_AHEAD
        for section in ('workExperience', 'education', 'projects'):
            for entry in processed_data.get(section) or []:
                if not isinstance(entry, dict):
                    continue
                start = str(entry.get('startDate') or '')
                end = str(entry.get('endDate') or '')
                ## The same range copied into both fields, or left whole in one of them
                if start and end in ("", start) and self._split_date_range(start):
                    start, end = self._split_date_range(start)
                if start and not end:
                    parts = re.split(r"\s*(?:\s-\s|–|—|\sto\s)\s*", start, maxsplit=1)
                    if len(parts) == 2:
                        start, end = parts
                    elif self._normalize_date(start) == "Present":
                        start, end = "", start  # "(Current)" written as the start date
                for key, raw in (('startDate', start), ('endDate', end)):
                    if key not in entry and not raw:
                        continue
                    value = self._normalize_date(raw)
                    year = self._date_year(value)
                    if year is not None and not (_MIN_YEAR <= year <= max_year):
                        issues.append(f"{section}: blanked implausible {key} '{raw}'")
                        value = ""
                    entry[key] = value

    def _extract_pages_info(self, processed_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract page metadata from processed data"""
        pages_info = []

        for page_num, page_data in enumerate(processed_data):
            page_info = {
                'page_number': page_num + 1,
                'has_text': 'text' in page_data and bool(page_data['text']),
                'has_image': 'image' in page_data
            }

            pages_info.append(page_info)

        return pages_info

    def _normalize_for_comparison(self, text: str) -> str:
        """
        Normalize text for comparison: remove special chars and spaces, lowercase.

        Args:
            text: Text to normalize

        Returns:
            str: Normalized text
        """
        if not text:
            return ""

        normalized = str(text).lower()
        normalized = re.sub(r'[^\w]', '', normalized)
        return normalized

    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode text for comparison.

        Args:
            text: Text to normalize

        Returns:
            str: Normalized text
        """
        return unicodedata.normalize('NFKC', text).strip()

    def _appears_in_text(self, value: str, normalized_full_text: str, text_words: set) -> bool:
        """
        True when value is in the resume: verbatim (ignoring case, spaces and
        punctuation), or with at least 75% of its words present ("Univ. of Notre
        Dame" vs "University of Notre Dame" still passes; "Lakeview Robotics Club"
        does not).
        """
        normalized = self._normalize_for_comparison(value)
        if not normalized:
            return False
        if normalized in normalized_full_text:
            return True
        words = [w for w in re.findall(r"\w+", unicodedata.normalize('NFKC', value).lower()) if len(w) >= 2]
        if not words:
            return False
        found = sum(1 for w in words if w in text_words)
        return found >= max(1, math.ceil(len(words) * _NAME_WORD_COVERAGE))

    def _validate_fields_in_text(self, processed_data: Dict[str, Any], text_lines: List[str],
                                 issues: Optional[List[str]] = None) -> None:
        """
        Drop entries whose name is not in the resume (copied from the prompt's
        example or invented).

        The entry's identifying field must appear in the text: the company for a job
        (the position only when there is no company - a common title like "President"
        would otherwise vouch for an invented club), the school for education (the
        major only when there is no school) and the name for a project.
        """
        if issues is None:
            issues = []
        full_text = '\n'.join(text_lines)
        normalized_full_text = self._normalize_for_comparison(full_text)
        text_words = set(re.findall(r"\w+", unicodedata.normalize('NFKC', full_text).lower()))

        checks = (
            ('workExperience', 'companyName', 'position'),
            ('education', 'school', 'major'),
            ('projects', 'name', None),
        )
        for section, primary, fallback in checks:
            if section not in processed_data:
                continue
            kept = []
            for entry in processed_data[section]:
                main_value = str(entry.get(primary) or '').strip()
                backup_value = str(entry.get(fallback) or '').strip() if fallback else ''
                if main_value:
                    valid = self._appears_in_text(main_value, normalized_full_text, text_words)
                    label = main_value
                else:
                    valid = bool(backup_value) and self._appears_in_text(
                        backup_value, normalized_full_text, text_words)
                    label = backup_value or '(empty entry)'
                if valid:
                    kept.append(entry)
                else:
                    issues.append(f"{section}: dropped '{label}' (not found in resume)")
            processed_data[section] = kept

    def _entry_identity(self, section: str, entry: Dict[str, Any]) -> Tuple[str, ...]:
        """What makes two entries the same, ignoring dates (a loop often only changes those)."""
        fields = {
            'workExperience': ('companyName', 'position'),
            'education': ('school', 'degreeLevel', 'major'),
            'projects': ('name',),
        }[section]
        return tuple(self._normalize_for_comparison(str(entry.get(f) or '')) for f in fields)

    _DESCRIPTION_FIELDS = {
        'workExperience': ('jobDescription', 'jobDescription_refer_index_range'),
        'projects': ('description', 'description_refer_index_range'),
        'education': ('educationDescription', None),
    }

    def _merge_into(self, section: str, kept: Dict[str, Any], dup: Dict[str, Any]) -> None:
        """
        Fold a duplicate into the entry kept: a model sometimes writes one job as
        several entries, one per group of bullets, so their descriptions and
        technologies belong together.
        """
        text_key, range_key = self._DESCRIPTION_FIELDS[section]
        kept_text, dup_text = str(kept.get(text_key) or ''), str(dup.get(text_key) or '')
        if dup_text and dup_text not in kept_text:
            kept[text_key] = f"{kept_text}\n{dup_text}" if kept_text else dup_text
        if range_key:
            ranges = [r for r in (kept.get(range_key), dup.get(range_key))
                      if isinstance(r, list) and len(r) == 2 and all(isinstance(i, int) for i in r)]
            if ranges:
                kept[range_key] = [min(r[0] for r in ranges), max(r[1] for r in ranges)]
        techs = list(kept.get('technologies') or [])
        for tech in dup.get('technologies') or []:
            if tech not in techs:
                techs.append(tech)
        if 'technologies' in kept or techs:
            kept['technologies'] = techs
        for key, value in dup.items():
            if value not in ('', [], None) and kept.get(key) in ('', [], None):
                kept[key] = value

    def _dedupe_entries(self, processed_data: Dict[str, Any], issues: List[str]) -> None:
        """Merge entries that are identical apart from their dates and descriptions."""
        for section in ('workExperience', 'education', 'projects'):
            entries = processed_data.get(section)
            if not entries:
                continue
            first: Dict[Tuple[str, ...], Dict[str, Any]] = {}
            kept, merged = [], 0
            for entry in entries:
                identity = self._entry_identity(section, entry)
                if any(identity) and identity in first:
                    self._merge_into(section, first[identity], entry)
                    merged += 1
                    continue
                first[identity] = entry
                kept.append(entry)
            if merged:
                issues.append(f"{section}: merged {merged} duplicate entr{'y' if merged == 1 else 'ies'}")
            processed_data[section] = kept

    def _resolve_work_project_overlap(self, processed_data: Dict[str, Any], issues: List[str]) -> None:
        """
        One item extracted as both a job and a project (a mixed "EXPERIENCE & PROJECTS"
        section is read by both prompts) is kept once: as a project when the job's
        position calls it a project ("Academic Project"), otherwise as a job.
        """
        works = processed_data.get('workExperience') or []
        projects = processed_data.get('projects') or []
        if not works or not projects:
            return

        drop_work, drop_project = set(), set()
        for p_idx, project in enumerate(projects):
            name = self._normalize_for_comparison(str(project.get('name') or ''))
            if not name:
                continue
            for w_idx, work in enumerate(works):
                company = self._normalize_for_comparison(str(work.get('companyName') or ''))
                position = self._normalize_for_comparison(str(work.get('position') or ''))
                ## Exact matches only: "<Club> Data Project" is a real project of <Club>
                if name not in {company, position, company + position, position + company} - {""}:
                    continue
                if 'project' in position:
                    drop_work.add(w_idx)
                else:
                    drop_project.add(p_idx)
                break

        for idx in sorted(drop_work):
            issues.append(f"workExperience: moved '{works[idx].get('companyName', '')}' to projects only")
        for idx in sorted(drop_project):
            issues.append(f"projects: dropped '{projects[idx].get('name', '')}' (already a job)")
        processed_data['workExperience'] = [w for i, w in enumerate(works) if i not in drop_work]
        processed_data['projects'] = [p for i, p in enumerate(projects) if i not in drop_project]

    # -- basicInfo cleaners -- #
    def _clean_person_name(self, name: Any) -> str:
        """Collapse spaces; an ALL CAPS name becomes Title Case ("BENJAMIN D. KIM" -> "Benjamin D. Kim")."""
        name = self._clean_text(name)
        letters = [ch for ch in name if ch.isalpha()]
        if letters and all(ch.isupper() for ch in letters):
            name = re.sub(r"[^\W\d_]+", lambda m: m.group(0).capitalize(), name.lower())
        return name

    def _clean_phone(self, phone: Any) -> str:
        """
        Digits only, with a leading "+" for numbers that carry a country code
        ("1-(779)-875-4805" -> "+17798754805", "(908) 873-7124" -> "9088737124").
        Several numbers ("+44 7565 009271 / +1 (574) 400-3972") are kept apart,
        joined with ", ". Text without 7+ digits is returned cleaned but unchanged.
        """
        text = self._clean_text(phone)
        numbers = []
        for part in _PHONE_SPLIT.split(text):
            digits = re.sub(r"\D", "", part)
            if len(digits) < 7:
                continue
            numbers.append("+" + digits if part.strip().startswith("+") or len(digits) > 10 else digits)
        return ", ".join(dict.fromkeys(numbers)) if numbers else text

    @staticmethod
    def _url_key(url: str) -> str:
        return re.sub(r"^(?:https?://)?(?:www\.)?", "", url.strip().lower()).rstrip("/")

    @classmethod
    def _clean_urls(cls, urls: Any) -> List[str]:
        """
        Strip and de-duplicate links (ignoring scheme, "www." and a trailing "/").
        Link labels ("LinkedIn", "GitHub") are dropped: the PDF text often shows
        only the label, and the address comes from the PDF's link annotations.
        """
        if isinstance(urls, str):
            urls = [u for u in re.split(r"[,\s]+", urls) if u]
        if not isinstance(urls, list):
            return []
        seen, kept = set(), []
        for url in urls:
            url = str(url).strip()
            key = cls._url_key(url)
            if url and _URL_LIKE.match(url) and key not in seen:
                seen.add(key)
                kept.append(url)
        return kept

    def merge_pdf_links(self, result: Dict[str, Any], pdf_path: str) -> None:
        """
        Add the profile links of a PDF (link annotations in the top of page 1, e.g.
        a "LinkedIn" label linking to the real profile) to basicInfo.URLs. A link
        the model wrote is replaced by the PDF's full address for the same page.
        """
        basic = result.get('basicInfo')
        if not isinstance(basic, dict) or not str(pdf_path).lower().endswith('.pdf'):
            return
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                if not pdf.pages:
                    return
                page = pdf.pages[0]
                limit = float(page.height) * _HEADER_LINK_ZONE
                links = [str(h.get('uri') or '').strip() for h in page.hyperlinks
                         if float(h.get('top', page.height)) <= limit]
        except Exception as e:
            logger.warning("Could not read PDF links from %s: %s", pdf_path, e)
            return
        links = [u for u in links if u.lower().startswith(('http://', 'https://'))]
        if not links:
            return
        merged = self._clean_urls(links)
        keys = {self._url_key(u) for u in merged}
        merged += [u for u in self._clean_urls(basic.get('URLs') or []) if self._url_key(u) not in keys]
        basic['URLs'] = merged

    @classmethod
    def _blank_placeholders(cls, value: Any) -> Any:
        """Turn "Unknown", "N/A", ... into "" (and drop them from lists)."""
        if isinstance(value, str):
            return "" if value.strip().lower() in _PLACEHOLDERS else value
        if isinstance(value, dict):
            return {key: cls._blank_placeholders(item) for key, item in value.items()}
        if isinstance(value, list):
            cleaned = [cls._blank_placeholders(item) for item in value]
            return [item for item in cleaned if item != ""]
        return value
