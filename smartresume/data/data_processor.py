#!/usr/bin/env python3
"""
Data processor
Responsible for processing and cleaning extracted resume data
"""
import re

from statistics import median
from typing import Dict, Any, List, Tuple
import tiktoken
import unicodedata
from smartresume.utils.config import config


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
            processed_result = self.process_resume_data(structure_output, text_lines)

            pages_info = self._extract_pages_info(processed_data)
            if pages_info:
                processed_result['pages'] = pages_info

            processed_result['metadata'] = {
                'text_lines_count': len(text_lines),
                'pages_count': len(processed_data)
            }

            return processed_result

        except Exception:
            return structure_output

    def process_resume_data(self, raw_data: Dict[str, Any], text_lines: List[str]) -> Dict[str, Any]:
        """
        Process raw resume data.

        Args:
            raw_data: Raw data
            text_lines: Text lines

        Returns:
            Dict: Processed data
        """
        try:
            processed_data = {}

            if 'basicInfo' in raw_data:
                processed_data['basicInfo'] = self._process_basic_info(raw_data['basicInfo'])

            if 'workExperience' in raw_data:
                processed_data['workExperience'] = self._process_work_experience(raw_data['workExperience'], text_lines)

            if 'education' in raw_data:
                processed_data['education'] = self._process_education(raw_data['education'], text_lines)

            if 'gpa' in raw_data or 'honors' in raw_data:
                processed_data['gpa'] = self._process_gpa_info(raw_data)

            self._validate_fields_in_text(processed_data, text_lines)

            return processed_data

        except Exception:
            return raw_data

    def _process_basic_info(self, basic_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process basic info"""
        processed = {}

        if 'name' in basic_info:
            processed['name'] = self._clean_text(basic_info['name'])

        if 'phoneNumber' in basic_info:
            processed['phoneNumber'] = self._clean_text(basic_info['phoneNumber'])

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

        for exp in work_exp:
            processed_exp = {}

            if 'companyName' in exp:
                processed_exp['companyName'] = self._clean_company_name(exp['companyName'])

            if 'position' in exp:
                processed_exp['position'] = self._clean_text(exp['position'])

            if 'employmentPeriod' in exp:
                processed_exp['employmentPeriod'] = self._process_time_period(exp['employmentPeriod'])

            if 'jobDescription_refer_index_range' in exp:
                processed_exp['jobDescription_refer_index_range'] = exp['jobDescription_refer_index_range']
                processed_exp['jobDescription'] = self._extract_description_from_range(
                    exp['jobDescription_refer_index_range'], text_lines, processed_exp['companyName'], processed_exp["position"]
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

        for edu in education:
            processed_edu = {}

            if 'school' in edu:
                processed_edu['school'] = self._clean_school_name(edu['school'])

            if 'major' in edu:
                processed_edu['major'] = self._clean_text(edu['major'])

            if 'degreeLevel' in edu:
                processed_edu['degreeLevel'] = self._clean_text(edu['degreeLevel'])

            if 'period' in edu:
                processed_edu['period'] = self._process_time_period(edu['period'])

            if 'gpa' in edu:
                processed_edu['gpa'] = self._clean_text(edu['gpa'])
                processed_edu['gpaNum'] = self._extract_gpa_number(edu['gpa'])

            for key, value in edu.items():
                if key not in processed_edu:
                    processed_edu[key] = self._clean_text(value) if isinstance(value, str) else value

            processed_list.append(processed_edu)

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
        if not index_range or len(index_range) != 2:
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

    def _normalize_date(self, date_str: str) -> str:
        """Normalize date format"""
        if not date_str:
            return ""

        date_str = str(date_str).strip()

        # Handle "to present" and similar markers
        if date_str in ['至今', '现在', '目前', 'present', 'now']:
            return 'present'

        date_patterns = [
            r'(\d{4})[年.-](\d{1,2})[月.-]?(\d{1,2})?',
            r'(\d{4})[年.-](\d{1,2})',
            r'(\d{4})',
        ]

        for pattern in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                year = match.group(1)
                month = match.group(2) if len(match.groups()) > 1 else None
                day = match.group(3) if len(match.groups()) > 2 else None

                if month:
                    month = month.zfill(2)
                    if day:
                        return f"{year}.{month}.{day.zfill(2)}"
                    else:
                        return f"{year}.{month}"
                else:
                    return year

        return date_str

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

    def _validate_fields_in_text(self, processed_data: Dict[str, Any], text_lines: List[str]) -> None:
        """
        Validate that key fields appear in the original text; remove otherwise.

        Args:
            processed_data: Processed data
            text_lines: Original text lines
        """
        full_text = ''.join(text_lines)
        normalized_full_text = self._normalize_for_comparison(full_text)

        if 'workExperience' in processed_data:
            valid_works = []
            for work in processed_data['workExperience']:
                company_name = work.get('companyName', '').strip()
                position = work.get('position', '').strip()

                normalized_company_name = self._normalize_for_comparison(company_name)
                normalized_position = self._normalize_for_comparison(position)

                if ((normalized_company_name and normalized_company_name in normalized_full_text) or
                        (normalized_position and normalized_position in normalized_full_text)):
                    valid_works.append(work)

            processed_data['workExperience'] = valid_works

        if 'education' in processed_data:
            valid_educations = []
            for edu in processed_data['education']:
                school = edu.get('school', '').strip()
                major = edu.get('major', '').strip()

                normalized_school = self._normalize_for_comparison(school)
                normalized_major = self._normalize_for_comparison(major)

                if ((normalized_school and normalized_school in normalized_full_text) or
                        (normalized_major and normalized_major in normalized_full_text)):
                    valid_educations.append(edu)

            processed_data['education'] = valid_educations
