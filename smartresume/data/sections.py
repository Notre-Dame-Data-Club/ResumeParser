"""
Split indexed resume text into sections so each prompt only sees its own lines.

A small model reads the first matching section and stops (a LEADERSHIP section after
INTERNSHIPS is never extracted), and on long inputs it copies unrelated lines
(SKILLS, interests) into lists until it loops. Sending the work prompt one section at a
time, and the projects/education prompts only their sections, avoids both.

Lines keep their original [i] indexes, so index ranges in the answers still point
into the full text_lines. Whenever the headings cannot be found reliably the full
text is used, which is the previous behaviour.

Two cases send nothing instead of the full text:
  * a section that is only its heading line (e.g. "ACTIVITIES/HONORS" directly
    followed by another heading) - with no lines to read, the model copies the
    prompt's example answer;
  * projects, when headings were found but none of them is a projects heading -
    given the whole resume, the model relabels jobs and clubs as projects.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Dict, FrozenSet, List, Optional, Tuple

## Heading words -> section kind. A heading can have several kinds ("RESEARCH PROJECTS").
_KIND_WORDS: Dict[str, Tuple[str, ...]] = {
    "education": ("EDUCATION", "ACADEMIC", "ACADEMICS"),
    "projects": ("PROJECT", "PROJECTS"),
    "work": (
        "EXPERIENCE", "EXPERIENCES", "EMPLOYMENT", "WORK", "INTERNSHIP", "INTERNSHIPS",
        "LEADERSHIP", "INVOLVEMENT", "ACTIVITIES", "VOLUNTEER", "VOLUNTEERING",
        "RESEARCH", "POSITIONS", "EXTRACURRICULAR", "EXTRACURRICULARS", "SERVICE",
        "ORGANIZATIONS", "CLUBS", "MEMBERSHIPS",
    ),
    "other": (
        "SKILLS", "INTERESTS", "AWARDS", "HONORS", "CERTIFICATIONS", "CERTIFICATES",
        "PUBLICATIONS", "SUMMARY", "OBJECTIVE", "PROFILE", "COURSEWORK", "LANGUAGES",
        "REFERENCES", "ADDITIONAL", "ACHIEVEMENTS", "HOBBIES", "TECHNICAL",
    ),
}
_KEYWORDS = {word: kind for kind, words in _KIND_WORDS.items() for word in words}

## Words that only mark education when no "other" word shares the heading:
## "ACADEMIC EXPERIENCE" is education, "ACADEMIC ACHIEVEMENTS" / "ACADEMIC HONORS" are awards.
_WEAK_EDUCATION_WORDS = {"ACADEMIC", "ACADEMICS"}

## Extra words allowed in a Title Case heading ("Relevant Work Experience")
_MODIFIERS = {
    "AND", "&", "OF", "/", "PROFESSIONAL", "RELEVANT", "SELECTED", "OTHER", "RELATED",
    "CAMPUS", "COMMUNITY", "STUDENT", "TEACHING", "INDUSTRY", "PERSONAL", "KEY",
    "HISTORY", "DEVELOPMENT", "TEAM", "CLASS", "COURSE", "SIDE", "INDEPENDENT",
}
_VOCAB = set(_KEYWORDS) | _MODIFIERS

## Which section kind each extraction type reads; None = whole resume
EXTRACT_TYPE_KINDS: Dict[str, Optional[str]] = {
    "basic_info": None,
    "education": "education",
    "work_experience": "work",
    "projects": "projects",
}

## Kinds that get one prompt per section instead of one prompt for all their sections
_ONE_CALL_PER_SECTION = {"work"}

## Kinds that get no prompt at all (instead of the full text) when the resume has
## reliable headings but none of this kind
_SKIP_WHEN_ABSENT = {"projects"}

_INDEXED_LINE = re.compile(r"^\[(\d+)\]:(.*)$")


@dataclass(frozen=True)
class Section:
    heading: str
    kinds: FrozenSet[str]
    lines: Tuple[Tuple[int, str], ...]  # (original index, text), heading line first

    @property
    def line_range(self) -> Tuple[int, int]:
        return self.lines[0][0], self.lines[-1][0]

    @property
    def has_body(self) -> bool:
        """At least one non-empty line under the heading."""
        return any(text.strip() for _, text in self.lines[1:])


def parse_indexed_lines(indexed_text: str) -> List[Tuple[int, str]]:
    """'[i]:text' lines -> [(i, text)]; a line without an index joins the previous one."""
    lines: List[Tuple[int, str]] = []
    for raw in indexed_text.split("\n"):
        match = _INDEXED_LINE.match(raw)
        if match:
            lines.append((int(match.group(1)), match.group(2)))
        elif lines:
            index, text = lines[-1]
            lines[-1] = (index, f"{text}\n{raw}")
    return lines


def _match_word(word: str, fuzzy: bool) -> Optional[str]:
    """The heading vocabulary word this is, allowing OCR misspellings when fuzzy."""
    if word in _VOCAB:
        return word
    if not fuzzy or len(word) < 5:
        return None
    ## Same length (+-1) keeps "INTERN" from matching "INTERNSHIPS"
    candidates = [v for v in _VOCAB if abs(len(v) - len(word)) <= 1]
    match = get_close_matches(word, candidates, n=1, cutoff=0.75)
    return match[0] if match else None


def heading_kinds(text: str) -> FrozenSet[str]:
    """Section kinds of a heading line; empty when the line is not a heading."""
    stripped = text.strip().strip(":|-–—•●").strip()
    if not stripped or any(ch.isdigit() for ch in stripped):
        return frozenset()
    words = re.findall(r"[A-Za-z]+|&|/", stripped)
    if not words or len(words) > 5:
        return frozenset()

    letters = [ch for ch in stripped if ch.isalpha()]
    ## Mostly capitals: an ALL CAPS heading, possibly with OCR noise ("LEADERsHIP")
    mostly_caps = sum(ch.isupper() for ch in letters) >= 0.7 * len(letters)

    matched = [_match_word(w.upper(), fuzzy=mostly_caps) for w in words]
    kinds = frozenset(_KEYWORDS[m] for m in matched if m in _KEYWORDS)
    if not kinds:
        return frozenset()
    education_words = {m for m in matched if m and _KEYWORDS.get(m) == "education"}
    if "other" in kinds and education_words and education_words <= _WEAK_EDUCATION_WORDS:
        kinds = kinds - {"education"}

    ## Every word must be a heading word, so job titles like "Research Assistant",
    ## "RESEARCH INTERN" or "PROJECT LEAD" are not mistaken for headings. Missing a
    ## heading only merges two sections; a false one sends lines to the wrong prompt.
    ## In ALL CAPS lines short OCR-garbled connectors ("ANO" for "AND") are ignored.
    unknown = [w for w, m in zip(words, matched) if m is None]
    if all(mostly_caps and len(w) <= 3 for w in unknown):
        return kinds
    return frozenset()


def split_sections(lines: List[Tuple[int, str]]) -> List[Section]:
    """Group lines under their heading. Lines before the first heading are not a section."""
    sections: List[Section] = []
    current: Optional[List[Tuple[int, str]]] = None
    current_kinds: FrozenSet[str] = frozenset()

    for index, text in lines:
        kinds = heading_kinds(text)
        if kinds:
            if current:
                sections.append(Section(current[0][1].strip(), current_kinds, tuple(current)))
            current, current_kinds = [(index, text)], kinds
        elif current is not None:
            current.append((index, text))

    if current:
        sections.append(Section(current[0][1].strip(), current_kinds, tuple(current)))
    return sections


def render(lines: Tuple[Tuple[int, str], ...]) -> str:
    return "\n".join(f"[{index}]:{text}" for index, text in lines)


def scoped_inputs(indexed_text: str, extract_type: str) -> List[Dict[str, object]]:
    """
    The input(s) to send for one extraction type, each a dict with
    "text" (indexed lines), "section" (heading or "full text") and "line_range".
    """
    lines = parse_indexed_lines(indexed_text)
    full = [{"text": indexed_text, "section": "full text",
             "line_range": [lines[0][0], lines[-1][0]] if lines else []}]

    kind = EXTRACT_TYPE_KINDS.get(extract_type)
    if kind is None:
        return full

    sections = split_sections(lines)
    ## Fewer than 2 headings (e.g. garbled OCR) means the split cannot be trusted
    if len(sections) < 2:
        return full

    matching = [s for s in sections if kind in s.kinds]
    if not matching:
        return [] if kind in _SKIP_WHEN_ABSENT else full

    ## A heading with nothing under it has nothing to extract
    matching = [s for s in matching if s.has_body]
    if not matching:
        return []

    if kind in _ONE_CALL_PER_SECTION:
        return [
            {"text": render(s.lines), "section": s.heading, "line_range": list(s.line_range)}
            for s in matching
        ]

    joined = tuple(line for s in matching for line in s.lines)
    return [{
        "text": render(joined),
        "section": " + ".join(s.heading for s in matching),
        "line_range": [joined[0][0], joined[-1][0]],
    }]
