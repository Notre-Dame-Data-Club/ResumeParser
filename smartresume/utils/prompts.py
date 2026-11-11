"""
LLM prompts for resume parsing (written for small local models such as Gemma 4 E4B).

Each section prompt has the same four parts, in this order:
  1. RULES     short numbered rules, one idea per line
  2. TEMPLATE  the exact output shape, as *valid* JSON with empty values
  3. EXAMPLE   one worked input (indexed lines) -> output pair
  4. a closing reminder to output JSON only

Why this shape for a small model:
  * It copies what it sees. The template is pure JSON (no "#" comments, no
    example values inside it), so the thing it copies is already valid output.
  * One concrete example teaches index ranges, entry boundaries and
    per-entry technologies better than paragraphs of explanation.
  * Dates are flat startDate/endDate keys copied verbatim, never reformatted.
  * Example people and companies are fictional so a copied value is
    recognisable as a leak instead of passing as real data.

TEMPLATE and EXAMPLE_OUTPUT constants are parsed by schema.validate_csv_schema,
so they must stay valid JSON and use only the keys the CSV schema reads.
"""
from __future__ import annotations

from typing import Dict

SYSTEM_PROMPT = (
    "You are a resume parser. The user message is a resume. "
    "Each line starts with its index, like [12]:text. "
    "Return one JSON object and nothing else: compact, on a single line, "
    "without ``` code fences.\n"
)


def _compose(rules: str, template: str, example_input: str, example_output: str) -> str:
    """Join the four parts with the same headings in every section prompt."""
    return (
        f"\nRules:\n{rules.strip()}\n"
        f"\nJSON format:\n{template.strip()}\n"
        f"\nExample input:\n{example_input.strip()}\n"
        f"\nExample output:\n{example_output.strip()}\n"
        "\nThe example is not the resume. Read the user's resume and output JSON only.\n"
    )


# -- basicInfo -- #
BASIC_INFO_RULES = """
1. Copy each value exactly as written in the resume header. Use "" when missing.
2. name: the candidate's full name, usually line [0].
3. currentLocation: the candidate's own city and state or country. Not a school or employer location.
4. URLs: every link (LinkedIn, GitHub, portfolio). Use [] when there are none.
"""

BASIC_INFO_TEMPLATE = """
{"basicInfo": {"name": "", "personalEmail": "", "phoneNumber": "", "currentLocation": "", "URLs": []}}
"""

BASIC_INFO_EXAMPLE_INPUT = """
[0]:Jordan Vale
[1]:Springfield, IL | jvale@examplemail.com | (555) 010-2030 | linkedin.com/in/jvale-example
[2]:EDUCATION
"""

BASIC_INFO_EXAMPLE_OUTPUT = """
{"basicInfo": {"name": "Jordan Vale", "personalEmail": "jvale@examplemail.com", "phoneNumber": "(555) 010-2030", "currentLocation": "Springfield, IL", "URLs": ["linkedin.com/in/jvale-example"]}}
"""

BASIC_INFO_PROMPT = _compose(
    BASIC_INFO_RULES, BASIC_INFO_TEMPLATE, BASIC_INFO_EXAMPLE_INPUT, BASIC_INFO_EXAMPLE_OUTPUT
)


# -- Shared example for workExperience and projects -- #
# The same lines are used by both prompts so the model sees where one section
# stops and the other starts: [1]-[7] are work, [9]-[10] are a project.
_WORK_AND_PROJECTS_EXAMPLE_INPUT = """
[0]:EXPERIENCE
[1]:Northwind Freight Chicago, IL
[2]:Data Analyst Intern Jun 2024 - Aug 2024
[3]:- Built SQL dashboards for 12 warehouses
[4]:- Automated weekly reports in Python
[5]:Lakeview Robotics Club
[6]:President Sep 2023 - Present
[7]:- Led a team of 15 members
[8]:PROJECTS
[9]:Route Planner | Jan 2024
[10]:- Wrote a shortest-path solver in Java
[11]:SKILLS
[12]:Excel, Tableau
"""


# -- workExperience -- #
WORK_EXPERIENCE_RULES = """
1. One object per job, internship, research role, leadership role, club, sport, activity or volunteer/community service item. Keep resume order. List each one once.
   Lines like "Youth Camp Volunteer - 35 Hours" are entries too. In a section that mixes honors and roles, list the roles (coach, tutor, captain, member) and skip pure awards.
   Bullet lines under a role belong to that role's description; they never start a new object.
2. Copy companyName, position, startDate and endDate exactly as written. Use "" when missing. endDate may be "Present".
3. The organization and the role can be on different lines. Fill both.
4. jobDescription_refer_index_range is [first, last]: the indexes of that entry's description lines. Start after the organization, role and date lines. Stop before the next entry or heading. Use [] when there are no description lines.
5. technologies: tools named in that entry's own description lines. Use [] when none. Do not copy from other entries or from SKILLS.
6. internship: 1 if the role is an internship, else 0. isLeadership: 1 if the role leads people (president, captain, lead, founder, manager), else 0.
7. Skip PROJECTS, EDUCATION and SKILLS sections.
8. Use only the keys in the JSON format. Do not add keys. Do not put entries inside entries.
"""

WORK_EXPERIENCE_TEMPLATE = """
{"workExperience": [{"companyName": "", "position": "", "startDate": "", "endDate": "", "internship": 0, "isLeadership": 0, "jobDescription_refer_index_range": [], "technologies": []}]}
"""

WORK_EXPERIENCE_EXAMPLE_OUTPUT = """
{"workExperience": [{"companyName": "Northwind Freight", "position": "Data Analyst Intern", "startDate": "Jun 2024", "endDate": "Aug 2024", "internship": 1, "isLeadership": 0, "jobDescription_refer_index_range": [3, 4], "technologies": ["SQL", "Python"]}, {"companyName": "Lakeview Robotics Club", "position": "President", "startDate": "Sep 2023", "endDate": "Present", "internship": 0, "isLeadership": 1, "jobDescription_refer_index_range": [7, 7], "technologies": []}]}
"""

WORK_EXPERIENCE_PROMPT = _compose(
    WORK_EXPERIENCE_RULES,
    WORK_EXPERIENCE_TEMPLATE,
    _WORK_AND_PROJECTS_EXAMPLE_INPUT,
    WORK_EXPERIENCE_EXAMPLE_OUTPUT,
)


# -- education -- #
EDUCATION_RULES = """
1. One object per school or degree. Keep resume order. List each one once.
2. school: the institution name only, without its city or state.
3. degreeLevel: exactly one of "High School", "Associate", "Bachelors", "Masters", "Doctorate", "Certificate", or "" when unclear. B.S., B.A. and BBA are "Bachelors". M.S., MBA and M.A. are "Masters". Ph.D. is "Doctorate".
4. department: the college or school inside the institution. Use "" when missing.
5. major: the field of study, copied as written. Join a double major with " and ".
6. startDate and endDate: copy exactly as written. Use "" when missing.
7. GPA: the GPA as written, like "3.7/4.0". Use "" when missing.
8. educationDescription: other lines about that degree (honors, coursework, awards, study abroad) copied as one string. Use "" when none.
9. Use only the keys in the JSON format. Do not add keys.
"""

EDUCATION_TEMPLATE = """
{"education": [{"school": "", "degreeLevel": "", "department": "", "major": "", "startDate": "", "endDate": "", "GPA": "", "educationDescription": ""}]}
"""

EDUCATION_EXAMPLE_INPUT = """
[0]:EDUCATION
[1]:Lakeview University Springfield, IL
[2]:B.S. in Computer Science, College of Engineering Aug 2022 - May 2026
[3]:GPA: 3.7/4.0 | Dean's List
[4]:EXPERIENCE
"""

EDUCATION_EXAMPLE_OUTPUT = """
{"education": [{"school": "Lakeview University", "degreeLevel": "Bachelors", "department": "College of Engineering", "major": "Computer Science", "startDate": "Aug 2022", "endDate": "May 2026", "GPA": "3.7/4.0", "educationDescription": "Dean's List"}]}
"""

EDUCATION_PROMPT = _compose(
    EDUCATION_RULES, EDUCATION_TEMPLATE, EDUCATION_EXAMPLE_INPUT, EDUCATION_EXAMPLE_OUTPUT
)


# -- projects -- #
PROJECTS_RULES = """
1. The user message is the PROJECTS section of a resume. List every titled item in it, to the last line, including research and club or team projects listed there. Return {"projects": []} only if it has no items.
2. When the section also holds jobs or internships (a mixed "EXPERIENCE & PROJECTS" section), list only the projects.
3. One object per project title. A description bullet is never its own project. List each project once.
4. name: the project title exactly as written, without dates.
5. startDate and endDate: copy exactly as written. Use "" when missing.
6. description_refer_index_range is [first, last]: the indexes of that project's description lines. Start after the title line. Stop before the next project or heading. Use [] when there are none.
7. technologies: tools named in that project's own lines. Use [] when none. Do not copy from other projects or from SKILLS.
8. Use only the keys in the JSON format. Do not add keys. Do not put entries inside entries.
"""

PROJECTS_TEMPLATE = """
{"projects": [{"name": "", "startDate": "", "endDate": "", "description_refer_index_range": [], "technologies": []}]}
"""

PROJECTS_EXAMPLE_OUTPUT = """
{"projects": [{"name": "Route Planner", "startDate": "Jan 2024", "endDate": "", "description_refer_index_range": [10, 10], "technologies": ["Java"]}]}
"""

PROJECTS_PROMPT = _compose(
    PROJECTS_RULES, PROJECTS_TEMPLATE, _WORK_AND_PROJECTS_EXAMPLE_INPUT, PROJECTS_EXAMPLE_OUTPUT
)


def get_prompts() -> Dict[str, str]:
    """
    Return the system prompt for each extraction type.

    Thinking is switched off through the chat template (enable_thinking=False in
    llm_client), so no model-specific tag is appended here.
    """
    return {
        "basic_info": SYSTEM_PROMPT + BASIC_INFO_PROMPT,
        "work_experience": SYSTEM_PROMPT + WORK_EXPERIENCE_PROMPT,
        "education": SYSTEM_PROMPT + EDUCATION_PROMPT,
        "projects": SYSTEM_PROMPT + PROJECTS_PROMPT,
    }


EXTRACT_TYPES = tuple(get_prompts())
