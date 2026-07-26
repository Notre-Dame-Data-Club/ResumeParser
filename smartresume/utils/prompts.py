"""
LLM Prompts for resume parsing.
Translated from the original Mandarin prompts in the smartresume repo.
"""
from __future__ import annotations

from typing import Dict

SYSTEM_PROMPT = """ACT: You are a resume parsing system. TASK: Transform resume content into the set JSON format."""

BASIC_INFO_PROMPT = """
EXTRACT provided information. INSERT into the JSON strcture，if field missing, return "".
{
  "basicInfo": {
    "name": "", # Name. Ex: Riley Leonard, Marcus Freeman, Daniel Rojas.
    "personalEmail": "", # Email. Ex: jsmith@nd.edu, mateo.kim333@gmail.com
    "phoneNumber": "", # Telephone/Cellphone number. Ex: 5745869012, +1（201）706 1136
    "currentLocation": "",  # City/Town Name, State/Country 'sampletown/cityname, state' or 'cityname, country'  **DIFFERENT from target location if present.**
    "URLs": "[""]" # URL list with relevant portfolios or social media. Seperate items with ", " . Ex: https://linkedin.com/in/xxx, github.com/xxx, www.website.com/.
  },
}
"""

WORK_EXPERIENCE_PROMPT = """{
  "workExperience": [  # Work Experience, Job History, Past Internships
    {
      "companyName": "", # Company Name. Ex: Amazon, Gazmin, Aunalytics
      "employmentPeriod": {  #  Job Start & End date
        "startDate": "", # START date or anticipated start date. if missing, return "". Format: Month [EX: %M, %MM, MONTH, MNT, ""], Year [EX: %YY, %YYYY, ""], Label ["UPCOMING", "TBD", ""]
        "endDate": "" # END date, anticipated end date or still on going label. if missing, return "". Format: Month [EX: %M, %MM, MONTH, MNT, ""], Year [EX: %YY, %YYYY, ""], Label ["TBD", "PRESENT", ""]
      },
      "position": "", # Role Ex: Senior Architect, Software Engineer, Business Consultant. USE EXACT WORDING FROM THE SOURCE.
      "internship": 0, #Was the experience an internship. TRUE = 1,  FALSE = 0
      "jobDescription_refer_index_range": [start_index,end_index]   # List
       # IF missing, return [],
       # jobDescription_refer_index_range: The paragraph index range referencing the raw job text description. 
       # Normally covers work achievements, performance, main duties, project background, tech stack, job description, etc. Include as much as possible until the next work experience section starts.
       # jobDescription_refer_index_range DON'T INCLUDE companyName, employmentPeriod, position.
       # Ex 1:
       # [22]: Alibaba 2021.11-2022.11
       # [22]: Job Description: Engaged in field marketing, achieved XX performance
       # [23]: Rated A in field marketing tasks
       # [24]: Company: Alibaba Cloud
       # "jobDescription_refer_index_range": [22, 23] covers all paragraph content from index 22 to 23 inclusive ($22 + 23$).
       # Ex 2:
       # [22]: Job Description: Engaged in field marketing, achieved XX performance
       # [23]: Rated A in field marketing tasks
       # [...]
       # [40]: Contributed XXX sales performance to the successful conclusion of field marketing activities.
       # "jobDescription_refer_index_range": [22, 40] covers all paragraph content from index 22 to 40 inclusive ($22 + 23 + 24 + ... + 39 + 40).
    }, ...
  ]
  }
"""

EDUCATION_PROMPT = """
{
  "education": [  # Education History
    {
      "degreeLevel": "", # Degree Ex: Associate, Bachelors, Masters, PHD/Doctorate, High School Diploma if missing, return""
      "period": {  #  Degree Start & End date
        "startDate": "", # START date or anticipated start date. If missing, return "". Format: Month [EX: %M, %MM, MONTH, MNT, ""], Year [EX: %YY, %YYYY, ""], Label ["UPCOMING", "TBD", ""]
        "endDate":""   # END date, anticipated end date or still on going label. If missing, return "". Format: Month [EX: %M, %MM, MONTH, MNT, ""], Year [EX: %YY, %YYYY, ""], Label ["TBD", "PRESENT", ""]
      },
      "school": "", # Institution Name Ex: MIT, University of Notre Dame, Howard College, Universidad Catolica de Chile, Tsinghua University.
      "department": "", # Institution School/Department. If missing, return "". Ex: Mendoza Business School, Cornell David A. Duffield College of Engineering, Peking University Health Science Center.
      "major": "", # Major: Ex: Business Analytics, Computer Science, Biology, Speech-Language Pathology.
      "GPA": "". # GPA score: Numerical degree wide grade. Normally out of a maximum score /4.0, /100, /7. Extract directly from source. If missing, return "".
      "educationDescription": "" # Education Description: Other fields such as coursework grades, research focus, and honors/awards.  **DIFFERENT from the degree and GPA.**. If missing, return "".
    }, ...
  ]
}
"""

WORK_EXPERIENCE_HINT = """
- Sections titled WORK EXPERIENCE, EXPERIENCE, EMPLOYMENT, LEADERSHIP, ACTIVITIES, or PROJECTS can contain valid experience entries when an organization and role are listed.
- Company or organization names may appear on one line and the role title on the next line. Capture both instead of leaving either field empty.
- Preserve the original company and position wording exactly as written in the resume.
"""


THINK_TAG = " /no_think"


def get_prompts() -> Dict[str, str]:
    """Return prompts for external script use."""
    return {
        "basic_info": SYSTEM_PROMPT + BASIC_INFO_PROMPT + THINK_TAG,
        "work_experience": SYSTEM_PROMPT + WORK_EXPERIENCE_PROMPT + WORK_EXPERIENCE_HINT + THINK_TAG,
        "education": SYSTEM_PROMPT + EDUCATION_PROMPT + THINK_TAG,
    }

EXTRACT_TYPES = tuple(get_prompts())
