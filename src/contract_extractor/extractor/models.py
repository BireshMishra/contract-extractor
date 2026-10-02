from pydantic import BaseModel
from datetime import date


class ContractSummary(BaseModel):
    parties : list[str]
    effective_date : date|None
    term_months : int|None
    termination_notice_days : int|None 
    governing_law : str|None
    liability_cap : str|None


