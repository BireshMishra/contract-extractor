from datetime import date

from pydantic import BaseModel, Field


class ContractSummary(BaseModel):
    parties: list[str] = Field(
        description="Full legal names of the parties to the contract, as written in it."
    )
    effective_date: date | None = Field(
        description="Date the contract takes effect, as an ISO 8601 date (YYYY-MM-DD)."
    )
    term_months: int | None = Field(
        description="Initial term of the contract in whole months (e.g. 2 years = 24)."
    )
    termination_notice_days: int | None = Field(
        description="Days of written notice required to terminate for convenience."
    )
    governing_law: str | None = Field(
        description="Jurisdiction whose law governs the contract, e.g. 'England and Wales'."
    )
    liability_cap: str | None = Field(
        description="The limitation-of-liability cap, in the contract's own terms (amount or formula)."
    )
