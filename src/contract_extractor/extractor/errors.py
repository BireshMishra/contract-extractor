class ExtractionError(Exception):
    """The extraction could not be completed; the message is safe to show the user."""


class SchemaError(ExtractionError):
    """The model kept returning output that does not match ContractSummary."""
