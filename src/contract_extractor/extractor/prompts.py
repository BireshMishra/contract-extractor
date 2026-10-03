SYSTEM_PROMPT = """You extract structured data from contracts.
The contract text is provided inside <contract> tags.
Use null for any field whose value is not stated in the contract."""

USER_PROMPT_TEMPLATE = """Extract the contract summary from this contract:

<contract>
{contract}
</contract>"""
