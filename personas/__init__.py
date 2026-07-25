"""The Partners' Room — deterministic investor-persona voting layer.

Architecture inspired by virattt/ai-hedge-fund (persona agents + risk and
portfolio managers), rebuilt with zero LLM calls: every persona votes from
its codified rules in knowledge/rules.yaml plus a deterministic scoring lens.
"""
