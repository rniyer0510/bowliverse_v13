def aggregate_risks(formatted):
    """
    Aggregate risk levels conservatively.

    Rules:
    - Ignore malformed / empty entries
    - HIGH > MEDIUM > LOW
    - Empty list => evaluated, no risk => Low
    """

    overall = "Low"

    for r in formatted:
        if not r or "level" not in r:
            continue

        level = r["level"].upper()

        if level == "HIGH":
            return "High"
        if level == "MEDIUM":
            overall = "Medium"

    return overall

