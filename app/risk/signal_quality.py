def signal_quality(values, min_len=6, jitter_tol=0.15):
    """
    Returns quality ∈ [0,1]
    - Too short → 0
    - Excess jitter → low confidence
    """
    if values is None or len(values) < min_len:
        return 0.0

    diffs = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
    mean = sum(diffs) / (len(diffs) + 1e-9)

    if mean > jitter_tol:
        return 0.4

    return 1.0

