"""

"""

from datetime import date


def year_fraction(start: date, end: date, day_count_convention: str):
    """Tested for negative numbers also"""
    if day_count_convention == "ACT360":
        return (end - start).days / 360
    if day_count_convention == "ACT365":
        return (end - start).days / 365
