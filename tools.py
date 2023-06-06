"""
tools.py - a collection of general utility functions.

This module contains a collection of general-purpose functions.

Functions:
    year_fraction(start: date, end: date, day_count_convention: str) -> float:
        Calculate the year fraction between two dates based on a given day count convention.

    error_function(result: np.array, objective: np.array, criterion="abs_sqr") -> float:
        Calculate the error between two arrays using a given criterion.

Classes:
    Caplet:
        Represents an interest rate caplet product with specified parameters.
    
    MarketCaplet:
        Represents an interest rate caplet on the market with .
"""

from datetime import date
import numpy as np


def year_fraction(start: date, end: date, day_count_convention="ACT365") -> float:
    """
    Calculate the year fraction between two dates based on a given day count convention.

    Args:
        start (date): The start date.
        end (date): The end date.
        day_count_convention (str): The day count convention to use for the calculation.
            Currently supported conventions are "ACT360" and "ACT365".

    Returns:
        float: The year fraction between the start and end dates.

    Raises:
        ValueError: If the day_count_convention is not supported.
    """
    if day_count_convention == "ACT360":
        return (end - start).days / 360
    if day_count_convention == "ACT365":
        return (end - start).days / 365
    raise ValueError(f"Unsupported day count convention: {day_count_convention}")


def error_function(result: np.array, objective: np.array, criterion="abs_sqr") -> float:
    """
    Calculate the error between two arrays using a given criterion.

    Args:
        result (np.array): The resulting array.
        objective (np.array): The objective array.
        criterion (str): The error criterion to use.
            Currently supported criteria are "abs_abs", "rel_abs", "rel_sqr", and "abs_sqr".

    Returns:
        float: The error between the two arrays based on the chosen criterion.

    Raises:
        ValueError: If an unsupported criterion is provided.
    """
    if criterion.lower() == "sqrt":
        errors = np.abs(result - objective) ** (1 / 3)
    elif criterion.lower() == "abs_abs":
        errors = np.abs(result - objective)
    elif criterion.lower() == "abs_sqr":
        errors = (result - objective) ** 2
    elif criterion == "rel_abs":  # TEST
        errors = np.abs(result - objective) / (objective)
    elif criterion.lower() == "rel_sqr":  # TEST
        errors = ((result - objective) / (objective)) ** 2
    else:
        raise ValueError(f"Unsupported criterion: {criterion}")
    return np.sum(errors)


# TODO Formatting
# TODO docstring
def callback_function(xk, convergence):
    # xk is the current best candidate solution
    # convergence is the current best value of the objective function
    # step_number is the current iteration number
    print(f"\talpha\t{xk[0]:.4f}\n\trho\t{xk[1]:.4f}\n\tnu\t{xk[2]:.4f}")


def eff_rho_bounds(start, end, exp, in_accrual):
    # TEST
    if in_accrual:
        zeta = (1 / (4 * exp + 3)) * (
            1 / (2 * exp + 1) + (2 * exp) / (3 * exp + 2) ** 2
        )
        return 2 / (np.sqrt(zeta) * (3 * exp + 2))
    tau = 2 * exp * start + end
    gamma_1 = (
        2 * tau**3
        + end**3
        + (4 * exp**2 - 2 * exp) * start**3
        + 6 * exp * start**2 * end
    )
    gamma_2 = 3 * tau**2 - end**2 + 5 * exp * start**2 + 4 * start * end
    gamma = (tau * gamma_1) / ((4 * exp + 3) * (2 * exp + 1)) + (
        3 * exp * (end - start) ** 2 * gamma_2
    ) / ((4 * exp + 3) * (3 * exp + 2) ** 2)
    return (3 * tau**2 + 2 * exp * start**2 + end**2) / (
        np.sqrt(gamma) * (6 * exp + 4)
    )
