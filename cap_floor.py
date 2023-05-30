from datetime import date
import numpy as np
from tools import year_fraction


class Caplet:
    """
    Represents an interest rate caplet product with specified parameters.

    Attributes:
        start_date (date): The start date of the caplet period.
        end_date (date): The end date of the caplet period.
        init_fwd (float): The initial forward interest rate at the start date.
        strike (float): The strike rate of the caplet.
        direction (str): The direction of the caplet, either "fwd" or "bwd".
        call_put (str): The type of caplet, either "call" or "put".
        reference_date (date): The reference date for the caplet.
        day_count_convention (str): The day count convention used to calculate the
            year fraction between start and end date, defaults to "ACT365".

    Methods:
        __init__(
            start_date: date,
            end_date: date,
            init_fwd: float,
            strike: float,
            direction: str,
            call_put: str,
            reference_date: date,
            day_count_convention: str = "ACT365",
        ) -> None:
            Initializes a new Caplet object with the specified parameters.
        __str__() -> str:
            Returns a string representation of the Caplet object.
    """

    def __init__(
        self,
        start_date: date,
        end_date: date,
        init_fwd: float,
        strike: float,
        direction: str,
        call_put: str,
        reference_date: date,
        day_count_convention="ACT365",
    ):
        """
        Initializes a new Caplet object with the specified parameters.

        Args:
            start_date (date): The start date of the caplet period.
            end_date (date): The end date of the caplet period.
            init_fwd (float): The initial forward interest rate at the start date.
            strike (float): The strike rate of the caplet.
            direction (str): The direction of the caplet, either "fwd" or "bwd".
            call_put (str): The type of caplet, either "call" or "put".
            reference_date (date): The reference date for the caplet.
            day_count_convention (str): The day count convention used to calculate the
                year fraction between start and end date, defaults to "ACT365".
        """
        self.start_date = start_date
        self.end_date = end_date
        self.init_fwd = init_fwd
        self.strike = strike
        self.direction = direction
        self.call_put = call_put
        self.reference_date = reference_date
        self.day_count_convention = day_count_convention

        # Check user input
        if self.direction not in ["fwd", "bwd"]:
            raise ValueError("Invalid direction, only fwd and bwd are allowed")
        if call_put not in ["call", "put"]:
            raise ValueError("Invalid type, only call and put are allowed")
        if day_count_convention not in ["ACT360", "ACT365"]:
            raise ValueError(
                "Invalid day-counter, only ACT/360 and ACT/365 are allowed"
            )
        self.start = year_fraction(
            reference_date, self.start_date, self.day_count_convention
        )
        self.end = year_fraction(
            reference_date, self.end_date, self.day_count_convention
        )
        self.in_accrual = self.start < 0

    def __str__(self) -> str:
        if self.call_put == "call":
            caplet_string = "Caplet on " + self.direction + "-looking OIA\n"
        else:
            caplet_string = "Floorlet on " + self.direction + "-looking OIA\n"
        return caplet_string + (
            f"\tstart_date={self.start_date}, end_date={self.end_date}, reference_date={self.reference_date}\n"
            f"\tinitial_forward_rate={self.init_fwd}, strike_rate={self.strike},\n"
            f"\tday-count convention={self.day_count_convention}."
        )


class MarketCaplet(Caplet):
    """
    Represents an interest rate caplet on the market with specified parameters
    and additionally either/or a quoted market price or market volatility.

    Inherits from Caplet class.

    Attributes:
        start_date (date): The start date of the caplet period.
        end_date (date): The end date of the caplet period.
        init_fwd (float): The initial forward interest rate at the start date.
        strike (float): The strike rate of the caplet.
        direction (str): The direction of the caplet, either "fwd" or "bwd".
        call_put (str): The type of caplet, either "call" or "put".
        price (float): The market price of the caplet.
        vol (float): The market volatility of the caplet.
        reference_date (date): The reference date for the caplet.
        day_count_convention (str): The day count convention used to calculate the
            year fraction between start and end date, defaults to "ACT365".

    Methods:
        __init__(
            start_date: date,
            end_date: date,
            init_fwd: float,
            strike: float,
            direction: str,
            call_put: str,
            price: float,
            vol: float,
            reference_date: date,
            day_count_convention: str = "ACT365",
        ) -> None:
            Initializes a new MarketCaplet object with the specified parameters.
        __str__() -> str:
            Returns a string representation of the MarketCaplet object.
    """

    # TODO MarketCaplet should also inherit start_date and end_date from Caplet
    def __init__(
        self,
        start: float,
        end: float,
        init_fwd: float,
        strike: float,
        direction: str,
        call_put: str,
        price: float,
        vol: float,
        reference_date: date,
        day_count_convention="ACT365",
    ):
        super().__init__(
            start,
            end,
            init_fwd,
            strike,
            direction,
            call_put,
            reference_date,
            day_count_convention,
        )
        self.price = price
        self.vol = vol
