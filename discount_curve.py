"""

"""

import csv
from datetime import date, datetime
import numpy as np


class Curve:
    """ """

    def __init__(self, filename: str, reference_date: date, day_count_convention: str):
        self.reference_date = reference_date
        self.zero_rates = {}
        self.day_count_convention = day_count_convention
        self._read_csv_file(filename)

        assert day_count_convention in [
            "ACT360",
            "ACT365",
        ], "Day-count convention should be either ACT/360 or ACT/365"
        self.zero_rates = dict(sorted(self.zero_rates.items()))  # Sorts the dictionary

    def _read_csv_file(self, filename):
        """Assumes filename.csv's dates are sorted"""
        with open(filename + ".csv", newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile, delimiter=";")
            next(reader)  # skip the header row
            for row in reader:
                zero_date = datetime.strptime(row[0], "%d-%m-%Y").date()
                zero_rate = float(row[1].replace(",", "."))
                self.zero_rates[zero_date] = zero_rate
        if not (self.reference_date in self.zero_rates.keys()):
            self.zero_rates[self.reference_date] = 0.0

    def _interpolate_zero_rate(self, zero_date: date):
        """ """
        # find the last date in the curve before the given date
        last_date = None
        for curve_date in self.zero_rates.keys():
            if curve_date > zero_date:
                break
            last_date = curve_date
        assert last_date is not None
        zero_rate_last_date = self.zero_rates[last_date]

        # find the first date in the curve after the given date
        first_date = None
        for curve_date in self.zero_rates.keys():
            if curve_date > zero_date:
                first_date = curve_date
                break
        assert first_date is not None
        zero_rate_first_date = self.zero_rates[first_date]

        slope = (zero_rate_first_date - zero_rate_last_date) / (
            first_date - last_date
        ).days
        return zero_rate_last_date + (zero_date - last_date).days * slope

    def _get_zero_rate(self, zero_date: date):
        """ """
        assert zero_date >= self.reference_date
        if zero_date in self.zero_rates:
            return self.zero_rates[zero_date]
        return self._interpolate_zero_rate(zero_date)

    def _year_fraction(self, start_date: date, end_date: date):
        """ """
        if self.day_count_convention == "ACT360":
            return (end_date - start_date).days / 360
        if self.day_count_convention == "ACT365":
            return (end_date - start_date).days / 365

    def get_discount_factor(self, end_date: date):
        """ """
        zero_rate_end = self._get_zero_rate(end_date)
        return np.exp(
            self._year_fraction(self.reference_date, end_date) * (zero_rate_end / 100)
        )

    def get_future_discount_factor(self, start_date: date, end_date: date):
        """Yet to be tested"""
        discount_factor_start = self.get_discount_factor(start_date)
        discount_factor_end = self.get_discount_factor(end_date)
        return discount_factor_start / discount_factor_end


if __name__ == "__main__":
    TODAY = date(2023, 4, 19)
    DISCOUNT_CURVE = Curve("SOFR_YC_USD_19042023", TODAY, "ACT365")

    FUTURE_DATE = date(2023, 4, 20)
    print(DISCOUNT_CURVE.get_discount_factor(FUTURE_DATE))

    FUTURE_DATE = date(2023, 4, 23)
    print(DISCOUNT_CURVE.get_discount_factor(FUTURE_DATE))

    FUTURE_DATE = date(2043, 8, 21)
    print(DISCOUNT_CURVE.get_discount_factor(FUTURE_DATE))

    # TODAY = date(2020, 5, 13)
    # DISCOUNT_CURVE = Curve("curve", TODAY, "ACT365")
