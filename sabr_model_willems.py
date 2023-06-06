"""
SABR Model Implementation

This module implements the SABR model for pricing options on various underlying assets. 
The SABR model is a stochastic volatility model that was introduced by Hagan et al. 
in their 2002 paper "Managing Smile Risk".

References:
    [1] Willems, S. (2020). SABR smiles for RFR caplets
    [2] Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002). Managing smile risk.
"""

import time
from datetime import date
import numpy as np
from scipy.stats import norm
from scipy.optimize import bisect, differential_evolution, dual_annealing
from discount_curve import Curve
from tools import eff_rho_bounds, error_function, callback_function
from cap_floor import Caplet, MarketCaplet


class SabrModel:
    """
    A class that implements the SABR model for pricing caplets on overnight index averages.
    Notation is taken from [1].
    """

    # DOCSTRING
    # TODO Write __str__

    def __init__(
        self,
        alpha: float,
        beta: float,
        rho: float,
        volvol: float,
        exp: float,
        caplet: Caplet,
        discount_curve: Curve,
    ):
        self.sabr = {"alpha": alpha, "beta": beta, "rho": rho, "volvol": volvol}
        self.exp = exp
        self.caplet = caplet
        self.discount_curve = discount_curve

        # Check for valid input values
        if not np.isclose(self.sabr["beta"], 0) and np.isclose(self.caplet.strike, 0):
            raise ValueError("We only allow zero strike if beta=0")
        if not np.isclose(self.sabr["beta"], 0) and (
            self.caplet.init_fwd <= 0 or self.caplet.strike < 0
        ):
            raise ValueError("We only allow negative rates if beta=0")
        self.__no_black_model_support = (
            self.caplet.init_fwd <= 0 or self.caplet.strike <= 0
        )

        if self.__no_black_model_support:
            print("WARNING: Black model not supported")

        self.sabr_eff = {"alpha": alpha, "beta": beta, "rho": rho, "volvol": volvol}
        self.eff_parameters()

    def _chi(self, zeta: float, rho: float) -> float:
        """
        Determines the value of the χ(ζ) function as described in equation (2.17c) [1].

        Args
            zeta (float): The input value ζ.

        Returns
            float: The value of χ(ζ).
        """
        return np.log(
            (np.sqrt(1 - 2 * rho * zeta + zeta**2) + zeta - rho) / (1 - rho)
        )

    def _rel_time(self, exp: float) -> float:
        """
        Determines the relative time ((end_date - start_date)/(start_date))^exp as in [1].

        Args
            exp (float): The input value exp.

        Returns
            float: The value of ((start_date - end_date)/(start_date))^exp.
        """
        return ((self.caplet.end() - self.caplet.start()) / self.caplet.end()) ** exp

    def _tau(self):
        """ """
        # DOCSTRING
        return 2 * self.exp * self.caplet.start() + self.caplet.end()

    def _zeta(self, rho):
        """
        Determines the value of zeta as in Theorem 4.1 [1].
        """
        return (1 / (4 * self.exp + 3)) * (
            1 / (2 * self.exp + 1)
            + (2 * self.exp * rho**2) / ((3 * self.exp + 2) ** 2)
        )

    def _delta_sqr(self):
        """
        Function that determines Δ^2 in Appendix C [1].
        """
        # DOCSTRING
        if self.caplet.in_accrual:  # Theorem 4.1 in [1]
            return (self.sabr["alpha"] ** 2) / (
                (2 * self.exp + 1) * self._rel_time(2 * self.exp)
            )
        # Theorem 4.2 in [1]
        return (self.sabr["alpha"] ** 2 * self._tau()) / (
            self.caplet.end() * (2 * self.exp + 1)
        )

    def _b(self):
        """
        Function that determines b in Appendix C [1].
        """
        # DOCSTRING
        if self.caplet.in_accrual:  # Theorem 4.1 in [1]
            term_1 = (self.sabr["volvol"] * self.sabr["rho"] * (4 * self.exp + 2)) / (
                self.sabr["alpha"] * (3 * self.exp + 2)
            )
            return term_1 * self._rel_time(self.exp)
        # Theorem 4.2 in [1]
        term_1 = (
            self.sabr["volvol"] * self.sabr["rho"] * (2 * self.exp + 1)
        ) / self.sabr["alpha"]
        term_2_num = (
            3 * self._tau() ** 2
            + 2 * self.exp * self.caplet.start() ** 2
            + self.caplet.end() ** 2
        )
        term_2_denominator = 2 * self._tau() ** 2 * (3 * self.exp + 2)
        return term_1 * (term_2_num / term_2_denominator)

    def _gamma(self, rho):
        """
        Determines the value of gamma as in Theorem 4.2 in [1]
        """
        gamma_1_num = (
            2 * self._tau() ** 3
            + self.caplet.end() ** 3
            + (4 * self.exp**2 - 2 * self.exp) * self.caplet.start() ** 3
            + 6 * self.exp * self.caplet.start() ** 2 * self.caplet.end()
        )
        gamma_1 = (self._tau() * gamma_1_num) / (
            (4 * self.exp + 3) * (2 * self.exp + 1)
        )

        gamma_2_num = (
            3 * self._tau() ** 2
            - self.caplet.end() ** 2
            + 5 * self.exp * self.caplet.start() ** 2
            + 4 * self.caplet.start() * self.caplet.end()
        )
        gamma_2 = gamma_2_num / ((4 * self.exp + 3) * (3 * self.exp + 2) ** 2)

        return (
            gamma_1
            + 3
            * self.exp
            * rho**2
            * (self.caplet.end() - self.caplet.start()) ** 2
            * gamma_2
        )

    def _H(self):
        """
        Determines the value of H as in Theorem 4.2 [1]
        """
        fraction = (
            self._tau() ** 2
            + 2 * self.exp * self.caplet.start() ** 2
            + self.caplet.end() ** 2
        ) / (2 * self.caplet.end() * self._tau() * (self.exp + 1))
        return self.sabr["volvol"] * fraction - self.sabr_eff["volvol"] ** 2

    def _c(self):
        """
        Function that determines c in Appendix C [1].
        """
        # DOCSTRING
        # TODO Variable names
        if self.caplet.in_accrual:  # Theorem 4.1 in [1]
            term_1 = (
                self.sabr["volvol"] ** 2 * self._rel_time(2**self.exp)
            ) / self.sabr["alpha"] ** 2
            term_2 = (2 * self.exp + 1) / ((4 * self.exp + 3) * (3 * self.exp + 2) ** 2)
            term_3 = (3 * self.exp + 2) ** 2 + self.sabr["rho"] ** 2 * (
                4 * self.exp**2 + 2 * self.exp
            )
            return term_1 * term_2 * term_3

        # Theorem 4.2 in [1]
        term_1 = (self.sabr["volvol"] ** 2 * (2 * self.exp + 1) ** 2) / (
            self.sabr["alpha"] ** 2 * self._tau() ** 4
        )

        return term_1 * self._gamma(self.sabr["rho"])

    def _G(self):
        """
        Function that determines G in Appendix C [1].
        """
        # DOCSTRING
        if self.caplet.in_accrual:  # Theorem 4.1 in [1]
            return (
                self.sabr["volvol"] ** 2 / (self._delta_sqr() * (self.exp + 1))
                - self._c()
            )
        # Theorem 4.2 in [1]
        numerator = (self.sabr["volvol"] ** 2) * (
            self._tau() ** 2
            + 2 * self.exp * self.caplet.start() ** 2
            + self.caplet.end() ** 2
        )
        denominator = self._delta_sqr() * (
            2 * self.caplet.end() * self._tau() * (self.exp + 1)
        )
        return (numerator / denominator) - self._c()

    def eff_parameters(self):
        """
        Determines the effective SABR parameters based on the
        expressions b, c, G and Δ^2 in Appendix C [1].
        """
        # DOCSTRING
        # TEST in_accrual also ok?
        delta = np.sqrt(self._delta_sqr())
        b = self._b()
        c = self._c()
        G = self._G()

        self.sabr_eff["alpha"] = delta * np.exp(
            (delta**2 * G * self.caplet.end()) / 4
        )
        self.sabr_eff["beta"] = self.sabr["beta"]
        self.sabr_eff["rho"] = b / np.sqrt(c)
        self.sabr_eff["volvol"] = delta * np.sqrt(c)
        # TODO Check here for extremely small numbers
        if np.isclose(self.sabr_eff["alpha"], 0):
            print("Effective alpha not suitable", self.sabr)
            raise ValueError("Effective alpha not suitable")
        return self.sabr_eff["alpha"], self.sabr_eff["rho"], self.sabr_eff["volvol"]

    def _eff_correlation(self, rho):
        if self.caplet.in_accrual:  # Theorem 4.1 in [1]
            return (2 * rho) / (np.sqrt(self._zeta(rho)) * (3 * self.exp + 2))
        # Theorem 4.2 in [1]
        numerator = (
            3 * self._tau() ** 2
            + 2 * self.exp * self.caplet.start() ** 2
            + self.caplet.end() ** 2
        )
        return (rho * numerator) / (
            np.sqrt(self._gamma(self.sabr["rho"])) * (6 * self.exp + 4)
        )

    def orig_parameters(self):
        """
        Function that determines the original SABR parameters from the
        set of effective SABR parameters.
        """
        upper_bound = eff_rho_bounds(
            self.caplet.start(), self.caplet.end(), self.exp, self.caplet.in_accrual
        )

        if np.abs(self.sabr_eff["rho"]) < upper_bound:

            def objective_function(rho):
                return self._eff_correlation(rho) - self.sabr_eff["rho"]

            self.sabr["rho"] = bisect(objective_function, -1, 1)
        else:
            raise ValueError("Effective rho can't be inverted")

        if self.caplet.in_accrual:
            # Inverting volvol
            self.sabr["volvol"] = np.sqrt(
                (self.sabr_eff["volvol"] ** 2)
                / ((2 * self.exp + 1) * self._zeta(self.sabr["rho"]))
            )

            # Inverting alpha
            self.sabr["alpha"] = np.sqrt(
                (
                    self.sabr_eff["alpha"] ** 2
                    * (2 * self.exp + 1)
                    * self._rel_time(-2 * self.exp)
                    * np.exp(
                        (-self.caplet.end() / 2)
                        * (
                            (self.sabr["volvol"] ** 2 / (self.exp + 1))
                            - self.sabr_eff["volvol"] ** 2
                        )
                    )
                )
            )

            return [self.sabr["alpha"], self.sabr["rho"], self.sabr["volvol"]]
        # Inverting volvol
        self.sabr["volvol"] = np.sqrt(
            (self.sabr_eff["volvol"] ** 2 * self._tau() ** 3 * self.caplet.end())
            / (self._gamma(self.sabr["rho"]) * (2 * self.exp + 1))
        )

        # Inverting alpha
        self.sabr["alpha"] = np.sqrt(
            self.sabr_eff["alpha"] ** 2
            * ((2 * self.exp + 1) * self.caplet.end())
            / self._tau()
            * np.exp((-self.caplet.end() / 2) * self._H())
        )

        # Inverting beta
        self.sabr["beta"] = self.sabr_eff["beta"]

    def _equivalent_volatility_1(self, model: str):
        """ """
        # DOCSTRING
        # TEST for in_accrual

        # Chooses the right set of SABR parameters
        if self.caplet.direction == "bwd":
            sabr = self.sabr_eff
        else:
            sabr = self.sabr

        if model.lower() == "black":
            if not self.__no_black_model_support:
                if np.isclose(self.caplet.init_fwd, self.caplet.strike):  # ATM
                    return sabr["alpha"] * self.caplet.init_fwd ** (sabr["beta"] - 1)
                zeta = (sabr["volvol"] / sabr["alpha"]) * (
                    (self.caplet.init_fwd * self.caplet.strike)
                    ** ((1 - sabr["beta"]) / 2)
                    * np.log(self.caplet.init_fwd / self.caplet.strike)
                )
                numerator = (
                    sabr["alpha"]
                    * zeta
                    * (self.caplet.init_fwd * self.caplet.strike)
                    ** ((sabr["beta"] - 1) / 2)
                )
                adj_log_money = (1 - sabr["beta"]) * np.log(
                    self.caplet.init_fwd / self.caplet.strike
                )  # (1-β) log(F/K)
                denominator = (
                    1 + adj_log_money**2 / 24 + adj_log_money**4 / 1920
                ) * self._chi(zeta, sabr["rho"])
                return numerator / denominator
            raise ValueError("Black model not supported for currents sabr")
        if model.lower() == "bachelier":
            if self.caplet.init_fwd <= 0 or self.caplet.strike <= 0:
                if np.isclose(sabr["beta"], 0):
                    if np.isclose(self.caplet.init_fwd, self.caplet.strike):  # ATM case
                        return sabr["alpha"]
                    zeta = (
                        sabr["volvol"] * (self.caplet.init_fwd - self.caplet.strike)
                    ) / sabr["alpha"]
                    return (sabr["alpha"] * zeta) / (self._chi(zeta, sabr["rho"]))
                raise ValueError("Negative F or K only supported for beta = 0")
            if np.isclose(self.caplet.init_fwd, self.caplet.strike):  # ATM case
                return sabr["alpha"] * self.caplet.init_fwd ** sabr["beta"]
            if np.isclose(sabr["beta"], 1):  # β = 1
                zeta = (
                    sabr["volvol"] * (self.caplet.init_fwd - self.caplet.strike)
                ) / (sabr["alpha"] * np.sqrt(self.caplet.init_fwd * self.caplet.strike))
                return (
                    sabr["alpha"] * (self.caplet.init_fwd - self.caplet.strike) * zeta
                ) / (
                    (np.log(self.caplet.init_fwd) - np.log(self.caplet.strike))
                    * self._chi(zeta, sabr["rho"])
                )
            zeta = (sabr["volvol"] * (self.caplet.init_fwd - self.caplet.strike)) / (
                sabr["alpha"]
                * (self.caplet.init_fwd * self.caplet.strike) ** (sabr["beta"] / 2)
            )
            chi_zeta = self._chi(zeta, sabr["rho"])
            return (
                sabr["alpha"]
                * (1 - sabr["beta"])
                * (self.caplet.init_fwd - self.caplet.strike)
                * zeta
            ) / (
                (
                    self.caplet.init_fwd ** (1 - sabr["beta"])
                    - self.caplet.strike ** (1 - sabr["beta"])
                )
                * chi_zeta
            )

    def _equivalent_volatility_2(self, model: str):
        """ """
        # TEST
        # DOCSTRING

        # Chooses the right set of SABR parameters
        if self.caplet.direction == "bwd":
            sabr = self.sabr_eff
        else:
            sabr = self.sabr

        if model.lower() == "black":
            term_1 = (
                (1 - sabr["beta"]) ** 2
                * sabr["alpha"] ** 2
                * (self.caplet.init_fwd * self.caplet.strike) ** (sabr["beta"] - 1)
            ) / 24
        elif model.lower() == "bachelier":
            if np.isclose(sabr["beta"], 0):
                term_1 = 0
            else:
                term_1 = (
                    sabr["alpha"] ** 2
                    * sabr["beta"]
                    * (sabr["beta"] - 2)
                    * (self.caplet.init_fwd * self.caplet.strike) ** (sabr["beta"] - 1)
                ) / 24
        if np.isclose(sabr["beta"], 0):
            term_2 = 0
        else:
            term_2 = (
                sabr["alpha"]
                * sabr["beta"]
                * sabr["rho"]
                * sabr["volvol"]
                * (self.caplet.init_fwd * self.caplet.strike)
                ** ((sabr["beta"] - 1) / 2)
            ) / 4
        term_3 = ((2 - 3 * sabr["rho"] ** 2) * sabr["volvol"] ** 2) / 24
        return np.real(term_1 + term_2 + term_3)

    def equivalent_volatility(self, model: str):
        """ """
        # TEST to fwd and bwd-looking caplets
        # DOCSTRING
        if self.caplet.direction == "fwd" and self.caplet.in_accrual:
            raise ValueError("Can't price fwd-looking caplet in accrual period")
        if model.lower() == "bachelier":
            i_1 = self._equivalent_volatility_1("bachelier")
            i_2 = self._equivalent_volatility_2("bachelier")
        elif model.lower() == "black" and not self.__no_black_model_support:
            i_1 = self._equivalent_volatility_1("black")
            i_2 = self._equivalent_volatility_2("black")
        elif model.lower() == "black" and self.__no_black_model_support:
            raise ValueError("Black model not supported")
        else:
            raise ValueError(f"Unknown model: {model}")

        if any(np.isnan([i_1, i_2])):
            raise ValueError(f"NaN found in {model}-volatility")
        if self.caplet.direction == "fwd":  # BUG Is this correct?
            return i_1 * (1 + i_2 * self.caplet.start())
        return i_1 * (1 + i_2 * self.caplet.end())

    def _black_formula(self, volatility):
        """Determines Black option premium"""
        # DOCSTRING
        # TODO check maturity, start or end?
        d_plus = (
            np.log(self.caplet.init_fwd / self.caplet.strike)
            + 0.5 * volatility**2 * self.caplet.end()
        ) / (volatility * np.sqrt(self.caplet.end()))
        d_min = (
            np.log(self.caplet.init_fwd / self.caplet.strike)
            - 0.5 * volatility**2 * self.caplet.end()
        ) / (volatility * np.sqrt(self.caplet.end()))

        call_premium = self.caplet.init_fwd * norm.cdf(
            d_plus
        ) - self.caplet.strike * norm.cdf(d_min)

        if self.caplet.call_put == "call":
            return call_premium
        return call_premium + (self.caplet.strike - self.caplet.init_fwd)

    def _bachelier_formula(self, volatility):
        """ """
        # DOCSTRING
        # TODO check maturity, start or end?
        norm_argument = (self.caplet.init_fwd - self.caplet.strike) / (
            volatility * np.sqrt(self.caplet.end())
        )
        if self.caplet.call_put == "call":
            return (self.caplet.init_fwd - self.caplet.strike) * norm.cdf(
                norm_argument
            ) + volatility * np.sqrt(self.caplet.end()) * norm.pdf(norm_argument)
        return (self.caplet.strike - self.caplet.init_fwd) * norm.cdf(
            -norm_argument
        ) + volatility * np.sqrt(self.caplet.end()) * norm.pdf(norm_argument)

    def option_price(self, model: str, discount=True):
        """ """
        # DOCSTRING
        # self.eff_parameters()
        if discount:
            discount_factor = self.discount_curve.get_discount_factor(
                self.caplet.end_date
            )
        else:
            discount_factor = 1
        if self.caplet.direction == "fwd" and self.caplet.in_accrual:
            raise ValueError("Can't price fwd-looking caplet in accrual period")
        if self.__no_black_model_support:
            model = "bachelier"
            print("Switching to Bachelier model")
        if model == "black":
            volatility = self.equivalent_volatility(model="black")
            return discount_factor * self._black_formula(volatility)
        if model == "bachelier":
            volatility = self.equivalent_volatility(model="bachelier")
            return discount_factor * self._bachelier_formula(volatility)
        raise ValueError(f"Unsupported model: {model}")

    def market_option_price(self, market_caplet: MarketCaplet, model: str):
        """
        Function that determines the option premium of market_caplet
        based on the current set of SABR parameters.
        """
        # DOCSTRING
        # FIXME
        self._set_market_caplet_as_inner_caplet(market_caplet)
        discount_factor = self.discount_curve.get_discount_factor(self.caplet.end_date)
        if model.lower() == "black":
            return discount_factor * self.option_price(model="black")
        if model.lower() == "bachelier":
            return discount_factor * self.option_price(model="bachelier")
        raise ValueError(f"Unsupported model: {model}")

    def market_equiv_vol(self, market_caplet: MarketCaplet, model: str):
        """ """
        # DOCSTRING
        # FIXME
        self._set_market_caplet_as_inner_caplet(market_caplet)
        if model.lower() == "black":
            return self.equivalent_volatility(model="black")
        if model.lower() == "bachelier":
            return self.equivalent_volatility(model="bachelier")
        raise ValueError(f"Unsupported model: {model}")

    def _set_market_caplet_as_inner_caplet(self, market_caplet: MarketCaplet):
        """
        Sets market_caplet as self.caplet, makes it possible to compute
        the option premium or equivalent volatility of market_caplet for a particular
        set of SABR parameters.
        """
        # DOCSTRING
        self.caplet.start_date = market_caplet.start_date
        self.caplet.end_date = market_caplet.end_date
        self.caplet.init_fwd = market_caplet.init_fwd
        self.caplet.strike = market_caplet.strike
        self.caplet.direction = market_caplet.direction
        self.caplet.call_put = market_caplet.call_put
        self.caplet.reference_date = market_caplet.reference_date
        self.caplet.day_count_convention = market_caplet.day_count_convention
        # self.eff_parameters()

    def implied_vol(
        self, price: float, model="black", lower_bound=0.01, upper_bound=1.0
    ):
        """ """
        # DOCSTRING
        if model.lower() == "black":

            def objective_function(volatility):
                return self._black_formula(volatility) - price

        elif model.lower() == "bachelier":

            def objective_function(volatility):
                return self._bachelier_formula(volatility) - price

        try:
            return bisect(objective_function, lower_bound, upper_bound)
        except ValueError:
            print(
                f"{price:.4f}\t[{self._bachelier_formula(lower_bound):.4f}, {self._bachelier_formula(upper_bound):.4f}] - {self.caplet.strike:.4f}",
            )

    def market_calibration(
        self,
        market_caplets: list,
        model: str,
        disp=False,
        objective="price",
        set_new_params=False,
        criterion="abs_abs",
        method="de",
    ):
        """
        Calibrate the SABR model parameters to market caplets using differential evolution.
        Does not calibrate on beta
        """
        # DOCSTRING
        market_prices = np.array(
            [market_caplet.price for market_caplet in market_caplets]
        )
        market_vols = np.array([market_caplet.vol for market_caplet in market_caplets])
        eps = 0.001  # Number slightly larger than zero in order to not divide by zero
        rho_bound = eff_rho_bounds(
            self.caplet.start(), self.caplet.end(), self.exp, self.caplet.in_accrual
        )
        print("rho_hat bound", rho_bound)
        sabr_bounds = [(eps, 0.5), (-rho_bound, rho_bound), (0.2 + eps, 1.0)]

        def objective_function(arg, market_caplets):
            if market_caplets[0].direction == "fwd":
                self.sabr["alpha"], self.sabr["rho"], self.sabr["volvol"] = (
                    arg[0],
                    arg[1],
                    arg[2],
                )
            elif market_caplets[0].direction == "bwd":
                (
                    self.sabr_eff["alpha"],
                    self.sabr_eff["rho"],
                    self.sabr_eff["volvol"],
                ) = (
                    arg[0],
                    arg[1],
                    arg[2],
                )

            if objective == "price":
                model_prices = np.array(
                    [
                        self.market_option_price(market_caplet, model=model)
                        for market_caplet in market_caplets
                    ]
                )
                return error_function(model_prices, market_prices, criterion)
            model_vols = np.array(
                [
                    self.market_equiv_vol(market_caplet, model=model)
                    for market_caplet in market_caplets
                ]
            )
            return error_function(model_vols, market_vols, criterion)

        start_time = time.time()
        if method.lower() == "de":
            result = differential_evolution(
                objective_function,
                bounds=sabr_bounds,
                args=([market_caplets]),
                disp=disp,
                callback=callback_function,
            )
        elif method.lower() == "da":
            result = dual_annealing(
                objective_function, bounds=sabr_bounds, args=([market_caplets])
            )
        else:
            raise ValueError(f"Unsupported method: {method}")
        elapsed_time = time.time() - start_time

        assert result.success, "Minimization failed."
        print(f"alpha\t{result.x[0]}")
        print(f"rho\t{result.x[1]}")
        print(f"nu\t{result.x[2]}")
        print(f"time\t{elapsed_time:.2f} s")

        if set_new_params:
            if market_caplets[0].direction == "fwd":
                self.sabr["alpha"], self.sabr["rho"], self.sabr["volvol"] = result.x
                self.eff_parameters()
            elif market_caplets[0].direction == "bwd":
                (
                    self.sabr_eff["alpha"],
                    self.sabr_eff["rho"],
                    self.sabr_eff["volvol"],
                ) = result.x
                self.orig_parameters()

        return result.x


if __name__ == "__main__":
    # Define Backwards SABR Model Parameters
    ALPHA = 0.2
    BETA = 1
    RHO = -0.2
    VOLVOL = 0.4
    Q = 1.5

    # Define Caplet
    REFERENCE_DATE = date(2023, 4, 19)
    ACCRUAL_START = date(2024, 4, 19)
    ACCRUAL_END = date(2025, 4, 19)
    DAY_COUNT_CONVENTION = "ACT365"
    INIT_FWD = 0.05
    STRIKE = 0.01
    CAPLET = Caplet(
        ACCRUAL_START,
        ACCRUAL_END,
        INIT_FWD,
        STRIKE,
        "bwd",
        "call",
        REFERENCE_DATE,
        DAY_COUNT_CONVENTION,
    )

    # Import Discount Curve
    DISCOUNT_CURVE = Curve("Data/SOFR_YC_USD_19042023", REFERENCE_DATE, "ACT365")

    # Initialize SABR Model
    SABR = SabrModel(ALPHA, BETA, RHO, VOLVOL, Q, CAPLET, DISCOUNT_CURVE)

    print("Bachelier")
    print(SABR.equivalent_volatility("bachelier"))
    price = SABR.option_price("bachelier", discount=False)
    print(price)
    print(SABR.implied_vol(price, "bachelier"))
    print(SABR.implied_vol(price + 0.0002, "bachelier"))

    # print(SABR.implied_vol(price - 0.0002, "bachelier"))
    # print("\nBlack")
    # print(SABR.equivalent_volatility("black"))
    # price = SABR.option_price("black", discount=False)
    # print(price)
    # iv = SABR.implied_vol(price, "black")
    # print(iv)
    # print(SABR.implied_vol(price + 0.0002, "black"))
