"""
SABR Model Implementation

This module implements the SABR model for pricing options on various underlying assets. 
The SABR model is a stochastic volatility model that was introduced by Hagan et al. 
in their 2002 paper "Managing Smile Risk".

References:
    [1] Willems, S. (2020). SABR smiles for RFR caplets.
    [2] Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002). Managing smile risk.
    [3] Taipe, M. (2022). Tuning the FMM-SABR for RFR caplets.
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
        alpha: list,
        beta: float,
        rho: list,
        volvol: list,
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

        self.sabr_eff = {
            "alpha": alpha.copy(),
            "beta": beta,
            "rho": rho.copy(),
            "volvol": volvol.copy(),
        }
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
        Function that determines Δ^2 in Appendix B [3].
        """
        # DOCSTRING
        delta_0 = (
            self.sabr["alpha"][0] ** 2 * self.caplet.start()
            - (self.sabr["alpha"][1] ** 2 * (self.caplet.start() - self.caplet.end()))
            / (2 * self.exp + 1)
        ) / self.caplet.end()
        delta_1 = ((self.sabr["alpha"][1] ** 2) * self._rel_time(-2 * self.exp)) / (
            2 * self.exp + 1
        )
        return delta_0, delta_1

    def _b(self):
        """
        Function that determines b in Appendix B [3].
        """
        # DOCSTRING
        # TODO Also calculate b_1
        delta_0 = self._delta_sqr()[0]

        b_0_1 = (
            self.sabr["alpha"][0]
            * self.sabr["volvol"][0]
            * self.sabr["rho"][0]
            * self.caplet.start()
            * (
                self.sabr["alpha"][0] ** 2 * self.caplet.start() * (2 * self.exp + 1)
                + 2
                * self.sabr["alpha"][1] ** 2
                * (self.caplet.end() - self.caplet.start())
            )
        ) / (2 * (2 * self.exp + 1))
        b_0_2 = (
            self.sabr["alpha"][1] ** 3
            * self.sabr["volvol"][1]
            * self.sabr["rho"][1]
            * (self.caplet.end() - self.caplet.start()) ** 2
        ) / ((2 * self.exp + 1) * (3 * self.exp + 2))

        b_0 = (2 * (b_0_1 + b_0_2)) / (delta_0**2 * self.caplet.end() ** 2)

        b_1 = (
            (self.sabr["volvol"][1] * self.sabr["rho"][1] * (4 * self.exp + 2))
            / (self.sabr["alpha"][1] * (3 * self.exp + 2))
        ) * self._rel_time(self.exp)
        return b_0, b_1

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

    def _lambda(self):
        """
        Determines lambda as in Appendix B [3]
        """
        return self.sabr["alpha"][0] ** 2 * (
            self.caplet.start() + 2 * self.exp * self.caplet.start()
        ) + self.sabr["alpha"][1] ** 2 * (self.caplet.end() - self.caplet.start())

    def _c(self):
        b_0, b_1 = self._b()

        tau_1 = self.sabr["alpha"][0] ** 2 * self.caplet.start() + (
            self.sabr["alpha"][1] ** 2 * (self.caplet.end() - self.caplet.start())
        ) / (2 * self.exp + 1)

        # Compute Integral 1
        int_1_1 = (
            self.sabr["volvol"][0] ** 2
            * self.sabr["alpha"][0] ** 4
            * self.caplet.start() ** 3
        ) / 3
        int_1_2 = (
            self.sabr["volvol"][0] ** 2
            * self.sabr["alpha"][0] ** 2
            * self.sabr["alpha"][1] ** 2
            * (self.caplet.end() - self.caplet.start())
            * self.caplet.start() ** 2
        ) / (2 * self.exp + 1)
        int_1_3 = (
            self.sabr["volvol"][0] ** 2
            * self.sabr["alpha"][1] ** 4
            * (self.caplet.end() - self.caplet.start()) ** 2
            * self.caplet.start()
        ) / ((2 * self.exp + 1) ** 2)
        int_1_4 = (
            self.sabr["volvol"][1] ** 2
            * self.sabr["alpha"][1] ** 4
            * (self.caplet.end() - self.caplet.start()) ** 3
        ) / ((2 * self.exp + 1) ** 2 * (4 * self.exp + 3))
        int_1 = int_1_1 + int_1_2 + int_1_3 + int_1_4

        # Compute Integral 2
        int_2_1 = (
            self.sabr["rho"][0] ** 2
            * self.sabr["volvol"][0] ** 2
            * self.sabr["alpha"][0] ** 4
            * self.caplet.start() ** 3
        ) / 3
        int_2_2 = (
            self.sabr["alpha"][0] ** 2
            * self.sabr["volvol"][0] ** 2
            * self.sabr["rho"][0] ** 2
            * self.caplet.start() ** 2
        ) / (2 * self.exp + 1)
        int_2_3 = (
            2
            * self.sabr["alpha"][0]
            * self.sabr["alpha"][1]
            * self.sabr["volvol"][0]
            * self.sabr["volvol"][1]
            * self.sabr["rho"][0]
            * self.sabr["rho"][1]
            * self.caplet.start()
            * (self.caplet.end() - self.caplet.start())
        ) / ((2 * self.exp + 1) * (3 * self.exp + 2))
        int_2_4 = (
            2
            * self.sabr["alpha"][1] ** 2
            * self.sabr["volvol"][1] ** 2
            * self.sabr["rho"][1] ** 2
            * (self.caplet.end() - self.caplet.start()) ** 2
        ) / ((2 * self.exp + 1) * (3 * self.exp + 2) * (4 * self.exp + 3))
        int_2 = int_2_1 + self.sabr["alpha"][1] ** 2 * (
            self.caplet.end() - self.caplet.start()
        ) * (int_2_2 + int_2_3 + int_2_4)

        c0 = (3 / (tau_1**3)) * (int_1 + 3 * int_2) - 3 * b_0**2
        return c0, np.nan

    def _G(self):
        """
        Function that determines G in Appendix B [3].
        """
        # DOCSTRING
        # TODO Also calculate G_1
        delta_0, delta_1 = self._delta_sqr()
        c_0, c_1 = self._c()

        g_0_nu_0 = (self.sabr["alpha"][0] ** 2 * self.caplet.start() ** 2) / 2 + (
            self.sabr["alpha"][1] ** 2
            * self.caplet.start()
            * (self.caplet.end() - self.caplet.start())
        ) / (2 * self.exp + 1)
        g_0_nu_1 = (
            self.sabr["alpha"][1] ** 2 * (self.caplet.end() - self.caplet.start()) ** 2
        ) / ((2 * self.exp + 1) * (2 * self.exp + 2))
        g_0_int = (
            self.sabr["volvol"][0] ** 2 * g_0_nu_0
            + self.sabr["volvol"][1] ** 2 * g_0_nu_1
        )
        tau_t1 = self.caplet.end() * delta_0

        g_0 = (2 * g_0_int) / ((self.caplet.end() * delta_0) ** 2) - c_0

        # TEST G_1 calculations
        g_1 = ((self.sabr["volvol"][1] ** 2) / (delta_1 * (self.exp + 1))) - c_1

        return g_0, g_1

    def eff_parameters(self):
        """
        Determines the effective SABR parameters based on the
        expressions b, c, G and Δ^2 in Appendix B [3].
        """
        # DOCSTRING
        delta_0_sqr, delta_1_sqr = self._delta_sqr()
        b_0, b_1 = self._b()
        c_0, c_1 = self._c()
        g_0, g_1 = self._G()

        self.sabr_eff["alpha"][0] = np.sqrt(delta_0_sqr) * np.exp(
            (delta_0_sqr * g_0 * self.caplet.end()) / 4
        )
        # self.sabr_eff["alpha"][1] = np.sqrt(delta_1_sqr) * np.exp(
        #     (delta_1_sqr * g_1 * self.caplet.end()) / 4
        # )

        self.sabr_eff["rho"][0] = b_0 / np.sqrt(c_0)
        # self.sabr_eff["rho"][1] = b_1 / np.sqrt(c_1)

        self.sabr_eff["volvol"][0] = np.sqrt(delta_0_sqr) * np.sqrt(c_0)
        # self.sabr_eff["volvol"][1] = np.sqrt(delta_1) * np.sqrt(c_1)

        return (
            self.sabr_eff["alpha"][0],
            self.sabr_eff["rho"][0],
            self.sabr_eff["volvol"][0],
        )

    def orig_parameters(self, alpha_eff, rho_eff, volvol_eff):
        """
        Function that determines the original SABR parameters from the
        set of effective SABR parameters.
        """

        def objective_function(arg, eff_params):
            self.sabr = {
                "alpha": [arg[0], arg[1]],
                "beta": self.sabr["beta"],
                "rho": [arg[2], arg[3]],
                "volvol": [arg[4], arg[5]],
            }
            alpha, rho, volvol = self.eff_parameters()
            return error_function(np.array([alpha, rho, volvol]), np.array(eff_params))

        eps = 0.001
        sabr_bounds = [
            (eps, 0.2),
            (eps, 0.2),
            (-1 + eps, 1 - eps),
            (-1 + eps, 1 - eps),
            (eps, 1),
            (eps, 1),
        ]
        eff_params = [alpha_eff, rho_eff, volvol_eff]

        result = differential_evolution(
            objective_function,
            bounds=sabr_bounds,
            args=[eff_params],
            disp=True,
            polish=False,
        )

        return result.x

    def _equivalent_volatility_1(self, model: str):
        """ """
        # DOCSTRING
        # TEST for in_accrual
        # TODO set correct SABR parameters in _equivalent_volatility

        # Chooses the right set of SABR parameters
        if self.caplet.direction == "bwd" and not self.caplet.in_accrual:
            sabr = {
                "alpha": self.sabr_eff["alpha"][0],
                "beta": self.sabr_eff["beta"],
                "rho": self.sabr_eff["rho"][0],
                "volvol": self.sabr_eff["volvol"][0],
            }
        elif self.caplet.direction == "bwd" and self.caplet.in_accrual:
            sabr = {
                "alpha": self.sabr_eff["alpha"][1],
                "beta": self.sabr_eff["beta"],
                "rho": self.sabr_eff["rho"][1],
                "volvol": self.sabr_eff["volvol"][1],
            }
        else:
            sabr = {
                "alpha": self.sabr["alpha"][0],
                "beta": self.sabr["beta"],
                "rho": self.sabr["rho"][0],
                "volvol": self.sabr["volvol"][0],
            }

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
        if self.caplet.direction == "bwd" and not self.caplet.in_accrual:
            sabr = {
                "alpha": self.sabr_eff["alpha"][0],
                "beta": self.sabr_eff["beta"],
                "rho": self.sabr_eff["rho"][0],
                "volvol": self.sabr_eff["volvol"][0],
            }
        elif self.caplet.direction == "bwd" and self.caplet.in_accrual:
            sabr = {
                "alpha": self.sabr_eff["alpha"][1],
                "beta": self.sabr_eff["beta"],
                "rho": self.sabr_eff["rho"][1],
                "volvol": self.sabr_eff["volvol"][1],
            }
        else:
            sabr = {
                "alpha": self.sabr["alpha"][0],
                "beta": self.sabr["beta"],
                "rho": self.sabr["rho"][0],
                "volvol": self.sabr["volvol"][0],
            }

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

    def option_premium(self, model: str):
        vol = self.equivalent_volatility(model)
        if model == "bachelier":
            return self._bachelier_formula(vol)
        return self._black_formula(vol)

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

    def market_calibration(
        self,
        market_caplets: list,
        model: str,
        disp=False,
        set_new_params=False,
        criterion="abs_abs",
    ):
        market_vols = np.array([market_caplet.vol for market_caplet in market_caplets])
        eps = 0.001
        sabr_bounds = [
            (eps, 0.2),
            (eps, 0.2),
            (-1 + eps, 1 - eps),
            (-1 + eps, 1 - eps),
            (eps, 1),
            (eps, 1),
        ]

        def objective_function(arg, market_caplets):
            self.sabr = {
                "alpha": [arg[0], arg[1]],
                "beta": self.sabr["beta"],
                "rho": [arg[2], arg[3]],
                "volvol": [arg[4], arg[5]],
            }
            self.eff_parameters()

            model_vols = np.array(
                [
                    self.market_equiv_vol(market_caplet, model=model)
                    for market_caplet in market_caplets
                ]
            )
            return error_function(model_vols, market_vols, criterion)

        start_time = time.time()
        result = differential_evolution(
            objective_function,
            bounds=sabr_bounds,
            args=([market_caplets]),
            disp=disp,
            callback=callback_function,
        )
        elapsed_time = time.time() - start_time

        assert result.success, "Minimization failed."
        print(result.x)
        print(f"time\t{elapsed_time:.2f} s")

        # if set_new_params:
        #     if market_caplets[0].direction == "fwd":
        #         self.sabr["alpha"], self.sabr["rho"], self.sabr["volvol"] = result.x
        #         self.eff_parameters()
        #     elif market_caplets[0].direction == "bwd":
        #         (
        #             self.sabr_eff["alpha"],
        #             self.sabr_eff["rho"],
        #             self.sabr_eff["volvol"],
        #         ) = result.x
        #         self.orig_parameters()

        return result.x


if __name__ == "__main__":
    # Define Backwards SABR Model Parameters
    ALPHA = [0.04, 0.025]
    BETA = 0
    RHO = [-0.4, -0.4]
    VOLVOL = [0.4, 0.48]
    Q = 1

    # Define Caplet
    REFERENCE_DATE = date(2023, 4, 19)
    ACCRUAL_START = date(2024, 4, 19)
    ACCRUAL_END = date(2025, 4, 19)
    DAY_COUNT_CONVENTION = "ACT365"
    INIT_FWD = 0.05
    STRIKE = 0.04

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

    alpha_res = SABR.sabr_eff["alpha"][0]
    rho_res = SABR.sabr_eff["rho"][0]
    volvol_res = SABR.sabr_eff["volvol"][0]

    print(SABR.orig_parameters(alpha_res, rho_res, volvol_res))
