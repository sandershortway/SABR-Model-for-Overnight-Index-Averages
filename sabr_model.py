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
from scipy.optimize import bisect, differential_evolution
from discount_curve import Curve


class Caplet:
    """
    A class that holds the necessary information for a OIA caplet/floorlet
    """

    def __init__(
        self,
        start: float,
        end: float,
        init_fwd: float,
        strike: float,
        direction: str,
        call_put: str,
    ):
        self.start = start
        self.end = end
        self.init_fwd = init_fwd
        self.strike = strike
        self.direction = direction
        self.call_put = call_put

        assert direction in ["fwd", "bwd"]
        assert call_put in ["call", "put"]
        assert init_fwd > 0
        assert strike > 0
        self.in_accrual = start < 0


class MarketCaplet(Caplet):
    """
    A class that holds the necessary information for a OIA caplet/floorlet that
    we observe in the market, it holds the contract details but additionally
    a market price or market volatility
    """

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
    ):
        super().__init__(start, end, init_fwd, strike, direction, call_put)
        self.price = price
        self.vol = vol


class SabrModel:
    """
    A class that implements the SABR model for pricing caplets on overnight index averages.
    Notation is taken from [1].
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        rho: float,
        volvol: float,
        exp: float,
        caplet: Caplet,
    ):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.volvol = volvol
        self.exp = exp
        self.caplet = caplet
        self.alpha_eff = alpha
        self.rho_eff = rho
        self.volvol_eff = volvol
        self.tau = 2 * self.exp * self.caplet.start + self.caplet.end

    def _chi(self, zeta):
        """
        Function that determines Chi(zeta) as in equation (2.17c) [1].
        """
        numerator = np.sqrt(1 - 2 * self.rho * zeta + zeta**2) + zeta - self.rho
        return np.log(numerator / (1 - self.rho))

    def _rel_time(self, exp):
        """
        Function that determines ((tau_1 - tau_0)/(tau_1))^exp [1].
        """
        return ((self.caplet.end - self.caplet.start) / self.caplet.end) ** exp

    def _delta_sqr(self):
        """
        Function that determines Δ^2 in Appendix C [1].
        """
        if self.caplet.in_accrual:  # Theorem 4.1 in [1]
            return (self.alpha**2) / (
                (2 * self.exp + 1) * self._rel_time(2 * self.exp)
            )
        # Theorem 4.2 in [1]
        return (self.alpha**2 * self.tau) / (self.caplet.end * (2 * self.exp + 1))

    def _b(self):
        """
        Function that determines b in Appendix C [1].
        """
        if self.caplet.in_accrual:  # Theorem 4.1 in [1]
            term_1 = (self.volvol * self.rho * (4 * self.exp + 2)) / (
                self.alpha * (3 * self.exp + 2)
            )
            return term_1 * self._rel_time(self.exp)
        # Theorem 4.2 in [1]
        term_1 = (self.volvol * self.rho * (2 * self.exp + 1)) / self.alpha
        term_2_num = (
            3 * self.tau**2
            + 2 * self.exp * self.caplet.start**2
            + self.caplet.end**2
        )
        term_2_denom = 2 * self.tau**2 * (3 * self.exp + 2)
        return term_1 * (term_2_num / term_2_denom)

    def _c(self):
        """
        Function that determines c in Appendix C [1].
        """
        if self.caplet.in_accrual:  # Theorem 4.1 in [1]
            term_1 = (
                self.volvol**2 * self._rel_time(2**self.exp)
            ) / self.alpha**2
            term_2 = (2 * self.exp + 1) / ((4 * self.exp + 3) * (3 * self.exp + 2) ** 2)
            gamma_1 = (3 * self.exp + 2) ** 2
            gamma_2 = 4 * self.exp**2 + 2 * self.exp
            gamma = gamma_1 + self.rho**2 * gamma_2
            return term_1 * term_2 * gamma

        # Theorem 4.2 in [1]
        term_1 = (self.volvol**2 * (2 * self.exp + 1) ** 2) / (
            self.alpha**2 * self.tau**4
        )
        gamma_1_num = (
            2 * self.tau**3
            + self.caplet.end**3
            + (4 * self.exp**2 - 2 * self.exp) * self.caplet.start**3
            + 6 * self.exp * self.caplet.start**2 * self.caplet.end
        )
        gamma_1 = (self.tau * gamma_1_num) / ((4 * self.exp + 3) * (2 * self.exp + 1))

        gamma_2_num = (
            3 * self.tau**2
            - self.caplet.end**2
            + 5 * self.exp * self.caplet.start**2
            + 4 * self.caplet.start * self.caplet.end
        )
        gamma_2 = gamma_2_num / ((4 * self.exp + 3) * (3 * self.exp + 2) ** 2)

        gamma = (
            gamma_1
            + 3
            * self.exp
            * self.rho**2
            * (self.caplet.end - self.caplet.start) ** 2
            * gamma_2
        )

        return term_1 * gamma

    def _g(self):
        """
        Function that determines G in Appendix C [1].
        """
        if self.caplet.in_accrual:  # Theorem 4.1 in [1]
            return self.volvol**2 / (self._delta_sqr() * (self.exp + 1)) - self._c()
        # Theorem 4.2 in [1]
        num = (self.volvol**2) * (
            self.tau**2 + 2 * self.exp * self.caplet.start**2 + self.caplet.end**2
        )
        denom = self._delta_sqr() * (2 * self.caplet.end * self.tau * (self.exp + 1))
        return (num / denom) - self._c()

    def _eff_parameters(self):
        """
        Function that determines the effective SABR parameters based on the
        expressions b, c, G and Δ^2 in Appendix C [1].
        """
        delta = np.sqrt(self._delta_sqr())
        b = self._b()
        c = self._c()
        g = self._g()

        self.alpha_eff = delta * np.exp((1 / 4) * delta**2 * g * self.caplet.end)
        self.rho_eff = b / np.sqrt(c)
        self.volvol_eff = delta * np.sqrt(c)

    def show(self):
        """
        Function that neatly prints parameters, effective parameters,
        Black implied volatility and cap/floor price.
        """
        print("Today\t\t0")
        print(f"Accrual\t\t[{self.caplet.start}, {self.caplet.end}]")
        if self.caplet.in_accrual:
            print("Theorem\t\tWillems 4.1 - in accrual", "\n")
        else:
            print("Theorem\t\tWillems 4.2 - before accrual", "\n")

        self._eff_parameters()
        print("Param\t\tValue\tEff Value")
        print(f"alpha\t\t{self.alpha}\t{self.alpha_eff}")
        print(f"beta\t\t{self.beta}\t-")
        print(f"rho\t\t{self.rho}\t{self.rho_eff}")
        print(f"volvol\t\t{self.volvol}\t{self.volvol_eff}\n")

        print(f"imp vol\t\t{self.black_equiv_vol()}")
        print(f"forward\t\t{self.caplet.init_fwd}")
        print(f"strike\t\t{self.caplet.strike}")
        if self.caplet.call_put == "call":
            print(f"caplet\t\t{self.option_premium()}")
        else:
            print(f"floorlet\t{self.option_premium()}")

    def black_equiv_vol(self):
        """
        Function that computes the implied volatility for the Black model
        to obtain option premium.
        """
        # Compute i_1
        if np.isclose(self.caplet.init_fwd, self.caplet.strike):
            i_1 = self.alpha_eff * self.caplet.init_fwd ** (self.beta - 1)
        else:
            zeta = (self.volvol_eff / self.alpha_eff) * (
                (self.caplet.init_fwd * self.caplet.strike)
                ** ((1 - self.beta**2) / 2)
                * np.log(self.caplet.init_fwd / self.caplet.strike)
            )
            chi_zeta = self._chi(zeta)

            denom_2 = ((1 - self.beta) ** 2 / 24) * (
                np.log(self.caplet.init_fwd / self.caplet.strike)
            ) ** 2
            denom_3 = ((1 - self.beta) ** 2 / 1920) * (
                np.log(self.caplet.init_fwd / self.caplet.strike)
            ) ** 4
            denom = (1 + denom_2 + denom_3) * chi_zeta

            i_1 = (
                self.alpha_eff
                * zeta
                * ((self.caplet.init_fwd * self.caplet.strike) ** ((self.beta - 1) / 2))
            ) / denom

        # Compute i_2
        i_2_1 = (
            (self.alpha_eff**2)
            * (1 - self.beta) ** 2
            * (self.caplet.init_fwd * self.caplet.strike) ** (self.beta - 1)
        ) / 24
        i_2_2 = (
            self.alpha_eff
            * self.beta
            * self.rho_eff
            * self.volvol_eff
            * (self.caplet.init_fwd * self.caplet.strike) ** ((self.beta - 1) / 2)
        ) / 4
        i_2_3 = ((2 - 3 * self.rho_eff**2) * self.volvol_eff**2) / 24

        i_2 = i_2_1 + i_2_2 + i_2_3

        return i_1 * (1 + self.caplet.end * i_2)

    def _black_formula(self, volatility):
        """ """
        d_plus = (
            np.log(self.caplet.init_fwd / self.caplet.strike)
            + 0.5 * volatility**2 * self.caplet.end
        ) / (volatility * np.sqrt(self.caplet.end))
        d_min = (
            np.log(self.caplet.init_fwd / self.caplet.strike)
            - 0.5 * volatility**2 * self.caplet.end
        ) / (volatility * np.sqrt(self.caplet.end))

        call_premium = self.caplet.init_fwd * norm.cdf(
            d_plus
        ) - self.caplet.strike * norm.cdf(d_min)

        if self.caplet.call_put == "call":
            return call_premium
        return call_premium + (self.caplet.strike - self.caplet.init_fwd)

    def option_premium(self):
        """
        Function that determines the option premium of a cap/floor
        using the Black option pricing formula
        """
        self._eff_parameters()
        black_vol = self.black_equiv_vol()
        return self._black_formula(black_vol)

    def market_option_premium(self, market_caplet: MarketCaplet):
        """
        Function that determines the option premium of market_caplet
        based on the current set of SABR parameters.
        """
        self._set_market_caplet_as_inner_caplet(market_caplet)
        self._eff_parameters()
        return self.option_premium()

    def market_black_equiv_vol(self, market_caplet: MarketCaplet):
        """
        Function that determines the Black model equivalent volatility of market_caplet
        based on the current set of SABR parameters.
        """
        self._set_market_caplet_as_inner_caplet(market_caplet)
        self._eff_parameters()
        return self.black_equiv_vol()

    def _set_market_caplet_as_inner_caplet(self, market_caplet: MarketCaplet):
        """
        Function that sets market_caplet as self.caplet, makes it possible to compute
        the option premium or black equivalent volatility of market_caplet for a particular
        set of SABR parameters.
        """
        self.caplet.start = market_caplet.start
        self.caplet.end = market_caplet.end
        self.caplet.init_fwd = market_caplet.init_fwd
        self.caplet.strike = market_caplet.strike
        self.caplet.direction = market_caplet.direction
        self.caplet.call_put = market_caplet.call_put

    def black_implied_vol(self, price):
        """
        Compute the Black model implied volatility based on the price of the option
        using the bisection root finding method.
        """

        def objective_function(volatility):
            return self._black_formula(volatility) - price

        return bisect(objective_function, 0.0001, 1)

    def market_calibration(self, market_caplets, disp=False, objective="price"):
        """
        Calibrate the SABR model parameters to market caplets using differential evolution.
        """
        market_prices = np.array(
            [market_caplet.price for market_caplet in market_caplets]
        )
        market_vols = np.array([market_caplet.vol for market_caplet in market_caplets])

        def objective_function(sabr, market_caplets, market_prices):
            self.alpha, self.beta, self.rho, self.volvol = (
                sabr[0],
                sabr[1],
                sabr[2],
                sabr[3],
            )
            if objective == "price":
                model_prices = np.array(
                    [
                        self.market_option_premium(market_caplet)
                        for market_caplet in market_caplets
                    ]
                )
                return np.sum(np.abs(model_prices - market_prices))
            model_vols = np.array(
                [
                    self.market_black_equiv_vol(market_caplet)
                    for market_caplet in market_caplets
                ]
            )
            return np.sum(np.abs(model_vols - market_vols))

        start_time = time.time()
        result = differential_evolution(
            objective_function,
            bounds=[(0.0001, 1), (0.0001, 1), (-1, 1), (0.0001, 1)],
            args=(market_caplets, market_prices),
            disp=disp,
        )
        elapsed_time = time.time() - start_time

        assert result.success, "Differential Evolution failed."
        print("Differential Evolution Results")
        print(f"alpha\t{result.x[0]}")
        print(f"beta\t{result.x[1]}")
        print(f"rho\t{result.x[2]}")
        print(f"nu\t{result.x[3]}\n")
        print(f"time\t{elapsed_time:.2f} s")


if __name__ == "__main__":
    # Define Backwards SABR Model Parameters
    ALPHA = 0.1
    BETA = 0
    RHO = -0.5
    VOLVOL = 0.5
    Q = 1

    # Define caplet object
    REFERENCE_DATE = date(2023, 4, 19)
    ACCRUAL_START = 0.5
    ACCRUAL_END = 1.0
    INIT_FWD = 0.05
    STRIKE = 0.05
    CAPLET = Caplet(ACCRUAL_START, ACCRUAL_END, INIT_FWD, STRIKE, "bwd", "call")

    # Initialize SABR Model
    SABR = SabrModel(ALPHA, BETA, RHO, VOLVOL, Q, CAPLET)

    # Market Calibration Example
    MARKET_CAPLETS = []

    DISCOUNT_CURVE = Curve("SOFR_YC_USD_19042023", REFERENCE_DATE, "ACT365")

    for STRIKE_VALUE in np.arange(0.03, 0.07, 0.001):
        CAPLET.strike = STRIKE_VALUE
        PRICE = SABR.option_premium()
        VOL = SABR.black_equiv_vol()

        MARKET_CAPLETS.append(
            MarketCaplet(
                ACCRUAL_START,
                ACCRUAL_END,
                INIT_FWD,
                STRIKE_VALUE,
                "bwd",
                "call",
                PRICE,
                VOL,
            )
        )

    SABR.market_calibration(MARKET_CAPLETS)
    SABR.market_calibration(MARKET_CAPLETS, objective="vol")
