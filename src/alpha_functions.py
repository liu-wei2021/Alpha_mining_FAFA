# src/alpha_functions.py

from typing import Dict
import pandas as pd
import numpy as np

# your rolling/window helpers
from .utils.ts_functions import (
    ts_rank,
    delta,
    ts_sum,
    ts_min,
    ts_max,
    decay_linear,
    ts_argmax,
    ts_argmin,
    delay
)
# your statistical helpers
from .utils.stat_helpers import (
    stddev,
    correlation,
    rank,
    sma,
    scale,
    sign,
    covariance
)

from .utils.math_helpers import(
    product
)



class AlphaFactory:
    @staticmethod
    def alpha001(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#1: rank(Ts_ArgMax(SignedPower(((returns<0)?stddev(returns,20):close),2),5)) - 0.5
        """
        inner = df["close"].copy()
        mask = df["returns"] < 0
        inner.loc[mask] = stddev(df["returns"], window=20).loc[mask]
        # argmax over a rolling window of 5, then rank and shift to last element
        argmax = ts_argmax(inner**2, window=5)
        return rank(argmax) - 0.5

    @staticmethod
    def alpha002(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#2: -1 * corr(rank(delta(log(volume),2)), rank((close-open)/open), 6)
        """
        a = rank(delta(np.log(df["volume"]), period=2))
        b = rank((df["close"] - df["open"]) / df["open"])
        corr = correlation(a, b, window=6)
        return (-1 * corr).replace([np.inf, -np.inf], 0).fillna(0)

    @staticmethod
    def alpha003(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#3: -1 * corr(rank(open), rank(volume), 10)
        """
        a = rank(df["open"])
        b = rank(df["volume"])
        corr = correlation(a, b, window=10)
        return (-1 * corr).replace([np.inf, -np.inf], 0).fillna(0)

    # … continue the same pattern for alpha004, alpha005, etc. …
    # For example:

    @staticmethod
    def alpha004(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#4: -1 * ts_rank(rank(low), 9)
        """
        return -1 * ts_rank(rank(df["low"]), window=9)

    @staticmethod
    def alpha005(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#5: rank(open - sma(vwap,10)) * (-1 * abs(rank(close - vwap)))
        """
        part1 = rank(df["open"] - ts_sum(df["vwap"], window=10) / 10)
        part2 = -1 * abs(rank(df["close"] - df["vwap"]))
        return part1 * part2

    @classmethod
    def all_alphas(cls, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Run every alphaXXX method and collect into a dict.
        """
        out: Dict[str, pd.Series] = {}
        for name, fn in cls.__dict__.items():
            if name.startswith("alpha") and callable(fn):
                out[name] = fn(df)
        return out

    def alpha006(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#6: -1 * correlation(open, volume, 10)
        """
        corr = -1 * correlation(df["open"], df["volume"], window=10)
        return corr.replace([np.inf, -np.inf], 0).fillna(0)

    @staticmethod
    def alpha007(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#7: (adv20 < volume) ?
                  ((-1 * ts_rank(abs(delta(close,7)),60)) * sign(delta(close,7)))
                : -1
        """
        adv20 = sma(df["volume"], window=20)
        base = -1 * ts_rank(df["close"].pipe(lambda x: abs(delta(x, 7))), window=60) \
               * sign(delta(df["close"], 7))
        # if adv20 >= volume then -1
        result = base.copy()
        result.loc[adv20 >= df["volume"]] = -1
        return result

    @staticmethod
    def alpha008(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#8: -1 * rank((sum(open,5)*sum(returns,5) - delay(sum(open,5)*sum(returns,5),10)))
        """
        expr = ts_sum(df["open"], 5) * ts_sum(df["returns"], 5)
        delayed = delta(expr, period=10)
        return -1 * rank(expr - delayed)

    @staticmethod
    def alpha009(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#9: (0 < ts_min(delta(close,1),5)) ? delta(close,1)
                : ((ts_max(delta(close,1),5) < 0)? delta(close,1) : -delta(close,1))
        """
        dc = delta(df["close"], 1)
        cond1 = ts_min(dc, window=5) > 0
        cond2 = ts_max(dc, window=5) < 0

        result = -dc.copy()
        mask = cond1 | cond2
        result.loc[mask] = dc.loc[mask]
        return result

    @staticmethod
    def alpha010(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#10: rank( same logic as alpha009 )
        """
        dc = delta(df["close"], 1)
        cond1 = ts_min(dc, window=4) > 0
        cond2 = ts_max(dc, window=4) < 0

        temp = -dc.copy()
        temp.loc[cond1 | cond2] = dc.loc[cond1 | cond2]
        return rank(temp)

    @staticmethod
    def alpha011(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#11: (rank(ts_max(vwap-close,3)) + rank(ts_min(vwap-close,3))) * rank(delta(volume,3))
        """
        diff = df["vwap"] - df["close"]
        a = rank(ts_max(diff, window=3))
        b = rank(ts_min(diff, window=3))
        c = rank(delta(df["volume"], 3))
        return (a + b) * c

    @staticmethod
    def alpha012(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#12: sign(delta(volume,1)) * (-1 * delta(close,1))
        """
        return sign(delta(df["volume"], 1)) * (-1 * delta(df["close"], 1))

    def alpha013(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#13: -1 * rank(covariance(rank(close), rank(volume), 5))
        """
        cov = covariance(rank(df["close"]), rank(df["volume"]), window=5)
        return -1 * rank(cov)

    @staticmethod
    def alpha014(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#14: (-1 * rank(delta(returns, 3))) * correlation(open, volume, 10)
        """
        rd = rank(delta(df["returns"], period=3))
        corr = correlation(df["open"], df["volume"], window=10) \
               .replace([np.inf, -np.inf], 0).fillna(0)
        return -1 * rd * corr

    @staticmethod
    def alpha015(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#15: -1 * ts_sum(rank(correlation(rank(high), rank(volume), 3)), 3)
        """
        corr = correlation(rank(df["high"]), rank(df["volume"]), window=3) \
               .replace([np.inf, -np.inf], 0).fillna(0)
        return -1 * ts_sum(rank(corr), window=3)

    @staticmethod
    def alpha016(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#16: -1 * rank(covariance(rank(high), rank(volume), 5))
        """
        cov = covariance(rank(df["high"]), rank(df["volume"]), window=5)
        return -1 * rank(cov)

    @staticmethod
    def alpha017(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#17: (-1 * rank(ts_rank(close, 10))) *
                  rank(delta(delta(close, 1), 1)) *
                  rank(ts_rank(volume/adv20, 5))
        """
        adv20 = sma(df["volume"], window=20)
        p1 = rank(ts_rank(df["close"], window=10))
        p2 = rank(delta(delta(df["close"], period=1), period=1))
        p3 = rank(ts_rank(df["volume"] / adv20, window=5))
        return -1 * (p1 * p2 * p3)

    @staticmethod
    def alpha018(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#18: -1 * rank(stddev(abs(close-open),5) + (close-open) + corr(close,open,10))
        """
        diff = (df["close"] - df["open"]).abs()
        s = stddev(diff, window=5)
        corr = correlation(df["close"], df["open"], window=10) \
               .replace([np.inf, -np.inf], 0).fillna(0)
        expr = s + (df["close"] - df["open"]) + corr
        return -1 * rank(expr)

    @staticmethod
    def alpha019(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#19: (-1 * sign((close-delay(close,7)) + delta(close,7))) *
                  (1 + rank(1 + ts_sum(returns,250)))
        """
        expr1 = (df["close"] - delay(df["close"], period=7)) + delta(df["close"], period=7)
        part1 = -1 * sign(expr1)
        part2 = 1 + rank(1 + ts_sum(df["returns"], window=250))
        return part1 * part2

    @staticmethod
    def alpha020(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#20: (-1*rank(open-delay(high,1))) *
                  rank(open-delay(close,1)) *
                  rank(open-delay(low,1))
        """
        p1 = -1 * rank(df["open"] - delay(df["high"], period=1))
        p2 = rank(df["open"] - delay(df["close"], period=1))
        p3 = rank(df["open"] - delay(df["low"], period=1))
        return p1 * p2 * p3

    @staticmethod
    def alpha021(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#21:
        if (sma(close,8)+stddev(close,8)) < sma(close,2)       → -1
        elif sma(close,2) < (sma(close,8)-stddev(close,8))     → +1
        elif (volume/adv20) >= 1                               → +1
        else                                                   → -1
        """
        adv20 = sma(df["volume"], window=20)
        m1 = ts_sum(df["close"], 8) / 8 + stddev(df["close"], window=8)
        m2 = ts_sum(df["close"], 2) / 2
        cond1 = m1 < m2
        cond2 = m2 < (ts_sum(df["close"], 8) / 8 - stddev(df["close"], window=8))
        cond3 = (df["volume"] / adv20) >= 1

        out = pd.Series(-1, index=df.index)
        out.loc[cond2 & ~cond1] = 1
        out.loc[~cond1 & ~cond2 & cond3] = 1
        return out

    @staticmethod
    def alpha022(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#22: -1 * ( delta(corr(high,volume,5), 5) * rank(stddev(close,20)) )
        """
        corr = correlation(df["high"], df["volume"], window=5) \
               .replace([np.inf, -np.inf], 0).fillna(0)
        return -1 * delta(corr, period=5) * rank(stddev(df["close"], window=20))

    @staticmethod
    def alpha023(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#23: if sma(high,20) < high → -delta(high,2); else → 0
        """
        cond = sma(df["high"], window=20) < df["high"]
        out = pd.Series(0.0, index=df.index)
        out.loc[cond] = -delta(df["high"], period=2).fillna(0)
        return out

    @staticmethod
    def alpha024(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#24:
        if delta(sma(close,100),100)/delay(close,100) <= 0.05
          → -(close - ts_min(close,100))
        else
          → -delta(close,3)
        """
        frac = delta(ts_sum(df["close"], 100) / 100, period=100) / delay(df["close"], period=100)
        base = -delta(df["close"], period=3)
        out = base.copy()
        out.loc[frac <= 0.05] = -(df["close"] - ts_min(df["close"], window=100))
        return out

    @staticmethod
    def alpha025(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#25: rank( (((-1 * returns) * adv20) * vwap) * (high - close) )
        """
        adv20 = sma(df["volume"], window=20)
        expr = (-1 * df["returns"]) * adv20 * df["vwap"] * (df["high"] - df["close"])
        return rank(expr)

    @staticmethod
    def alpha026(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#26: -1 * ts_max( corr(ts_rank(volume,5), ts_rank(high,5),5), 3 )
        """
        corr = correlation(
            ts_rank(df["volume"], window=5),
            ts_rank(df["high"], window=5),
            window=5
        ).replace([np.inf, -np.inf], 0).fillna(0)
        return -1 * ts_max(corr, window=3)

    @staticmethod
    def alpha027(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#27:
        rank( ts_sum( corr(rank(volume), rank(vwap), 6), 2 ) / 2 )
        then: >0.5 → -1, else → +1
        """
        a = correlation(rank(df["volume"]), rank(df["vwap"]), window=6) \
            .replace([np.inf, -np.inf], 0).fillna(0)
        expr = ts_sum(a, window=2) / 2.0
        out = rank(expr)
        out.loc[out > 0.5] = -1
        out.loc[out <= 0.5] = 1
        return out
    
    @staticmethod
    def alpha028(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#28: scale((corr(adv20,low,5) + (high+low)/2) - close)
        """
        adv20 = sma(df["volume"], window=20)
        corr = (
            correlation(adv20, df["low"], window=5)
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )
        expr = corr + (df["high"] + df["low"]) / 2 - df["close"]
        return scale(expr)

    @staticmethod
    def alpha029(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#29: ts_min(rank(rank(scale(log(ts_sum(rank(rank(-rank(delta(close-1,5)))),2))))) ,5)
                 + ts_rank(delay(-returns,6),5)
        """
        # part A
        d = delta(df["close"] - 1, period=5)
        r1 = rank(d)
        r2 = rank(r1)
        neg = -r2
        r3 = rank(neg)
        s = ts_sum(r3, window=2)
        l = np.log(s.replace(0, np.nan)).fillna(0)
        scaled = scale(l)
        doubled = rank(rank(scaled))
        part1 = ts_min(doubled, window=5)
        # part B
        part2 = ts_rank(delay(-df["returns"], period=6), window=5)
        return part1 + part2

    @staticmethod
    def alpha030(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#30: ((1 - rank(sign(delta(close,1))+sign(delay(delta(close,1),1))+sign(delay(delta(close,1),2))))
                   * ts_sum(volume,5)) / ts_sum(volume,20)
        """
        dc = delta(df["close"], period=1)
        inner = sign(dc) + sign(delay(dc, 1)) + sign(delay(dc, 2))
        num = (1.0 - rank(inner)) * ts_sum(df["volume"], window=5)
        den = ts_sum(df["volume"], window=20)
        return num / den

    @staticmethod
    def alpha031(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#31: rank(rank(rank(decay_linear(-rank(rank(delta(close,10))),10))))
                  + rank(-delta(close,3))
                  + sign(scale(corr(adv20,low,12)))
        """
        adv20 = sma(df["volume"], window=20)
        corr = (
            correlation(adv20, df["low"], window=12)
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )
        p1 = rank(rank(rank(decay_linear(-rank(rank(delta(df["close"], 10))), period=10))))
        p2 = rank(-delta(df["close"], period=3))
        p3 = sign(scale(corr))
        return p1 + p2 + p3

    @staticmethod
    def alpha032(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#32: scale((sma(close,7)-close)) + 20 * scale(corr(vwap,delay(close,5),230))
        """
        part1 = scale(sma(df["close"], window=7) - df["close"])
        corr = correlation(df["vwap"], delay(df["close"], period=5), window=230)
        part2 = 20 * scale(corr.replace([np.inf, -np.inf], 0).fillna(0))
        return part1 + part2

    @staticmethod
    def alpha033(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#33: rank(-1 + open/close)
        """
        expr = -1 + df["open"] / df["close"]
        return rank(expr)

    @staticmethod
    def alpha034(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#34: rank((1 - rank(stddev(returns,2)/stddev(returns,5))) + (1 - rank(delta(close,1))))
        """
        ratio = stddev(df["returns"], window=2) / stddev(df["returns"], window=5)
        ratio = ratio.replace([np.inf, -np.inf], 1).fillna(1)
        expr = (1 - rank(ratio)) + (1 - rank(delta(df["close"], period=1)))
        return rank(expr)

    @staticmethod
    def alpha035(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#35: ts_rank(volume,32)*(1 - ts_rank(close+high-low,16))*(1 - ts_rank(returns,32))
        """
        return (
            ts_rank(df["volume"], window=32)
            * (1 - ts_rank(df["close"] + df["high"] - df["low"], window=16))
            * (1 - ts_rank(df["returns"], window=32))
        )

    @staticmethod
    def alpha036(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#36: combination of weighted correlations, rks, and sums as specified
        """
        adv20 = sma(df["volume"], window=20)
        term1 = 2.21 * rank(correlation(df["close"] - df["open"], delay(df["volume"], 1), window=15))
        term2 = 0.7 * rank(df["open"] - df["close"])
        term3 = 0.73 * rank(ts_rank(delay(-df["returns"], 6), window=5))
        term4 = rank(abs(correlation(df["vwap"], adv20, window=6)))
        term5 = 0.6 * rank((sma(df["close"], window=200) - df["open"]) * (df["close"] - df["open"]))
        return term1 + term2 + term3 + term4 + term5

    @staticmethod
    def alpha037(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#37: rank(corr(delay(open-close,1), close,200)) + rank(open-close)
        """
        part1 = rank(correlation(delay(df["open"] - df["close"], 1), df["close"], window=200))
        part2 = rank(df["open"] - df["close"])
        return part1 + part2

    @staticmethod
    def alpha038(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#38: -rank(ts_rank(close,10)) * rank(close/open)
        """
        inner = df["close"] / df["open"]
        inner = inner.replace([np.inf, -np.inf], 1).fillna(1)
        return -1 * rank(ts_rank(df["close"], window=10)) * rank(inner)

    @staticmethod
    def alpha039(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#39: -rank(delta(close,7)*(1 - rank(decay_linear(volume/adv20,9)))) * (1 + rank(sma(returns,250)))
        """
        adv20 = sma(df["volume"], window=20)
        decay = decay_linear((df["volume"] / adv20), period=9)
        expr = delta(df["close"], 7) * (1 - rank(decay))
        return -rank(expr) * (1 + rank(sma(df["returns"], window=250)))

    @staticmethod
    def alpha040(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#40: -rank(stddev(high,10)) * correlation(high, volume, 10)
        """
        return -rank(stddev(df["high"], window=10)) * correlation(df["high"], df["volume"], window=10)

    @staticmethod
    def alpha041(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#41: sqrt(high * low) - vwap
        """
        return np.sqrt(df["high"] * df["low"]) - df["vwap"]

    @staticmethod
    def alpha042(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#42: rank(vwap - close) / rank(vwap + close)
        """
        num = rank(df["vwap"] - df["close"])
        den = rank(df["vwap"] + df["close"])
        return num / den

    @staticmethod
    def alpha043(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#43: ts_rank(volume/adv20,20) * ts_rank(-delta(close,7),8)
        """
        adv20 = sma(df["volume"], window=20)
        part1 = ts_rank(df["volume"] / adv20, window=20)
        part2 = ts_rank(-delta(df["close"], period=7), window=8)
        return part1 * part2

    @staticmethod
    def alpha044(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#44: -1 * correlation(high, rank(volume), 5)
        """
        corr = correlation(df["high"], rank(df["volume"]), window=5)
        return (-1 * corr).replace([np.inf, -np.inf], 0).fillna(0)

    @staticmethod
    def alpha045(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#45: -1 * [ rank(sma(delay(close,5),20)) * corr(close,volume,2) * rank(corr(sum(close,5),sum(close,20),2)) ]
        """
        p1 = rank(sma(delay(df["close"], period=5), window=20))
        p2 = correlation(df["close"], df["volume"], window=2).replace([np.inf, -np.inf], 0).fillna(0)
        p3 = rank(correlation(ts_sum(df["close"], window=5),
                              ts_sum(df["close"], window=20), window=2))
        return -1 * (p1 * p2 * p3)

    @staticmethod
    def alpha046(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#46:
        inner = ((delay(close,20)-delay(close,10))/10) - ((delay(close,10)-close)/10)
        result = -delta(close,1)
        inner<0 → +1; inner>0.25 → -1
        """
        inner = ((delay(df["close"],20) - delay(df["close"],10)) / 10) \
                - ((delay(df["close"],10) - df["close"]) / 10)
        out = -delta(df["close"], period=1)
        out.loc[inner < 0]   =  1
        out.loc[inner > 0.25]= -1
        return out

    @staticmethod
    def alpha047(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#47:
        ((((rank(1/close)*volume)/adv20) * ((high*rank(high-close))/(sma(high,5)/5)))
         - rank(vwap-delay(vwap,5)))
        """
        adv20 = sma(df["volume"], window=20)
        term1 = rank(1 / df["close"]) * df["volume"] / adv20
        term2 = (df["high"] * rank(df["high"] - df["close"])) / (sma(df["high"],5) / 5)
        term3 = rank(df["vwap"] - delay(df["vwap"], period=5))
        return term1 * term2 - term3

    @staticmethod
    def alpha049(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#49:
        inner = ((delay(close,20)-delay(close,10))/10) - ((delay(close,10)-close)/10)
        if inner < -0.1 → +1 else → -delta(close,1)
        """
        inner = ((delay(df["close"],20) - delay(df["close"],10)) / 10) \
                - ((delay(df["close"],10) - df["close"]) / 10)
        out = -delta(df["close"], period=1)
        out.loc[inner < -0.1] = 1
        return out

    @staticmethod
    def alpha050(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#50: -1 * ts_max(rank(correlation(rank(volume),rank(vwap),5)),5)
        """
        corr = correlation(rank(df["volume"]), rank(df["vwap"]), window=5)
        return -1 * ts_max(rank(corr), window=5)

    @staticmethod
    def alpha051(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#51: same inner as #49 but threshold -0.05
        """
        inner = ((delay(df["close"],20) - delay(df["close"],10)) / 10) \
                - ((delay(df["close"],10) - df["close"]) / 10)
        out = -delta(df["close"], period=1)
        out.loc[inner < -0.05] = 1
        return out

    @staticmethod
    def alpha052(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#52: ((-1 * delta(ts_min(low,5),5)) *
                  rank((ts_sum(returns,240)-ts_sum(returns,20))/220)) *
                  ts_rank(volume,5)
        """
        part1 = -delta(ts_min(df["low"], window=5), period=5)
        expr  = (ts_sum(df["returns"],240) - ts_sum(df["returns"],20)) / 220
        part2 = rank(expr)
        part3 = ts_rank(df["volume"], window=5)
        return part1 * part2 * part3

    @staticmethod
    def alpha053(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#53: -1 * delta((((close-low)-(high-close))/(close-low)),9)
        """
        denom = (df["close"] - df["low"]).replace(0, 0.0001)
        inner = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / denom
        return -delta(inner, period=9)

    @staticmethod
    def alpha054(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#54: (-1 * ((low-close)*open^5)) / ((low-high)*close^5)
        """
        denom = (df["low"] - df["high"]).replace(0, -0.0001)
        num   = (df["low"] - df["close"]) * (df["open"]**5)
        return -num / (denom * (df["close"]**5))

    @staticmethod
    def alpha055(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#55: -1 * correlation(rank((close-ts_min(low,12))/(ts_max(high,12)-ts_min(low,12))),
                                   rank(volume), 6)
        """
        div = (ts_max(df["high"],12) - ts_min(df["low"],12)).replace(0,0.0001)
        inner = (df["close"] - ts_min(df["low"],12)) / div
        corr  = correlation(rank(inner), rank(df["volume"]), window=6)
        return -corr.replace([np.inf, -np.inf], 0).fillna(0)
    


    @staticmethod
    def alpha057(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#57: -(close - vwap) / decay_linear(rank(ts_argmax(close,30)), 2)
        """
        ra = ts_argmax(df["close"], window=30)
        r  = rank(ra)
        dec = decay_linear(r, period=2)
        return -(df["close"] - df["vwap"]) / dec

    @staticmethod
    def alpha060(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#60: -(2*scale(rank(((close-low)-(high-close))*volume/(high-low))) - scale(rank(ts_argmax(close,10))))
        """
        denom = (df["high"] - df["low"]).replace(0, 0.0001)
        inner = ((df["close"] - df["low"]) - (df["high"] - df["close"])) * df["volume"] / denom
        part1 = scale(rank(inner))
        part2 = scale(rank(ts_argmax(df["close"], window=10)))
        return -(2 * part1 - part2)

    @staticmethod
    def alpha061(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#61: (rank(vwap - ts_min(vwap,16)) < rank(correlation(vwap, adv180,18)))
        """
        adv180 = sma(df["volume"], window=180)
        left  = rank(df["vwap"] - ts_min(df["vwap"], window=16))
        right = rank(correlation(df["vwap"], adv180, window=18))
        return left < right

    @staticmethod
    def alpha062(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#62: ((rank(corr(vwap, sma(adv20,22),10)) < rank((rank(open)+rank(open))<(rank((high+low)/2)+rank(high)))) * -1)
        """
        adv20 = sma(df["volume"], window=20)
        a = rank(correlation(df["vwap"], sma(adv20, window=22), window=10))
        cond = (rank(df["open"]) + rank(df["open"])) < (
                   rank((df["high"] + df["low"]) / 2) + rank(df["high"])
               )
        b = rank(cond.astype(float))
        return (a < b) * -1

    @staticmethod
    def alpha064(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#64: ((rank(corr(sma(open*0.178404+low*0.821596,13), sma(adv120,13),17))
                   < rank(delta((high+low)/2*0.178404 + vwap*0.821596,4))) * -1)
        """
        adv120 = sma(df["volume"], window=120)
        expr1 = sma(df["open"] * 0.178404 + df["low"] * 0.821596, window=13)
        a = rank(correlation(expr1, sma(adv120, window=13), window=17))
        expr2 = ((df["high"] + df["low"]) / 2) * 0.178404 + df["vwap"] * 0.821596
        b = rank(delta(expr2, period=4))
        return (a < b) * -1

    @staticmethod
    def alpha065(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#65: ((rank(corr(open*0.00817205+vwap*0.99182795, sma(adv60,9),6))
                   < rank(open - ts_min(open,14))) * -1)
        """
        adv60 = sma(df["volume"], window=60)
        expr1 = df["open"] * 0.00817205 + df["vwap"] * 0.99182795
        a = rank(correlation(expr1, sma(adv60, window=9), window=6))
        b = rank(df["open"] - ts_min(df["open"], window=14))
        return (a < b) * -1

    @staticmethod
    def alpha066(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#66: (rank(decay_linear(delta(vwap,4),7)) + ts_rank(decay_linear(expr,11),7)) * -1
        where expr = ((low*0.96633 + low*0.03367) - vwap)/(open - (high+low)/2)
        """
        part1 = rank(decay_linear(delta(df["vwap"], period=4), window=7))
        expr = ((df["low"] * 0.96633 + df["low"] * 0.03367) - df["vwap"]) \
               / (df["open"] - (df["high"] + df["low"]) / 2)
        part2 = ts_rank(decay_linear(expr, window=11), window=7)
        return (part1 + part2) * -1

    @staticmethod
    def alpha068(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#68: ((ts_rank(corr(rank(high),rank(sma(volume,15)),9),14)
                   < rank(delta(close*0.518371+low*0.481629,2))) * -1)
        """
        adv15 = sma(df["volume"], window=15)
        a = ts_rank(correlation(rank(df["high"]), rank(adv15), window=9), window=14)
        expr = df["close"] * 0.518371 + df["low"] * 0.481629
        b = rank(delta(expr, period=2))
        return (a < b) * -1

    @staticmethod
    def alpha071(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#71: max(p1,p2) where
        p1 = ts_rank(decay_linear(corr(ts_rank(close,3),ts_rank(sma(volume,180),12),18),4),16)
        p2 = ts_rank(decay_linear(rank((low+open)-(vwap+vwap))**2,16),4)
        """
        adv180 = sma(df["volume"], window=180)
        p1 = ts_rank(
            decay_linear(correlation(ts_rank(df["close"],3),
                                      ts_rank(adv180,12), window=18), window=4),
            window=16
        )
        diff = (df["low"] + df["open"]) - (df["vwap"] + df["vwap"])
        p2 = ts_rank(decay_linear(rank(diff) ** 2, window=16), window=4)
        return pd.concat([p1, p2], axis=1).max(axis=1)

    @staticmethod
    def alpha072(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#72: rank(decay_linear(corr((high+low)/2, sma(volume,40),9),10))
                  / rank(decay_linear(corr(ts_rank(vwap,4),ts_rank(volume,19),7),3))
        """
        adv40 = sma(df["volume"], window=40)
        num = rank(decay_linear(correlation((df["high"] + df["low"]) / 2,
                                            adv40, window=9), window=10))
        denom = rank(decay_linear(correlation(ts_rank(df["vwap"],4),
                                              ts_rank(df["volume"],19), window=7), window=3))
        return num / denom

    @staticmethod
    def alpha073(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#73: -max(p1,p2) where
        p1 = rank(decay_linear(delta(vwap,5),3))
        p2 = ts_rank(decay_linear(delta(open*0.147155+low*0.852845,2)/expr*-1,3),17)
        """
        p1 = rank(decay_linear(delta(df["vwap"], period=5), window=3))
        expr = delta(df["open"] * 0.147155 + df["low"] * 0.852845, period=2) \
               / ((df["open"] * 0.147155) + (df["low"] * 0.852845))
        p2 = ts_rank(decay_linear(-expr, window=3), window=17)
        return pd.concat([p1, p2], axis=1).max(axis=1) * -1

    @staticmethod
    def alpha074(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#74: ((rank(corr(close, ts_sum(sma(volume,30),37),15))
                   < rank(corr(rank(high*0.0261661+vwap*0.9738339),rank(volume),11))) * -1)
        """
        adv30 = sma(df["volume"], window=30)
        sum30 = ts_sum(adv30, window=37)
        a = rank(correlation(df["close"], sum30, window=15))
        b = rank(correlation(rank(df["high"] * 0.0261661 + df["vwap"] * 0.9738339),
                             rank(df["volume"]), window=11))
        return (a < b) * -1

    @staticmethod
    def alpha075(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#75: (rank(corr(vwap,volume,4)) < rank(corr(rank(low),rank(sma(volume,50)),12)))
        """
        adv50 = sma(df["volume"], window=50)
        a = rank(correlation(df["vwap"], df["volume"], window=4))
        b = rank(correlation(rank(df["low"]), rank(adv50), window=12))
        return a < b

    @staticmethod
    def alpha077(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#77: min(p1,p2) where
        p1 = rank(decay_linear(((((high+low)/2)+high)-(vwap+high)),20))
        p2 = rank(decay_linear(corr((high+low)/2, sma(volume,40),3),6))
        """
        adv40 = sma(df["volume"], window=40)
        expr1 = (((df["high"] + df["low"]) / 2) + df["high"]) - (df["vwap"] + df["high"])
        p1 = rank(decay_linear(expr1, window=20))
        p2 = rank(decay_linear(correlation((df["high"] + df["low"]) / 2,
                                           adv40, window=3), window=6))
        return pd.concat([p1, p2], axis=1).min(axis=1)

    @staticmethod
    def alpha078(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#78: rank(corr(ts_sum(low*0.352233+vwap*0.647767,20),
                             ts_sum(sma(volume,40),20),7))
                  ** rank(corr(rank(vwap),rank(volume),6))
        """
        adv40 = sma(df["volume"], window=40)
        expr1 = ts_sum(df["low"] * 0.352233 + df["vwap"] * 0.647767, window=20)
        sum40 = ts_sum(adv40, window=20)
        a = rank(correlation(expr1, sum40, window=7))
        b = rank(correlation(rank(df["vwap"]), rank(df["volume"]), window=6))
        return a.pow(b)
    

    @staticmethod
    def alpha081(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#81:
        ((rank(log(product(rank(rank(corr(vwap, ts_sum(adv10,50),8).pow(4))),15)))
         < rank(corr(rank(vwap),rank(volume),5))) * -1
        """
        adv10 = sma(df["volume"], window=10)
        corr1 = correlation(df["vwap"], ts_sum(adv10, window=50), window=8).pow(4)
        r1 = rank(rank(corr1))
        prod = product(r1, window=15)  # elementwise or rolling product as defined
        logp = np.log(prod.replace(0, np.nan)).fillna(0)
        part1 = rank(logp)
        part2 = rank(correlation(rank(df["vwap"]), rank(df["volume"]), window=5))
        return (part1 < part2) * -1

    @staticmethod
    def alpha083(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#83:
        (rank(delay((high-low)/(ts_sum(close,5)/5),2)) * rank(rank(volume)))
        / (((high-low)/(ts_sum(close,5)/5)) / (vwap-close))
        """
        expr = (df["high"] - df["low"]) / (ts_sum(df["close"], window=5) / 5)
        num = rank(delay(expr, period=2)) * rank(rank(df["volume"]))
        den = expr / (df["vwap"] - df["close"])
        return num / den

    @staticmethod
    def alpha084(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#84:
        SignedPower(ts_rank(vwap - ts_max(vwap,15),21), delta(close,5))
        """
        base = ts_rank(df["vwap"] - ts_max(df["vwap"], window=15), window=21)
        expn = delta(df["close"], period=5)
        return np.power(base, expn)

    @staticmethod
    def alpha085(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#85:
        rank(corr(high*0.876703 + close*0.123297, adv30,10))
        ^ rank(corr(ts_rank((high+low)/2,4), ts_rank(volume,10),7))
        """
        adv30 = sma(df["volume"], window=30)
        a = rank(correlation(df["high"] * 0.876703 + df["close"] * 0.123297, adv30, window=10))
        b = rank(correlation(ts_rank((df["high"] + df["low"]) / 2, window=4),
                             ts_rank(df["volume"], window=10), window=7))
        return a.pow(b)

    @staticmethod
    def alpha086(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#86:
        (ts_rank(corr(close, ts_sum(adv20,15),6),20) < rank((open+close)-(vwap+open))) * -1
        """
        adv20 = sma(df["volume"], window=20)
        a = ts_rank(correlation(df["close"], ts_sum(adv20, window=15), window=6), window=20)
        b = rank((df["open"] + df["close"]) - (df["vwap"] + df["open"]))
        return (a < b) * -1

    @staticmethod
    def alpha088(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#88:
        min(
          rank(decay_linear(((rank(open)+rank(low))-(rank(high)+rank(close))),8)),
          ts_rank(decay_linear(corr(ts_rank(close,8), ts_rank(sma(volume,60),21),8),7),3)
        )
        """
        adv60 = sma(df["volume"], window=60)
        expr1 = (rank(df["open"]) + rank(df["low"])) - (rank(df["high"]) + rank(df["close"]))
        p1 = rank(decay_linear(expr1, window=8))
        corr2 = correlation(ts_rank(df["close"], window=8),
                            ts_rank(adv60, window=21), window=8)
        p2 = ts_rank(decay_linear(corr2, window=7), window=3)
        return pd.concat([p1, p2], axis=1).min(axis=1)
    

    @staticmethod
    def alpha092(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#92: min(
          ts_rank(decay_linear(((high+low)/2 + close < low+open), 15), 19),
          ts_rank(decay_linear(correlation(rank(low), rank(adv30), 8), 7), 7)
        )
        """
        expr1 = (((df["high"] + df["low"]) / 2) + df["close"]) < (df["low"] + df["open"])
        p1 = ts_rank(decay_linear(expr1.astype(float), window=15), window=19)

        adv30 = sma(df["volume"], window=30)
        corr2 = correlation(rank(df["low"]), rank(adv30), window=8) \
                .replace([np.inf, -np.inf], 0).fillna(0)
        p2 = ts_rank(decay_linear(corr2, window=7), window=7)

        return pd.concat([p1, p2], axis=1).min(axis=1)

    @staticmethod
    def alpha094(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#94: - ( rank(vwap - ts_min(vwap,12)) ^ ts_rank(corr(ts_rank(vwap,20), ts_rank(adv60,4),18), 3) )
        """
        adv60 = sma(df["volume"], window=60)
        a = rank(df["vwap"] - ts_min(df["vwap"], window=12))
        b = ts_rank(correlation(ts_rank(df["vwap"], window=20),
                                 ts_rank(adv60, window=4), window=18), window=3)
        return (a.pow(b)) * -1

    @staticmethod
    def alpha095(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#95: ( rank(open - ts_min(open,12)) < ts_rank(rank(corr(sum((high+low)/2,19), sum(adv40,19),13)^5), 12) )
        """
        adv40 = sma(df["volume"], window=40)
        a = rank(df["open"] - ts_min(df["open"], window=12))

        avg = (df["high"] + df["low"]) / 2
        sum1 = ts_sum(avg, window=19)
        sum2 = ts_sum(adv40, window=19)
        corr = correlation(sum1, sum2, window=13).pow(5)
        b = ts_rank(rank(corr), window=12)

        return a < b

    @staticmethod
    def alpha096(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#96: - max(
          ts_rank(decay_linear(corr(rank(vwap), rank(volume),4), 4), 8),
          ts_rank(decay_linear(ts_argmax(corr(ts_rank(close,7), ts_rank(adv60,4),4),13),14), 13)
        )
        """
        adv60 = sma(df["volume"], window=60)

        c1 = correlation(rank(df["vwap"]), rank(df["volume"]), window=4)
        p1 = ts_rank(decay_linear(c1, window=4), window=8)

        c2 = correlation(ts_rank(df["close"], window=7), ts_rank(adv60, window=4), window=4)
        idx = ts_argmax(c2, window=13)
        p2  = ts_rank(decay_linear(idx, window=14), window=13)

        return pd.concat([p1, p2], axis=1).max(axis=1) * -1

    @staticmethod
    def alpha098(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#98: rank(decay_linear(corr(vwap, ts_sum(adv5,26),5),7))
                  - rank(decay_linear(ts_rank(ts_argmin(corr(rank(open), rank(adv15),21),9),7),8))
        """
        adv5  = sma(df["volume"], window=5)
        adv15 = sma(df["volume"], window=15)

        c1 = correlation(df["vwap"], ts_sum(adv5, window=26), window=5)
        p1 = rank(decay_linear(c1, window=7))

        c2 = correlation(rank(df["open"]), rank(adv15), window=21)
        idx = ts_argmin(c2, window=9)
        p2  = rank(decay_linear(ts_rank(idx, window=7), window=8))

        return p1 - p2

    @staticmethod
    def alpha099(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#99: ( rank(corr(ts_sum((high+low)/2,20), ts_sum(adv60,20),9))
                   < rank(corr(low, volume,6)) ) * -1
        """
        adv60 = sma(df["volume"], window=60)

        sum1 = ts_sum((df["high"] + df["low"]) / 2, window=20)
        sum2 = ts_sum(adv60, window=20)
        a    = rank(correlation(sum1, sum2, window=9))

        b    = rank(correlation(df["low"], df["volume"], window=6))

        return (a < b) * -1

    @staticmethod
    def alpha101(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#101: (close - open) / (high - low + 0.001)
        """
        return (df["close"] - df["open"]) / ((df["high"] - df["low"]) + 0.001)





    @classmethod
    def all_alphas(cls, df: pd.DataFrame) -> Dict[str, pd.Series]:
        out: Dict[str, pd.Series] = {}
        for name, fn in cls.__dict__.items():
            if name.startswith("alpha") and callable(fn):
                out[name] = fn(df)
        return out