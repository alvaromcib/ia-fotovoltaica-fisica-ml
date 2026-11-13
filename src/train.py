"""PV plant production: physical vs machine-learning vs hybrid model,
plus detection of under-performing inverters.

Data: public 34-day dataset of a ~22-inverter PV plant (15-min resolution),
generation per inverter + plant weather sensor (irradiation, ambient & module temperature).

Usage: python src/train.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "results"
OUT.mkdir(exist_ok=True)

GAMMA = -0.004        # power temperature coefficient of c-Si modules (1/°C)
TEST_DAYS = 10        # last days held out
PRICE_EUR_MWH = 60    # illustrative energy price for the loss estimate


def load():
    gen = pd.read_csv(DATA / "Plant_1_Generation_Data.csv")
    gen["ts"] = pd.to_datetime(gen["DATE_TIME"], format="%d-%m-%Y %H:%M")
    wx = pd.read_csv(DATA / "Plant_1_Weather_Sensor_Data.csv", parse_dates=["DATE_TIME"])
    wx = wx.rename(columns={"DATE_TIME": "ts", "IRRADIATION": "G",
                            "AMBIENT_TEMPERATURE": "t_amb", "MODULE_TEMPERATURE": "t_mod"})
    return gen, wx[["ts", "G", "t_amb", "t_mod"]]


def physical(df, k):
    """Expected AC power from irradiance and module temperature."""
    return np.clip(k * df["G"] * (1 + GAMMA * (df["t_mod"] - 25)), 0, None)


def metrics(y, p, cap):
    e = p - y
    day = y > 0.02 * cap
    return {"nrmse_pct": round(100 * np.sqrt(np.mean(e[day] ** 2)) / cap, 2),
            "nmae_pct": round(100 * np.mean(np.abs(e[day])) / cap, 2),
            "energy_error_pct": round(100 * (p.sum() - y.sum()) / y.sum(), 2)}


def main():
    gen, wx = load()

    # ---------------- Plant-level model ----------------
    plant = gen.groupby("ts", as_index=False)["AC_POWER"].sum().rename(columns={"AC_POWER": "P"})
    df = plant.merge(wx, on="ts", how="inner").sort_values("ts").reset_index(drop=True)
    df["P"] /= 1000  # kW -> MW
    df["hour"] = df["ts"].dt.hour + df["ts"].dt.minute / 60
    df["doy"] = df["ts"].dt.dayofyear
    cap = float(df["P"].quantile(0.999))

    cut = df["ts"].dt.normalize().max() - pd.Timedelta(days=TEST_DAYS - 1)
    tr, te = df[df["ts"] < cut].copy(), df[df["ts"] >= cut].copy()

    # Physical: single gain k fitted by least squares on training data (one parameter)
    x = tr["G"] * (1 + GAMMA * (tr["t_mod"] - 25))
    k = float((x * tr["P"]).sum() / (x * x).sum())
    te["p_phys"] = physical(te, k)
    tr["p_phys"] = physical(tr, k)

    feats = ["G", "t_amb", "t_mod", "hour"]
    params = dict(objective="regression", learning_rate=0.05, num_leaves=15, min_data_in_leaf=20,
                  verbose=-1, seed=7)
    ml = lgb.train(params, lgb.Dataset(tr[feats], tr["P"]), 400)
    te["p_ml"] = np.clip(ml.predict(te[feats]), 0, None)

    # Hybrid: ML learns only the residual of the physical model
    hy = lgb.train(params, lgb.Dataset(tr[feats + ["p_phys"]], tr["P"] - tr["p_phys"]), 400)
    te["p_hyb"] = np.clip(te["p_phys"] + hy.predict(te[feats + ["p_phys"]]), 0, None)

    y = te["P"].values
    models = {"Físico": metrics(y, te["p_phys"].values, cap),
              "Machine learning": metrics(y, te["p_ml"].values, cap),
              "Híbrido": metrics(y, te["p_hyb"].values, cap)}

    # Robustness: train on only 5 days (data-scarce new plant)
    tr5 = tr[tr["ts"] < tr["ts"].min().normalize() + pd.Timedelta(days=5)]
    x5 = tr5["G"] * (1 + GAMMA * (tr5["t_mod"] - 25))
    k5 = float((x5 * tr5["P"]).sum() / (x5 * x5).sum())
    ml5 = lgb.train(params, lgb.Dataset(tr5[feats], tr5["P"]), 400)
    tr5 = tr5.assign(p_phys=physical(tr5, k5))
    hy5 = lgb.train(params, lgb.Dataset(tr5[feats + ["p_phys"]], tr5["P"] - tr5["p_phys"]), 400)
    te5_phys = physical(te, k5)
    small = {"Físico": metrics(y, te5_phys.values, cap),
             "Machine learning": metrics(y, np.clip(ml5.predict(te[feats]), 0, None), cap),
             "Híbrido": metrics(y, np.clip(te5_phys.values + hy5.predict(te[feats].assign(p_phys=te5_phys)), 0, None), cap)}

    # ---------------- Inverter health ----------------
    inv = gen.merge(wx, on="ts")
    inv = inv[inv["G"] > 0.3]  # daylight, meaningful irradiance only
    med = inv.groupby("ts")["AC_POWER"].transform("median")
    inv = inv.assign(ratio=inv["AC_POWER"] / med.replace(0, np.nan))
    inv["day"] = inv["ts"].dt.date
    daily = inv.groupby(["SOURCE_KEY", "day"])["ratio"].median().unstack()
    keys = daily.median(axis=1).sort_values().index
    names = {k: f"INV-{i + 1:02d}" for i, k in enumerate(sorted(daily.index))}
    perf = daily.median(axis=1).sort_values()
    flagged = perf[perf < 0.97]

    # Energy lost by flagged inverters vs. the plant median inverter
    e = gen.copy()
    e["med"] = e.groupby("ts")["AC_POWER"].transform("median")
    lost_kwh = {}
    for kkey in flagged.index:
        s = e[e["SOURCE_KEY"] == kkey]
        lost_kwh[kkey] = float(((s["med"] - s["AC_POWER"]).clip(lower=0)).sum() * 0.25)  # kW·15min -> kWh
    days = df["ts"].dt.normalize().nunique()
    lost_mwh_year = sum(lost_kwh.values()) / 1000 * 365 / days
    plant_mwh = float(df["P"].sum() * 0.25)

    results = {
        "dataset": "Planta FV pública, 22 inversores, 34 días, resolución 15 min",
        "capacity_mw": round(cap, 2), "k_gain": round(k, 4),
        "train_days": int(tr["ts"].dt.normalize().nunique()), "test_days": TEST_DAYS,
        "models": models, "models_5_days": small,
        "test_series": {"ts": te["ts"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
                        "real": te["P"].round(3).tolist(),
                        "phys": te["p_phys"].round(3).tolist(),
                        "ml": te["p_ml"].round(3).tolist(),
                        "hyb": te["p_hyb"].round(3).tolist(),
                        "G": te["G"].round(3).tolist()},
        "inverters": [{"id": names[kk], "ratio": round(float(perf[kk]), 4),
                       "flag": bool(kk in flagged.index),
                       "daily": [None if pd.isna(v) else round(float(v), 3) for v in daily.loc[kk]]}
                      for kk in keys],
        "days": [str(d) for d in daily.columns],
        "losses": {"flagged": len(flagged), "lost_mwh_period": round(sum(lost_kwh.values()) / 1000, 1),
                   "lost_mwh_year": round(lost_mwh_year, 0),
                   "lost_pct_of_production": round(100 * sum(lost_kwh.values()) / 1000 / plant_mwh, 2),
                   "price_eur_mwh": PRICE_EUR_MWH,
                   "lost_eur_year": int(round(lost_mwh_year * PRICE_EUR_MWH, -2))},
    }
    (OUT / "results.json").write_text(json.dumps(results))
    print(json.dumps({k: results[k] for k in ("capacity_mw", "models", "models_5_days", "losses")}, indent=2, ensure_ascii=False))
    print([(i["id"], i["ratio"]) for i in results["inverters"][:5]])


if __name__ == "__main__":
    main()
