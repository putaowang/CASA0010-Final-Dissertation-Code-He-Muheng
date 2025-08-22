# -*- coding: utf-8 -*-
"""
4.5 PTAL × Income 交互分析 一键脚本
产出：
- 表：table_45_ols_interaction.csv, table_45_qr_tau25.csv 等, table_45_key_numbers.csv
- 图：fig_45a_marginal_effect_ptal_over_income.png
      fig_45b_pdp_2d_ptal_income.png
      fig_45c_qr_beta_interaction_over_quantiles.png

依赖：pandas numpy matplotlib seaborn statsmodels scikit-learn
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay

warnings.filterwarnings("ignore")

# ========== 路径 ==========
CSV_PATH = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_full_summary.csv"
OUTDIR = "./out_45_ptal_income"
os.makedirs(OUTDIR, exist_ok=True)

# ========== 列名 ==========
COL_PTAL   = "adjusted_ptal"
COL_DENS   = "dwelling_density"
COL_INCOME = "Net annual income before housing costs (£)"
COL_DIST   = "dist_to_center_km"
COL_TYPE   = "type_mode"
COL_Y      = "mixadj_price"

# ========== 工具函数 ==========
def to_float_strict(s: pd.Series) -> pd.Series:
    """去掉逗号、货币符号、空格，仅保留数值；无法转换的置 NaN"""
    s = s.astype(str).str.replace(r"[£$,％%]", "", regex=True).str.replace(r"\s+", "", regex=True)
    s = s.str.replace(r"[^0-9eE\.\-+]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def normalize_type_mode(s: pd.Series) -> pd.Series:
    """把多种写法统一到 Terraced / Semi-Detached / Detached；未识别为 NaN"""
    s0 = s.astype(str).str.strip().str.lower()
    out = pd.Series(index=s.index, dtype="object")
    # Terraced
    terr = s0.str.match(r"^(t|terr?ac?ed|terraced\s*house)$", na=False)
    # Semi-Detached
    semi = s0.str.match(r"^(s|semi[-\s_]?det(ached)?|semidetached)$", na=False)
    # Detached
    deta = s0.str.match(r"^(d|det(ached)?|house\s*-\s*detached)$", na=False)
    out.loc[terr] = "Terraced"
    out.loc[semi] = "Semi-Detached"
    out.loc[deta] = "Detached"
    out.loc[~(terr | semi | deta)] = np.nan
    return out

# ========== 读取与基础清洗 ==========
df = pd.read_csv(CSV_PATH)

# 强制数值化关键列
for c in [COL_INCOME, COL_DENS]:
    if c in df.columns:
        df[c] = to_float_strict(df[c])

# 目标与核心自变量存在且有效
must_have = [COL_Y, COL_PTAL, COL_INCOME, COL_DIST, COL_DENS, COL_TYPE]
df = df.dropna(subset=[c for c in must_have if c in df.columns]).copy()
df = df[df[COL_Y] > 1e4].copy()

# 规范化住房类型，做 one-hot（Terraced 作为基线）
df["type_norm"] = normalize_type_mode(df[COL_TYPE])
type_dum = pd.get_dummies(df["type_norm"], prefix="type_mode")
# 移除 Terraced 作为基线（如果存在）
for base in ["type_mode_Terraced"]:
    if base in type_dum.columns:
        type_dum = type_dum.drop(columns=[base])
        break

# 合并
dfm = pd.concat([df[[COL_Y, COL_PTAL, COL_INCOME, COL_DIST, COL_DENS]], type_dum], axis=1)

# ========== 标准化（z-score）==========
for c in [COL_PTAL, COL_INCOME, COL_DIST, COL_DENS]:
    dfm[f"{c}_z"] = (dfm[c] - dfm[c].mean()) / dfm[c].std()

# 交互项
dfm["ptal_x_income"] = dfm[f"{COL_PTAL}_z"] * dfm[f"{COL_INCOME}_z"]

# 清理 Inf/NaN
dfm.replace([np.inf, -np.inf], np.nan, inplace=True)

# 设计矩阵列
X_cols_base = [f"{COL_PTAL}_z", f"{COL_INCOME}_z", "ptal_x_income",
               f"{COL_DIST}_z", f"{COL_DENS}_z"] + list(type_dum.columns)

# 丢缺失，统一 float
dfm_clean = dfm.dropna(subset=[COL_Y] + X_cols_base).copy()
for c in [COL_Y] + X_cols_base:
    dfm_clean[c] = pd.to_numeric(dfm_clean[c], errors="coerce").astype(float)
dfm_clean = dfm_clean.dropna(subset=[COL_Y] + X_cols_base).copy()

# ========== 1) 交互 OLS（HC3）==========
X = sm.add_constant(dfm_clean[X_cols_base].astype(float), has_constant="add")
y = dfm_clean[COL_Y].astype(float).values
ols = sm.OLS(y, X).fit(cov_type="HC3")
print(ols.summary())
ols.summary2().tables[1].to_csv(os.path.join(OUTDIR, "table_45_ols_interaction.csv"))

# ========== 2) 边际效应图：dPrice/dPTAL 随 Income 变化（HC3 CI）==========
b_ptal = float(ols.params[f"{COL_PTAL}_z"])
b_int  = float(ols.params["ptal_x_income"])
cov = ols.cov_params()
V11 = float(cov.loc[f"{COL_PTAL}_z", f"{COL_PTAL}_z"])
V22 = float(cov.loc["ptal_x_income", "ptal_x_income"])
C12 = float(cov.loc[f"{COL_PTAL}_z", "ptal_x_income"])

z_vals = np.linspace(dfm_clean[f"{COL_INCOME}_z"].quantile(0.05),
                     dfm_clean[f"{COL_INCOME}_z"].quantile(0.95), 200)
me = b_ptal + b_int * z_vals
se = np.sqrt(V11 + (z_vals**2)*V22 + 2*z_vals*C12)
ci_low, ci_hi = me - 1.96*se, me + 1.96*se

plt.figure(figsize=(8,5))
plt.plot(z_vals, me, lw=2, label="Marginal effect of PTAL")
plt.fill_between(z_vals, ci_low, ci_hi, alpha=0.2, label="95% CI")
for q, ls in zip([0.25, 0.5, 0.75], ["--", ":", "--"]):
    qv = dfm_clean[f"{COL_INCOME}_z"].quantile(q)
    plt.axvline(qv, color="k", lw=1, ls=ls, alpha=0.6)
plt.xlabel("Income (z-score)")
plt.ylabel("d Price / d PTAL  (GBP per 1 SD PTAL)")
plt.title("Marginal Effect of PTAL across Income (HC3-OLS)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig_45a_marginal_effect_ptal_over_income.png"), dpi=220)
plt.close()

# ========== 3) 2D PDP（PTAL × Income）基于随机森林 ==========
rf_features = [f"{COL_PTAL}_z", f"{COL_INCOME}_z", f"{COL_DIST}_z", f"{COL_DENS}_z"] + list(type_dum.columns)
X_rf = dfm_clean[rf_features].astype(float)
y_rf = dfm_clean[COL_Y].astype(float)

Xtr, Xte, ytr, yte = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
rf = RandomForestRegressor(
    n_estimators=300, random_state=42, n_jobs=-1,
    min_samples_leaf=5, max_features="sqrt", oob_score=True
)
rf.fit(Xtr, ytr)

plt.figure(figsize=(7.2,5.6))
_ = PartialDependenceDisplay.from_estimator(
    rf, X_rf, [(rf_features.index(f"{COL_PTAL}_z"), rf_features.index(f"{COL_INCOME}_z"))],
    grid_resolution=35, kind="average"
)
plt.suptitle("2D Partial Dependence: PTAL × Income (Random Forest)", y=1.02, fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig_45b_pdp_2d_ptal_income.png"), dpi=230, bbox_inches="tight")
plt.close()

# ========== 4) 分位数回归（τ=0.25/0.5/0.75）==========
X_qr = sm.add_constant(dfm_clean[[f"{COL_PTAL}_z", f"{COL_INCOME}_z", "ptal_x_income",
                                  f"{COL_DIST}_z", f"{COL_DENS}_z"] + list(type_dum.columns)].astype(float),
                       has_constant="add")
y_qr = dfm_clean[COL_Y].astype(float)

taus = [0.25, 0.5, 0.75]
beta_int, ci_lo, ci_hi = [], [], []
for t in taus:
    res = QuantReg(y_qr, X_qr).fit(q=t)
    b = float(res.params["ptal_x_income"])
    lo, hi = res.conf_int().loc["ptal_x_income"].astype(float)
    beta_int.append(b); ci_lo.append(lo); ci_hi.append(hi)
    res.params.to_csv(os.path.join(OUTDIR, f"table_45_qr_tau{str(t).replace('.','')}.csv"))

plt.figure(figsize=(6.6,4.6))
yerr = [np.array(beta_int)-np.array(ci_lo), np.array(ci_hi)-np.array(beta_int)]
plt.errorbar(taus, beta_int, yerr=yerr, fmt="o-", capsize=4)
plt.axhline(0, color="k", lw=1, ls="--")
plt.xlabel("Quantile τ")
plt.ylabel("Coef. of PTAL × Income")
plt.title("Quantile Regression: Interaction across the Price Distribution")
plt.xticks(taus, [".25", ".50", ".75"])
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig_45c_qr_beta_interaction_over_quantiles.png"), dpi=230)
plt.close()

# 关键数值
key = pd.DataFrame({
    "metric": ["OLS_b_ptal_z","OLS_b_income_z","OLS_b_ptalXincome","R2_OLS",
               "RF_train_R2","RF_test_R2","RF_OOB_R2"],
    "value":[
        float(ols.params.get(f"{COL_PTAL}_z", np.nan)),
        float(ols.params.get(f"{COL_INCOME}_z", np.nan)),
        float(ols.params.get("ptal_x_income", np.nan)),
        float(ols.rsquared),
        float(rf.score(Xtr, ytr)),
        float(rf.score(Xte, yte)),
        float(getattr(rf, "oob_score_", np.nan))
    ]
})
key.to_csv(os.path.join(OUTDIR, "table_45_key_numbers.csv"), index=False)

print("Done. 输出目录：", os.path.abspath(OUTDIR))


