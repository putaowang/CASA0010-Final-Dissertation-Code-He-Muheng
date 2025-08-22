# -*- coding: utf-8 -*-
"""
4.2 全局基线与分区对比（强化清洗：type_mode 兼容 T/S/D + 全称；Terraced 为基准；RF 学习曲线；S/T/D 全显）

输入：
- CSV：/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_full_summary.csv
- HEX：/Users/muhenghe/Documents/BYLW/start/pythonProject2/accessible_hexagons.geojson
- Borough：/Users/muhenghe/Documents/BYLW/start/map data/statistical-gis-boundaries-london/ESRI/London_Borough_Excluding_MHW.shp

输出目录：./out_4_2
产物：
- fig_4_9_rf_feature_importance.png
- fig_4_9b_rf_oob_curve.png
- fig_4_10_zonal_coef_adjptal.png
- fig_4_11_residual_hist_baseline.png
- fig_4_12_spatial_residuals_baseline.png
- table_baseline_ols.csv
- table_rf_importance.csv
- table_zonal_adjptal_coef.csv
- baseline_predictions_residuals.csv
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

# ---------- 固定路径 ----------
CSV_PATH = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_full_summary.csv"
HEX_GEOJSON_PATH = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/accessible_hexagons.geojson"
BOROUGH_SHP_PATH = "/Users/muhenghe/Documents/BYLW/start/map data/statistical-gis-boundaries-london/ESRI/London_Borough_Excluding_MHW.shp"

OUT_DIR = "./out_4_2"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 列名配置 ----------
TARGET = "mixadj_price"
FEATURES_NUMERIC = [
    "adjusted_ptal",
    "dist_to_center_km",
    "Net annual income before housing costs (£)",
    "dwelling_density",
]
CAT_COL = "type_mode"
ID_COL = "hex_id"
LAT_COL = "hex_lat"
LON_COL = "hex_lon"

# ---------- 分区与学习曲线参数 ----------
CENTER_LAT = 51.5074      # 北/南分界
CITY_THRESHOLD_KM = 10.0   # 市区/郊区分界
LOG_PRICE = False

N_START, N_STOP, N_STEP = 50, 1000, 50
PLATEAU_DELTA = 0.001
PLATEAU_STEPS = 3

# ---------- 工具函数 ----------
def to_numeric_strict(s: pd.Series) -> pd.Series:
    """
    强清洗：移除所有非数字字符（含£/$/%/空格/千分位等），仅保留 0-9 . - + e/E
    """
    if s.dtype == object:
        s = (s.astype(str)
               .str.replace(r"[£$,％%]", "", regex=True)
               .str.replace(r"\s+", "", regex=True)
               .str.replace(r"[^0-9eE\.\-+]", "", regex=True))
    return pd.to_numeric(s, errors="coerce")

def check_columns(df, cols, where="dataframe"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{where} 缺少必要列: {missing}")

def read_geo(path):
    if os.path.exists(path):
        try:
            return gpd.read_file(path)
        except Exception as e:
            warnings.warn(f"读取矢量文件失败：{path}\n错误：{e}")
            return None
    return None

def clean_design_matrix(X: pd.DataFrame, y: pd.Series, name="(global)"):
    """把 X,y 全部转 float，丢弃含 NaN 的行；打印诊断。"""
    Xc = X.copy()
    for c in Xc.columns:
        Xc[c] = to_numeric_strict(Xc[c])
    yc = to_numeric_strict(y)

    print(f"\n[诊断] 清洗前 {name}:")
    print("X 非空计数：\n", Xc.notnull().sum().sort_values())
    print("y 非空计数：", yc.notnull().sum())

    mask = Xc.notnull().all(axis=1) & yc.notnull()
    n_kept = int(mask.sum()); n_total = int(len(mask))
    print(f"[诊断] 清洗后 {name}: 保留 {n_kept}/{n_total} 行")
    if n_kept == 0:
        col_bad = Xc.columns[Xc.notnull().sum().argmin()]
        raise ValueError(f"[致命] 设计矩阵被清空：列 `{col_bad}` 有效值为 0。请检查该列的原始格式/单位。")
    return Xc.loc[mask].astype(float), yc.loc[mask].astype(float), mask

def detect_plateau(x_list, y_list, delta=PLATEAU_DELTA, steps=PLATEAU_STEPS):
    if len(y_list) < steps + 1:
        return x_list[np.argmax(y_list)]
    improves = np.diff(y_list)
    for i in range(len(improves) - steps + 1):
        if np.all(improves[i:i+steps] < delta):
            return x_list[i + steps]
    return x_list[np.argmax(y_list)]

# ---------- 读取 ----------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"未找到 CSV：{CSV_PATH}")

df = pd.read_csv(CSV_PATH)
check_columns(df, [TARGET, ID_COL, LAT_COL, LON_COL, CAT_COL] + FEATURES_NUMERIC, "CSV")

# ---------- 数值列清洗 ----------
for col in FEATURES_NUMERIC + [TARGET, LAT_COL, LON_COL]:
    if col in df.columns:
        df[col] = to_numeric_strict(df[col])

# ---------- type_mode 规范化（兼容 T/S/D + 全称 + 常见变体） ----------
def normalize_type_mode(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.lower()
    # 常见同义 / 缩写映射
    mapping = {
        # Terraced
        r"^t$": "Terraced",
        r"^ter{0,1}r?ac?ed?$": "Terraced",                    # terrace / terraced / teraced / terr
        r"^terraced\s*house$": "Terraced",
        # Semi-Detached
        r"^s$": "Semi-Detached",
        r"^semi[-\s_]?detached$": "Semi-Detached",
        r"^semi\s*det$": "Semi-Detached",
        r"^semidetached$": "Semi-Detached",
        # Detached
        r"^d$": "Detached",
        r"^detached$": "Detached",
        r"^det$": "Detached",
        r"^house\s*-\s*detached$": "Detached",
    }
    out = pd.Series(index=s.index, dtype="object")
    matched_mask = pd.Series(False, index=s.index)
    for pattern, target in mapping.items():
        m = s.str.match(pattern, na=False)
        out.loc[m] = target
        matched_mask |= m
    # 没匹配到的直接填 None
    out.loc[~matched_mask] = None
    return out

df[CAT_COL + "_norm"] = normalize_type_mode(df[CAT_COL])

# 诊断：看规范化结果
print("\n[诊断] type_mode 规范化后频次：")
print(df[CAT_COL + "_norm"].value_counts(dropna=False))

# 仅保留三类（Terraced / Semi-Detached / Detached）
df = df[df[CAT_COL + "_norm"].isin(["Terraced", "Semi-Detached", "Detached"])].copy()
df[CAT_COL] = pd.Categorical(df[CAT_COL + "_norm"],
                             categories=["Terraced", "Semi-Detached", "Detached"],
                             ordered=True)
df = df.drop(columns=[CAT_COL + "_norm"])

# 若还有关键列缺失，剔除
need_cols = list(set([TARGET, ID_COL, LAT_COL, LON_COL, CAT_COL] + FEATURES_NUMERIC))
df = df.dropna(subset=need_cols).copy()

# ---------- 分区（北/南 × 市区/郊区） ----------
def assign_zone(row):
    lat = row[LAT_COL]
    dist = row["dist_to_center_km"]
    if pd.isna(lat) or pd.isna(dist):
        return np.nan
    ns = "N" if lat >= CENTER_LAT else "S"
    cs = "City" if dist <= CITY_THRESHOLD_KM else "Suburb"
    return f"{ns}-{cs}"

df["zone"] = df.apply(assign_zone, axis=1)
df = df.dropna(subset=["zone"]).copy()

# ---------- OLS（Terraced 基准；HC3） ----------
X_ols_num = df[FEATURES_NUMERIC].copy()
dum_ols = pd.get_dummies(df[CAT_COL], drop_first=True).rename(
    columns={"Semi-Detached": "type_S", "Detached": "type_D"}
)
for c in ["type_S", "type_D"]:
    if c not in dum_ols.columns:
        dum_ols[c] = 0.0
X_ols = pd.concat([X_ols_num, dum_ols], axis=1)

y_ols = df[TARGET].copy()
if LOG_PRICE:
    y_ols = np.log1p(y_ols)

X_ols_clean, y_ols_clean, mask_used = clean_design_matrix(X_ols, y_ols, name="OLS")
X_const = sm.add_constant(X_ols_clean, has_constant="add").astype(float)
ols = sm.OLS(y_ols_clean, X_const).fit(cov_type="HC3")

summary_table = ols.summary2().tables[1].copy()
summary_table.reset_index(inplace=True)
summary_table.rename(columns={"index": "variable"}, inplace=True)
summary_table.to_csv(os.path.join(OUT_DIR, "table_baseline_ols.csv"), index=False)

df_used = df.loc[mask_used].copy()
df_used["_pred"] = ols.predict(X_const)
df_used["_resid"] = y_ols_clean - df_used["_pred"]
df_used[[ID_COL, TARGET, "_pred", "_resid", LAT_COL, LON_COL, "zone"]].to_csv(
    os.path.join(OUT_DIR, "baseline_predictions_residuals.csv"), index=False
)

# 图4-11：残差直方图
plt.figure()
plt.hist(df_used["_resid"].dropna(), bins=50)
plt.xlabel("Residuals (baseline OLS)")
plt.ylabel("Frequency")
plt.title("Residual Histogram — Baseline Global Model")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_4_11_residual_hist_baseline.png"), dpi=200)
plt.close()

# 图4-12：空间残差（多边形优先；失败则点图），叠加 Borough
def read_geo_silent(path):
    try:
        return read_geo(path)
    except Exception:
        return None

ghex = read_geo_silent(HEX_GEOJSON_PATH)
gbor = read_geo_silent(BOROUGH_SHP_PATH)

if ghex is not None and ID_COL in ghex.columns and "geometry" in ghex.columns:
    gplot = ghex.merge(df_used[[ID_COL, "_resid"]], on=ID_COL, how="left")
    ax = gplot.plot(column="_resid", legend=True)
    if gbor is not None:
        try:
            if gbor.crs != gplot.crs:
                gbor = gbor.to_crs(gplot.crs)
            gbor.plot(ax=ax, facecolor="none")
        except Exception as e:
            warnings.warn(f"叠加 Borough 边界失败（多边形）：{e}")
    ax.set_axis_off()
    plt.title("Spatial Residuals — Baseline Global Model (Polygons)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_4_12_spatial_residuals_baseline.png"), dpi=300)
    plt.close()
else:
    plt.figure()
    sc = plt.scatter(df_used[LON_COL], df_used[LAT_COL], c=df_used["_resid"], s=6)
    plt.colorbar(sc, label="Residuals")
    if gbor is not None:
        try:
            gbor_4326 = gbor.to_crs(4326)
            ax = plt.gca()
            gbor_4326.plot(ax=ax, facecolor="none")
        except Exception as e:
            warnings.warn(f"叠加 Borough 边界失败（点图）：{e}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Spatial Residuals — Baseline Global Model (Points)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_4_12_spatial_residuals_baseline.png"), dpi=300)
    plt.close()

# ---------- RF：S/T/D 全显 + OOB 学习曲线 ----------
dum_rf = pd.get_dummies(df[CAT_COL], drop_first=False).rename(
    columns={"Terraced": "type_T", "Semi-Detached": "type_S", "Detached": "type_D"}
)
for c in ["type_T", "type_S", "type_D"]:
    if c not in dum_rf.columns:
        dum_rf[c] = 0.0
X_rf_all = pd.concat([df[FEATURES_NUMERIC], dum_rf[["type_T", "type_S", "type_D"]]], axis=1)
y_rf_all = df[TARGET].copy()
if LOG_PRICE:
    y_rf_all = np.log1p(y_rf_all)

X_rf_all, y_rf_all, mask_rf = clean_design_matrix(X_rf_all, y_rf_all, name="RF")

ns = list(range(N_START, N_STOP + 1, N_STEP))
oob_scores = []
for n in ns:
    rf_tmp = RandomForestRegressor(
        n_estimators=n, bootstrap=True, oob_score=True, random_state=42, n_jobs=-1
    )
    rf_tmp.fit(X_rf_all, y_rf_all)
    oob_scores.append(rf_tmp.oob_score_)

best_n = detect_plateau(ns, oob_scores, delta=PLATEAU_DELTA, steps=PLATEAU_STEPS)
print(f"\n[INFO] RF 平台期选择的树数 best_n = {best_n}，对应 OOB R² = {oob_scores[ns.index(best_n)]:.4f}")

plt.figure(figsize=(9,5))
plt.plot(ns, oob_scores, marker="o")
plt.axvline(best_n, linestyle="--")
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("OOB R²")
plt.title("Random Forest — OOB Learning Curve")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_4_9b_rf_oob_curve.png"), dpi=220)
plt.close()

rf_final = RandomForestRegressor(
    n_estimators=best_n, bootstrap=True, oob_score=True, random_state=42, n_jobs=-1
)
rf_final.fit(X_rf_all, y_rf_all)

imp_df = (pd.DataFrame({"feature": X_rf_all.columns, "importance": rf_final.feature_importances_})
          .sort_values("importance", ascending=True))
imp_df.to_csv(os.path.join(OUT_DIR, "table_rf_importance.csv"), index=False)

plt.figure(figsize=(12,6))
plt.barh(imp_df["feature"], imp_df["importance"])
plt.xlabel("Feature Importance")
plt.title("Random Forest — Global Feature Importances (with type_T/S/D)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_4_9_rf_feature_importance.png"), dpi=220)
plt.close()

# ---------- 分区 OLS（adjusted_ptal；Terraced 基准） ----------
zones_order = ["N-City", "N-Suburb", "S-City", "S-Suburb"]
rows = []
for z in zones_order:
    sub = df[df["zone"] == z].copy()
    if len(sub) < 50:
        warnings.warn(f"Zone {z} 样本量较小（n={len(sub)}），跳过。")
        continue

    Xz_num = sub[FEATURES_NUMERIC].copy()
    dz = pd.get_dummies(sub[CAT_COL], drop_first=True).rename(
        columns={"Semi-Detached": "type_S", "Detached": "type_D"}
    )
    for c in ["type_S", "type_D"]:
        if c not in dz.columns:
            dz[c] = 0.0
    Xz = pd.concat([Xz_num, dz], axis=1)
    yz = sub[TARGET].copy()
    if LOG_PRICE:
        yz = np.log1p(yz)

    Xz_clean, yz_clean, maskz = clean_design_matrix(Xz, yz, name=f"Zone {z}")
    if len(Xz_clean) < 30:
        warnings.warn(f"Zone {z} 清洗后有效样本不足（n={len(Xz_clean)}），跳过。")
        continue

    Xz_const = sm.add_constant(Xz_clean, has_constant="add").astype(float)
    mz = sm.OLS(yz_clean, Xz_const).fit(cov_type="HC3")

    if "adjusted_ptal" in mz.params.index:
        b = float(mz.params["adjusted_ptal"])
        se = float(mz.bse["adjusted_ptal"])
        ci_l, ci_u = b - 1.96*se, b + 1.96*se
        rows.append({"zone": z, "beta": b, "ci_low": ci_l, "ci_high": ci_u, "n": int(len(Xz_clean))})

zonal_df = pd.DataFrame(rows)
zonal_df.to_csv(os.path.join(OUT_DIR, "table_zonal_adjptal_coef.csv"), index=False)

if len(zonal_df) > 0:
    x = np.arange(len(zonal_df))
    plt.figure(figsize=(8,5))
    plt.bar(x, zonal_df["beta"])
    yerr = np.vstack([zonal_df["beta"] - zonal_df["ci_low"], zonal_df["ci_high"] - zonal_df["beta"]])
    plt.errorbar(x, zonal_df["beta"], yerr=yerr, fmt="none", capsize=4)
    plt.xticks(x, zonal_df["zone"])
    plt.ylabel("Coefficient of adjusted_ptal (OLS, HC3)")
    plt.title("Zonal OLS Coefficient — adjusted_ptal (Baseline=Terraced)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_4_10_zonal_coef_adjptal.png"), dpi=220)
    plt.close()
else:
    warnings.warn("无可用分区估计（样本量不足或清洗后不足），未生成 fig_4_10。")

print("Done. All figures and tables saved to:", os.path.abspath(OUT_DIR))


# ========= 追加：PDP（Partial Dependence） for adjusted_ptal =========
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import numpy as np
import os

# 输出目录与特征名
PDP_OUT_DIR = OUT_DIR  # 与前面一致：./out_4_2
feat_name = "adjusted_ptal"

# 1) 全城 PDP -----------------------------------------------------------
if feat_name not in X_rf_all.columns:
    raise ValueError(f"PDP 需要的列 `{feat_name}` 不在 X_rf_all 中。现有列：{list(X_rf_all.columns)}")

fig, ax = plt.subplots(figsize=(8,5))
PartialDependenceDisplay.from_estimator(
    rf_final, X_rf_all, [feat_name],
    grid_resolution=50, kind="average", ax=ax
)
ax.set_title("Partial Dependence — adjusted_ptal (Global)")
ax.set_xlabel("adjusted_ptal")
ax.set_ylabel("Partial dependence of predicted price" + (" (log scale)" if LOG_PRICE else ""))
plt.tight_layout()
plt.savefig(os.path.join(PDP_OUT_DIR, "fig_pdp_ptal_global.png"), dpi=220)
plt.close(fig)

# 2) 分区 PDP（N/S × City/Suburb） ---------------------------------------
zones_order = ["N-City", "N-Suburb", "S-City", "S-Suburb"]
# 注意：X_rf_all 的索引是清洗后保留的行，按这个索引去取对应的 zone
zone_series = df.loc[X_rf_all.index, "zone"]

fig, axes = plt.subplots(2, 2, figsize=(11,7), sharey=True)
for i, z in enumerate(zones_order):
    r, c = divmod(i, 2)
    mask_z = (zone_series == z)
    Xz = X_rf_all.loc[mask_z]
    ax = axes[r, c]
    if len(Xz) < 50:
        # 样本太少则给提示
        ax.text(0.5, 0.5, f"{z}\nInsufficient samples (n={len(Xz)})",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        continue

    PartialDependenceDisplay.from_estimator(
        rf_final, Xz, [feat_name],
        grid_resolution=50, kind="average", ax=ax
    )
    ax.set_title(z)
    ax.set_xlabel("adjusted_ptal")
    if c == 0:
        ax.set_ylabel("Partial dependence" + (" (log)" if LOG_PRICE else ""))

# 收尾保存
plt.suptitle("Partial Dependence by Zone — adjusted_ptal", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PDP_OUT_DIR, "fig_pdp_ptal_by_zone.png"), dpi=220, bbox_inches="tight")
plt.close(fig)
# ========= 追加结束 =========
