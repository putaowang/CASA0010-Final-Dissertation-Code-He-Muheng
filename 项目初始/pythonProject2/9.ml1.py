import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import math

# ===== 路径（按你提供）=====
csv_path = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_full_summary.csv"
boundary_path = "/Users/muhenghe/Documents/BYLW/start/map data/statistical-gis-boundaries-london/ESRI/London_Borough_Excluding_MHW.shp"

# ===== 参数 =====
SIDE_LEN_M = 200.0   # 六边形边长 200 m（flat-topped）
DPI = 300
ALPHA = 0.7          # adjusted PTAL 衰减系数
# 制图坐标系：英国国家格网（米制，便于画正六边形），再转 WGS84 展示
CRS_METRIC = 27700   # EPSG:27700
CRS_WGS84 = 4326

# ===== 读取 CSV 并检查列 =====
df = pd.read_csv(csv_path)
required = {"hex_lon", "hex_lat"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"CSV 缺少必要列：{missing}. 需要 hex_lon/hex_lat 来生成六边形。")

# ===== 生成中心点 GeoDataFrame（WGS84 -> 英国平面坐标）=====
gdf_pts = gpd.GeoDataFrame(
    df.copy(),
    geometry=gpd.points_from_xy(df["hex_lon"], df["hex_lat"]),
    crs=f"EPSG:{CRS_WGS84}"
).to_crs(CRS_METRIC)

# ===== 基于中心点生成“平顶正六边形”（flat-topped）=====
# 几何关系：对平顶六边形，顶点到中心的半径 R = 边长 s
R = SIDE_LEN_M

def hex_polygon(center_x, center_y, R):
    # 平顶六边形顶点角度：0, 60, 120, 180, 240, 300 度
    angles = [0, 60, 120, 180, 240, 300]
    coords = [(center_x + R*math.cos(math.radians(a)),
               center_y + R*math.sin(math.radians(a))) for a in angles]
    return Polygon(coords)

gdf_hex = gdf_pts.copy()
gdf_hex["geometry"] = gdf_pts.geometry.apply(lambda p: hex_polygon(p.x, p.y, R))
gdf_hex.set_crs(CRS_METRIC, inplace=True)

# ===== 行政边界（投影到英制坐标，便于叠加裁剪/显示）=====
boundary = gpd.read_file(boundary_path)
if boundary.crs is None:
    # 若无 crs，尝试按文件常见坐标系处理；否则直接设成 EPSG:27700
    boundary.set_crs(CRS_METRIC, inplace=True)
else:
    boundary = boundary.to_crs(CRS_METRIC)

# （可选）裁剪六边形到伦敦边界内，保证边界外不画
try:
    gdf_hex = gpd.overlay(gdf_hex, boundary[["geometry"]], how="intersection")
except Exception:
    pass

# ===== 若缺失 adjusted_ptal，自动计算 =====
if "adjusted_ptal" not in gdf_hex.columns:
    if {"mean_AI", "avg_walk_km"}.issubset(gdf_hex.columns):
        gdf_hex["adjusted_ptal"] = gdf_hex["mean_AI"] * np.exp(-ALPHA * gdf_hex["avg_walk_km"])
        print("ℹ️ 已依据 mean_AI 与 avg_walk_km 计算 adjusted_ptal。")
    else:
        print("⚠️ 未找到 adjusted_ptal，也缺少 mean_AI/avg_walk_km，后续将跳过 adjusted PTAL 图。")

# ===== 投影到 WGS84 以便标准底图坐标展示（也可保持 EPSG:27700 直接输出）=====
gdf_hex_plot = gdf_hex.to_crs(CRS_WGS84)
boundary_plot = boundary.to_crs(CRS_WGS84)

# ===== 工具函数 =====
def _robust_limits(series, qmin=0.02, qmax=0.98):
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return None, None
    vmin, vmax = s.quantile(qmin), s.quantile(qmax)
    if vmin == vmax:
        return None, None
    return vmin, vmax

def plot_hex_coverage(hex_gdf, boundary_gdf, title, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=DPI)
    hex_gdf.boundary.plot(ax=ax, linewidth=0.2, color="#999999")
    boundary_gdf.boundary.plot(ax=ax, linewidth=1.0, color="black")
    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

def plot_hex_continuous(hex_gdf, boundary_gdf, value_col, title, cmap, legend_label, out_path,
                        qmin=0.02, qmax=0.98):
    if value_col not in hex_gdf.columns:
        print(f"⚠️ 列 {value_col} 不存在，跳过：{title}")
        return
    vmin, vmax = _robust_limits(hex_gdf[value_col], qmin, qmax)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=DPI)
    hex_gdf.plot(
        column=value_col, ax=ax, cmap=cmap, legend=True,
        vmin=vmin, vmax=vmax, linewidth=0, edgecolor="none",
        missing_kwds={"color": "lightgray", "label": "No data"}
    )
    boundary_gdf.boundary.plot(ax=ax, linewidth=1.0, color="black")
    if legend_label:
        try:
            cax = ax.get_figure().axes[-1]
            cax.set_ylabel(legend_label)
        except Exception:
            pass
    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

# ===== 1) 六边形覆盖图 =====
plot_hex_coverage(
    gdf_hex_plot, boundary_plot,
    title="Greater London Hexagon Coverage (200 m)",
    out_path="fig_hex_coverage.png"
)

# ===== 2) adjusted PTAL 分布图 =====
plot_hex_continuous(
    gdf_hex_plot, boundary_plot,
    value_col="adjusted_ptal",
    title="Adjusted PTAL (200 m Hexagons)",
    cmap="viridis",
    legend_label="Adjusted PTAL",
    out_path="fig_adjusted_ptal.png"
)

# ===== 3) mix-adjusted house price 分布图 =====
plot_hex_continuous(
    gdf_hex_plot, boundary_plot,
    value_col="mixadj_price",
    title="Mix-adjusted House Price (£)",
    cmap="plasma",
    legend_label="Price (£)",
    out_path="fig_mixadj_price.png"
)

print("✅ 已导出：fig_hex_coverage.png, fig_adjusted_ptal.png, fig_mixadj_price.png")