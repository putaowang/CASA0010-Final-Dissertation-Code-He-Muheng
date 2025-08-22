import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.spatial import cKDTree
from tqdm import tqdm

# ========== 1. 路径与数据加载 ==========
csv_path = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_full_summary.csv"
hex_path = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/accessible_hexagons.geojson"
boundary_path = "/Users/muhenghe/Documents/BYLW/start/map data/statistical-gis-boundaries-london/ESRI/London_Borough_Excluding_MHW.shp"

df = pd.read_csv(csv_path)
hex_gdf = gpd.read_file(hex_path)
hex_gdf['hex_id'] = hex_gdf.index
boundary = gpd.read_file(boundary_path).to_crs(hex_gdf.crs)

df['Net annual income before housing costs (£)'] = (
    df['Net annual income before housing costs (£)'].astype(str).str.replace(',', '').astype(float)
)
hex_all = pd.merge(hex_gdf[['hex_id', 'geometry']], df, on='hex_id', how='inner')
hex_all = hex_all.dropna(subset=[
    'adjusted_ptal', 'dist_to_center_km',
    'Net annual income before housing costs (£)', 'dwelling_density', 'mixadj_price'
])
hex_all = hex_all[hex_all['mixadj_price'] > 10000]

# ========== 2. 局部多项式回归 ==========
hex_all = hex_all.to_crs(epsg=32630)
hex_all['centroid'] = hex_all.centroid
coords = np.vstack([hex_all['centroid'].x.values, hex_all['centroid'].y.values]).T
tree = cKDTree(coords)
radius = 3000

coef_results = []
r2_results = []
y_pred_poly = []

print("⏳ Running local polynomial regression...")
for i in tqdm(range(len(hex_all))):
    idxs = tree.query_ball_point(coords[i], r=radius)
    sub = hex_all.iloc[idxs]
    if len(sub) < 30:
        coef_results.append(np.nan)
        r2_results.append(np.nan)
        y_pred_poly.append(np.nan)
        continue
    X = sub[['adjusted_ptal', 'dist_to_center_km',
             'Net annual income before housing costs (£)', 'dwelling_density']].values
    y = sub['mixadj_price'].values
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    coef_results.append(model.coef_[0])  # PTAL一次项
    r2_results.append(model.score(X_poly, y))
    xi = hex_all.iloc[i][['adjusted_ptal', 'dist_to_center_km',
                          'Net annual income before housing costs (£)', 'dwelling_density']].values.reshape(1, -1)
    xi_poly = poly.transform(xi)
    y_pred_poly.append(model.predict(xi_poly)[0])

# 投回WGS84
hex_all = hex_all.to_crs(epsg=4326)
hex_all['local_poly_coef_adjusted_ptal'] = coef_results
hex_all['local_poly_r2'] = r2_results
hex_all['local_poly_pred'] = y_pred_poly
hex_all['local_poly_resid'] = hex_all['mixadj_price'] - hex_all['local_poly_pred']

# ========== 3. 空间分布图 ==========
boundary = gpd.read_file(boundary_path).to_crs(epsg=4326)

# 3.1 PTAL系数空间分布
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
hex_all.plot(
    column='local_poly_coef_adjusted_ptal', cmap='coolwarm', ax=ax,
    legend=True, edgecolor='gray', linewidth=0.1,
    vmin=-200000, vmax=200000,
    missing_kwds={'color': 'lightgray', 'label': 'No Data'}
)
boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.8, alpha=0.7)
plt.title("Local Polynomial Coefficient (PTAL) on House Price\n(Control: distance, income, density)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig("ptal_poly_coef_map.png", dpi=300)
plt.show()

# 3.2 局部R2空间分布
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
hex_all.plot(
    column='local_poly_r2', cmap='YlGnBu', ax=ax,
    legend=True, edgecolor='gray', linewidth=0.1,
    vmin=0, vmax=1, missing_kwds={'color': 'lightgray', 'label': 'No Data'}
)
boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.8, alpha=0.7)
plt.title("Local R² of Polynomial Regression on House Price (2km Window)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig("ptal_poly_r2_map.png", dpi=300)
plt.show()

# ========== 4. 统计与分组可视化 ==========
hex_all['income_group'] = pd.qcut(
    hex_all['Net annual income before housing costs (£)'],
    q=4, labels=['Low', 'Mid-low', 'Mid-high', 'High']
)
income_groups = ['Low', 'Mid-low', 'Mid-high', 'High']

# 小提琴图
plt.figure(figsize=(8, 5))
sns.violinplot(
    x='income_group', y='local_poly_coef_adjusted_ptal',
    data=hex_all, inner="quartile", cut=0
)
plt.title("Distribution of PTAL Linear Coefficient (Violin Plot, by Income Group)")
plt.ylabel("PTAL Linear Coefficient")
plt.xlabel("Income Group")
plt.tight_layout()
plt.savefig("ptal_coef_violin_income.png", dpi=300)
plt.show()

# 分组柱状图/箱线图
import numpy as np
from scipy.stats.mstats import winsorize

results = []
for g in income_groups:
    vals = hex_all.loc[hex_all['income_group']==g, 'local_poly_coef_adjusted_ptal'].dropna()
    if len(vals) < 30: continue
    vals_clip = winsorize(vals, limits=[0.01, 0.01])  # 1%截断
    results.append({'income_group': g, 'mean': np.mean(vals_clip), 'median': np.median(vals_clip), 'std': np.std(vals_clip), 'count': len(vals_clip)})
df_stat = pd.DataFrame(results)
plt.figure(figsize=(8,5))
plt.bar(df_stat['income_group'].astype(str), df_stat['mean'])
plt.ylabel("Mean PTAL Linear Coefficient (winsorized)")
plt.xlabel("Income Group")
plt.title("Winsorized Mean Local PTAL Coefficient by Income Group")
plt.tight_layout()
plt.savefig("ptal_coef_bar_income.png", dpi=300)
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(
    x='income_group', y='local_poly_coef_adjusted_ptal',
    data=hex_all, showfliers=False
)
plt.title("Boxplot of PTAL Linear Coefficient (Outliers Hidden)")
plt.ylabel("PTAL Linear Coefficient")
plt.xlabel("Income Group")
plt.tight_layout()
plt.savefig("ptal_coef_box_income.png", dpi=300)
plt.show()

# ========== 5. 残差/负PTAL/重合空间分析 ==========
# 5.1 残差空间分布
plt.figure(figsize=(12,10))
hex_all.plot(
    column='local_poly_resid', cmap='bwr', legend=True,
    edgecolor='gray', linewidth=0.1, vmin=-2e6, vmax=2e6,
    missing_kwds={'color':'lightgray','label':'No Data'}
)
boundary.plot(ax=plt.gca(), facecolor='none', edgecolor='black', linewidth=0.7)
plt.title("Unexplained House Price")
plt.tight_layout()
plt.savefig("ptal_poly_resid_map.png", dpi=300)
plt.show()

# 5.2 PTAL负系数区域
plt.figure(figsize=(12,10))
hex_all.plot(
    column=(hex_all['local_poly_coef_adjusted_ptal'] < 0).astype(int),
    cmap='coolwarm', legend=True,
    edgecolor='gray', linewidth=0.1, missing_kwds={'color':'lightgray','label':'No Data'}
)
boundary.plot(ax=plt.gca(), facecolor='none', edgecolor='black', linewidth=0.7)
plt.title("PTAL Negative Linear Coefficient (Red=Negative)")
plt.tight_layout()
plt.savefig("ptal_poly_negcoef_map.png", dpi=300)
plt.show()

# 5.3 残差极值区（绝对值Top10%）
abs_resid = np.abs(hex_all['local_poly_resid'])
thres = np.percentile(abs_resid, 90)
hex_all['high_resid'] = abs_resid > thres

# 5.4 重合分析
hex_all['neg_ptal'] = hex_all['local_poly_coef_adjusted_ptal'] < 0
hex_all['neg_and_high_resid'] = hex_all['neg_ptal'] & hex_all['high_resid']

num_total = len(hex_all)
num_neg = hex_all['neg_ptal'].sum()
num_high_resid = hex_all['high_resid'].sum()
num_overlap = hex_all['neg_and_high_resid'].sum()

print(f"Total hexes: {num_total}")
print(f"PTAL linear coefficient < 0: {num_neg} ({num_neg/num_total:.1%})")
print(f"Top 10% high residuals: {num_high_resid} ({num_high_resid/num_total:.1%})")
print(f"Overlap (both PTAL negative and high residual): {num_overlap} ({num_overlap/num_total:.1%})")
print(f"Overlap within PTAL negative: {num_overlap/num_neg:.1%}")
print(f"Overlap within high residual: {num_overlap/num_high_resid:.1%}")

# 5.5 空间重合区
plt.figure(figsize=(12,10))
hex_all.plot(
    column='neg_and_high_resid', cmap='autumn_r', legend=True,
    edgecolor='gray', linewidth=0.1, missing_kwds={'color':'lightgray','label':'No Data'}
)
boundary.plot(ax=plt.gca(), facecolor='none', edgecolor='black', linewidth=0.7)
plt.title("PTAL Negative Coefficient & High Residual (Unexplained)")
plt.tight_layout()
plt.savefig("ptal_poly_overlap_map.png", dpi=300)
plt.show()

# 5.6 残差分布直方图
plt.figure(figsize=(8,5))
sns.histplot(hex_all['local_poly_resid'], bins=50, kde=True)
plt.title("Distribution of Model Residuals (Poly Regression)")
plt.xlabel("Residual (Observed - Predicted)")
plt.tight_layout()
plt.savefig("ptal_poly_resid_hist.png", dpi=300)
plt.show()

# 5.7 统计极端值信息
for g in income_groups:
    vals = hex_all.loc[hex_all['income_group'] == g, 'local_poly_coef_adjusted_ptal']
    print(f"{g} group quartiles:", np.percentile(vals.dropna(), [1, 10, 25, 50, 75, 90, 99]))

print("== 代码运行完毕，所有结果和图片已保存！ ==")