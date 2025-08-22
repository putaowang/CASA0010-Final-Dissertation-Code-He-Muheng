import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import seaborn as sns

# --------------------- 1. 数据与路径配置 ---------------------

hex_path = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/accessible_hexagons.geojson"
csv_path = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_full_summary.csv"
boundary_path = "/Users/muhenghe/Documents/BYLW/start/map data/statistical-gis-boundaries-london/ESRI/London_Borough_Excluding_MHW.shp"
borough_path = boundary_path

# --------------------- 2. 读取空间单元、属性与边界 ---------------------

hex_gdf = gpd.read_file(hex_path)
hex_gdf['hex_id'] = hex_gdf.index
df = pd.read_csv(csv_path)
df['Net annual income before housing costs (£)'] = df['Net annual income before housing costs (£)'].astype(str).str.replace(',', '').astype(float)
boundary = gpd.read_file(boundary_path).to_crs('EPSG:4326')
borough_gdf = gpd.read_file(borough_path).to_crs(hex_gdf.crs)

# --------------------- 3. 空间连接 Borough -> zone，并合并 ---------------------

for gdf in [hex_gdf, borough_gdf]:
    if 'index_right' in gdf.columns:
        gdf.drop(columns='index_right', inplace=True)
hex_with_borough = gpd.sjoin(hex_gdf, borough_gdf[['geometry', 'NAME']], how='left', predicate='intersects')

borough_to_zone = {
    'Camden': 'N-City', 'Hackney': 'N-City', 'Islington': 'N-City',
    'Kensington and Chelsea': 'N-City', 'Westminster': 'N-City',
    'Tower Hamlets': 'N-City', 'City of London': 'N-City',
    'Hammersmith and Fulham': 'N-City', 'Haringey': 'N-Suburb',
    'Havering': 'N-Suburb', 'Barnet': 'N-Suburb', 'Brent': 'N-Suburb',
    'Ealing': 'N-Suburb', 'Enfield': 'N-Suburb', 'Harrow': 'N-Suburb',
    'Hillingdon': 'N-Suburb', 'Hounslow': 'N-Suburb',
    'Barking and Dagenham': 'N-Suburb', 'Newham': 'N-Suburb',
    'Redbridge': 'N-Suburb', 'Waltham Forest': 'N-Suburb',
    'Lambeth': 'S-City', 'Lewisham': 'S-City', 'Southwark': 'S-City',
    'Wandsworth': 'S-City', 'Bexley': 'S-Suburb', 'Bromley': 'S-Suburb',
    'Croydon': 'S-Suburb', 'Greenwich': 'S-Suburb',
    'Kingston upon Thames': 'S-Suburb', 'Merton': 'S-Suburb',
    'Richmond upon Thames': 'S-Suburb', 'Sutton': 'S-Suburb'
}
hex_with_borough['zone'] = hex_with_borough['NAME'].map(borough_to_zone)
zone_info = hex_with_borough[['hex_id', 'zone']]

# --------------------- 4. 构造主表 hex_all（属性合并+zone合并） ---------------------

hex_all = pd.merge(hex_gdf[['hex_id', 'geometry']], df, on='hex_id', how='inner')
hex_all = hex_all.dropna(subset=[
    'adjusted_ptal', 'dist_to_center_km',
    'Net annual income before housing costs (£)', 'dwelling_density', 'mixadj_price'
])
hex_all = hex_all[hex_all['mixadj_price'] > 10000]
hex_all = hex_all.merge(zone_info, on='hex_id', how='left')

# --------------------- 5. GWRF建模并输出所有变量重要性 ---------------------

hex_all = hex_all.to_crs(epsg=32630)
coords = np.vstack([hex_all.geometry.centroid.x, hex_all.geometry.centroid.y]).T
tree = cKDTree(coords)
feature_cols = [
    'adjusted_ptal', 'dist_to_center_km',
    'Net annual income before housing costs (£)', 'dwelling_density'
]
target_col = 'mixadj_price'
k_neigh = 40

preds, local_r2, importances = [], [], []

print(f"⏳ 正在为每个hex执行GWRF（邻域k={k_neigh}）...")
for i in tqdm(range(len(hex_all))):
    idxs = tree.query(coords[i], k=k_neigh)[1]
    sub = hex_all.iloc[idxs]
    X = sub[feature_cols].values
    y = sub[target_col].values
    if len(y) < 15:
        preds.append(np.nan)
        local_r2.append(np.nan)
        importances.append([np.nan] * len(feature_cols))
        continue
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    xi = hex_all.iloc[i][feature_cols].values.reshape(1, -1)
    preds.append(rf.predict(xi)[0])
    local_r2.append(rf.score(X, y))
    importances.append(rf.feature_importances_)

hex_all = hex_all.to_crs(epsg=4326)
hex_all['gwrf_pred'] = preds
hex_all['gwrf_r2'] = local_r2
for j, col in enumerate(feature_cols):
    hex_all[f'gwrf_imp_{col}'] = [imp[j] for imp in importances]

# --------------------- 6. 全市和分区空间变量重要性热图 ---------------------

zones = ['N-City', 'N-Suburb', 'S-City', 'S-Suburb']

# 全市热图
for col in feature_cols:
    plt.figure(figsize=(10,8))
    hex_all.plot(
        column=f'gwrf_imp_{col}',
        cmap='OrRd', legend=True,
        edgecolor='gray', linewidth=0.1, vmin=0, vmax=1,
        missing_kwds={'color':'lightgray','label':'No Data'}
    )
    boundary.plot(ax=plt.gca(), facecolor='none', edgecolor='black', linewidth=0.7)
    plt.title(f"GWRF: Local Importance of {col} (All London)")
    plt.tight_layout()
    plt.savefig(f'gwrf_imp_{col}_london.png', dpi=400)
    plt.show()


# --------------------- 7. 分区变量重要性均值统计表 ---------------------

summary_rows = []
for col in feature_cols:
    row = {'Variable': col, 'London Mean': np.nanmean(hex_all[f'gwrf_imp_{col}'])}
    for zone in zones:
        row[f'{zone} Mean'] = np.nanmean(hex_all.loc[hex_all['zone']==zone, f'gwrf_imp_{col}'])
    summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)
print("\n======= 变量空间重要性均值表（全市+各分区） =======")
print(summary_df.to_string(index=False))

# --------------------- 8. 箱线图：分区变量重要性分布 ---------------------

for col in feature_cols:
    plt.figure(figsize=(7,5))

    sns.boxplot(x='zone', y=f'gwrf_imp_{col}', data=hex_all, palette='Set2')

    plt.title(f"GWRF: {col} Importance by Zone (Boxplot)")
    plt.ylabel('Importance')
    plt.xlabel('Zone')
    plt.tight_layout()
    plt.savefig(f'gwrf_imp_{col}_boxplot.png', dpi=400)
    plt.show()

# --------------------- 9. 空间残差热图与局部R²热图 ---------------------

hex_all['gwrf_resid'] = hex_all['mixadj_price'] - hex_all['gwrf_pred']

plt.figure(figsize=(12,10))
hex_all.plot(
    column='gwrf_resid', cmap='bwr', legend=True,
    edgecolor='gray', linewidth=0.1,
    vmin=-60000, vmax=60000,
    missing_kwds={'color':'lightgray','label':'No Data'}
)
boundary.plot(ax=plt.gca(), facecolor='none', edgecolor='black', linewidth=0.8, alpha=0.7)
plt.title("Residuals of House Price Prediction (Observed - Predicted)")
plt.tight_layout()
plt.savefig('gwrf_residuals.png', dpi=400)
plt.show()

# ============ 全局RF模型，计算全局R² =============
from sklearn.metrics import r2_score

# 全局随机森林训练
X_global = hex_all[feature_cols].values
y_global = hex_all[target_col].values

rf_global = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
rf_global.fit(X_global, y_global)
y_pred_global = rf_global.predict(X_global)
global_r2 = r2_score(y_global, y_pred_global)

print(f"\n🌍 全局随机森林 R²（All London）：{global_r2:.4f}")
print(f"🌍 GWRF 平均局部R²（空间均值）：{np.nanmean(hex_all['gwrf_r2']):.4f}")
# --------------------- 10. 保存GWRF空间主表 ---------------------
hex_all.to_file("/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_gwrf_results.geojson", driver='GeoJSON')
print("\n✅ 全部GWRF空间分析完成，所有结果已保存！")