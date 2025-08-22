import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv("/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_full_summary.csv")

print(df.columns)
print(df.head())
print(df.info())

# 选用核心特征
features = [
    'adjusted_ptal',
    'dwelling_density',
    'Net annual income before housing costs (£)',
    'dist_to_center_km',
    'type_mode'
]
target = 'mixadj_price'
cols_to_fix = ['Net annual income before housing costs (£)', 'dwelling_density']
for c in cols_to_fix:
    df[c] = (
        df[c]
        .astype(str)
        .str.replace(",", "")   # 去掉逗号
        .str.strip()            # 去掉首尾空格
        .replace("", np.nan)    # 空字符串变nan
        .astype(float)
    )
# 1. 去除缺失和无效房价
df = df[df[target].notnull() & (df[target] > 10000)]  # >1万防止极端异常

# 2. 只保留特征与目标
df_model = df[features + [target]].dropna()

# 3. type_mode编码
df_model = pd.get_dummies(df_model, columns=['type_mode'], prefix='type')

# 显示数据分布
print(df_model.describe())

from sklearn.model_selection import train_test_split

X = df_model.drop(columns=[target])
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

rf = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("R² score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

import matplotlib.pyplot as plt
import numpy as np

# 特征重要性
importances = rf.feature_importances_
feat_names = X.columns

plt.figure(figsize=(8,5))
plt.barh(feat_names, importances)
plt.xlabel('Feature Importance')
plt.title('Random Forest: Feature Importances')
plt.tight_layout()
plt.show()

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# 路径配置
hex_csv = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_full_summary.csv"
hex_shp = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/accessible_hexagons.geojson"
boundary_shp = "/Users/muhenghe/Documents/BYLW/start/map data/statistical-gis-boundaries-london/ESRI/London_Borough_Excluding_MHW.shp"

# 读取数据
hex_df = pd.read_csv(hex_csv)
hex_gdf = gpd.read_file(hex_shp)
hex_gdf['hex_id'] = hex_gdf.index

# 合并
plot_gdf = hex_gdf.merge(hex_df[['hex_id', 'Net annual income before housing costs (£)']], on='hex_id', how='left')

# 读取伦敦边界
boundary = gpd.read_file(boundary_shp).to_crs('EPSG:4326')

# 可视化
plt.figure(figsize=(12, 10))
plot_gdf.plot(column='Net annual income before housing costs (£)', ax=plt.gca(), legend=False, cmap='YlGnBu',
              edgecolor='gray', linewidth=0.1, missing_kwds={'color': 'lightgray', "label": "No Data"})
boundary.plot(ax=plt.gca(), facecolor="none", edgecolor="black", linewidth=0.7, alpha=0.7)
plt.title("London Net Annual Income Before Housing Costs (Hexagon)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()

# 直接用上一段的hex_full_summary.csv
plot_gdf = hex_gdf.merge(hex_df[['hex_id', 'mean_AI']], on='hex_id', how='left')

plt.figure(figsize=(12, 10))
plot_gdf.plot(column='mean_AI', ax=plt.gca(), legend=True, cmap='viridis',
              edgecolor='gray', linewidth=0.1, missing_kwds={'color': 'lightgray', "label": "No Data"})
boundary.plot(ax=plt.gca(), facecolor="none", edgecolor="black", linewidth=0.7, alpha=0.7)
plt.title("London Public Transport Accessibility (mean_AI, Hexagon)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()

# 1. Adjusted PTAL spatial distribution
plt.figure(figsize=(12, 10))
plot_gdf = hex_gdf.merge(hex_df[['hex_id', 'adjusted_ptal']], on='hex_id', how='left')
plot_gdf.plot(column='adjusted_ptal', ax=plt.gca(), legend=True, cmap='viridis',
              edgecolor='gray', linewidth=0.1, missing_kwds={'color': 'lightgray', "label": "No Data"})
boundary.plot(ax=plt.gca(), facecolor="none", edgecolor="black", linewidth=0.7, alpha=0.7)
plt.title("London Adjusted PTAL (with walking decay), Hexagon")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()

# 2. Distance to Central London (km)
plt.figure(figsize=(12, 10))
plot_gdf = hex_gdf.merge(hex_df[['hex_id', 'dist_to_center_km']], on='hex_id', how='left')
plot_gdf.plot(column='dist_to_center_km', ax=plt.gca(), legend=True, cmap='YlOrRd',
              edgecolor='gray', linewidth=0.1, missing_kwds={'color': 'lightgray', "label": "No Data"})
boundary.plot(ax=plt.gca(), facecolor="none", edgecolor="black", linewidth=0.7, alpha=0.7)
plt.title("Distance to Central London (km), Hexagon")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()

# 3. Dwelling density (per ha)
plt.figure(figsize=(12, 10))
plot_gdf = hex_gdf.merge(hex_df[['hex_id', 'dwelling_density']], on='hex_id', how='left')
plot_gdf.plot(column='dwelling_density', ax=plt.gca(), legend=True, cmap='YlGnBu',
              edgecolor='gray', linewidth=0.1, missing_kwds={'color': 'lightgray', "label": "No Data"})
boundary.plot(ax=plt.gca(), facecolor="none", edgecolor="black", linewidth=0.7, alpha=0.7)
plt.title("Dwelling Density (per ha), Hexagon")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()

# 4. n_samples per hex (heatmap of transaction sample counts)
plt.figure(figsize=(12, 10))
plot_gdf = hex_gdf.merge(hex_df[['hex_id', 'n_samples']], on='hex_id', how='left')
plot_gdf.plot(column='n_samples', ax=plt.gca(), legend=True, cmap='plasma',
              edgecolor='gray', linewidth=0.1, missing_kwds={'color': 'lightgray', "label": "No Data"})
boundary.plot(ax=plt.gca(), facecolor="none", edgecolor="black", linewidth=0.7, alpha=0.7)
plt.title("Sample Size per Hexagon (n_samples), Hexagon")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()

# 5. radius_m (adaptive price estimation radius)
plt.figure(figsize=(12, 10))
plot_gdf = hex_gdf.merge(hex_df[['hex_id', 'radius_m']], on='hex_id', how='left')
plot_gdf.plot(column='radius_m', ax=plt.gca(), legend=True, cmap='cividis',
              edgecolor='gray', linewidth=0.1, missing_kwds={'color': 'lightgray', "label": "No Data"})
boundary.plot(ax=plt.gca(), facecolor="none", edgecolor="black", linewidth=0.7, alpha=0.7)
plt.title("Estimation Radius (meters) for Hexagon Price", fontsize=16)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()