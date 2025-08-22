import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from scipy.spatial import cKDTree
from tqdm import tqdm

# 路径
hex_path = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/accessible_hexagons.geojson"
price_path = "/Users/muhenghe/Documents/BYLW/start/map data/pp_2024-2020_london_only_cleaned.csv"
output_path = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_mixadj_price_typeonly.csv"

# 参数
min_n = 150
radii = [170, 200, 300, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]  # 单位：米
alpha = 0.2  # type不一致的权重

# 1. 读取六边形并投影，生成中心点
hex_gdf = gpd.read_file(hex_path)
hex_gdf['hex_id'] = hex_gdf.index
hex_gdf = hex_gdf.to_crs(32630)   # 投影到米制
hex_gdf['centroid'] = hex_gdf.centroid

# 生成中心点坐标数组
hex_coords = np.vstack([
    hex_gdf['centroid'].x.values,
    hex_gdf['centroid'].y.values
]).T

# 2. 读取房价点（价格在1，type在4列，long/lat字段已包含）
price_df = pd.read_csv(price_path)
price_gdf = gpd.GeoDataFrame(
    price_df, geometry=gpd.points_from_xy(price_df['long'], price_df['lat']), crs="EPSG:4326"
).to_crs(32630)
price_coords = np.vstack([price_gdf.geometry.x, price_gdf.geometry.y]).T

# 3. 建立KD树
tree = cKDTree(price_coords)

# 4. 本地聚合（mix-adjusted，仅type）
records = []
print(" 正在为每个六边形计算mix-adjusted房价...")

for i, (hx, hy) in tqdm(enumerate(hex_coords), total=len(hex_coords)):
    # 自动扩圈
    local_idx = []
    for R in radii:
        idx = tree.query_ball_point([hx, hy], r=R)
        if len(idx) >= min_n or R == radii[-1]:
            local_idx = idx
            break
    if not local_idx:  # 无样本
        records.append({
            'hex_id': hex_gdf.iloc[i]['hex_id'],
            'hex_lon': hex_gdf.iloc[i]['centroid'].x,
            'hex_lat': hex_gdf.iloc[i]['centroid'].y,
            'mixadj_price': np.nan,
            'n_samples': 0,
            'radius_m': radii[-1],
            'type_mode': None
        })
        continue

    # 取数据
    prices = price_gdf.iloc[local_idx]['1'].astype(float).values
    types = price_gdf.iloc[local_idx]['4'].astype(str)
    dists = np.linalg.norm(price_coords[local_idx] - np.array([hx, hy]), axis=1)
    # 主流type
    type_mode = types.mode().values[0] if not types.mode().empty else None
    # 权重
    dists_km = dists / 1000
    w_dist = 1 / (dists_km + 1)
    w_type = np.where(types.values == type_mode, 1, 0.5)
    w_total = w_dist * w_type
    weighted_price = np.sum(prices * w_total) / np.sum(w_total)
    # 记录
    records.append({
        'hex_id': hex_gdf.iloc[i]['hex_id'],
        'hex_lon': hex_gdf.iloc[i]['centroid'].x,
        'hex_lat': hex_gdf.iloc[i]['centroid'].y,
        'mixadj_price': weighted_price,
        'n_samples': len(local_idx),
        'radius_m': R,
        'type_mode': type_mode
    })

# 5. 保存结果
df_out = pd.DataFrame(records)
df_out.to_csv(output_path, index=False)
print(f" 完成，每个hex已聚合mix-adjusted本地房价 保存为：{output_path}")
print(df_out.head())

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# 路径
hex_csv = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_mixadj_price_typeonly.csv"
hex_shp = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/accessible_hexagons.geojson"
boundary_shp = "/Users/muhenghe/Documents/BYLW/start/map data/statistical-gis-boundaries-london/ESRI/London_Borough_Excluding_MHW.shp"
output_img = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_mixadj_price_typeonly_map.png"

# 1. 读取属性表和六边形shp
hex_df = pd.read_csv(hex_csv)
hex_gdf = gpd.read_file(hex_shp)
hex_gdf['hex_id'] = hex_gdf.index

# 合并
plot_gdf = hex_gdf.merge(hex_df[['hex_id', 'mixadj_price']], on='hex_id', how='left')

# 2. 读取伦敦行政边界
boundary = gpd.read_file(boundary_shp).to_crs('EPSG:4326')

# 3. 可视化
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
# 1. 画六边形房价（空缺变灰）
plot_gdf.plot(column='mixadj_price', ax=ax, legend=True, cmap='plasma',
              edgecolor='gray', linewidth=0.1,
              missing_kwds={'color': 'lightgray', "label": "No Data"})
# 2. 叠加行政边界
boundary.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.7, alpha=0.7)
plt.title("London Local Mix-adjusted House Price (Hexagon, by Main Type)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig(output_img, dpi=300)
plt.show()
print(" 六边形mix-adjusted房价空间分布已保存：", output_img)


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 路径
hex_csv = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_mixadj_price_typeonly.csv"
hex_shp = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/accessible_hexagons.geojson"
boundary_shp = "/Users/muhenghe/Documents/BYLW/start/map data/statistical-gis-boundaries-london/ESRI/London_Borough_Excluding_MHW.shp"
output_img = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_typemode_map.png"

# 1. 读取数据
hex_df = pd.read_csv(hex_csv)
hex_gdf = gpd.read_file(hex_shp)
hex_gdf['hex_id'] = hex_gdf.index

# 合并
plot_gdf = hex_gdf.merge(hex_df[['hex_id', 'type_mode']], on='hex_id', how='left')

# 2. 类型与颜色映射
type_labels = {
    'D': 'Detached',
    'S': 'Semi-Detached',
    'T': 'Terraced',
    'F': 'Flat/Maisonette',
    'O': 'Other',
    None: 'No Data'
}
# 给每种类型指定颜色（可根据喜好调整）
type_colors = {
    'D': '#1f77b4',  # 蓝
    'S': '#ff7f0e',  # 橙
    'T': '#2ca02c',  # 绿
    'F': '#d62728',  # 红
    'O': '#9467bd',  # 紫
    None: 'lightgray'
}

plot_gdf['type_mode_label'] = plot_gdf['type_mode'].map(type_labels)
plot_gdf['type_color'] = plot_gdf['type_mode'].map(type_colors)
plot_gdf['type_color'] = plot_gdf['type_color'].fillna('lightgray')  # 填充nan为灰色

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
plot_gdf.plot(ax=ax, color=plot_gdf['type_color'], edgecolor='gray', linewidth=0.1)


boundary = gpd.read_file(boundary_shp).to_crs('EPSG:4326')
boundary.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.7, alpha=0.7)
plt.title("London Hexagons: Main Housing Type (type_mode)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# 自定义legend
import matplotlib.patches as mpatches
handles = [
    mpatches.Patch(color=type_colors[t], label=type_labels[t])
    for t in ['D','S','T','F','O',None]
]
plt.legend(handles=handles, title='Main Type', bbox_to_anchor=(1.01, 1), loc='upper left')

plt.tight_layout()
plt.savefig(output_img, dpi=300)
plt.show()
print(" type_mode空间分布图已保存：", output_img)

import pandas as pd

# 路径
file1 = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_velocity_score_income_borough_dwelling_cleaned.csv"
file2 = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_mixadj_price_typeonly.csv"
output = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_full_summary.csv"

# 读取数据
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# 找到重复的列（除hex_id外）
dup_cols = [col for col in df2.columns if col in df1.columns and col != 'hex_id']

# 合并前，删除df2中和df1重复的非hex_id列
df2 = df2.drop(columns=dup_cols)

# 合并
merged = df1.merge(df2, on='hex_id', how='left')

# 保存
merged.to_csv(output, index=False)
print(" 合并并去除重名字段成功，已保存为：", output)