import pandas as pd
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import numpy as np

# File path
hex_csv = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_to_station_walk_matrix_with_msoa_lsoa_code.csv"
ptal_csv = "/Users/muhenghe/Documents/BYLW/start/map data/LSOA_aggregated_PTAL_stats_2023.csv"
output_csv = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_to_station_walk_matrix_with_ptal.csv"

# Read the hex hexagonal table
hex_df = pd.read_csv(hex_csv)

# Read the LSOA PTAL statistics table and retain only the specified fields
ptal_df = pd.read_csv(ptal_csv, low_memory=False)
fields = ['LSOA21CD', 'mean_AI', 'MEDIAN_AI', 'Shape__Area']
ptal_df = ptal_df[fields]

# Merge, left join, and retain all hex
merged = hex_df.merge(ptal_df, left_on='lsoa_code', right_on='LSOA21CD', how='left')

# After merging, remove the LSOA21CD column
merged = merged.drop(columns=['LSOA21CD'])

# save
merged.to_csv(output_csv, index=False)
print("The PTAL metrics that only retain mean_AI, MEDIAN_AI, and Shape__Area have been merged and exported：", output_csv)

hex_csv = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_to_station_walk_matrix_with_ptal.csv"
output_csv = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_velocity_score.csv"


# 1. Read the hexagonal data containing all fields
df = pd.read_csv(hex_csv)

# 2. Calculate the average walking distance for each hex_id
avg_dist_df = df.groupby('hex_id')['walk_distance_m'].mean().reset_index().rename(columns={'walk_distance_m':'avg_walk_distance'})

# 3. Obtain the basic attributes of hex
meta_cols = ['hex_id', 'hex_lon', 'hex_lat', 'msoa_code', 'lsoa_code', 'mean_AI', 'MEDIAN_AI']
hex_base = df[meta_cols].drop_duplicates('hex_id')

# 4. Merge the basic attributes and avg_walk_distance
merged = hex_base.merge(avg_dist_df, on='hex_id', how='left')

# 5. Calculate the adjusted PTAL (using the Hansen exponential decay function)
alpha = 0.7  # Hansen exponential decay function
merged['avg_walk_km'] = merged['avg_walk_distance'] / 1000
merged['adjusted_ptal'] = merged['mean_AI'] * np.exp(-alpha * merged['avg_walk_km'])

# 7. Retain columns as needed
final_cols = ['hex_id', 'hex_lon', 'hex_lat', 'msoa_code', 'lsoa_code',
              'avg_walk_distance', 'avg_walk_km', 'mean_AI', 'adjusted_ptal']
merged[final_cols].to_csv(output_csv, index=False)
print(" The speed score has been calculated and exported：", output_csv)


hex_csv = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_velocity_score.csv"
hex_gdf_file = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/accessible_hexagons.geojson"
boundary_file = "/Users/muhenghe/Documents/BYLW/start/map data/statistical-gis-boundaries-london/ESRI/LSOA_2011_London_gen_MHW.shp"
output_img = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_velocity_score_map.png"

# Read data
hex_df = pd.read_csv(hex_csv)
hex_gdf = gpd.read_file(hex_gdf_file)
boundary = gpd.read_file(boundary_file).to_crs(epsg=4326)

# hex_id matching
hex_gdf['hex_id'] = hex_gdf.index

# Merge attributes
plot_gdf = hex_gdf.merge(hex_df[['hex_id','adjusted_ptal']], on='hex_id', how='left')
# visualization
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
# 1.Draw hexagons (Adjusted PTAL coloring)
plot_gdf.plot(column='adjusted_ptal', ax=ax, legend=True, cmap='plasma',
              edgecolor='gray', linewidth=0.15, missing_kwds={"color": "lightgray", "label": "无数据"})
plt.title("London Hexagon-based Adjusted PTAL Score")

# 2. Superimpose administrative boundaries
boundary.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.8, alpha=0.7)
plt.title("London Hexagon-based Velocity Score ")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig(output_img, dpi=300)
plt.show()
print("The spatial distribution map of the speed score has been saved：", output_img)
print("There is no hex number of mean_AI:", merged['mean_AI'].isna().sum())
print("total hex:", merged.shape[0])