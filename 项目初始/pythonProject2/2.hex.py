import pandas as pd
import os
import geopandas as gpd

OUTPUT_DIR = "/Users/muhenghe/Documents/BYLW/start/pythonProject2"

# 1. Read the original csv
df = pd.read_csv(os.path.join(OUTPUT_DIR, "hex_to_station_walk_matrix.csv"))
boundary = gpd.read_file("/Users/muhenghe/Documents/BYLW/start/map data/statistical-gis-boundaries-london/ESRI/LSOA_2011_London_gen_MHW.shp").to_crs(epsg=4326)

# 2. Delete the straight-line distance
df = df.drop(columns=['straight_distance_m'])

# 3. New walking time (minutes)，5km/h = 83.33 m/min
df['walk_time_min'] = df['walk_distance_m'] / 83.33

# 4. Save the new csv
df.to_csv(os.path.join(OUTPUT_DIR, "hex_to_station_walk_time.csv"), index=False)
print("The simplified csv has been saved：", os.path.join(OUTPUT_DIR, "hex_to_station_walk_time.csv"))

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import os

OUTPUT_DIR = "/Users/muhenghe/Documents/BYLW/start/pythonProject2"

# 1. Read the aggregated hexagonal geojson (or directly use final_hex_centroids/accessible_hexagons)
final_hex_centroids = gpd.read_file(os.path.join(OUTPUT_DIR, "accessible_hexagons.geojson"))

# 2. Read the csv of the walking time
df = pd.read_csv(os.path.join(OUTPUT_DIR, "hex_to_station_walk_time.csv"))

# 3. Aggregation: The shortest walking time from each hexagon to the nearest station
min_walk_time = df.groupby('hex_id')['walk_time_min'].min().reset_index()

# 4. Merge the attributes back to the GeoDataFrame
final_hex_centroids['hex_id'] = final_hex_centroids.index  # 如果hex_id是index
plot_gdf = final_hex_centroids.merge(min_walk_time, on='hex_id', how='left')

# 5. visualization
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# 1. Hexagonal main image
plot_gdf.plot(column='walk_time_min', ax=ax, legend=True, cmap='viridis',
              edgecolor='gray', linewidth=0.2, missing_kwds={"color": "lightgray"})

# 2. Superimposed boundary
boundary.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.2, alpha=0.4)

plt.title("Shortest walking time from the hexagon to the nearest subway station (minutes)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hex_min_walk_time_map_v2.svg"))
plt.show()

print("The number of hexagons without walking time：", plot_gdf['walk_time_min'].isna().sum())


# 1. Read the hexagonal GeoDataFrame to generate the center point and coordinates
hex_gdf = gpd.read_file(os.path.join(OUTPUT_DIR, "accessible_hexagons.geojson"))
hex_gdf['hex_id'] = hex_gdf.index
hex_gdf['hex_centroid'] = hex_gdf.centroid
hex_gdf['hex_lon'] = hex_gdf['hex_centroid'].x
hex_gdf['hex_lat'] = hex_gdf['hex_centroid'].y

# 2. Read the LSOA boundary file and extract the code fields
boundary = gpd.read_file("/Users/muhenghe/Documents/BYLW/start/map data/statistical-gis-boundaries-london/ESRI/LSOA_2011_London_gen_MHW.shp").to_crs(epsg=4326)
msoa_code_field = 'MSOA11CD'
lsoa_code_field = 'LSOA11CD'
fields_needed = ['geometry']
if msoa_code_field in boundary.columns: fields_needed.append(msoa_code_field)
if lsoa_code_field in boundary.columns: fields_needed.append(lsoa_code_field)
boundary = boundary[fields_needed]

# 3. Generate the center point gdf
hex_centroids_gdf = hex_gdf[['hex_id', 'hex_centroid', 'hex_lon', 'hex_lat']].copy()
hex_centroids_gdf = gpd.GeoDataFrame(hex_centroids_gdf, geometry='hex_centroid', crs='EPSG:4326')

# 4. Spatial join: Obtain MSOA/LSOA code
joined = gpd.sjoin(hex_centroids_gdf, boundary, how='left', predicate='within')

# 5. Take out the required fields and rename them
col_map = {}
if msoa_code_field in joined.columns: col_map[msoa_code_field] = 'msoa_code'
if lsoa_code_field in joined.columns: col_map[lsoa_code_field] = 'lsoa_code'
joined = joined.rename(columns=col_map)

hex_centroids_df = joined[['hex_id', 'hex_lon', 'hex_lat'] + list(col_map.values())]

# 6. Read the csv and merge
df = pd.read_csv(os.path.join(OUTPUT_DIR, "hex_to_station_walk_matrix.csv"))
df = df.merge(hex_centroids_df, on='hex_id', how='left')

# 7. Sort and save (only output the code field, not the name)
main_cols = ['hex_id', 'hex_lon', 'hex_lat']
if 'msoa_code' in hex_centroids_df.columns: main_cols.append('msoa_code')
if 'lsoa_code' in hex_centroids_df.columns: main_cols.append('lsoa_code')
cols = main_cols + [c for c in df.columns if c not in main_cols]
df = df[cols]

df.to_csv(os.path.join(OUTPUT_DIR, "hex_to_station_walk_matrix_with_msoa_lsoa_code.csv"), index=False)
print('New center point coordinates, MSOA and LSOA code columns have been added and saved')

# LSOA - How many different hex_id correspond to each LSOA
lsoa_hex_counts = df.groupby('lsoa_code')['hex_id'].nunique()
avg_hex_per_lsoa = lsoa_hex_counts.mean()
print(f"hex_id correspond to each LSOA {avg_hex_per_lsoa:.2f}  hex")

# MSOA - How many different hex_id correspond to each MSOA
msoa_hex_counts = df.groupby('msoa_code')['hex_id'].nunique()
avg_hex_per_msoa = msoa_hex_counts.mean()
print(f"hex_id correspond to each MSOA {avg_hex_per_msoa:.2f}  hex")