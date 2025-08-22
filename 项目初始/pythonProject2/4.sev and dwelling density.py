import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# read
income_csv = "/Users/muhenghe/Documents/BYLW/start/map data/net income before housing costs.csv"
hex_csv = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_velocity_score.csv"
output_csv = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_velocity_score_with_income.csv"

# 1. read
income_df = pd.read_csv(income_csv)

# check
print(income_df.columns.tolist())

# 2. only keep data which needed
keep_cols = [
    'MSOA code',
    'Net annual income before housing costs (£)',
    'Upper confidence limit (£)',
    'Lower confidence limit (£)',
    'Confidence interval (£)'
]
income_df = income_df[keep_cols]

# 3. read
hex_df = pd.read_csv(hex_csv)

# 4. join
merged = hex_df.merge(income_df, left_on='msoa_code', right_on='MSOA code', how='left')

# 5. delete data not needed
merged = merged.drop(columns=['MSOA code'])

# 6. save result
merged.to_csv(output_csv, index=False)
print("save successfully：", output_csv)

# read
hex_csv = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_velocity_score_with_income.csv"
hex_gdf_file = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/accessible_hexagons.geojson"
borough_shp = "/Users/muhenghe/Documents/BYLW/start/map data/statistical-gis-boundaries-london/ESRI/London_Borough_Excluding_MHW.shp"
density_csv = "/Users/muhenghe/Documents/BYLW/start/map data/Number_and_density_of_dwellings_by_borough.csv"
output_csv = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_velocity_score_income_borough_dwelling.csv"

# 1. read
hex_df = pd.read_csv(hex_csv)
hex_gdf = gpd.read_file(hex_gdf_file)
hex_gdf['hex_id'] = hex_gdf.index

# 2. Generate the hex center point GeoDataFrame
hex_centroids_gdf = pd.merge(
    hex_df[['hex_id','hex_lon','hex_lat']],
    hex_gdf[['hex_id','geometry']],
    on='hex_id',
    how='left'
)
hex_centroids_gdf['geometry'] = hex_centroids_gdf.apply(lambda row: Point(row['hex_lon'], row['hex_lat']), axis=1)
hex_centroids_gdf = gpd.GeoDataFrame(hex_centroids_gdf, geometry='geometry', crs='EPSG:4326')

# 3. read Borough shapefile
borough_gdf = gpd.read_file(borough_shp).to_crs(epsg=4326)
# check borough
print(borough_gdf.columns)
borough_name_field = 'NAME'

# 4.  join
hex_centroids_gdf = gpd.sjoin(hex_centroids_gdf, borough_gdf[[borough_name_field, 'geometry']], how='left', predicate='within')
hex_centroids_gdf = hex_centroids_gdf.rename(columns={borough_name_field: 'borough_name'})
hex_borough_df = hex_centroids_gdf[['hex_id', 'borough_name']]

# 5. add hex  detail
result_df = hex_df.merge(hex_borough_df, on='hex_id', how='left')

# 6. read table，only keep Area name、2019
density_df = pd.read_csv(density_csv)
print(density_df.columns)
area_name_field = 'Area name'
density_field = '2019'
density_df = density_df[[area_name_field, density_field]]
density_df = density_df.rename(columns={area_name_field: 'borough_name', density_field: 'dwelling_density'})

# 7. join
result_df = result_df.merge(density_df, on='borough_name', how='left')

# 8. save
result_df.to_csv(output_csv, index=False)
print("✅ Borough和dwelling density已合并，结果已保存：", output_csv)

# 9. delete empty data
before_rows = result_df.shape[0]
result_df = result_df.dropna()
after_rows = result_df.shape[0]
removed_rows = before_rows - after_rows

# 10. save
cleaned_csv = output_csv.replace('.csv', '_cleaned.csv')
result_df.to_csv(cleaned_csv, index=False)
print(f"delete {removed_rows} data。result save：{cleaned_csv}")

from geopy.distance import geodesic

csv_file = "/Users/muhenghe/Documents/BYLW/start/pythonProject2/hex_velocity_score_income_borough_dwelling_cleaned.csv"

# read
df = pd.read_csv(csv_file)

# location urban centre（Charing Cross）
city_center = (51.507351, -0.127758)  # (lat, lon)

# calculate distance
df['dist_to_center_km'] = df.apply(
    lambda row: geodesic((row['hex_lat'], row['hex_lon']), city_center).km, axis=1
)


# save
df.to_csv(csv_file, index=False)
print("distance saved")