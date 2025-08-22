import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# 1. read
ppd_paths = [
    "/Users/muhenghe/Documents/BYLW/start/map data/pp-2020.csv",
    "/Users/muhenghe/Documents/BYLW/start/map data/pp-2021.csv",
    "/Users/muhenghe/Documents/BYLW/start/map data/pp-2022.csv",
    "/Users/muhenghe/Documents/BYLW/start/map data/pp-2023.csv",
    "/Users/muhenghe/Documents/BYLW/start/map data/pp-2024.csv"
]
ons_path = "/Users/muhenghe/Documents/BYLW/start/map data/ONSPD_FEB_2024_UK.csv"
london_shp = "/Users/muhenghe/Documents/BYLW/start/map data/statistical-gis-boundaries-london/ESRI/London_Borough_Excluding_MHW.shp"
output_path = "/Users/muhenghe/Documents/BYLW/start/map data/pp_2024-2022_london_only.csv"

# 2. read
dfs = [pd.read_csv(p, header=None, encoding="latin1") for p in ppd_paths]
ppd = pd.concat(dfs, ignore_index=True)

# 3. read ONS postcode location
ons = pd.read_csv(ons_path, usecols=['pcd', 'lat', 'long'])
ons['pcd'] = ons['pcd'].astype(str).str.strip().str.upper()

# 4. stander postcode
ppd['postcode'] = ppd[3].astype(str).str.strip().str.upper()

# 5. join
ppd = ppd.merge(ons, left_on='postcode', right_on='pcd', how='left')

# 6. delete empty
ppd = ppd.dropna(subset=['lat', 'long'])

# 7. transform to GeoDataFrame
ppd['lat'] = ppd['lat'].astype(float)
ppd['long'] = ppd['long'].astype(float)
gdf = gpd.GeoDataFrame(ppd, geometry=gpd.points_from_xy(ppd['long'], ppd['lat']), crs="EPSG:4326")

# 8. read boundary
london = gpd.read_file(london_shp).to_crs("EPSG:4326")
london_union = london.unary_union

# 9. keep point inside London
gdf['in_london'] = gdf.geometry.within(london_union)
gdf_london = gdf[gdf['in_london']].copy()

# 10. output
gdf_london.drop(columns=['geometry','in_london']).to_csv(output_path, index=False)
print("sample number:", len(gdf_london))

# read
price_path = "/Users/muhenghe/Documents/BYLW/start/map data/pp_2024-2022_london_only.csv"
output_path = "/Users/muhenghe/Documents/BYLW/start/map data/pp_2024-2020_london_only_cleaned.csv"

# 1. read
df = pd.read_csv(price_path)

# 2. delete type L
filtered = df[(df['4'] != 'O') & (df['6'] != 'L')]

# 3. save
filtered.to_csv(output_path, index=False)
print(f"delete type='O'和'6'='L'，rest sample：{filtered.shape[0]}")