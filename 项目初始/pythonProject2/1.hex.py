import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import os
import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
from shapely.geometry import Point
from tqdm import tqdm
from networkx import multi_source_dijkstra


tqdm.pandas()
OUTPUT_DIR = "/Users/muhenghe/Documents/BYLW/start/pythonProject2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Read map boundaries and subway station data
boundary = gpd.read_file("/Users/muhenghe/Documents/BYLW/start/map data/statistical-gis-boundaries-london/ESRI/LSOA_2011_London_gen_MHW.shp").to_crs(epsg=4326)
transport_points = gpd.read_file("/Users/muhenghe/Documents/BYLW/start/map data/transport point.geojson")
transport_points = transport_points[transport_points.geometry.is_valid]

# 2. Generate a hexagonal mesh
minx, miny, maxx, maxy = boundary.total_bounds
edge_length = 0.002
dx = math.sqrt(3) * edge_length
dy = 1.5 * edge_length
cols = np.arange(minx, maxx + dx, dx)
rows = np.arange(miny, maxy + dy, dy)

polygons = []
for j, x in enumerate(cols):
    for i, y in enumerate(rows):
        x0 = x + (dx / 2 if i % 2 else 0)
        hexagon = Polygon([
            (x0 + edge_length * math.cos(math.radians(a)),
             y  + edge_length * math.sin(math.radians(a)))
            for a in range(0, 360, 60)
        ])
        if boundary.contains(hexagon.centroid).any():
            polygons.append(hexagon)

hexgrid = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")

# 3.Spatial connection: hex containing at least one subway station
joined = gpd.sjoin(hexgrid, transport_points, how="left", predicate="contains")
hex_with_stops = joined[~joined.index_right.isna()].drop_duplicates("geometry")
print(f"A hexagon containing the site: {len(hex_with_stops)}")

# 4. hex, which is within a 25-minute walk (2.083 km) to the subway station
hex_utm = hexgrid.to_crs(epsg=32630)
tp_utm  = transport_points.to_crs(epsg=32630)
union  = tp_utm.geometry.union_all()
def dist(p): return p.centroid.distance(union)
hex_utm["distance_to_stop"] = hex_utm.geometry.progress_apply(dist)
hex_near = hex_utm[hex_utm.distance_to_stop <= 2083]

# Merge all reachable hexagon
final_hex = pd.concat([hex_with_stops.to_crs(32630), hex_near]).drop_duplicates("geometry").to_crs(4326)
final_hex.reset_index(drop=True, inplace=True)
final_hex.to_file(os.path.join(OUTPUT_DIR, "accessible_hexagons.geojson"), driver="GeoJSON")
print(f"It can eventually reach a hexagon: {len(final_hex)}")

# 5. Visualization
fig, ax = plt.subplots(1,1,figsize=(10,10))
boundary.plot(ax=ax, color="white", edgecolor="black", linewidth=0.5)
final_hex.plot(ax=ax, color="lightblue", edgecolor="gray", alpha=0.6, linewidth=0.3)
transport_points.plot(ax=ax, color="red", markersize=5)
plt.title("15-min Accessible Hexagons in Greater London")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "map.svg"))
plt.show()

# =======================
# 6. Real OSM walking distance modeling (calculated only for final_hex)
# =======================
print("Download the London Pedestrian Network (OSMnx)...")
G = ox.graph_from_place("London, UK", network_type='walk', simplify=True)
print("The number of nodes in the pedestrian road network:", len(G.nodes))

# Generate the center point
final_hex_centroids = final_hex.copy()
final_hex_centroids["geometry"] = final_hex_centroids.centroid
final_hex_centroids = final_hex_centroids.set_geometry("geometry").to_crs(epsg=4326)
tube_points = transport_points.to_crs(epsg=4326)

# Find the nearest node of the OSM walking network
print(" Match OSM network nodes ...")
xs = final_hex_centroids.geometry.x.values
ys = final_hex_centroids.geometry.y.values
final_hex_centroids['osmid'] = ox.nearest_nodes(G, xs, ys)
tube_points['osmid'] = tube_points.geometry.apply(
    lambda p: ox.nearest_nodes(G, p.x, p.y)
)

# Save the OSM node list of the subway station in advance
station_osmids = tube_points['osmid'].tolist()
station_ids = tube_points['id'].tolist()
station_names = tube_points['name'].tolist()

nearest_station_ids = []
nearest_station_names = []
walk_distances = []
# Switch to UTM (meters) coordinates for straight-line distances
final_hex_centroids_utm = final_hex_centroids.to_crs(epsg=32630)
tube_points_utm = tube_points.to_crs(epsg=32630)

hex_osmids = final_hex_centroids['osmid'].tolist()
hex_ids = final_hex_centroids.index.tolist()
hex_geoms = final_hex_centroids_utm.geometry.tolist()

station_osmids = tube_points['osmid'].tolist()
station_ids = tube_points['id'].tolist()
station_names = tube_points['name'].tolist()
station_geoms = tube_points_utm.geometry.tolist()

results = []
print(" Batch filter each hexagon one by one and calculate the actual walking distance ...")
for i, (hex_osmid, hex_geom) in tqdm(enumerate(zip(hex_osmids, hex_geoms)), total=len(hex_osmids)):
    hex_id = hex_ids[i]
    # Step 1
    dists = [hex_geom.distance(sg) for sg in station_geoms]
    close_idx = [j for j, d in enumerate(dists) if d <= 2083]
    if not close_idx:
        continue
    # Step 2
    for j in close_idx:
        station_osmid = station_osmids[j]
        station_id = station_ids[j]
        station_name = station_names[j]
        line_dist = dists[j]
        try:
            walk_dist = nx.shortest_path_length(G, hex_osmid, station_osmid, weight='length')
            # Step 3
            if walk_dist <= 2083:
                results.append({
                    "hex_id": hex_id,
                    "station_id": station_id,
                    "station_name": station_name,
                    "straight_distance_m": line_dist,
                    "walk_distance_m": walk_dist
                })
        except Exception:
            continue

# 导出为csv
df = pd.DataFrame(results)
df.to_csv(os.path.join(OUTPUT_DIR, "hex_to_station_walk_matrix.csv"), index=False)
print("The csv of the real distance from hex to the site has been exported. Location：", os.path.join(OUTPUT_DIR, "hex_to_station_walk_matrix.csv"))