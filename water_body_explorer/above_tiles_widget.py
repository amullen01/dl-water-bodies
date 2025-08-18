import ipyleaflet as ipl
import geopandas as gpd
import json
from ipywidgets import HTML, VBox
from IPython.display import display
from ipywidgets import Layout

def create_interactive_polygon_map(gdf, grid_id_column='grid_id'):
    """
    Create an interactive ipyleaflet map for visualizing polygons with click functionality.
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        The GeoDataFrame containing polygon geometries
    grid_id_column : str
        The column name containing grid_id values (default: 'grid_id')
    
    Returns:
    --------
    VBox widget containing the map and info display
    """
    
    # Ensure the GeoDataFrame is in WGS84 (EPSG:4326) for leaflet
    if gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')
    
    # Calculate map center from bounds
    bounds = gdf.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Create the map
    m = ipl.Map(
        center=[center_lat, center_lon],
        zoom=10,
        basemap=ipl.basemaps.OpenStreetMap.Mapnik
    )
    
    # Create info widget to display clicked polygon info
    info_widget = HTML(
        value="<b>Click on a polygon to see its grid_id</b>",
        layout={'margin': '10px'}
    )
    
    # Convert polygons to GeoJSON and add to map
    polygon_layers = []
    
    for idx, row in gdf.iterrows():
        # Convert geometry to GeoJSON
        geom = row.geometry
        if geom.geom_type == 'Polygon':
            coords = [list(geom.exterior.coords)]
        elif geom.geom_type == 'MultiPolygon':
            coords = [list(polygon.exterior.coords) for polygon in geom.geoms]
        else:
            continue
            
        # Create GeoJSON feature
        geojson_feature = {
            "type": "Feature",
            "geometry": {
                "type": geom.geom_type,
                "coordinates": coords
            },
            "properties": {
                grid_id_column: row[grid_id_column],
                "index": idx
            }
        }
        
        # Create GeoJSON layer
        geo_json = ipl.GeoJSON(
            data=geojson_feature,
            style={
                'color': 'blue',
                'weight': 2,
                'fillColor': 'lightblue',
                'fillOpacity': 0.3
            },
            hover_style={
                'color': 'red',
                'weight': 3,
                'fillOpacity': 0.5
            }
        )
        
        # Add click handler
        def handle_click(feature, **kwargs):
            grid_id = feature['properties'][grid_id_column]
            info_widget.value = f"<b>Selected Grid ID:</b> {grid_id}"
        
        geo_json.on_click(handle_click)
        
        # Add layer to map
        m.layout = Layout(width='800px', height='300px')
        m.add_layer(geo_json)
        polygon_layers.append(geo_json)
    
    # Fit map to polygon bounds
    southwest = [bounds[1], bounds[0]]
    northeast = [bounds[3], bounds[2]]
    m.fit_bounds([southwest, northeast])
    
    # Create widget container
    map_widget = VBox([info_widget, m])
    
    return map_widget
