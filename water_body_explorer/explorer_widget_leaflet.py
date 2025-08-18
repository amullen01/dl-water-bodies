import pandas as pd
import geopandas as gpd
import ipyleaflet as ipl
import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
import json
from shapely.geometry import mapping
import branca.colormap as cm
import seaborn as sns
import cmcrameri

warnings.filterwarnings('ignore')

class WaterBodyExplorer:
    def __init__(self, database_path, above_tiles, tile_id):
        """Initialize the Water Body Explorer"""
        self.db_path = Path(database_path)
        self.above_tiles = above_tiles
        self.above_tile_id = tile_id
        self.load_data()
        self.setup_widgets()
        self.create_interface()
        
    def load_data(self):
        print('loading water bodies')
        catalog_path = self.db_path / "metadata" / "water_bodies_catalog.parquet"
        
        self.water_bodies = gpd.read_parquet(catalog_path)
        
        self.above_tiles = self.above_tiles.to_crs(self.water_bodies.crs)
        bounds = self.above_tiles.loc[self.above_tiles['grid_id']==self.above_tile_id].total_bounds

        print('filtering')
        self.water_bodies = self.water_bodies.cx[
                                    bounds[0]:bounds[2], 
                                    bounds[1]:bounds[3]
                                  ]
        
        if self.water_bodies.crs != 'EPSG:4326':
            self.water_bodies = self.water_bodies.to_crs('EPSG:4326')

        self.measurements = self.load_measurements()
        self.create_summary_stats()

    def load_measurements(self):
        print('loading measurements')
        path = self.db_path / "measurements"
        files = list(path.rglob("*.parquet"))
        dfs = [pd.read_parquet(f) for f in files if f.exists()]
        df = pd.concat(dfs, ignore_index=True)
        df = df.loc[df['water_body_id'].isin(self.water_bodies['water_body_id'].unique())]
        df['measurement_date'] = pd.to_datetime(df['measurement_date'])
        return df
    
    def create_summary_stats(self, filtered_measurements = None):
        """Create summary statistics for filtering"""
        if self.measurements.empty:
            self.summary_stats = pd.DataFrame()
            return
        
        if filtered_measurements is None:
            measurements = self.measurements

        else:
            measurements = filtered_measurements
        # Calculate summary statistics per water body
        self.summary_stats = measurements.groupby('water_body_id').agg({
            'surface_area': ['mean', 'median', 'std', 'min', 'max', lambda x: x.quantile(0.25), 
                            lambda x: x.quantile(0.75), 'count']
        }).round(2)
        
        # Flatten column names
        self.summary_stats.columns = ['surface_area_mean', 'surface_area_median',
                                      'surface_area_std', 'surface_area_min', 'surface_area_max', 
                                      'surface_area_q25', 'surface_area_q75', 'surface_area_count']
        self.summary_stats = self.summary_stats.reset_index()
        
        # Merge with geometry
        self.summary_stats = self.summary_stats.merge(
            self.water_bodies[['water_body_id', 'geometry', 'reference_area']], 
            on='water_body_id', 
            how='left'
        )
        
        # Calculate change metrics
        self.summary_stats['surface_area_dynamism'] = (
            (self.summary_stats['surface_area_q75'] - self.summary_stats['surface_area_q25']) / 
            self.summary_stats['surface_area_median'] * 100
        )
        
    def setup_widgets(self):

        """Set up all interactive widgets"""
        
        # Area filters
        if not self.summary_stats.empty:
            min_area = self.summary_stats['surface_area_min'].min()
            max_area = self.summary_stats['surface_area_max'].max()
            
            self.area_range = widgets.FloatRangeSlider(
                value=[min_area, max_area],
                min=min_area,
                max=max_area,
                step=(max_area - min_area) / 100,
                description='Area Range (m²):',
                style={'description_width': 'initial'},
                readout_format='.0f'
            )
            
            # Change percentage filter
            min_change = self.summary_stats['surface_area_dynamism'].min()
            max_change = self.summary_stats['surface_area_dynamism'].max()
            
            self.change_range = widgets.FloatRangeSlider(
                value=[min_change, max_change],
                min=min_change,
                max=max_change,
                step=(max_change - min_change) / 100,
                description='Area dynamism (%):',
                style={'description_width': 'initial'},
                readout_format='.1f'
            )

            # Change percentage filter
            min_doy = self.measurements['measurement_date'].dt.dayofyear.min()
            max_doy = self.measurements['measurement_date'].dt.dayofyear.max()
            
            self.doy_range = widgets.FloatRangeSlider(
                value=[min_doy, max_doy],
                min=min_doy,
                max=max_doy,
                step=1,
                description='Day of year:',
                style={'description_width': 'initial'},
                readout_format='.1f'
            )
        
        # Color by option
        self.color_by = widgets.Dropdown(
            options=['Mean Area', 'Dynamism %'],
            value='Mean Area',
            description='Color By:'
        )
        
        # Selected water body info
        self.selected_info = widgets.HTML(
            value="<b>Click on a water body to see details and plots</b>",
            layout=widgets.Layout(height='150px', overflow='auto')
        )
        
        # Update button
        self.update_button = widgets.Button(
            description='Update Map',
            button_style='primary',
            tooltip='Apply filters and update map'
        )
        
        # Export button
        self.export_button = widgets.Button(
            description='Export Filtered Data',
            button_style='info',
            tooltip='Export current selection as CSV'
        )
        
        # Clear plots button
        self.clear_plots_button = widgets.Button(
            description='Clear Plots',
            button_style='warning',
            tooltip='Clear all plots'
        )
        
        # Connect button callbacks
        self.update_button.on_click(self.update_map)
        self.export_button.on_click(self.export_data)
        self.clear_plots_button.on_click(self.clear_plots)
        
    def create_interface(self):
        """Create the main interface layout"""
        
        # Filter controls
        filter_controls = widgets.VBox([
            widgets.HTML("<h3>🌊 Water Body Explorer</h3>"),
            self.change_range,
            self.area_range,
            self.doy_range,
            widgets.HBox([self.color_by]),
            widgets.HBox([self.update_button, self.export_button, self.clear_plots_button]),
            self.selected_info
        ])
        
        # Create the map
        self.create_map()
        
        # Plot output
        self.plot_output = widgets.Output()
        
        # Create main layout
        self.main_layout = widgets.VBox([
            widgets.VBox([filter_controls], layout=widgets.Layout(width='100%')),
            widgets.VBox([
                self.map,
                self.plot_output
            ], layout=widgets.Layout(width='100%'))
        ])
        
        # Initial map update
        self.update_map(None)
        
    def create_map(self):
        """Create the ipyleaflet map"""
        # Default center (will be updated when data is loaded)
        if not self.water_bodies.empty:
            bounds = self.water_bodies.total_bounds
            center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        else:
            center = [64.0, -155.0]
        
        # Create map
        self.map = ipl.Map(
            center=center,
            zoom=6,
            layout=widgets.Layout(width='100%')
        )
        
        # Store reference to current geo_json layer
        self.current_layer = None
        
        # Add basemap options
        self.update_basemap()
        
    def update_basemap(self):
        """Update the basemap based on selection"""
        # Clear existing basemaps (keep only the first default layer)
        if len(self.map.layers) > 1:
            self.map.layers = self.map.layers[:1]

        imagery_layer = ipl.TileLayer(
                url='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                name='Esri World Imagery',
                attribution='Tiles © Esri'
            )
        self.map.add_layer(imagery_layer)
    
    def filter_data(self):
        """Apply current filters to the data"""

        doy_min, doy_max = self.doy_range.value

        filtered_measurements = self.measurements.loc[
            (self.measurements['measurement_date'].dt.day_of_year >= doy_min) & 
            (self.measurements['measurement_date'].dt.day_of_year <= doy_max)
        ]

        self.create_summary_stats(filtered_measurements)

        if self.summary_stats.empty:
            return gpd.GeoDataFrame()
        
        filtered = self.summary_stats.copy()
        
        # Apply filters
        
        area_min, area_max = self.area_range.value
        filtered = filtered[
            (filtered['surface_area_mean'] >= area_min) & 
            (filtered['surface_area_mean'] <= area_max)
        ]
        
        change_min, change_max = self.change_range.value
        filtered = filtered[
            (filtered['surface_area_dynamism'] >= change_min) & 
            (filtered['surface_area_dynamism'] <= change_max)
        ]
        
        
        
        return gpd.GeoDataFrame(filtered, crs=self.water_bodies.crs)
    
    def get_color_values(self, filtered_gdf):
        """Get values for coloring based on selection"""
        if filtered_gdf.empty:
            return []
        
        color_map = {
            'Mean Area': 'surface_area_mean',
            'Dynamism %': 'surface_area_dynamism'
        }
        
        column = color_map[self.color_by.value]
        return filtered_gdf[column].fillna(0).values
    
    def create_geojson_layer(self, filtered_gdf):
        """Create GeoJSON layer for ipyleaflet"""
        if filtered_gdf.empty:
            return None
        
        # Get color values
        color_values = self.get_color_values(filtered_gdf)
        
        if len(color_values) == 0:
            return None
        
        if self.color_by.value == 'Dynamism %':
            # Create colormap
            colormap = cm.LinearColormap(
                colors=['blue','red'],
                vmin=0,
                vmax=100
            )
        
        else:
            # Create colormap
            colormap = cm.LinearColormap(
                colors=['blue', 'green', 'yellow', 'red'],
                vmin=min(color_values),
                vmax=max(color_values)
            )
        
        # Create features with properties
        features = []
        for idx, row in filtered_gdf.iterrows():
            if pd.isna(row.geometry):
                continue
            
            # Get color for this feature
            color = colormap(color_values[idx])
            
            feature = {
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": {
                    "water_body_id": row['water_body_id'],
                    "surface_area_mean": row['surface_area_mean'],
                    "surface_area_dynamism": row['surface_area_dynamism'],
                    "surface_area_count": row['surface_area_count'],
                    "color": color,
                    "style": {
                        "fillColor": color,
                        "color": "black",
                        "weight": 1,
                        "fillOpacity": 0.7
                    }
                }
            }
            features.append(feature)
        
        # Create GeoJSON object
        geojson_data = {
            "type": "FeatureCollection",
            "features": features
        }
        
        # Create GeoJSON layer
        geo_json = ipl.GeoJSON(
            data=geojson_data,
            style={
                "fillOpacity": 0.3,
                "weight": 1,
                "color": "black"
            },
            hover_style={
                "fillOpacity": 0.9,
                "weight": 3
            }
        )
        
        # Add click handler - try multiple approaches for robustness
        def handle_click(**kwargs):
            self.on_feature_click(**kwargs)
        
        geo_json.on_click(handle_click)
        
        return geo_json
    
    def on_feature_click(self, event=None, feature=None, **kwargs):
        """Handle feature click events"""
        # Debug print to see what we're getting
        print(f"Click event received: {event}")
        
        # Try different ways to access the feature data
        if feature is not None:
            properties = feature.get('properties', {})
        elif hasattr(event, 'feature'):
            properties = event.feature.get('properties', {})
        elif isinstance(event, dict) and 'feature' in event:
            properties = event['feature'].get('properties', {})
        else:
            print("Could not extract feature data from click event")
            return
        
        if not properties:
            print("No properties found in feature")
            return
            
        water_body_id = properties.get('water_body_id')
        mean_area = properties.get('surface_area_mean', 0)
        change_pct = properties.get('surface_area_dynamism', 0)
        obs_count = properties.get('surface_area_count', 0)
        
        if not water_body_id:
            print("No water_body_id found in properties")
            return
        
        print(f"Processing click for water body: {water_body_id}")
        
        # Update info display
        info_html = f"""
        <b>Selected Water Body: {water_body_id}</b><br>
        <b>Mean Area:</b> {mean_area:.0f} m²<br>
        <b>Dynamism:</b> {change_pct:.1f}%<br>
        <b>Observations:</b> {obs_count}<br>
        <small>Generating plots...</small>
        """
        self.selected_info.value = info_html
        
        # Generate plots
        self.generate_plots(water_body_id)
    
    def generate_plots(self, water_body_id):
        """Generate plots for the selected water body"""
        with self.plot_output:
            clear_output(wait=True)
            
            if self.measurements.empty:
                print("No measurements available")
                return
            
            wb_data = self.measurements[self.measurements['water_body_id'] == water_body_id].copy()
            
            if wb_data.empty:
                print(f"No data available for water body {water_body_id}")
                return
            
            wb_data = wb_data.sort_values('measurement_date')
            
            self.create_time_series_plot(wb_data, water_body_id)
            
    
    def create_time_series_plot(self, wb_data, water_body_id):

        """Create seasonality plot"""
        # Extract month and filter May-September
        wb_data['month'] = wb_data['measurement_date'].dt.month
        wb_data['year'] = wb_data['measurement_date'].dt.year
        wb_data['doy'] = wb_data['measurement_date'].dt.dayofyear
        seasonal_data = wb_data[wb_data['month'].between(5, 9)].copy()
        
        if seasonal_data.empty:
            print("No seasonal data available (May-September)")
            return
        
        # Calculate monthly statistics
        monthly_stats = seasonal_data.groupby('month').agg({
            'surface_area': ['mean', 'std', 'count']
        }).reset_index()
        
        monthly_stats.columns = ['month', 'mean_area', 'std_area', 'count']
        
        fig, axes=plt.subplots(2,1, figsize=(10,4))
        # Approximate mid-points of each month in terms of day of year
        month_midpoints = [135, 166, 196, 227, 258]
        month_names = ['May', 'Jun', 'Jul', 'Aug', 'Sep']
        sns.scatterplot(x=wb_data['doy'], y=wb_data['surface_area'], hue=wb_data['year'], 
                        ax=axes[0], palette = 'Dark2', alpha=0.7, style = wb_data['year'])
        axes[0].set_xticks(month_midpoints, month_names)

        axes[0].legend(ncols=5, loc=(0.1,1.05), frameon=True)
        
        sns.barplot(x=monthly_stats['month'], y=monthly_stats['mean_area'],ax=axes[1])
        x_coords = [p.get_x() + 0.5 * p.get_width() for p in axes[1].patches]
        y_coords = [p.get_height() for p in axes[1].patches]
        axes[1].errorbar(x=x_coords, y=y_coords, yerr=monthly_stats["std_area"], fmt="none", c="k")

        axes[0].set_xlabel('')
        axes[1].set_xlabel('')
        axes[0].set_ylabel('surface area (m$^2$)')
        axes[1].set_ylabel('surface area (m$^2$)')

        axes[0].text(x=1,y=0.4,s='time series', transform=axes[0].transAxes, rotation=270)
        axes[1].text(x=1,y=0.4,s='seasonality', transform=axes[1].transAxes, rotation=270)

        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        plt.show()
        
    
    def update_map(self, button):
        """Update the map with current filters"""
        print("Applying filters...")
        filtered_gdf = self.filter_data().reset_index()
        
        # Update basemap
        self.update_basemap()
        
        # Add new layer
        if not filtered_gdf.empty:
            self.current_layer = self.create_geojson_layer(filtered_gdf)
            if self.current_layer is not None:
                self.map.add_layer(self.current_layer)
                
                # Fit map bounds
                bounds = filtered_gdf.total_bounds
                self.map.fit_bounds([
                    [bounds[1], bounds[0]],  # SW corner
                    [bounds[3], bounds[2]]   # NE corner
                ])
        
        print(f"Map updated! Showing {len(filtered_gdf)} water bodies")
        
        # Clear selection info
        self.selected_info.value = "<b>Click on a water body to see details and plots</b>"
        
        # Clear plots
        with self.plot_output:
            clear_output()
    
    def clear_plots(self, button):
        """Clear all plots"""
        with self.plot_output:
            clear_output()
        self.selected_info.value = "<b>Click on a water body to see details and plots</b>"
    
    def export_data(self, button):
        """Export filtered data to CSV"""
        filtered_gdf = self.filter_data()
        
        if filtered_gdf.empty:
            print("No data to export!")
            return
        
        # Drop geometry for CSV export
        export_df = filtered_gdf.drop('geometry', axis=1)
        
        # Export to CSV
        filename = f"water_bodies_filtered_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        export_df.to_csv(filename, index=False)
        
        print(f"✅ Exported {len(export_df)} water bodies to {filename}")
    
    def display(self):
        """Display the complete interface"""
        display(self.main_layout)

# Usage function
def create_water_body_explorer(database_path, above_tiles, tile_id):
    """
    Create and display the water body explorer widget
    
    Parameters:
    database_path: Path to your water bodies database
    above_tiles: GeoDataFrame of tiles
    tile_id: ID of the tile to focus on
    
    Returns:
    WaterBodyExplorer instance
    """
    explorer = WaterBodyExplorer(database_path, above_tiles, tile_id)
    explorer.display()
    return explorer
