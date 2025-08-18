import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

class WaterBodiesDB:
    """
    Helper class for querying and analyzing the water bodies dataset
    created by WaterBodyTracker
    """
    
    def __init__(self, database_path: str):
        """
        Initialize database connection
        
        Args:
            database_path: Path to the water bodies dataset directory
        """
        self.db_path = Path(database_path)
        
        # Verify database structure
        if not self.db_path.exists():
            raise ValueError(f"Database path does not exist: {database_path}")
        
        self.metadata_path = self.db_path / "metadata"
        self.measurements_path = self.db_path / "measurements"
        
        if not self.metadata_path.exists():
            raise ValueError(f"Metadata directory not found: {self.metadata_path}")
        
        # Load water bodies catalog
        catalog_path = self.metadata_path / "water_bodies_catalog.parquet"
        if catalog_path.exists():
            self.water_bodies_catalog = gpd.read_parquet(catalog_path)
        else:
            self.water_bodies_catalog = None
            warnings.warn("Water bodies catalog not found")
        
        # Cache for loaded measurements
        self._measurements_cache = {}
        self._full_dataset_cache = None
    
    def get_water_bodies_catalog(self) -> gpd.GeoDataFrame:
        """Get the water bodies catalog with metadata"""
        if self.water_bodies_catalog is None:
            raise ValueError("Water bodies catalog not available")
        return self.water_bodies_catalog.copy()
    
    def get_water_body_info(self, water_body_id: str) -> Dict:
        """
        Get detailed information about a specific water body
        
        Args:
            water_body_id: ID of the water body
            
        Returns:
            Dictionary with water body information
        """
        if self.water_bodies_catalog is None:
            raise ValueError("Water bodies catalog not available")
        
        wb_info = self.water_bodies_catalog[
            self.water_bodies_catalog['water_body_id'] == water_body_id
        ]
        
        if len(wb_info) == 0:
            raise ValueError(f"Water body {water_body_id} not found in catalog")
        
        wb_row = wb_info.iloc[0]
        
        # Get measurement statistics
        measurements = self.get_measurements_for_water_body(water_body_id)
        
        info = {
            'water_body_id': water_body_id,
            'reference_area': wb_row['reference_area'],
            'centroid': wb_row['centroid'],
            'bbox': (wb_row['bbox_minx'], wb_row['bbox_miny'], 
                    wb_row['bbox_maxx'], wb_row['bbox_maxy']),
            'geometry': wb_row['geometry'],
            'measurement_count': len(measurements),
            'date_range': (measurements['measurement_date'].min(), 
                          measurements['measurement_date'].max()) if len(measurements) > 0 else (None, None),
            'area_stats': {
                'min': measurements['surface_area'].min() if len(measurements) > 0 else None,
                'max': measurements['surface_area'].max() if len(measurements) > 0 else None,
                'mean': measurements['surface_area'].mean() if len(measurements) > 0 else None,
                'std': measurements['surface_area'].std() if len(measurements) > 0 else None
            }
        }
        
        return info
    
    def load_measurements(self, 
                         year: Optional[int] = None,
                         month: Optional[int] = None,
                         water_body_ids: Optional[List[str]] = None,
                         use_cache: bool = True) -> pd.DataFrame:
        """
        Load measurements from parquet files with optional filtering
        
        Args:
            year: Filter by year
            month: Filter by month
            water_body_ids: Filter by water body IDs
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with measurements
        """
        cache_key = f"{year}_{month}_{hash(tuple(water_body_ids or []))}"
        
        if use_cache and cache_key in self._measurements_cache:
            return self._measurements_cache[cache_key].copy()
        
        measurements_list = []
        
        # Determine which partitions to load
        if year is not None:
            year_dirs = [self.measurements_path / f"year={year}"]
        else:
            year_dirs = [d for d in self.measurements_path.iterdir() 
                        if d.is_dir() and d.name.startswith("year=")]
        
        for year_dir in year_dirs:
            if not year_dir.exists():
                continue
                
            if month is not None:
                month_dirs = [year_dir / f"month={month:02d}"]
            else:
                month_dirs = [d for d in year_dir.iterdir() 
                            if d.is_dir() and d.name.startswith("month=")]
            
            for month_dir in month_dirs:
                parquet_file = month_dir / "measurements.parquet"
                if parquet_file.exists():
                    try:
                        df = pd.read_parquet(parquet_file)
                        measurements_list.append(df)
                    except Exception as e:
                        warnings.warn(f"Error reading {parquet_file}: {e}")
        
        if not measurements_list:
            measurements_df = pd.DataFrame()
        else:
            measurements_df = pd.concat(measurements_list, ignore_index=True)
            
            # Convert date column
            if 'measurement_date' in measurements_df.columns:
                measurements_df['measurement_date'] = pd.to_datetime(
                    measurements_df['measurement_date']
                )
            
            # Filter by water body IDs if specified
            if water_body_ids is not None:
                measurements_df = measurements_df[
                    measurements_df['water_body_id'].isin(water_body_ids)
                ]
        
        # Cache the result
        if use_cache:
            self._measurements_cache[cache_key] = measurements_df.copy()
        
        return measurements_df
    
    def get_measurements_for_water_body(self, 
                                       water_body_id: str,
                                       start_date: Optional[Union[str, datetime]] = None,
                                       end_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Get all measurements for a specific water body
        
        Args:
            water_body_id: ID of the water body
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with measurements for the water body
        """
        # Load all measurements (cached)
        measurements = self.load_measurements(water_body_ids=[water_body_id])
        
        # Apply date filters if specified
        if start_date is not None:
            start_date = pd.to_datetime(start_date)
            measurements = measurements[measurements['measurement_date'] >= start_date]
        
        if end_date is not None:
            end_date = pd.to_datetime(end_date)
            measurements = measurements[measurements['measurement_date'] <= end_date]
        
        return measurements.sort_values('measurement_date')
    
    def get_time_series(self, 
                       water_body_id: str,
                       start_date: Optional[Union[str, datetime]] = None,
                       end_date: Optional[Union[str, datetime]] = None,
                       resample_frequency: Optional[str] = None) -> pd.DataFrame:
        """
        Get time series data for a water body
        
        Args:
            water_body_id: ID of the water body
            start_date: Optional start date filter
            end_date: Optional end date filter
            resample_frequency: Optional resampling frequency ('D', 'W', 'M', 'Q', 'Y')
            
        Returns:
            DataFrame with time series data
        """
        measurements = self.get_measurements_for_water_body(
            water_body_id, start_date, end_date
        )
        
        if len(measurements) == 0:
            return pd.DataFrame()
        
        # Create time series
        ts = measurements.set_index('measurement_date')[['surface_area']].copy()
        
        # Resample if requested
        if resample_frequency is not None:
            ts = ts.resample(resample_frequency).agg({
                'surface_area': ['mean', 'min', 'max', 'count']
            })
            ts.columns = ['area_mean', 'area_min', 'area_max', 'observation_count']
        
        return ts
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for all water bodies
        
        Returns:
            DataFrame with summary statistics
        """
        if self.water_bodies_catalog is None:
            raise ValueError("Water bodies catalog not available")
        
        # Load all measurements
        all_measurements = self.load_measurements()
        
        if len(all_measurements) == 0:
            # Return empty summary with catalog water bodies
            summary = self.water_bodies_catalog[['water_body_id', 'reference_area']].copy()
            summary['observation_count'] = 0
            summary['first_observation'] = None
            summary['last_observation'] = None
            summary['mean_area'] = None
            summary['min_area'] = None
            summary['max_area'] = None
            summary['area_std'] = None
            summary['area_cv'] = None
            return summary
        
        # Calculate statistics per water body
        stats = all_measurements.groupby('water_body_id').agg({
            'surface_area': ['count', 'mean', 'min', 'max', 'std'],
            'measurement_date': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        stats.columns = [
            'water_body_id', 'observation_count', 'mean_area', 'min_area', 
            'max_area', 'area_std', 'first_observation', 'last_observation'
        ]
        
        # Calculate coefficient of variation
        stats['area_cv'] = stats['area_std'] / stats['mean_area']
        
        # Merge with catalog information
        summary = self.water_bodies_catalog[['water_body_id', 'reference_area']].merge(
            stats, on='water_body_id', how='left'
        )
        
        # Fill NaN values for water bodies with no measurements
        summary['observation_count'] = summary['observation_count'].fillna(0).astype(int)
        
        return summary
    
    def find_water_bodies_by_area(self, 
                                 min_area: Optional[float] = None,
                                 max_area: Optional[float] = None,
                                 use_reference_area: bool = True) -> List[str]:
        """
        Find water bodies by area criteria
        
        Args:
            min_area: Minimum area threshold
            max_area: Maximum area threshold
            use_reference_area: Whether to use reference area from catalog
            
        Returns:
            List of water body IDs matching criteria
        """
        if use_reference_area:
            if self.water_bodies_catalog is None:
                raise ValueError("Water bodies catalog not available")
            area_series = self.water_bodies_catalog.set_index('water_body_id')['reference_area']
        else:
            # Use mean observed area
            summary = self.get_summary_statistics()
            area_series = summary.set_index('water_body_id')['mean_area'].dropna()
        
        mask = pd.Series(True, index=area_series.index)
        
        if min_area is not None:
            mask &= (area_series >= min_area)
        
        if max_area is not None:
            mask &= (area_series <= max_area)
        
        return area_series[mask].index.tolist()
    
    def find_water_bodies_by_location(self, 
                                     bounds: Tuple[float, float, float, float],
                                     buffer_distance: float = 0) -> List[str]:
        """
        Find water bodies within geographic bounds
        
        Args:
            bounds: (minx, miny, maxx, maxy) bounding box
            buffer_distance: Buffer distance around bounds
            
        Returns:
            List of water body IDs within bounds
        """
        if self.water_bodies_catalog is None:
            raise ValueError("Water bodies catalog not available")
        
        minx, miny, maxx, maxy = bounds
        
        if buffer_distance > 0:
            minx -= buffer_distance
            miny -= buffer_distance
            maxx += buffer_distance
            maxy += buffer_distance
        
        # Use spatial indexing for efficient search
        water_bodies_in_bounds = self.water_bodies_catalog.cx[minx:maxx, miny:maxy]
        
        return water_bodies_in_bounds['water_body_id'].tolist()
    
    def get_change_detection_results(self, 
                                   water_body_id: str,
                                   reference_period_days: int = 30,
                                   change_threshold: float = 0.2) -> pd.DataFrame:
        """
        Detect significant changes in water body area
        
        Args:
            water_body_id: ID of the water body
            reference_period_days: Number of days for reference period
            change_threshold: Threshold for significant change (as fraction)
            
        Returns:
            DataFrame with change detection results
        """
        measurements = self.get_measurements_for_water_body(water_body_id)
        
        if len(measurements) < 2:
            return pd.DataFrame()
        
        # Calculate rolling reference area
        measurements = measurements.sort_values('measurement_date')
        measurements['reference_area'] = measurements['surface_area'].rolling(
            window=reference_period_days, min_periods=1, center=True
        ).mean()
        
        # Calculate change metrics
        measurements['area_change'] = measurements['surface_area'] - measurements['reference_area']
        measurements['area_change_pct'] = measurements['area_change'] / measurements['reference_area']
        
        # Identify significant changes
        measurements['significant_change'] = (
            np.abs(measurements['area_change_pct']) > change_threshold
        )
        
        # Add change direction
        measurements['change_direction'] = np.where(
            measurements['area_change'] > 0, 'increase',
            np.where(measurements['area_change'] < 0, 'decrease', 'stable')
        )
        
        return measurements[['measurement_date', 'surface_area', 'reference_area', 
                           'area_change', 'area_change_pct', 'significant_change', 
                           'change_direction']]
    
    def get_seasonal_patterns(self, 
                            water_body_id: str,
                            aggregation_method: str = 'mean') -> pd.DataFrame:
        """
        Analyze seasonal patterns in water body area
        
        Args:
            water_body_id: ID of the water body
            aggregation_method: Method for aggregation ('mean', 'median', 'min', 'max')
            
        Returns:
            DataFrame with seasonal patterns
        """
        measurements = self.get_measurements_for_water_body(water_body_id)
        
        if len(measurements) == 0:
            return pd.DataFrame()
        
        # Add temporal components
        measurements['year'] = measurements['measurement_date'].dt.year
        measurements['month'] = measurements['measurement_date'].dt.month
        measurements['day_of_year'] = measurements['measurement_date'].dt.dayofyear
        measurements['season'] = measurements['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Aggregate by different temporal scales
        patterns = {}
        
        # Monthly patterns
        monthly = measurements.groupby('month')['surface_area'].agg([
            'mean', 'median', 'min', 'max', 'std', 'count'
        ])
        monthly.index.name = 'month'
        patterns['monthly'] = monthly
        
        # Seasonal patterns
        seasonal = measurements.groupby('season')['surface_area'].agg([
            'mean', 'median', 'min', 'max', 'std', 'count'
        ])
        patterns['seasonal'] = seasonal
        
        # Annual patterns
        annual = measurements.groupby('year')['surface_area'].agg([
            'mean', 'median', 'min', 'max', 'std', 'count'
        ])
        patterns['annual'] = annual
        
        return patterns
    
    def export_water_body_data(self, 
                              water_body_id: str,
                              output_path: str,
                              format: str = 'csv') -> None:
        """
        Export all data for a water body
        
        Args:
            water_body_id: ID of the water body
            output_path: Path for output file
            format: Output format ('csv', 'parquet', 'json')
        """
        # Get water body info
        info = self.get_water_body_info(water_body_id)
        measurements = self.get_measurements_for_water_body(water_body_id)
        
        # Combine metadata and measurements
        export_data = {
            'metadata': {
                'water_body_id': info['water_body_id'],
                'reference_area': info['reference_area'],
                'measurement_count': info['measurement_count'],
                'date_range': info['date_range'],
                'area_stats': info['area_stats']
            },
            'measurements': measurements.to_dict('records') if len(measurements) > 0 else []
        }
        
        output_path = Path(output_path)
        
        if format == 'csv':
            # Export measurements only as CSV
            if len(measurements) > 0:
                measurements.to_csv(output_path, index=False)
        elif format == 'parquet':
            if len(measurements) > 0:
                measurements.to_parquet(output_path, compression='snappy')
        elif format == 'json':
            import json
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_database_info(self) -> Dict:
        """
        Get general information about the database
        
        Returns:
            Dictionary with database information
        """
        info = {
            'database_path': str(self.db_path),
            'water_bodies_count': len(self.water_bodies_catalog) if self.water_bodies_catalog is not None else 0,
            'has_catalog': self.water_bodies_catalog is not None,
            'measurements_partitions': []
        }
        
        # Count measurements partitions
        if self.measurements_path.exists():
            for year_dir in self.measurements_path.iterdir():
                if year_dir.is_dir() and year_dir.name.startswith("year="):
                    year = year_dir.name.split("=")[1]
                    for month_dir in year_dir.iterdir():
                        if month_dir.is_dir() and month_dir.name.startswith("month="):
                            month = month_dir.name.split("=")[1]
                            parquet_file = month_dir / "measurements.parquet"
                            if parquet_file.exists():
                                try:
                                    df = pd.read_parquet(parquet_file)
                                    info['measurements_partitions'].append({
                                        'year': int(year),
                                        'month': int(month),
                                        'record_count': len(df),
                                        'unique_water_bodies': df['water_body_id'].nunique()
                                    })
                                except Exception as e:
                                    warnings.warn(f"Error reading partition {year}-{month}: {e}")
        
        # Calculate total measurements
        info['total_measurements'] = sum(p['record_count'] for p in info['measurements_partitions'])
        
        return info
    
    def close(self):
        """Close database connection and clear caches"""
        self._measurements_cache.clear()
        self._full_dataset_cache = None
        print("Database connection closed and caches cleared")

# Utility functions for analysis
def calculate_area_statistics(measurements_df: pd.DataFrame) -> Dict:
    """Calculate comprehensive area statistics"""
    if len(measurements_df) == 0:
        return {}
    
    areas = measurements_df['surface_area']
    
    stats = {
        'count': len(areas),
        'mean': areas.mean(),
        'median': areas.median(),
        'std': areas.std(),
        'min': areas.min(),
        'max': areas.max(),
        'q25': areas.quantile(0.25),
        'q75': areas.quantile(0.75),
        'cv': areas.std() / areas.mean() if areas.mean() > 0 else np.nan,
        'range': areas.max() - areas.min()
    }
    
    return stats

def detect_outliers(measurements_df: pd.DataFrame, 
                   method: str = 'iqr',
                   threshold: float = 1.5) -> pd.DataFrame:
    """Detect outliers in area measurements"""
    if len(measurements_df) == 0:
        return measurements_df
    
    df = measurements_df.copy()
    areas = df['surface_area']
    
    if method == 'iqr':
        q25, q75 = areas.quantile([0.25, 0.75])
        iqr = q75 - q25
        lower_bound = q25 - threshold * iqr
        upper_bound = q75 + threshold * iqr
        df['is_outlier'] = (areas < lower_bound) | (areas > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((areas - areas.mean()) / areas.std())
        df['is_outlier'] = z_scores > threshold
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    return df