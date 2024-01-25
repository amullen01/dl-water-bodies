"""
import json
import geopandas as gpd
import os
from planet import Auth
from planet import data_filter
from datetime import datetime
from planet import Session
import pandas as pd

path_to_api_key = '/explore/nobackup/people/almullen/smallsat_augmentation/code/PL_API_KEY.txt'
path_to_shapefile = '/explore/nobackup/people/almullen/smallsat_augmentation/data/rois/YKD/YKD_ABoVE_tiles_C.shp'
path_to_tile_downloads = '/explore/nobackup/people/almullen/smallsat_augmentation/data/orders/YKD'
#read the geojson as a dictionary to feed into the Planet search
#path_to_geojson = '/explore/nobackup/people/almullen/smallsat_augmentation/data/rois/YKD/YKD_ABoVE_tiles_C.shp'
#with open(path_to_geojson) as json_file:
#    geom = json.load(json_file)

#open the geojson with geopandas and plot to make sure this is the correct region   
shapefile = gpd.read_file(path_to_shapefile)
shapefile = shapefile.to_crs('EPSG:4326')

#tile_id='Ch010v026' #quinhagak
#tile_id= 'Ch009v024' #kwigillingok
tile_id = 'Ch013v023' #nunapitchuk

tile = shapefile.loc[shapefile['grid_id']==tile_id]
tile.plot()

#get json representation of tile
tile_json = json.loads(tile.to_json())
tile_json

#Authenticate Planet
with open(path_to_api_key) as f:
    pl_api_key = f.readline()
    auth = Auth.from_key(pl_api_key)
    auth.store()

#set up API orders data filter
sfilter = data_filter.and_filter([
    data_filter.permission_filter(),
    data_filter.date_range_filter('acquired', gt=datetime(2018, 5, 1, 1), lt=datetime(2023, 11, 1, 1)),
    data_filter.geometry_filter(tile_json),
    data_filter.asset_filter(['ortho_analytic_4b_sr', 'ortho_analytic_8b_sr']),
    data_filter.range_filter('cloud_percent', lte=20),
    data_filter.string_in_filter('instrument', ['PSB.SD', 'PS2.SD'])
])

#Query Planet Catalog
ids = []
async def search():
    async with Session() as sess:
        cl = sess.client('data')
        items = [i async for i in cl.search(['PSScene'], sfilter, limit=0)]
        
        print('{} results'.format(len(items)))
        for item in items:
            ids.append([item['properties']['acquired'], 
                       item['id'], 
                       item['properties']['instrument'], 
                       item['properties']['cloud_percent'], 
                       item['properties']['clear_percent'], 
                       item['properties']['clear_confidence_percent'],
                       {'geometry': item['geometry'], 'properties': {}}])
        
await search()

#Create dataframe with image ids and metadata
df_ids = gpd.GeoDataFrame(ids, columns = ['time', 'id', 'instrument', 'cloud_percent', 'clear_percent', 'clear_confidence_percent', 'geometry'])
df_ids['geometry'] = gpd.GeoDataFrame.from_features(df_ids['geometry'])['geometry']
df_ids = df_ids.set_geometry('geometry')
df_ids.crs='EPSG:4326'

#Filter for growing season
df_ids['time'] = pd.to_datetime(df_ids['time'])
df_ids['year'] = df_ids['time'].dt.year
df_ids['month'] = df_ids['time'].dt.month
df_ids = df_ids.loc[(df_ids['month']>=5) & (df_ids['month']<=10)]

df_ids = df_ids.sort_values(by='time')

df_ids

from planet import order_request

tools = [order_request.harmonize_tool(target_sensor='Sentinel-2'),
           order_request.clip_tool(aoi=tile_json),
           order_request.composite_tool()
          ]

from planet import reporting

for t in df_ids['time'].dt.date.unique():
    
    time = str(t)
    
    if not os.path.exists(os.path.join(path_to_tile_downloads, tile_id)):
        os.mkdir(os.path.join(path_to_tile_downloads, tile_id))
        
    if not os.path.exists(os.path.join(path_to_tile_downloads, tile_id, time)):
        os.mkdir(os.path.join(path_to_tile_downloads, tile_id, time))
    
    df_t = df_ids.loc[df_ids['time'].dt.date==t]

    #composite operation places last image in list on top, so sort first by cloud percent, 
    #descending, then clear confidence percent, ascending
    df_t = df_t.sort_values(by=['cloud_percent', 'clear_confidence_percent'], ascending=[False, True])

    #build request
    request = order_request.build_request(
        name='{}_{}'.format(tile_id, time),
        products=[
            order_request.product(item_ids=df_t['id'].to_list(),
                                            product_bundle='analytic_sr_udm2',
                                            item_type='PSScene')
            ],
        tools=tools
     )

    #place the request with asynchronous function
    async def main():
        async with Session() as sess:
            cl = sess.client('orders')
            with reporting.StateBar(state='creating') as bar:
                #place order
                order = await cl.create_order(request)
                
                with open(os.path.join(path_to_tile_downloads, tile_id, time, 'order_id.txt'), 'w') as file:
                    file.write(order['id'])
                with open(os.path.join(path_to_tile_downloads, tile_id, time, 'image_ids.txt'), 'w') as file:
                    file.write(str(df_t['id'].unique()))
                    
    await main()

"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def getParser():
    """
    Get parser object for main initialization.
    """
    desc = 'Use this to specify the regions and years to download ' + \
        'Landsat ARD. Regions follow the GLAD ARD tile system.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '-i', '--input-tiles', type=str,
        default='/explore/nobackup/projects/ilab/software/ilab-geoproc/' +
        'ilab_geoproc/landsat/Collection2_requests/ABoVE_Tiles_all.csv',
        required=False, dest='input_tiles',
        help='Full file path to a csv containing the tiles to download')

    """
    parser.add_argument(
        '-i', '--input-tiles', type=str,
        default='/explore/nobackup/projects/ilab/software/ilab-geoproc/' +
        'ilab_geoproc/landsat/Collection2_requests/ABoVE_Tiles_all.csv',
        required=False, dest='input_tiles',
        help='Full file path to a csv containing the tiles to download')

    parser.add_argument(
        '-u', '--username', type=str, default='glad', dest='username',
        help='Username for your GLAD account')

    parser.add_argument(
        '-p', '--password', type=str, default='ardpas', dest='password',
        help='Password for your GLAD account')

    parser.add_argument(
        '-o', '--output-path', type=str,
        default='/css/landsat/Collection2/GLAD_ARD/Native_Grid',
        dest='output_path', help='Parent directory for the Landsat ARD')

    parser.add_argument(
        '-s', '--interval-start', type=int, default=392, dest='interval_start',
        help='The first time interval to download')

    parser.add_argument(
        '-e', '--interval-end', type=int, default=1012, dest='interval_end',
        help='The last time interval to download')

    parser.add_argument(
        '-np', '--num-procs',
        type=int, default=cpu_count() * 2, dest='num_procs',
        help='Number of parallel processes')
    """

    return parser.parse_args()


def download_file(download_url: str):
    os.system(download_url)
    return


def main():

    # Process command-line args.
    args = getParser()

    """
    # Arguments
    tile_path = args.input_tiles
    out_path = args.output_path
    int_start = args.interval_start
    int_end = args.interval_end

    # Set logging
    logging.basicConfig(format='%(asctime)s %(message)s', level='INFO')
    timer = time.time()

    logging.info(f'Using {args.input_tiles} as data input.')

    assert os.path.isfile(tile_path), f'CSV file not found: {tile_path}'

    # Read input CSV with labels
    labels = pd.read_csv(tile_path)
    labels = labels['TILE'].values.tolist()
    logging.info(f'{str(len(labels))} tiles to process')

    # Get files required for download
    num = int(int_end - int_start)
    intervals = np.linspace(
        int_start, int_end, num=num, endpoint=True, dtype=int)
    logging.info(
        f'Downloading intervals: {int_start} to {int_end} ({num} total)')

    # List of files to download with their respective url
    download_urls = []

    logging.info('Gathering tile URLs')

    # Iterate over each tile
    for tile in tqdm(labels):

        # set latitude value
        lat = tile[-3:]

        for interval in intervals:

            # output filename
            output = os.path.join(out_path, lat, tile,  f'{interval}.tif')

            # if filename does not exist, start processing
            if not os.path.isfile(output):

                # create possible missing directory
                os.makedirs(os.path.dirname(output), exist_ok=True)

                # store curl command to execute
                download_command = \
                    f'curl -u {args.username}:{args.password} -X GET ' + \
                    'https://glad.umd.edu/dataset/glad_ard2/' + \
                    f'{lat}/{tile}/{interval}.tif -o {output}'
                download_urls.append(download_command)

    logging.info(f'Downloading {len(download_urls)} missing files.')

    # Set pool, start parallel multiprocessing
    p = Pool(processes=args.num_procs)
    p.map(download_file, download_urls)
    p.close()
    p.join()

    logging.info(
        f'Took {(time.time()-timer)/60.0:.2f} min, {args.output_path}.')
    """

# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    sys.exit(main())