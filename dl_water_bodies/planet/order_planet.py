import json
import geopandas as gpd
import os
from planet import Auth
from planet import data_filter
from datetime import datetime
from planet import Session
import pandas as pd
import planet_utils as pl_utils
import asyncio
from planet import order_request
from planet import reporting

#tile_id='Ch010v026' #quinhagak
#tile_id= 'Ch009v024' #kwigillingok
#tile_id = 'Ch013v023' #nunapitchuk

def get_json_from_shp(path_to_shapefile, id_str, id_attr='grid_id'):
    
    #open the geojson with geopandas and plot to make sure this is the correct region   
    shapefile = gpd.read_file(path_to_shapefile)
    shapefile = shapefile.to_crs('EPSG:4326')
    
    #get json representation from tile based on id col
    tile = shapefile.loc[shapefile[id_attr]==id_str]
    tile_json = json.loads(tile.to_json())

    return tile_json
    
def pl_order_analytic_sr(tile_json, yr_min=2018, month_min=5, day_min=1, yr_max=2023, month_max=11, day_max=1, cloud_max_pct=20):
    #set up API orders data filter
    
    sfilter = data_filter.and_filter([
        data_filter.permission_filter(),
        data_filter.date_range_filter('acquired', gt=datetime(yr_min, month_min, day_min, 1), lt=datetime(yr_max, month_max, day_max, 1)),
        data_filter.geometry_filter(tile_json),
        data_filter.asset_filter(['ortho_analytic_4b_sr', 'ortho_analytic_8b_sr']),
        data_filter.range_filter('cloud_percent', lte=cloud_max_pct),
        data_filter.string_in_filter('instrument', ['PSB.SD', 'PS2.SD'])
    ])
    
    return sfilter

async def search(sfilter, sess) -> asyncio.coroutine:
    ids = []
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
    return ids

async def search_catalog(sfilter, sess) -> asyncio.coroutine:
    #Query Planet Catalog
    ids = await search(sfilter, sess)
    
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
    
    return df_ids

async def place_order(request, path_to_order, image_ids, sess) -> asyncio.coroutine:
                cl = sess.client('orders')
                with reporting.StateBar(state='creating') as bar:
                    #place order
                    order = await cl.create_order(request)

                    with open(os.path.join(path_to_order, 'order_id.txt'), 'w') as file:
                        file.write(order['id'])
                    with open(os.path.join(path_to_order, 'image_ids.txt'), 'w') as file:
                        file.write(str(image_ids))

path_to_api_key = '/explore/nobackup/people/almullen/smallsat_augmentation/code/PL_API_KEY.txt'
path_to_shapefile = '/explore/nobackup/people/almullen/smallsat_augmentation/data/rois/YKD/YKD_ABoVE_tiles_C.shp'
path_to_tile_downloads = '/explore/nobackup/people/almullen/smallsat_augmentation/data/orders/YKD'    
grid_id_list=['Ch015v018']

async def iterate_shapefile_order(path_to_api_key, path_to_shapefile, path_to_tile_downloads, grid_id_list=None) -> asyncio.coroutine:
    
    shapefile = gpd.read_file(path_to_shapefile)
    auth = pl_utils.authenticate_planet(path_to_api_key) #authorize
    i=0
    async with Session() as sess:
        for index, row in shapefile.iterrows():
            print('----------')
            tile_id = row['grid_id']
            #get ch <=20 first
            print(tile_id)

            #if tile_id is not None and not tile_id in grid_id_list:
            #    continue

            if tile_id in os.listdir(path_to_tile_downloads) or row['ch']>=20:
                print('skipping')
                continue
            
            i = i+1
            if i>200:
                break

            tile_json = get_json_from_shp(path_to_shapefile, tile_id) #convert single tile to json format
            sfilter = pl_order_analytic_sr(tile_json) #build search filter
            df_ids = await search_catalog(sfilter, sess) #get image ids from Planet catalog
            print(str(len(df_ids)) + ' images to order')
            tools = pl_utils.get_ortho_tile_tools(tile_json) #get tools

            #set path to tile, create dir if not present
            path_to_tile = os.path.join(path_to_tile_downloads, tile_id)
            if not os.path.exists(path_to_tile):
                os.mkdir(path_to_tile)

            #iterate dates of available planet imagery
            for t in df_ids['time'].dt.date.unique():

                time = str(t)

                #set path to order, create dir if not present
                path_to_order = os.path.join(path_to_tile_downloads, tile_id, time)
                if not os.path.exists(path_to_order):
                    os.mkdir(path_to_order)
                else:
                    print('date already downloaded')
                    continue

                #filter planet images for single day
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

                #place the order
                await place_order(request, path_to_order, df_t['id'].unique(), sess)

            
asyncio.run(iterate_shapefile_order(path_to_api_key, path_to_shapefile, path_to_tile_downloads, grid_id_list=grid_id_list))
