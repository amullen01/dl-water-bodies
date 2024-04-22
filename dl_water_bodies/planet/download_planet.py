import json
import geopandas as gpd
import os
from planet import Session
import planet_utils as pl_utils
import asyncio
from planet.exceptions import ServerError

path_to_tiles = '/explore/nobackup/people/almullen/smallsat_augmentation/data/orders/YKD/' #kwigillingok
path_to_api_key = '/explore/nobackup/people/almullen/smallsat_augmentation/code/PL_API_KEY.txt'


#place the request with asynchronous function
async def download(order_id, download_dir, sess) -> asyncio.coroutine:
        cl = sess.client('orders')
        await cl.download_order(order_id, directory=download_dir)

async def iterate_downloads(path_to_tiles, path_to_api_key) -> asyncio.coroutine:
    auth = pl_utils.authenticate_planet(path_to_api_key) #authorize
    async with Session() as sess:
        for tile in os.listdir(path_to_tiles): #loop through tiles

            for date in os.listdir(os.path.join(path_to_tiles, tile)): #loop through dates

                order_id = os.path.join(path_to_tiles, tile, date, 'order_id.txt')
                download_dir = os.path.join(path_to_tiles, tile, date)

                with open(order_id, 'r') as file:
                    order_id = file.read()

                    if not os.path.isdir(os.path.join(download_dir, order_id)):
                        print(f'downloading: {order_id} to {download_dir}')
                        while True:
                            try:
                                await download(order_id, download_dir, sess)
                                break
                            except ServerError:
                                print('Planet server error, trying again')
                    else:
                        print(f'{order_id} already downloaded, skipping')
                        
asyncio.run(iterate_downloads(path_to_tiles, path_to_api_key))