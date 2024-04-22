from planet import Auth
from planet import order_request

def authenticate_planet(path_to_api_key):
    with open(path_to_api_key) as f:
        #Authenticate Planet
        pl_api_key = f.readline()
        auth = Auth.from_key(pl_api_key)
        auth.store()
        return auth

def get_ortho_tile_tools(tile_json):
    tools = [order_request.harmonize_tool(target_sensor='Sentinel-2'),
           order_request.clip_tool(aoi=tile_json),
           order_request.composite_tool()
          ]
    return tools