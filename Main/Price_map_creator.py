# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 03:35:13 2018

Heat map creator

@Author: PouyaRZ
"""

import numpy as np
import geopandas as gp
#import geoplot
import pandas as pd
from shapely.geometry import Point
import matplotlib.pyplot as plt

geojs = gp.read_file('../Data/neighbourhoods.geojson')

df = pd.read_csv('../Data/data_cleaned.csv')
df = df[['latitude','longitude','price']]

df['coords'] = list(zip(df.longitude, df.latitude))
df['coords'] = df['coords'].apply(Point)
df['price'] = np.exp(df['price'])
df = df[df['price']<=200]

#df.drop('longitude','latitude')


gdf = gp.GeoDataFrame(df, geometry='coords')

base = geojs.plot(color='white', edgecolor='black', linewidth = 1, figsize=(10,10))

gdf.plot(ax=base, marker='o', column='price', markersize=1, legend=True)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('NYC Airbnb Data Price Range')

plt.savefig('Price_map.svg', bbox_inches='tight')