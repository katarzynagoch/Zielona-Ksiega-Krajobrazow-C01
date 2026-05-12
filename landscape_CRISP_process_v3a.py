# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:06:55 2024

@author: Katarzyna Krasnodębska
"""
import geopandas as gp
import os
import rasterio
from rasterio.mask import mask
from rasterio.io import MemoryFile
import matplotlib.patches as mpatches
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from rasterio.warp import calculate_default_transform, reproject, Resampling
import csv
import scipy.stats
import pylandstats
import seaborn as sns

# Use Arial font, size 11
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 11
})
country='PL'
version = 'v3a'

landshp = r'C:\PROCESSING\2026_Ksiega_Krajobrazow\data\landscapes\landscape_database_%s_54009.gpkg'%version
datadir = r'C:\PROCESSING\2026_Ksiega_Krajobrazow\results_landscapes_%s'%version
if not os.path.exists(datadir):
    os.makedirs(datadir)

raster_bu = r'C:\DATA\GHSL_PRODUCTS\GHSL_R2025A\GHS_WUP_BUILT_S_E2020_GLOBE_R2025A_54009_1000_V1_0\GHS_WUP_BUILT_S_E2020_GLOBE_R2025A_54009_1000_V1_0.tif'

rasterize_landscapes=False
calc_stats = True

# We compute landscape metrics for each year
years=[2020,2100]
raster_template = raster_bu
    
def get_subset(bbox,currfile):
    # source: https://riptutorial.com/gdal/example/25844/read-subset-of-a-global-raster-defined-by-a-bounding-box
    ds = gdal.Open(currfile, gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    gt = ds.GetGeoTransform()
    # The inverse geotransform is used to convert lon/lat degrees to x/y pixel index
    inv_geotransform = gdal.InvGeoTransform(gt)            
    # Convert lon/lat degrees to x/y pixel for the dataset
    _x0, _y0 = gdal.ApplyGeoTransform(
        inv_geotransform, bbox[0], bbox[1])
    _x1, _y1 = gdal.ApplyGeoTransform(
        inv_geotransform, bbox[2], bbox[3])
    x0, y0 = min(_x0, _x1), min(_y0, _y1)
    x1, y1 = max(_x0, _x1), max(_y0, _y1)
    # Get subset of the raster as a numpy array
    data = band.ReadAsArray(int(x0), int(y0), int(x1-x0), int(y1-y0))
    nodataval = band.GetNoDataValue()
    data[data==nodataval]=0
    return data
    ds = None

if rasterize_landscapes:
    print('rasterize input polygons and generate landid identifier')
    # Load shapefile
    shp = gp.read_file(landshp)
    shp['landid_num'] = np.arange(1, len(shp) + 1)
    shapes = ((geom, value) for geom, value in zip(shp.geometry, shp.landid_num))
    
    # Reproject shapefile to match raster CRS
    with rasterio.open(raster_template) as src:
        shp = shp.to_crs(src.crs)
        
        # Mask (crop) the raster template with shapefile geometry
        out_image, out_transform = mask(src, shp.geometry, crop=True)
        out_meta = src.meta.copy()
    
    # Update metadata for the cropped raster
    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })
    
    # Use MemoryFile to temporarily hold the cropped raster
    with MemoryFile() as memfile:
        with memfile.open(**out_meta) as temp_raster:
            # The clipped template in memory is now the template raster
            # Rasterize your vector data to match this clipped template
            rasterized = rasterio.features.rasterize(
                shapes=shapes,
                out_shape=(temp_raster.height, temp_raster.width),
                fill=0,
                transform=temp_raster.transform,
                all_touched=True,
                default_value=1,
                dtype=np.uint32
            )
    
            # Update metadata for output
            kwargs = out_meta.copy()
            kwargs.update({
                'dtype': 'uint32',
                'count': 1,
                'nodata': 0,
                'compress': 'lzw'
            })
    
            # Save to file
            output_path = os.path.join(datadir, 'ref-landscapes-CRISP-2010-01m_54009_%s.tif'%version)
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                dst.write(rasterized, 1)
        
      
# Compute population, built-up sums for each landscape type
if calc_stats:
    for year in years:
        raster_bu = r"C:\DATA\GHSL_PRODUCTS\GHSL_R2025A\GHS_WUP_BUILT_S_EXXXX_GLOBE_R2025A_54009_1000_V1_0\GHS_WUP_BUILT_S_EXXXX_GLOBE_R2025A_54009_1000_V1_0.tif".replace('XXXX',str(year))
        raster_pop = r"C:\DATA\GHSL_PRODUCTS\GHSL_R2025A\GHS_WUP_POP_EXXXX_GLOBE_R2025A_54009_1000_V1_0\GHS_WUP_POP_EXXXX_GLOBE_R2025A_54009_1000_V1_0.tif".replace('XXXX',str(year))
    
        shp = gp.read_file(landshp).to_crs("ESRI:54009")
        shp['landid_num']=np.arange(1,len(shp)+1)    
        shp[['xmin','ymin','xmax','ymax']]=shp.bounds
        total_overall=len(shp)
        counter_overall=0
        
        for country,countrydf in shp.groupby('CNTR_CODE'):
            
            processdf = countrydf
            total=len(processdf)   
            landdata=[]
            counter=0
                    
            for i,row in processdf.iterrows():
                counter+=1
                counter_overall+=1
                typ_krajobrazu = row.nazwa
                landid = row.landid_num
                funkcja_gminy = row.FUN1_2025
                nazwa_gminy = row.PRG_nazwa
                gmina_id = row.ID6_2024
                gmina_typ = row.gminy_typ
                pow_km2 = row.pow_km2
                obwod_km = row.obwod_km
                
                try:
                    buarr = get_subset([row.xmin,row.ymin,row.xmax,row.ymax],raster_bu)   
                    poparr = get_subset([row.xmin,row.ymin,row.xmax,row.ymax],raster_pop)  
                    landarr = get_subset([row.xmin,row.ymin,row.xmax,row.ymax],os.path.join(datadir, 'ref-landscapes-CRISP-2010-01m_54009_%s.tif'%version))  
                except:
                    print('outside of domain - subset')
                    continue
                    # catch error if land areas are outside the raster data domain (eg overseas territories)
                                                  
                try:
                    curr_bu_bb_bin = buarr.copy()
                except:   
                    print('outside of domain - bu copy')                               
                    continue
                    # catch error if land areas are outside the raster data domain (eg overseas territories)  

                # Compute pop and bu stats
                # try: 
                bu = buarr.astype(float).copy()
                bu[landarr!=landid]=np.nan 
                pop = poparr.astype(float).copy()
                pop[landarr!=landid]=np.nan 
                    
                # except:   
                #     print('outside of domain - bu pop select')                               
                #     continue
                #     # catch error if land areas are outside the raster data domain (eg overseas territories)  
                    
                # plt.imshow(bu)
                # plt.show()         
                
                if np.nansum(bu)==0:
                    print('NBU land?')            
                    continue
                      
                total_bu = np.nansum(bu)
                # total_vol = np.nansum(vol)
                total_pop = np.nansum(pop)       
        
                landdata.append([landid,typ_krajobrazu,
                                 funkcja_gminy,
                                 nazwa_gminy,
                                 gmina_id,
                                 gmina_typ,
                                 pow_km2,
                                 obwod_km,
                                 total_bu,total_pop])
                
                print(year,country,counter,'/',total,counter_overall,'/',total_overall,landid)
                
            landdatadf=pd.DataFrame(landdata) 
            landdatadf.columns=['landid','typ_krajobrazu',
                                'funkcja_gminy',
                                'nazwa_gminy',
                                'gmina_id',
                                'gmina_typ',
                                'pow_km2',
                                'obwod_km',
                                'total_bu','total_pop']
            
            landdatadf.to_csv(datadir+os.sep+'landscapes_CRISP_stats_%s_%s_%s.csv' %(country,year, version),index=False) 
    
    # Concatenate bu and pop stats   
    result_gdf = shp.copy()
    for country,countrydf in shp.groupby('CNTR_CODE'):
        for year in years:
            df = pd.read_csv(datadir+os.sep+'landscapes_CRISP_stats_%s_%s_%s.csv' %(country,year, version))
            # Select only the necessary columns for merge
            df_subset = df[["landid", 'total_bu', "total_pop"]].copy()
            df_subset = df_subset.rename(columns={
                "total_bu": 'bu_%s'%year,
                "total_pop": 'pop_%s'%year,
            })
            
            # Rename 'landid' to match 'result_gdf' key 
            df_subset = df_subset.rename(columns={"landid": "landid_num"})
            
            # Merge without bringing in the 'landid' column explicitly
            result_gdf = result_gdf.merge(df_subset, how="left", on="landid_num")
    
    # Save the final database to a GPKG
    result_gdf.to_file(datadir+os.sep+'landscapes_CRISP_%s_GHSL_54009_%s.gpkg'%(country,version),driver='GPKG')

            
