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
input_landscapes = r"C:\DATA\2025_Landscapes\natural_landscapes_54009.gpkg"
landshp = r'C:\PROCESSING\2025_built_dynamics\data\landscapes\landscape_database_%s_54009.gpkg'%version
datadir = r'C:\PROCESSING\2025_built_dynamics\results_landscapes_%s'%version
if not os.path.exists(datadir):
    os.makedirs(datadir)

raster_bu = r'C:\DATA\GHSL_PRODUCTS\GHSL_R2023\DATA\GHS_BUILT_S_GLOBE_R2023A\GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100_V1_0\GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100_V1_0.tif'

create_polygons = False 
rasterize_landscapes=False

calc_stats = False
calc_agrm = False

plot_maps=False
generate_tables = False
plot_charts = False

plot_charts_types = True
plot_charts_types_agg = True

# We compute landscape metrics for each year
years=np.arange(1975,2021,5)
# years = np.delete(years, np.where(years == 1980)) # Check the 1980 GHSL input data
# We compute agreement between the first and the last observed year
agr_periods = [[1975, 2020]]

raster_template = raster_bu
    
# Define a dictionary mapping nazwa to colors
color_dict = {
    "1-1-1 glacjalne równinne": "#D9E5A5",
    "1-1-2 glacjalne pagórkowate": "#DFE484",
    "1-1-3 wzgórzowe": "#B8CE7B",
    "1-2-1 peryglacjalne równinne": "#E4E0DD",
    "1-2-2 peryglacjalne pagórkowate": "#BFBBB8",
    "1-2-3 peryglacjalne wzgórzowe": "#A3A49F",
    "1-3-1 fluwioglacjalne równinne": "#FDF6CD",
    "1-4-1 eoliczne pagórkowate": "#FAF07F",
    "2-1-1 wysoczyzny słabo rozcięte": "#F8AB65",
    "2-1-2 wysoczyzny silnie rozcięte": "#FD8469",
    "2-2-1 zwartych masywów": "#DF5B96",
    "2-2-2 izolowane połogie": "#F5589F",
    "2-2-3 płaskowyży falistych": "#FFAABF",
    "2-3-1 pogórzy": "#BE9577",
    "2-3-2 pojedycze wzniesienia": "#D0A68D",
    "3-1-1 regiel dolny": "#A88275",
    "3-1-2 regiel gorny": "#808178",
    "3-2 wysokogorskie": "#EAE1C9",
    "4-1-1 zalewowe dna dolin": "#8DBFDD",
    "4-2-1 tarasy nadzalewowe": "#BBE5F3",
    "4-3-0 deltowe akumulacyjne": "#CA9ED1",
    "4-4-0 równiny bagienne": "#AD80B7",
    "4-5-0 obniżeń denudacyjnych": "#DAB6D4"
}

fun_groups = [
    ['A'],['B'],['C'],[ 'D'],['E'],['F'],['G']
]

fun_names = {
    'A': 'A - ośrodki metropolitalne \ni ponadregionalne',
    'B': 'B - ośrodki regionalne',
    'C': 'C - ośrodki subregionalne',
    'D': 'D - ośrodki lokalne',
    'E': 'E - gminy podmiejskie',
    'F': 'F - małomiasteczkowe \ni zurbanizowane',
    'G': 'G - wiejskie'}

def _validate_input(pred: np.ndarray, ref: np.ndarray) -> None:
    """Validate that inputs are NumPy arrays of the same shape."""
    if not isinstance(pred, np.ndarray) or not isinstance(ref, np.ndarray):
        raise TypeError("Both `pred` and `ref` must be NumPy arrays.")
    if pred.shape != ref.shape:
        raise ValueError(f"Arrays must have the same shape. Found {pred.shape} and {ref.shape}.")

def cont_jaccard(pred: np.ndarray, ref: np.ndarray) -> float:
    """Continuous Jaccard Index (NaN-tolerant)."""
    _validate_input(pred, ref)
    numerator = np.nansum(np.minimum(pred, ref))
    denominator = np.nansum(np.maximum(pred, ref))
    return float(numerator / denominator) if denominator != 0 else np.nan

def cont_recall(pred: np.ndarray, ref: np.ndarray) -> float:
    """Continuous Recall (NaN-tolerant)."""
    _validate_input(pred, ref)
    numerator = np.nansum(np.minimum(pred, ref))
    denominator = np.nansum(ref)
    return float(numerator / denominator) if denominator != 0 else np.nan

def cont_precision(pred: np.ndarray, ref: np.ndarray) -> float:
    """Continuous Precision (NaN-tolerant)."""
    _validate_input(pred, ref)
    numerator = np.nansum(np.minimum(pred, ref))
    denominator = np.nansum(pred)
    return float(numerator / denominator) if denominator != 0 else np.nan

def fscore(precision: float, recall: float, beta: float = 1.0) -> float:
    """F-score given precision, recall, and beta (NaN-robust)."""
    if not np.isfinite(precision) or not np.isfinite(recall):
        return np.nan
    if beta <= 0:
        raise ValueError("`beta` must be greater than 0.")
    if precision == 0 and recall == 0:
        return np.nan
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)

def RMSD(pred, ref) -> float:
    """ Root Mean Square Deviation (nan-proof) """
    _validate_input(pred, ref)
    diff_squared = (pred - ref) ** 2
    return np.sqrt(np.nanmean(diff_squared))

def MAD(pred, ref) -> float:
    """ Mean Absolute Deviation (nan-proof) """
    _validate_input(pred, ref)
    abs_diff = np.abs(pred - ref)
    return np.nanmean(abs_diff)

def MD(pred, ref) -> float:
    """ Mean Deviation (nan-proof) """
    _validate_input(pred, ref)
    return np.nanmean(pred) - np.nanmean(ref)

def MAPE(pred, ref) -> float:
    """ Mean Absolute Percentage Error (nan-proof) """
    _validate_input(pred, ref)
    # Only compute where ref > 0 and neither pred nor ref is NaN
    mask = (ref > 0) & (~np.isnan(ref)) & (~np.isnan(pred))
    if not np.any(mask):
        return np.nan  # or raise an error depending on use case
    mape = np.nanmean(np.abs((pred[mask] - ref[mask]) / ref[mask])) * 100
    return mape

def CR(arr1, arr2) -> float:
    """ Change Rate (nan-proof) """
    _validate_input(arr1, arr2)
    sum1 = np.nansum(arr1)
    sum2 = np.nansum(arr2)
    if sum1 == 0:
        return np.nan  # avoid division by zero
    return (sum2 - sum1) / sum1

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
    

if create_polygons:
    # Load landscape data
    landscapes = gp.read_file(input_landscapes)
    landscapes = landscapes.drop(columns=['pow','obwod'])
    
    # Load communes
    communes = gp.read_file(r"C:\DATA\2025_Landscapes\PRG_A03_communes_20250812.gpkg")
    communes = communes[["JPT_KOD_JE", "JPT_NAZWA_", "geometry"]]
    communes["gminy_typ"] = communes["JPT_KOD_JE"].astype(str).str[-1].map({
        "1": "M",
        "2": "W",
        "3": "MW"
    }).fillna("inne")
    communes = communes.rename(columns={
        "JPT_KOD_JE": "PRG_kod",
        "JPT_NAZWA_": "PRG_nazwa"
    })
    
    # Remove last digit from the commune code, indicating the type
    communes["gmina_kod"] = communes['PRG_kod'].str[:-1] 
    
    # Load table with commune classification
    # class_file = r"C:\PROCESSING\2025_built_dynamics\data\communes\UZUPELNIACZ_2025_gminy_utf8.csv"
    class_file = r"C:\DATA\IGIPZ_klasyfikacja_funkcjonalna_gmin\UZUPELNIACZ_2025.xlsx"
    # classes2025 = pd.read_csv(class_file, dtype=object, on_bad_lines='warn',sep=";")  # READ AS CHAR not to loose zeros!!!!!!!
    classes2025 = pd.read_excel(class_file, sheet_name='pelny', dtype=object) 
    classes2025 = classes2025[["ID6_2024","FUN1_2025"]]
    
    # Add classification to gepackage with commune shapes
    communes_classified = communes.merge(classes2025, left_on='gmina_kod', right_on='ID6_2024', how='left').drop(columns='gmina_kod')
              
    # Ensure all are in the same CRS
    landscapes = landscapes.to_crs("ESRI:54009")
    communes_classified = communes_classified.to_crs("ESRI:54009")
    
    # Dissolve polygons by landscape type
    land_dissolved = landscapes.dissolve(by="nazwa")
    land_dissolved['nazwa'] = land_dissolved.index
    
    # Intersect landscapes with communes
    intersect = gp.overlay(land_dissolved, communes_classified, how="intersection")
    intersect = intersect.explode(ignore_index=True)
    
    # Calculate area and diameter
    intersect["pow_km2"] = intersect.geometry.area / 1000000  # from m² to km²
    intersect["obwod_km"] = intersect.geometry.length / 1000   # from m to km
    
    # Save to file
    intersect.to_file(landshp, driver="GPKG")

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
            output_path = os.path.join(datadir, 'ref-landscapes-2010-01m_54009_%s.tif'%version)
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                dst.write(rasterized, 1)
    
       
# Compute population, built-up sums for each landscape type
if calc_stats:
    for year in years:
        raster_bu = r'C:\DATA\GHSL_PRODUCTS\GHSL_R2023\DATA\GHS_BUILT_S_GLOBE_R2023A\GHS_BUILT_S_Exxxx_GLOBE_R2023A_54009_100_V1_0\GHS_BUILT_S_Exxxx_GLOBE_R2023A_54009_100_V1_0.tif'.replace('xxxx',str(year))
        raster_pop = r'C:\DATA\GHSL_PRODUCTS\GHSL_R2023\DATA\GHS_POP_GLOBE_R2023A\GHS_POP_Exxxx_GLOBE_R2023A_54009_100_V1_0\GHS_POP_Exxxx_GLOBE_R2023A_54009_100_V1_0.tif'.replace('xxxx',str(year))
    
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
                    landarr = get_subset([row.xmin,row.ymin,row.xmax,row.ymax],os.path.join(datadir, 'ref-landscapes-2010-01m_54009_%s.tif'%version))  
                except:
                    print('outside of domain')
                    continue
                    # catch error if land areas are outside the raster data domain (eg overseas territories)
                                                  
                try:
                    curr_bu_bb_bin = buarr.copy()
                except:   
                    print('outside of domain')                               
                    continue
                    # catch error if land areas are outside the raster data domain (eg overseas territories)  

                # Compute pop and bu stats
                try: 
                    bu = buarr.astype(float).copy()
                    bu[landarr!=landid]=np.nan 
                    pop = poparr.astype(float).copy()
                    pop[landarr!=landid]=np.nan 
                    
                except:   
                    print('outside of domain')                               
                    continue
                    # catch error if land areas are outside the raster data domain (eg overseas territories)  
                    
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
            
            landdatadf.to_csv(datadir+os.sep+'landscapes_stats_%s_%s_%s.csv' %(country,year, version),index=False) 
    
    # Concatenate bu and pop stats   
    result_gdf = shp.copy()
    for country,countrydf in shp.groupby('CNTR_CODE'):
        for year in years:
            df = pd.read_csv(datadir+os.sep+'landscapes_stats_%s_%s_%s.csv' %(country,year, version))
            # Select only the necessary columns for merge
            df_subset = df[["landid", 'total_bu', "total_pop"]].copy()
            df_subset = df_subset.rename(columns={
                "total_bu": 'bu_%s'%year,
                "total_pop": 'pop_%s'%year,
            })
            
            # Rename 'landid' to match 'result_gdf' key for direct merge, but without preserving it
            df_subset = df_subset.rename(columns={"landid": "landid_num"})
            
            # Merge without bringing in the 'landid' column explicitly
            result_gdf = result_gdf.merge(df_subset, how="left", on="landid_num")
    
    # Save the final database to a GPKG
    result_gdf.to_file(datadir+os.sep+'landscapes_%s_GHSL_54009_%s.gpkg'%(country,version),driver='GPKG')

# Compute agreement between two years in each polygon
if calc_agrm:
    print(' lets calculate agreement!')

    for y, period in enumerate(agr_periods):
        print(period)
        
        for s in ['BUILT_S', 'POP']:
            raster_start = r'C:\DATA\GHSL_PRODUCTS\GHSL_R2023\DATA\GHS_%s_GLOBE_R2023A\GHS_%s_E%s_GLOBE_R2023A_54009_100_V1_0\GHS_%s_E%s_GLOBE_R2023A_54009_100_V1_0.tif'%(s, s, period[0],s, period[0])
            raster_end = r'C:\DATA\GHSL_PRODUCTS\GHSL_R2023\DATA\GHS_%s_GLOBE_R2023A\GHS_%s_E%s_GLOBE_R2023A_54009_100_V1_0\GHS_%s_E%s_GLOBE_R2023A_54009_100_V1_0.tif'%(s, s, period[1],s, period[1])
            
            shp = gp.read_file(landshp,encoding='latin')
            shp['landid_num']=np.arange(1,len(shp)+1)    
            shp[['xmin','ymin','xmax','ymax']]=shp.bounds
            total_overall=len(shp)
            counter_overall=0
            for country,countrydf in shp.groupby('CNTR_CODE'):
                
                processdf = countrydf
                total=len(processdf)   
                agrmdata=[]
                counter=0
                
                for i,row in processdf.iterrows():
                    counter+=1
                    counter_overall+=1
                    aname = row.nazwa
                    
                    try:
                        arr_start = get_subset([row.xmin,row.ymin,row.xmax,row.ymax],raster_start)  
                        arr_end = get_subset([row.xmin,row.ymin,row.xmax,row.ymax],raster_end)  
                        landarr = get_subset([row.xmin,row.ymin,row.xmax,row.ymax],datadir+os.sep+'ref-landscapes-2010-01m_54009_%s.tif'%version)      
                    except:
                        print('outside of domain')
                        continue
                        # catch error if land areas are outside the raster data domain (eg overseas territories)            
                    
                    landid = row.landid_num
                    try:
                        _start = arr_start.copy()         
                        _end = arr_end.copy()
                    except:   
                        print('outside of domain')                               
                        continue
                        # catch error if land areas are outside the raster data domain (eg overseas territories)  
                        
                    _start_masked = _start.astype(float).copy()
                    _start_masked[landarr!=landid]=np.nan 
                    
                    _end_masked = _end.astype(float).copy()
                    _end_masked[landarr!=landid]=np.nan 
                    # plt.imshow(bu_start_masked)
                    # plt.show()         
                    # curr_bu_bb_masked[landarr!=landid]=0 
                    if np.nansum(_start_masked)==0 or np.nansum(_end_masked)==0:
                        print('NBU land?')            
                        continue
                    # calc agreement measures here and store in list of lists
                    try:
                        # Compute agreement metrics
                        cJaccard = cont_jaccard(_start_masked, _end_masked)
                        cPrecision = cont_precision(_end_masked, _start_masked)
                        cRecall = cont_recall(_end_masked, _start_masked)
                        
                        # Compute difference measures
                        _RMSD = RMSD(_end_masked, _start_masked)
                        _MAD = MAD(_end_masked, _start_masked)
                        _MD = MD(_end_masked, _start_masked)
                        difference = np.nansum(_end_masked)-np.nansum(_start_masked)
                        change_rate = CR(_start_masked, _end_masked)
                        
                    except:
                        continue
            
                    agrmdata.append([landid,aname, cJaccard, cPrecision, cRecall, _RMSD, _MAD, _MD, difference, change_rate])
                    
                    print(period,country,counter,'/',total,counter_overall,'/',total_overall,landid)
                    
                agrmdatadf=pd.DataFrame(agrmdata)
                agrmdatadf.columns=['landid','name','cJaccard','cPrecision','cRecall', '_RMSD', '_MAD', '_MD', 'difference', 'change_rate']
                
                agrmdatadf.to_csv(datadir+os.sep+'land_agrm_%s_%s_%s_%s.csv' %(s, country,period[0], period[1]),index=False) 

    print('join agreement measures into one geopackage!')
    for y, period in enumerate(agr_periods):
        print(period)
        for s in ['BUILT_S', 'POP']:

            shp = gp.read_file(input_landscapes,encoding='latin')
            shp.nazwa = shp.nazwa.map(str)
        
            shp['landid_num']=np.arange(1,len(shp)+1)   
            countries=shp.CNTR_CODE.unique()
            alldf = pd.DataFrame()
            for country in countries:
                try:
                    countrydf = pd.read_csv(datadir+os.sep+'land_agrm_%s_%s_%s_%s.csv' %(s, country,period[0], period[1]))
                except:
                    continue
                alldf = pd.concat([alldf,countrydf], ignore_index=True)
                print(country)
                
            shp_joined = shp.merge(alldf,left_on='landid_num',right_on='landid',how='left')
            
            # calc the ratios:
            shp_joined['year_start'] = int(period[0])
            shp_joined['year_end'] = int(period[1])
    
            # shp_joined.crs=None  
            del shp['nazwa']
            shp_joined.to_file(datadir+os.sep+'land_agrm_%s_results_%s_%s_%s.gpkg' %(s, version, period[0], period[1]),driver='GPKG')   
            
if plot_maps:
    # Load the final database with BU an POP counts per year
    map_gdf = gp.read_file(datadir+os.sep+'landscapes_%s_GHSL_54009_%s.gpkg'%(country,version))
    
    # Reproject the mad gdf to a Polish projection CRS Polkovo
    map_gdf = map_gdf.to_crs(2180)
    # Group polygons by the landscape polygon ID
    map_agg = (map_gdf.dissolve(
        by="nazwa",
        aggfunc={
            "pop_1975": "sum",
            "pop_2020": "sum",
            "bu_1975": "sum",
            "bu_2020": "sum",
            "pow_km2": "sum"} ))
    # -------------------------------------------------------
    # Calculate % change in built-up area by polygon between 1975 and 2020
    # -------------------------------------------------------
    map_agg["bu_change_pct"] = ((map_agg["bu_2020"] - map_agg["bu_1975"]) /
                                map_agg["bu_1975"] * 100)
    
    # Classify into growth categories
    bu_bins = [-float("inf"), 75,80,85,90,95,100,105,110,115,120,125,float("inf")]
    # Create labels for the bins
    bu_labels = ["<75%", "75–80%", "80–85%", "85–90%", "90–95%", "95–100%", 
                 "100–105%", "105–110%", "110–115%", "115–120%", "120–125%", 
                 ">125%"]
   
    # Assign categories
    map_agg["bu_class"] = pd.cut(map_agg["bu_change_pct"], bins=bu_bins, labels=bu_labels, right=False)

    
    # -------------------------------------------------------
    # Calculate population density in 1975 and 2020
    # -------------------------------------------------------
    map_agg["pop_density_1975"] = map_agg["pop_1975"] / map_agg["pow_km2"]
    map_agg["pop_density_2020"] = map_agg["pop_2020"] / map_agg["pow_km2"]
    
    # Calculate % change in population density
    map_agg["pop_density_change_pct"] = ((map_agg["pop_density_2020"] - map_agg["pop_density_1975"]) /
                                         map_agg["pop_density_1975"] * 100)
    
    # Classify into change categories
    # Here we include possible decline category (<0%)
    pop_bins = [-float("inf"), -10, -6, -2, 2, 6, 10, 14, 18, 22, 26, 30, float("inf")]
    pop_labels = [
        "< -10%", "-10% – -6%", "-6% – -2%", "-2% – 2%",
        "2% – 6%", "6% – 10%", "10% – 14%", "14% – 18%",
        "18% – 22%", "22% – 26%", "26% – 30%", "> 30%"
    ]

    map_agg["pop_class"] = pd.cut(map_agg["pop_density_change_pct"], bins=pop_bins, labels=pop_labels)
    
    # # ----------------
    # # Check geometries
    # # Check CRS
    # print(map_agg.crs)
    
    # # Check invalid geometries
    # invalid = map_agg[~map_agg.geometry.is_valid]
    # print("Invalid geometries:", len(invalid))
    
    # # Fix invalid
    # map_agg["geometry"] = map_agg.buffer(0)
    # map_agg["geometry"] = map_agg.simplify(50)  # 50 m tolerance

    # bounds = map_agg.total_bounds
    # print(bounds)
    
    # -------------------------------------------------------
    # Plot Figure 4.1: Built-up area change
    # -------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
    
    map_agg.plot(column="bu_class", cmap="RdYlBu_r", legend=True, ax=ax[0], edgecolor="black", linewidth=0.1)
    ax[0].set_title("Zmiana powierzchni terenów zabudowanych w latach 1975–2020", fontsize=14)
    ax[0].axis("off")
    
    # -------------------------------------------------------
    # Plot Figure 4.2: Population density change
    # -------------------------------------------------------
    map_agg.plot(column="pop_class", cmap="RdYlBu_r", legend=True, ax=ax[1], edgecolor="black", linewidth=0.1)
    ax[1].set_title("Zmiana gęstości zaludnienia w latach 1975–2020", fontsize=14)
    ax[1].axis("off")
    
    plt.tight_layout()
    fig.savefig(datadir + os.sep + 'zmiana_BU_POP_1975_2020_pc_map_%s.png'%version,dpi=150,bbox_inches='tight')
    plt.show()

if plot_charts:
    # Load the final database as a df
    result_df = gp.read_file(datadir+os.sep+'landscapes_%s_GHSL_54009_%s.gpkg'%(country,version))   
        
    # Fix the cut naming
    fixed_names = {
        '2-1-1': '2-1-1 wysoczyzny słabo rozcięte',
        '2-1-2': '2-1-2 wysoczyzny silnie rozcięte'}
    for landcode in fixed_names.keys():
        name_mask = result_df['nazwa'].str.contains(landcode, na=False)
        result_df.loc[name_mask, 'nazwa'] = fixed_names[landcode]
    
    # Group data by landscape type and aggregate sums
    agg_dict = {f'pop_{year}': 'sum' for year in range(1975, 2025, 5)}
    agg_dict.update({f'bu_{year}': 'sum' for year in range(1975, 2025, 5)})
    agg_dict['pow_km2'] = 'sum'
    agg_df = result_df.groupby('nazwa').agg(agg_dict).reset_index()
    
    # Calculate percentage changes
    agg_df['pop_change_1975_2020_pct'] = (agg_df['pop_2020'] - agg_df['pop_1975']) / agg_df['pop_1975'] * 100
    agg_df['bu_change_1975_2020_pct'] = (agg_df['bu_2020'] - agg_df['bu_1975']) / agg_df['bu_1975'] * 100
    
    # Calculate population density 2020 (people per km2)
    agg_df['pop_dens_2020_pp_km2'] = agg_df['pop_2020'] / agg_df['pow_km2']
    agg_df['pop_dens_2020_pp_km2']  = agg_df['pop_dens_2020_pp_km2'].round(0).astype(int)
    
    # Prepare table 4.1 (round values for clarity)
    table_4_1 = agg_df[['nazwa', 'pop_1975', 'pop_2020', 'pop_change_1975_2020_pct',
                        'bu_1975', 'bu_2020', 'bu_change_1975_2020_pct', 'pop_dens_2020_pp_km2']].copy()
    
    table_4_1['bu_1975'] = table_4_1['bu_1975'] / 1000  # convert from m to km
    table_4_1['bu_2020'] = table_4_1['bu_2020'] / 1000  # similarly
    table_4_1=table_4_1.rename(columns={"bu_1975": "bu_1975_km2","bu_2020": "bu_2020_km2"})
    
    table_4_1 = table_4_1.round({
        'pop_1975': 0,
        'pop_2020': 0,
        'pop_change_1975_2020_pct': 2,
        'bu_1975_km2': 2,
        'bu_2020_km2': 2,
        'bu_change_1975_2020_pct': 2,
        'pop_dens_2020_pp_km2': 2
    })
    table_4_1.to_csv(datadir + os.sep + 'BU_POP_change_1975-2020_%s.csv'%version, index=False)
    table_4_1.to_clipboard(index=False)
    print(table_4_1)
    
    # --- Wykres 4.1 ---   
    # Sort the entire DataFrame by absolute population change descending
    plot_df = agg_df.sort_values(by='pop_change_1975_2020_pct', ascending=False)
    
    # Calculate common y-axis limits based on min and max of both series
    ymin = min(plot_df['pop_change_1975_2020_pct'].min(), plot_df['bu_change_1975_2020_pct'].min())
    ymax = max(plot_df['pop_change_1975_2020_pct'].max(), plot_df['bu_change_1975_2020_pct'].max())
    
    # Optionally add some padding
    ymin -= (ymax - ymin) * 0.01
    ymax += (ymax - ymin) * 0.1

    # Plot population and built-up changes side by side
    fig, ax1 = plt.subplots(figsize=(12,6))
    
    bar_width = 0.35
    index = range(len(plot_df))
    
    # Bars for population change (%)
    pop_bars = ax1.bar(index, plot_df['pop_change_1975_2020_pct'], bar_width, label='Zmiana liczby ludności 1975–2020 [%]', color='tab:blue')
    
    # Bars for built-up change (%), shifted right
    ax2 = ax1.twinx()
    bu_bars = ax2.bar([i + bar_width for i in index], plot_df['bu_change_1975_2020_pct'],bar_width,label='Zmiana powierzchni terenów zabudowanych 1975–2020 [%]', color='tab:orange')
    
    # Set same limits on both y-axes
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)
    
    # X-axis labels
    ax1.set_xticks([i + bar_width/2 for i in index])
    ax1.set_xticklabels(plot_df['nazwa'], rotation=45, ha='right')
    
    # Labels and title
    ax1.set_ylabel('Zmiana liczby ludności [%]')
    ax2.set_ylabel('Zmiana powierzchni terenów zabudowanych [%]')
    plt.title('Zmiana powierzchni terenów zabudowanych i liczby ludności w latach 1975-2020 w typach krajobrazu naturalnego')
    
    # Legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    fig.savefig(datadir + os.sep + 'zmiana_BU_POP_1975_2020_pc_bars_%s.png'%version,dpi=300,bbox_inches='tight')
    plt.show()

    ####### Make line plots   
    # Columns for population and built-up area
    pop_cols = [f"pop_{year}" for year in range(1975, 2025, 5)]
    bu_cols = [f"bu_{year}" for year in range(1975, 2025, 5)]
    
    # Compute relative values and final pop relative in 2020 for sorting
    plot_data = []
    for krajobraz in agg_df['nazwa']:
        pop_values = agg_df.loc[agg_df['nazwa'] == krajobraz, pop_cols].values.flatten()
        pop_relative = pop_values / pop_values[0] * 100
        
        bu_values = agg_df.loc[agg_df['nazwa'] == krajobraz, bu_cols].values.flatten()
        bu_relative = bu_values / bu_values[0] * 100
        
        plot_data.append({
            "krajobraz": krajobraz,
            "pop_relative": pop_relative,
            "bu_relative": bu_relative,
            "pop_2020": pop_relative[-1]  # for sorting
        })
    
    # Sort by population in 2020 relative to 1975
    plot_data_sorted = sorted(plot_data, key=lambda x: x['pop_2020'], reverse=True)
    
    # Create figure with two subplots
    fig, (ax_bu, ax_pop) = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    
    years = range(1975, 2025, 5)
    
    # Plot lines in sorted order
    for data in plot_data_sorted:
        color = color_dict.get(data['krajobraz'], "#000000")  # default black if not in dict
        ax_pop.plot(years, data['pop_relative'], label=data['krajobraz'], color=color, lw=2)
        ax_bu.plot(years, data['bu_relative'], label=data['krajobraz'], color=color, lw=2)
    
    # Titles and labels
    ax_pop.set_title("Ludność względem 1975")
    ax_pop.set_ylabel("Względna liczba ludności (1975 = 100)")
    ax_pop.set_xlabel("Rok")
    ax_pop.set_xticks(years)
    
    ax_bu.set_title("Powierzchnia terenów zabudowanych względem 1975")
    ax_bu.set_ylabel("Powierzchnia terenów zabudowanych (1975 = 100)")
    ax_bu.set_xlabel("Rok")
    
    # Legend only on the right subplot
    ax_bu.legend().set_visible(False)
    ax_pop.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    fig.savefig(datadir + os.sep + 'zmiana_BU_POP_1975_2020_relative_lineplots_%s.png'%version,dpi=300,bbox_inches='tight')
    plt.show()

if plot_charts_types:
    # Load input
    result_all = gp.read_file(datadir + os.sep + f'landscapes_{country}_GHSL_54009_{version}.gpkg')

    # Fix naming
    fixed_names = {
        '2-1-1': '2-1-1 wysoczyzny słabo rozcięte',
        '2-1-2': '2-1-2 wysoczyzny silnie rozcięte'
    }
    for code, newname in fixed_names.items():
        mask = result_all['nazwa'].str.contains(code, na=False)
        result_all.loc[mask, 'nazwa'] = newname

    # 7 functional groups → 2×4 plot (last axis empty)
    fig, axes = plt.subplots(
        nrows=2, ncols=4,
        figsize=(12, 6),
        sharex=True,
        sharey=True
    )
    
    years = list(range(1975, 2025, 5))
    xticks_15y = [1975, 1990, 2005, 2020]
    
    legend_handles = []
    legend_labels = []
    
    for idx, fun_group in enumerate(fun_groups):
    
        ax = axes[idx // 4, idx % 4]
    
        result_df = result_all[result_all.FUN1_2025.str[0].isin(fun_group)]
    
        agg_dict = {f'pop_{y}': 'sum' for y in years}
        agg_df = result_df.groupby('nazwa').agg(agg_dict).reset_index()
    
        plot_data = []
        for krajobraz in agg_df['nazwa']:
            row = agg_df[agg_df['nazwa'] == krajobraz]
            pop_vals = row[[f'pop_{y}' for y in years]].values.flatten()
    
            if pop_vals[0] == 0 or row.pop_2020.iloc[0] < 1000:
                continue
    
            pop_rel = pop_vals / pop_vals[0] * 100
    
            plot_data.append({
                "krajobraz": krajobraz,
                "pop_rel": pop_rel,
                "pop_2020_rel": pop_rel[-1],
            })
    
        plot_data_sorted = sorted(
            plot_data, key=lambda x: x["pop_2020_rel"], reverse=True
        )
    
        for data in plot_data_sorted:
            color = color_dict.get(data["krajobraz"], "#000000")
    
            line, = ax.plot(
                years, data["pop_rel"],
                color=color, lw=2, label=data["krajobraz"]
            )
    
            if data["krajobraz"] not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(data["krajobraz"])
    
        group_label = ", ".join([fun_names[x] for x in fun_group])
        ax.set_title(group_label)
        ax.set_xticks(xticks_15y)
        ax.grid(True, linestyle="--", alpha=0.4)
    
    # 🔹 Wyłącz prawy dolny subplot
    axes[1, 3].set_visible(False)
    
    # 🔹 Jedna etykieta osi
    fig.supxlabel("Rok", y=0.08)
    fig.supylabel("Ludność względna (1975 = 100)", x=0.08)
    
    # 🔹 Legenda po prawej, 1 kolumna
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        title="Typy krajobrazów",
        loc="center right",
        bbox_to_anchor=(1.12, 0.5),
        ncol=1
    )
    
    plt.tight_layout(rect=[0.06, 0.06, 0.88, 1])
    
    fig.savefig(
        datadir + os.sep + f"relative_population_by_functional_groups_{version}.png",
        dpi=300,
        bbox_inches="tight"
    )
    
    plt.show()


###########################################

# Mapowanie pierwszej cyfry kodu Rychlinga -> klasy krajobrazu
klasy_krajobraz = {
    "1": "Krajobrazy nizin",
    "2": "Krajobrazy wyżyn i niskich gór",
    "3": "Krajobrazy gór średnich i wysokich",
    "4": "Krajobrazy dolin i obniżeń"
}

# Kolory 4 klas (delikatne, ale różne)
kolory_klas = {
    "1": "#4daf4a",    # zielony
    "2": "#ff7f00",    # pomarańczowy
    "3": "brown",#"#984ea3",    # fioletowy
    "4": "#984ea3"#"#e41a1c"     # czerwony
}

if plot_charts_types_agg:

    result_all = gp.read_file(
        datadir + os.sep + f'landscapes_{country}_GHSL_54009_{version}.gpkg'
    )

    # SAME FIXES AS BEFORE
    fixed_names = {
        '2-1-1': '2-1-1 wysoczyzny słabo rozcięte',
        '2-1-2': '2-1-2 wysoczyzny silnie rozcięte'
    }
    for code, newname in fixed_names.items():
        mask = result_all['nazwa'].str.contains(code, na=False)
        result_all.loc[mask, 'nazwa'] = newname

    # Extract Rychling class = first digit
    result_all["klasa"] = result_all["nazwa"].str[0]

    # ---- FIGURE: 2×4 (last empty) ----
    fig, axes = plt.subplots(
        nrows=2, ncols=4,
        figsize=(14, 6),
        sharex=True,
        sharey=True
    )

    years = list(range(1975, 2025, 5))

    legend_handles = []
    legend_labels = []

    for idx, fun_group in enumerate(fun_groups):

        ax = axes[idx // 4, idx % 4]

        # Only municipalities in this functional group
        result_df = result_all[
            result_all.FUN1_2025.str[0].isin(fun_group)
        ]

        # Aggregate population by landscape CLASS
        agg_dict = {f'pop_{y}': 'sum' for y in years}
        agg_df = result_df.groupby('klasa').agg(agg_dict).reset_index()

        for klasa in ["1", "2", "3", "4"]:

            row = agg_df[agg_df["klasa"] == klasa]
            if row.empty:
                continue

            pop_vals = row[[f'pop_{y}' for y in years]].values.flatten()

            if pop_vals[0] == 0:
                continue
            if row.pop_2020.iloc[0] < 1000:
                continue

            pop_rel = pop_vals / pop_vals[0] * 100

            line, = ax.plot(
                years,
                pop_rel,
                lw=2,
                color=kolory_klas[klasa],
                label=klasy_krajobraz[klasa]
            )

            if klasy_krajobraz[klasa] not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(klasy_krajobraz[klasa])

        group_label = ", ".join([fun_names[x] for x in fun_group])
        ax.set_title(group_label)

        ax.grid(True, linestyle="--", alpha=0.4)
        
        ax.set_ylim([80,160])
        ax.set_xticks([1975, 1990, 2005, 2020])

    # ---- Hide empty (8th) subplot ----
    # axes[1, 3].set_visible(False)
    
    # ---- Common labels ----
    fig.supxlabel("Rok", y=0.08)
    fig.supylabel("Ludność względna (1975 = 100)", x=0.08)
    
    plt.tight_layout(rect=[0.06, 0.06, 0.88, 1])
    
    # ---- Legend in bottom-right axis ----
    legend_ax = axes[1, 3]
    legend_ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        title="Klasy krajobrazów",
        loc="center",
        frameon=False
    )
    legend_ax.axis("off")

    fig.savefig(
        datadir + os.sep +
        f"relative_population_by_functional_groups_CLASSES_{version}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()



     
if generate_tables:
    # Load the final databse as a df
    result_df = gp.read_file(datadir+os.sep+'landscapes_%s_GHSL_54009_%s.gpkg'%(country,version),
                             ignore_geometry=True)
    # Fix the cut naming
    fixed_names = {
        '2-1-1': '2-1-1 wysoczyzny słabo rozcięte',
        '2-1-2': '2-1-2 wysoczyzny silnie rozcięte'}
    for landcode in fixed_names.keys():
        name_mask = result_df['nazwa'].str.contains(landcode, na=False)
        result_df.loc[name_mask, 'nazwa'] = fixed_names[landcode]
    
    # Find all pop_ and bu_ columns
    pop_cols = [c for c in result_df.columns if c.startswith("pop_")]
    bu_cols = [c for c in result_df.columns if c.startswith("bu_")]
    
    # Define the aggregation dict
    aggfunc = {col: "sum" for col in pop_cols + bu_cols}
    
    # Add surface function to the dict
    aggfunc["pow_km2"] = "sum"
    
    # Group by landscape name
    shp_grouped = result_df.groupby("nazwa", as_index=False).agg(aggfunc)
    shp_grouped["pow_km2"] = shp_grouped["pow_km2"].round(2)
    
    # Create population summary
    pop_table = shp_grouped[['nazwa', 'pow_km2'] + pop_cols].copy()
    
    # Round all population columns to 0 decimals (whole numbers)
    pop_table[pop_cols] = pop_table[pop_cols].round(0).astype(int)

    print(pop_table)
    pop_table.to_csv(datadir + os.sep + 'POP_1975-2020_%s.csv'%version, index=False)
    
    # Create built-up area summary
    bu_table = shp_grouped[['nazwa', 'pow_km2'] + bu_cols].copy()
    
    # Round all bu columns to km2
    bu_table[bu_cols] = bu_table[bu_cols]/1000000
    bu_table[bu_cols] = bu_table[bu_cols].round(2)
    col_dict = dict(zip(bu_cols, [b+'_km2' for b in bu_cols]))
    bu_table = bu_table.rename(columns=col_dict)

    print(bu_table)
    bu_table.to_csv(datadir + os.sep + 'BU_1975-2020_%s.csv'%version, index=False)
    
    # --- Step 1: Select only necessary columns ---
    cols = ["nazwa", "FUN1_2025", "pop_1975", "pop_2020"]
    df = result_df[cols].copy()
    
    # --- Step 2: Aggregate by landscape type and functional type ---
    
    summary = df.groupby(["nazwa", "FUN1_2025"]).agg(
        pop_1975_total=("pop_1975", "sum"),
        pop_2020_total=("pop_2020", "sum")
    ).reset_index()
    
    # --- Step 3: Compute relative change (%) ---
    summary["pop_change_rel"] = (summary["pop_2020_total"] - summary["pop_1975_total"]) / summary["pop_1975_total"] * 100
    
    # --- Step 4: Pivot table so that columns are functional types ---
    summary["FUN_short"] = summary["FUN1_2025"].str[0]
    pop2020_pivot = summary.pivot(index="nazwa", columns="FUN_short", values="pop_2020_total")
    relchange_pivot = summary.pivot(index="nazwa", columns="FUN_short", values="pop_change_rel")
    
    # --- Step 5: Combine into a single table with multi-level columns ---
    final_table = pd.concat([pop2020_pivot, relchange_pivot], axis=1, keys=["pop_2020", "rel_change"])
    final_table = final_table.sort_index(axis=1, level=1)  # sort functional types alphabetically
    
    # --- Step 8: Format values ---
    # pop_2020 → integer
    final_table["pop_2020"] = final_table["pop_2020"].fillna(0).astype(int)
    
    # rel_change → float with 2 decimals, NaN replaced with 0
    final_table["rel_change"] = final_table["rel_change"].fillna(0).round(2)
    
    # --- Step 9: Display ---
    final_table.head()
    final_table.to_csv(datadir + os.sep + 'POP_FUN_1975-2020_%s.csv'%version, index=True)    

    
    