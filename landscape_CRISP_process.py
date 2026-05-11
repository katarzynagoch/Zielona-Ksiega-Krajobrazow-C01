# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:06:55 2024

@author: uhljoha
@author: gochkat
"""
import geopandas as gp
import os
import rasterio
from rasterio.mask import mask
from rasterio.io import MemoryFile
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import pandas as pd

# Use Arial font, size 11
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 11
})

landshp = r'O:\03_MISC\2025_built_dynamics\data\landscapes\landscape_database_54009.gpkg'
datadir = r'O:\03_MISC\2025_built_dynamics\results_landscapes_CRISP_v1'
if not os.path.exists(datadir):
    os.makedirs(datadir)

# raster_bu = r'O:\02_REGIONAL\2024_GHSL_for_europe\ver_2\GHS_BUILT_S\100m\GHS_BUILT_S_E2020_EUROPE_R2023A_3035_100_V1_0.tif'
raster_bu = r'O:\01_GLOBAL\2022_POPTrends\BU-Population_grids\v12\BuiltUp_global_Y2020.tif'

rasterize_landscapes=False
calc_stats = False
plot_maps=True
plot_charts = True

# We compute  metrics for each year
years=[2020,2100]
raster_template = raster_bu
    
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
    
if rasterize_landscapes:
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
            output_path = os.path.join(datadir, 'ref-landscapes-2010-01m_54009.tif')
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                dst.write(rasterized, 1)

 
# Compute population, built-up sums for each landscape type
if calc_stats:
    for year in years:
        raster_bu = r'O:\01_GLOBAL\2022_POPTrends\BU-Population_grids\v12\BuiltUp_global_Yxxxx.tif'.replace('xxxx',str(year))
        # raster_vol = r'G:\GHSL_restricted_data\GHSL_PRODUCTS\RELEASE\GHSL_R2023\2_DATA\GHS_BUILT_V_GLOBE_R2023A\GHS_BUILT_V_Exxxx_GLOBE_R2023A_54009_100\V1-0\GHS_BUILT_V_Exxxx_GLOBE_R2023A_54009_100_V1_0.tif'.replace('xxxx',str(year))
        raster_pop = r'O:\01_GLOBAL\2022_POPTrends\BU-Population_grids\v12\Population_global_Yxxxx.tif'.replace('xxxx',str(year))
    
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
                aname = row.nazwa
                landid = row.landid_num
                
                try:
                    buarr = get_subset([row.xmin,row.ymin,row.xmax,row.ymax],raster_bu)  
                    # volarr = get_subset([row.xmin,row.ymin,row.xmax,row.ymax],raster_vol)  
                    poparr = get_subset([row.xmin,row.ymin,row.xmax,row.ymax],raster_pop)  
                    landarr = get_subset([row.xmin,row.ymin,row.xmax,row.ymax],datadir+os.sep+'ref-landscapes-2010-01m_54009.tif')  
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
                    # vol = volarr.astype(float).copy()
                    # vol[landarr!=landid]=np.nan 
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
        
                landdata.append([landid,aname,total_bu,total_pop])
                
                print(year,country,counter,'/',total,counter_overall,'/',total_overall,landid)
                
            landdatadf=pd.DataFrame(landdata) 
            landdatadf.columns=['landid','name','total_bu','total_pop']
            
            landdatadf.to_csv(datadir+os.sep+'landscape_stats_%s_%s.csv' %(country,year),index=False) 
    
    # Concatenate bu and pop stats   
    result_gdf = shp.copy()
    for country,countrydf in shp.groupby('CNTR_CODE'):
        for year in years:
            df = pd.read_csv(datadir+os.sep+'landscape_stats_%s_%s.csv' %(country,year))
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
    result_gdf.to_file(datadir+os.sep+'landscapes_mesoregions_CRISP_54009.gpkg',driver='GPKG')
    
if plot_maps:
    # Load the final database with BU an POP counts per year
    map_gdf = gp.read_file(datadir+os.sep+'landscapes_mesoregions_CRISP_54009.gpkg')
    # Reproject the mad gdf to a Polish projection CRS Polkovo
    map_gdf = map_gdf.to_crs(2180)
    # Group polygons by the landscape polygon ID
    map_agg = (map_gdf.dissolve(
        by="nazwa",
        aggfunc={
            "pop_2020": "sum",
            "pop_2100": "sum",
            "bu_2020": "sum",
            "bu_2100": "sum",
            "pow_km2": "sum"} ))
    # map_agg.to_file(datadir+os.sep+'CRISP_agg.gpkg',driver='GPKG')
    
    # -------------------------------------------------------
    # Calculate % change in built-up area between 1975 and 2020
    # -------------------------------------------------------
    map_agg["bu_change_pct"] = ((map_agg["bu_2100"] - map_agg["bu_2020"]) /
                                map_agg["bu_2020"] * 100)
    
    # Classify into growth categories
    bu_bins = [-float("inf"), 15,17,19,21,23, 25]
    # Create labels for the bins
    bu_labels = ["<15%", "15–17%", "17-19%","19-21%", "21-23%", 
                 "23-25%"]
    
    # Assign categories
    map_agg["bu_class"] = pd.cut(map_agg["bu_change_pct"], bins=bu_bins, labels=bu_labels, right=False)

    
    # -------------------------------------------------------
    # Calculate population density in 1975 and 2020
    # -------------------------------------------------------
    map_agg["pop_density_2020"] = map_agg["pop_2020"] / map_agg["pow_km2"]
    map_agg["pop_density_2100"] = map_agg["pop_2100"] / map_agg["pow_km2"] 
    
    # Calculate % change in population density
    map_agg["pop_density_change_pct"] = ((map_agg["pop_density_2100"] - map_agg["pop_density_2020"]) /
                                         map_agg["pop_density_2020"] * 100)
    
    # Classify into change categories
    # Here we include possible decline category (<0%)
    pop_bins = [-100, -90, -80,-70,-60,-50,-40,-30]
    pop_labels = ["-100% – -90%", "-90% – -80%", "-80% – -70%", "-70% – -60%", "-60% – -50%", "-50% – -40%", "-40% – -30%"]
    # pop_labels = ["< -100%", "-100% - -75%", "-75% - -50%",  "-50% - -25%", "-25% - 0%"]
    map_agg["pop_class"] = pd.cut(map_agg["pop_density_change_pct"], bins=pop_bins, labels=pop_labels)
    
    # -------------------------------------------------------
    # Plot Figure 4.1: Built-up area change
    # -------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
    
    map_agg.plot(column="bu_class", cmap="RdYlBu_r", legend=True, ax=ax[0], edgecolor="black", linewidth=0.1)
    ax[0].set_title("Zmiana powierzchni terenów zabudowanych w latach 2020-2100", fontsize=14)
    ax[0].axis("off")
    
    # -------------------------------------------------------
    # Plot Figure 4.2: Population density change
    # -------------------------------------------------------
    map_agg.plot(column="pop_class", cmap="RdYlBu_r", legend=True, ax=ax[1], edgecolor="black", linewidth=0.1)
    ax[1].set_title("Zmiena gęstości zaludnienia w latach 2020-2100", fontsize=14)
    ax[1].axis("off")
    
    plt.tight_layout()
    # fig.savefig(datadir + os.sep + 'zmiana_BU_POP_2020_2100.png',dpi=300,bbox_inches='tight')
    plt.show()

if plot_charts:
    # Load the final databse as a df
    result_df = gp.read_file(datadir+os.sep+'landscapes_mesoregions_CRISP_54009.gpkg',ignore_geometry=True)
    
    # Filter out uncertain instances
    filter_mask = ~result_df['nazwa'].str.contains('2-2-2 wysoczyzny silnie', na=False)
    result_df = result_df[filter_mask]       
        
    # Fix the cut naming
    fixed_names = {
        '2-1-1': '2-1-1 wysoczyzny słabo rozcięte',
        '2-1-2': '2-1-2 wysoczyzny silnie rozcięte'}
    for landcode in fixed_names.keys():
        name_mask = result_df['nazwa'].str.contains(landcode, na=False)
        result_df.loc[name_mask, 'nazwa'] = fixed_names[landcode]
    
    # Group data by landscape type and aggregate sums
    agg_df = result_df.groupby('nazwa').agg({
        'pop_2100': 'sum',
        'pop_2020': 'sum',
        'bu_2100': 'sum',
        'bu_2020': 'sum',
        'pow_km2': 'sum'
    }).reset_index()
    
    # Calculate percentage changes
    agg_df['pop_change_2020_2100_pct'] = (agg_df['pop_2100'] - agg_df['pop_2020']) / agg_df['pop_2020'] * 100
    agg_df['bu_change_2020_2100_pct'] = (agg_df['bu_2100'] - agg_df['bu_2020']) / agg_df['bu_2020'] * 100
    
    # Calculate population density 2020 (people per km2)
    agg_df['pop_dens_2100_pp_km2'] = agg_df['pop_2100'] / agg_df['pow_km2']
    
    # Prepare table 4.1 (round values for clarity)
    table_4_1 = agg_df[['nazwa', 'pop_2020', 'pop_2100', 'pop_change_2020_2100_pct',
                        'bu_2020', 'bu_2100', 'bu_change_2020_2100_pct', 'pop_dens_2100_pp_km2']].copy()
    
    table_4_1['bu_2020'] = table_4_1['bu_2020'] 
    table_4_1['bu_2100'] = table_4_1['bu_2100'] 
    table_4_1=table_4_1.rename(columns={"bu_2020": "bu_2020_km2", "bu_2100": "bu_2100_km2"})
    
    table_4_1 = table_4_1.round({
        'pop_2020': 0,
        'pop_2100': 0,
        'pop_change_2020_2100_pct': 2,
        'bu_2020_km2': 2,
        'bu_2100_km2': 2,
        'bu_change_2020_2100_pct': 2,
        'pop_dens_2100_pp_km2': 2
    })
    table_4_1.to_clipboard(index=False)
    print(table_4_1)
    
    # --- Wykres 4.1 ---
    
    # Calculate absolute population change (increase or decrease)
    # agg_df['abs_pop_change_1975_2020'] = agg_df['pop_change_1975_2020_pct'].abs()
    
    # Sort the entire DataFrame by absolute population change descending
    plot_df = agg_df.sort_values(by='pop_change_2020_2100_pct', ascending=False)
    
    # Calculate common y-axis limits based on min and max of both series
    ymin = min(plot_df['pop_change_2020_2100_pct'].min(), plot_df['bu_change_2020_2100_pct'].min())
    ymax = max(plot_df['pop_change_2020_2100_pct'].max(), plot_df['bu_change_2020_2100_pct'].max())
    
    # Optionally add some padding
    ymin -= (ymax - ymin) * 0.01
    ymax += (ymax - ymin) * 0.1

    # Plot population and built-up changes side by side
    fig, ax1 = plt.subplots(figsize=(12,6))
    
    bar_width = 0.35
    index = range(len(plot_df))
    
    # Bars for population change (%)
    pop_bars = ax1.bar(index, plot_df['pop_change_2020_2100_pct'], bar_width, label='Zmiana liczby ludności 2020-2100 [%]', color='tab:blue')
    
    # Bars for built-up change (%), shifted right
    ax2 = ax1.twinx()
    bu_bars = ax2.bar([i + bar_width for i in index], plot_df['bu_change_2020_2100_pct'],bar_width,label='Zmiana powierzchni terenów zabudowanych 2020-2100 [%]', color='tab:orange')
    
    # Set same limits on both y-axes
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)
    
    # X-axis labels
    ax1.set_xticks([i + bar_width/2 for i in index])
    ax1.set_xticklabels(plot_df['nazwa'], rotation=45, ha='right')
    
    # Labels and title
    ax1.set_ylabel('Zmiana liczby ludności [%]')
    ax2.set_ylabel('Zmiana powierzchni terenów zabudowanych [%]')
    plt.title('Zmiana powierzchni terenów zabudowanych i liczby ludności w latach 2020-2100 w typach krajobrazu naturalnego')
    
    # Legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

 
    ####### Make line plots
    # Define a dictionary mapping krajobraz_nazwa to colors
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
    
    


  
    
    
    
    