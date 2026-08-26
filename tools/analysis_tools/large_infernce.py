import numpy as np
import rasterio
from rasterio.features import shapes
from mmseg.apis import init_model, inference_model
import torch
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import shape
import os
# Load model configuration and weights
#config_file = r"C:\Users\202391\MMsegm2024\mmsegmentation\work_dirs\mask2former_DAViT_160k_ade20k-512x512_Accacia\mask2former_DAViT_160k_ade20k-512x512.py"
#checkpoint_path = r"C:\Users\202391\MMsegm2024\mmsegmentation\work_dirs\mask2former_DAViT_160k_ade20k-512x512_Accacia\best_mIoU_iter_80000.pth"

width,height = 1024,1024
x = np.linspace(-1, 1, width)
y = np.linspace(-1, 1, height)
X, Y = np.meshgrid(x, y)

# Compute Gaussian distribution centered at (0,0)
sigma = 0.4  # Standard deviation
gaussian = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Normalize to [0, 1] for visualization
gaussian_normalized = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())

# config_file =  r"configs/coatnet/upernet_r50.py"
# checkpoint_path = r"work_dirs/upernet_r50mamba_5000/best_mIoU_iter_45000.pth"

config_file =  r"configs/mask2former/mask2former_swin-s_8xb2-160k_ade20k-512x512.py"
checkpoint_path = r"work_dirs/mask2former_swin-s_8xb2-160k_ade20k-512x512fastvit_olddata/best_mIoU_iter_35000.pth"

# config_file = r"G:\experiments\mmsegmentation\configs\mask2former\mask2former_swin-s_8xb2-160k_ade20k-512x512.py"
# checkpoint_path = r"G:\experiments\mmsegmentation\work_dirs\mask2former_swin-s_8xb2-160k_ade20k-512x512_all_data\best_mIoU_iter_21000.pth" 
model = init_model(config_file, checkpoint_path, device='cuda:0')

# Sliding window parameters
tile_size = 1024  # Size of each prediction tile
overlap = 512  # Overlap between tiles
print("loading model")

#input_image = r"F:\Accacia Mapping from UAV Data\Dibba\Dibba_f02\Dibba_f02.tif"
#output_shapefile = r"F:\Accacia Mapping from UAV Data\Prediction_on_large-scale_images\Output\Dibba_f02_predicted_Swin.shp"

input_image = "/drone_imgs/Dibba_f02.tif"
output_shapefile = "/drone_imgs/Dibba_f02.shp"
output_shapefolder = "/output"
import glob 
print(os.listdir("/drone_imgs"))
for filepath in glob.glob("/drone_imgs/*", recursive=True):
    print(filepath)# Perform sliding window inference
with rasterio.open(input_image) as src:
    # Get metadata from the source image
    meta = src.meta.copy()
    meta.update({'count': 1, 'dtype': 'uint8', 'nodata': None})  # Update metadata for single-band output
    print("starting prediction")
    # Create an output array for the final prediction
    prediction = np.zeros((src.height, src.width), dtype=np.float32)
    # weight_matrix = np.zeros((src.height, src.width), dtype=np.float32)

    # Calculate the total number of tiles for the progress bar
    num_tiles_y = (src.height - overlap) // (tile_size - overlap) + 1
    num_tiles_x = (src.width - overlap) // (tile_size - overlap) + 1
    total_tiles = num_tiles_y * num_tiles_x

    # Iterate over the image with a sliding window
    with tqdm(total=total_tiles, desc="Processing tiles") as pbar:
        for y in range(0, src.height, tile_size - overlap):
            for x in range(0, src.width, tile_size - overlap):
                # Calculate the window size (handle boundaries)
                win_height = min(tile_size, src.height - y)
                win_width = min(tile_size, src.width - x)

                # Read the first three bands of the image window (RGB only)
                window = rasterio.windows.Window(x, y, win_width, win_height)
                img = src.read((1, 2, 3), window=window)  # Read bands 1, 2, and 3 (R, G, B)

                # Transpose the image to match the model's expected input format (H, W, C)
                img = np.transpose(img, (1, 2, 0))

                # Perform inference
                with torch.no_grad():
                    result = inference_model(model, img)
                    out = result.pred_sem_seg.data[0].cpu().numpy()

                # Convert to binary mask (e.g., 0 or 1)
                binary_mask = np.where(out > 0, 1, 0).astype(np.float32)
                binary_mask = binary_mask  * gaussian_normalized[0:win_height,0:win_width]
                # Add the prediction to the output array, managing overlap
                prediction[y:y + win_height, x:x + win_width] += binary_mask
                # weight_matrix[y:y + win_height, x:x + win_width] += 1
                
                # Update the progress bar
                pbar.update(1)
    results = []
    # output_shapefile = os.path.join(output_shapefolder,os.path.basename(tif_file).replace(".tif",".shp"))
    prediction = (prediction > .7).astype(np.uint8)
    with rasterio.open(input_image) as src:
        transform = src.transform
        for shape_data, value in shapes(prediction, transform=transform):
            if value == 1:  # Only take polygons for the positive class
                geom = shape(shape_data)
                area_m2 = geom.area
                # if area_m2 < 2:
                #     # skip tiny polygons
                #     print(f"Skipping polygon of area {area_m2:.2f} m²")
                #     continue
                if geom.is_empty:
                    print("Empty geometry encountered:", shape_data)
                else:
                    results.append({'geometry': geom, 'properties': {'class': 1}})
    
    gdf = gpd.GeoDataFrame(results, crs=src.crs)
    if gdf.shape[0]>1:
        gdf['area'] = gdf.geometry.area
        filtered_gdf = gdf[gdf['area'] > 2]
        
        # Save to shapefile
        filtered_gdf.to_file(output_shapefile, driver='ESRI Shapefile')
        print(f"Shapefile saved to {output_shapefile}")


    # Normalize by the weight matrix to handle overlapping regions
    # prediction = np.divide(prediction, weight_matrix, where=weight_matrix != 0)
    # prediction = (prediction > 0.5).astype(np.uint8)  # Threshold to obtain final binary mask

    # # Vectorize the raster prediction to polygons and save to a shapefile
    # results = []
    # with rasterio.open(input_image) as src:
    #     transform = src.transform
    #     for shape_data, value in shapes(prediction, transform=transform):
    #         if value == 1:  # Only take the polygons representing the positive class (e.g., trees)
    #             results.append({'geometry': shape(shape_data), 'properties': {'class': 1}})

    # # Convert to GeoDataFrame
    # gdf = gpd.GeoDataFrame(results, crs=src.crs)

    # # Save to shapefile
    # gdf.to_file(output_shapefile, driver='ESRI Shapefile')

print(f"Shapefile saved to {output_shapefile}")
