from flask import Flask, request, render_template
import os
import zipfile
import tarfile
import rasterio
import rasterio.mask
import numpy as np
from PIL import Image
import geopandas as gpd
import tempfile
import shutil
import json
import time
import threading

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MAX_FILE_SIZE_MB = 2048

BAND_PRESETS = {
    "Landsat-8": {"True Color": {"R": "B4", "G": "B3", "B": "B2"},
                  "False Color": {"R": "B5", "G": "B4", "B": "B3"}},
    "Landsat-9": {"True Color": {"R": "B4", "G": "B3", "B": "B2"},
                  "False Color": {"R": "B5", "G": "B4", "B": "B3"}},
    "Sentinel-2": {"True Color": {"R": "B04", "G": "B03", "B": "B02"},
                   "False Color": {"R": "B08", "G": "B04", "B": "B03"}},
    "Resourcesat-2 Series": {"False Color": {"R": "BAND4", "G": "BAND3", "B": "BAND2"}},
    "IRS-P6": {"False Color": {"R": "BAND4", "G": "BAND3", "B": "BAND2"}}
}

def validate_file_size(file_storage):
    file_storage.seek(0, os.SEEK_END)
    size_mb = file_storage.tell() / (1024 * 1024)
    file_storage.seek(0)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File '{file_storage.filename}' exceeds size limit ({MAX_FILE_SIZE_MB} MB).")

def validate_shapefile_zip(zip_file, tmp_dir):
    zip_path = os.path.join(tmp_dir, 'shapefile.zip')
    zip_file.save(zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)
    required_exts = {'.shp', '.shx', '.dbf'}
    extracted_exts = {os.path.splitext(f)[1].lower() for f in os.listdir(tmp_dir)}
    if not required_exts.issubset(extracted_exts):
        raise ValueError("Shapefile ZIP missing required files (.shp, .shx, .dbf)")
    shp_files = [f for f in os.listdir(tmp_dir) if f.endswith('.shp')]
    if not shp_files:
        raise ValueError("No .shp file found in shapefile ZIP.")
    gdf = gpd.read_file(os.path.join(tmp_dir, shp_files[0]))
    return [geom.__geo_interface__ for geom in gdf.geometry.values]

def read_band(path, mask_geom=None):
    with rasterio.open(path) as src:
        meta = src.meta.copy()
        if mask_geom:
            out_image, out_transform = rasterio.mask.mask(src, mask_geom, crop=True)
            band = out_image[0]
            meta.update({"height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})
        else:
            band = src.read(1)
        band = band.astype(np.float32)
        nodata = src.nodata if src.nodata is not None else 0
        band[band == nodata] = np.nan
        norm_band = ((band - np.nanmin(band)) / (np.nanmax(band) - np.nanmin(band)) * 255)
        norm_band = np.where(np.isnan(norm_band), 0, norm_band).astype(np.uint8)
        return norm_band, meta

def save_outputs(red, green, blue, meta):
    output_path = os.path.join(UPLOAD_FOLDER, 'output.tif')

    # Create NoData mask where all bands are zero
    nodata_mask = (red == 0) & (green == 0) & (blue == 0)

    # Update metadata
    meta.update({
        "count": 3,
        "dtype": 'uint8',
        "driver": "GTiff",
        "nodata": 0  
    })

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(red, 1)
        dst.write(green, 2)
        dst.write(blue, 3)
        dst.write_mask((~nodata_mask).astype(np.uint8) * 255)

    # Create PNG with transparency
    rgb = np.stack([red, green, blue], axis=2)
    alpha = np.where(nodata_mask, 0, 255).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    png_path = os.path.join(UPLOAD_FOLDER, 'output.png')
    Image.fromarray(rgba, 'RGBA').save(png_path)

    # Metadata for JSON
    from rasterio.transform import array_bounds
    bounds = array_bounds(meta['height'], meta['width'], meta['transform'])

    def file_size(path):
        return round(os.path.getsize(path) / (1024 * 1024), 2)

    metadata = {
        "crs": str(meta.get("crs")),
        "width": meta.get("width"),
        "height": meta.get("height"),
        "transform": str(meta.get("transform")),
        "bounding_box": {
            "min_y": bounds[0], "max_y": bounds[1],
            "min_x": bounds[2], "max_x": bounds[3]
        },
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "file_sizes_MB": {
            "output.tif": file_size(output_path),
            "output.png": file_size(png_path)
        }
    }

    meta_path = os.path.join(UPLOAD_FOLDER, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    # Zip all outputs
    zip_path = os.path.join(UPLOAD_FOLDER, 'output_bundle.zip')
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(output_path, 'output.tif')
        zipf.write(png_path, 'output.png')
        zipf.write(meta_path, 'metadata.json')

    return output_path, zip_path

def cleanup_old_files(folder, age_seconds=3600):
    now = time.time()
    for root, _, files in os.walk(folder):
        for file in files:
            full_path = os.path.join(root, file)
            try:
                if now - os.path.getmtime(full_path) > age_seconds:
                    os.remove(full_path)
            except Exception:
                pass

def start_cleanup_thread():
    def run():
        while True:
            cleanup_old_files(UPLOAD_FOLDER)
            time.sleep(1800)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/create_composite', methods=['POST'])
def create_composite():
    sat_model = request.form.get('sat_model')
    composite = request.form.get('composite')
    image_file = request.files.get('imagery')
    shapefile_zip = request.files.get('shapefile')
    tmp_dir = shape_dir = None

    try:
        validate_file_size(image_file)
        tmp_dir = tempfile.mkdtemp()
        archive_path = os.path.join(tmp_dir, image_file.filename)
        image_file.save(archive_path)

        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path) as z:
                z.extractall(tmp_dir)
        elif archive_path.endswith('.tar'):
            with tarfile.open(archive_path) as t:
                t.extractall(tmp_dir)

        band_codes = BAND_PRESETS[sat_model][composite]
        all_files = []
        for root, _, files in os.walk(tmp_dir):
            all_files.extend([os.path.join(root, f) for f in files])

        band_paths = {}
        for color, code in band_codes.items():
            for f in all_files:
                fname = f.lower().replace("-", "").replace("_", "")
                if code.lower() in fname and fname.endswith(('.tif', '.jp2')):
                    band_paths[color] = f
                    break

        if len(band_paths) != 3:
            raise FileNotFoundError("Missing one or more required bands.")

        with rasterio.open(next(iter(band_paths.values()))) as src:
            raster_crs = src.crs

        mask_geom = None
        if shapefile_zip and shapefile_zip.filename.endswith('.zip'):
            shape_dir = tempfile.mkdtemp()
            validate_file_size(shapefile_zip)
            mask_geom = validate_shapefile_zip(shapefile_zip, shape_dir)
            gdf = gpd.GeoSeries.from_file(os.path.join(shape_dir, [f for f in os.listdir(shape_dir) if f.endswith('.shp')][0]))
            mask_geom = gdf.to_crs(raster_crs)
            mask_geom = [g.__geo_interface__ for g in mask_geom]

        red, meta = read_band(band_paths['R'], mask_geom)
        green, _ = read_band(band_paths['G'], mask_geom)
        blue, _ = read_band(band_paths['B'], mask_geom)

        geo_path, zip_path = save_outputs(red, green, blue, meta)

        return render_template('result.html',
                               geo_url='static/outputs/output.tif',
                               zip_url='static/outputs/output_bundle.zip')

    except Exception as e:
        return f"Error: {e}", 500
    finally:
        for d in [tmp_dir, shape_dir]:
            if d and os.path.exists(d):
                shutil.rmtree(d)

@app.route('/create_manual', methods=['POST'])
def create_manual():
    red_file = request.files.get('red_band')
    green_file = request.files.get('green_band')
    blue_file = request.files.get('blue_band')
    shapefile_zip = request.files.get('shapefile')
    tmp_dir = shape_dir = None

    try:
        for f in [red_file, green_file, blue_file]:
            validate_file_size(f)

        tmp_dir = tempfile.mkdtemp()
        red_path = os.path.join(tmp_dir, 'R.tif')
        green_path = os.path.join(tmp_dir, 'G.tif')
        blue_path = os.path.join(tmp_dir, 'B.tif')
        red_file.save(red_path)
        green_file.save(green_path)
        blue_file.save(blue_path)

        with rasterio.open(red_path) as src:
            raster_crs = src.crs

        mask_geom = None
        if shapefile_zip and shapefile_zip.filename.endswith('.zip'):
            shape_dir = tempfile.mkdtemp()
            validate_file_size(shapefile_zip)
            mask_geom = validate_shapefile_zip(shapefile_zip, shape_dir)
            gdf = gpd.GeoSeries.from_file(os.path.join(shape_dir, [f for f in os.listdir(shape_dir) if f.endswith('.shp')][0]))
            mask_geom = gdf.to_crs(raster_crs)
            mask_geom = [g.__geo_interface__ for g in mask_geom]

        red, meta = read_band(red_path, mask_geom)
        green, _ = read_band(green_path, mask_geom)
        blue, _ = read_band(blue_path, mask_geom)

        geo_path, zip_path = save_outputs(red, green, blue, meta)

        return render_template('result.html',
                               geo_url='static/outputs/output.tif',
                               zip_url='static/outputs/output_bundle.zip')

    except Exception as e:
        return f"Error: {e}", 500
    finally:
        for d in [tmp_dir, shape_dir]:
            if d and os.path.exists(d):
                shutil.rmtree(d)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    start_cleanup_thread()
    app.run(debug=True)
