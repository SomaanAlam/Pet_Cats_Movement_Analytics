# Data Instructions

This project relies on two external spatial datasets (not stored in the repo to keep the repository lightweight):

## 1. Elevation (DEM)
- **Source**: FABDEM (Forest And Buildings removed Copernicus DEM, 30 m resolution).
- **Access**: Available from [https://data.worldpop.org/FABDEM](https://data.worldpop.org/FABDEM).
- **Format**: GeoTIFF (.tif).
- **Notes**: Used to enrich tracks with elevation. Please download the DEM covering your study area.

## 2. Land Cover
- **Source**: ESA WorldCover 2020 (10 m global land-cover map).
- **Access**: [https://worldcover2020.esa.int/](https://worldcover2020.esa.int/).
- **Format**: GeoTIFF (.tif).
- **Notes**: Used to classify each GPS fix into a land-cover type (grassland, forest, built-up, etc.).


---

### File placement
After download, place files into `data/`:

data/
├─ Pet Cats United Kingdom.csv # movement dataset

├─ FABDEM_UK.tif # DEM (example name)

├─ WorldCover_UK_2020.tif # Landcover (example name)

---

### Important
- Large raster files (`*.tif`) are **not tracked** in git (see `.gitignore`).
- If you want them in version control, enable [Git LFS](https://git-lfs.com/).
