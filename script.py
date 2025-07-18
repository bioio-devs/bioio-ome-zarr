from bioio_ome_zarr.writers.utils import add_zarr_level

path = "/Users/brian.whitney/Desktop/Repos/bioio-ome-zarr/s1_t7_c4_z3_Image_0.zarr"

add_zarr_level(path,[1.0,1.0,1.0,0.5,0.5])