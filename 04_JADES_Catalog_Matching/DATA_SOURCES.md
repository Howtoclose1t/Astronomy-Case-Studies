# Data sources and release roles

Raw catalogs are intentionally excluded from GitHub. Download them from the official release pages and keep the filenames shown below.

## Final DR4/DR5 workflow

Spectroscopy:

- Release page: `https://jades.herts.ac.uk/DR4/`
- Local file: `data/raw/dr45/Combined_DR4_external_v1.2.1.fits`

Photometry:

- Release page: `https://slate.ucsc.edu/~brant/jades-dr5/`
- Local files:
  - `data/raw/dr45/hlsp_jades_jwst_nircam_goods-n_photometry_v5.0_catalog.fits`
  - `data/raw/dr45/hlsp_jades_jwst_nircam_goods-s_photometry_v5.0_catalog.fits`

This is the final project-scale workflow, combining DR4 NIRSpec redshifts with DR5 GOODS-N and GOODS-S NIRCam photometry.

## DR3 GOODS-N workflow

Photometry:

`https://archive.stsci.edu/hlsps/jades/dr3/goods-n/catalogs/hlsp_jades_jwst_nircam_goods-n_photometry_v1.0_catalog.fits`

Spectroscopy:

`https://archive.stsci.edu/hlsps/jades/dr3/goods-n/catalogs/hlsp_jades_jwst_nirspec_goods-n_prism-line-fluxes_v1.1_catalog.fits`

Spectroscopy README:

`https://drive.google.com/file/d/1kPOzSoevgM22HSE5nx7016jxyC8Xcqx7/view`

Photometric extension documentation:

`https://archive.stsci.edu/hlsps/jades/hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog-ext-readme.pdf`

DR3 supplies photometry and spectroscopy in the same release and GOODS-N field. It was used to establish and explain the matching method on a smaller, scope-controlled dataset before the DR4/DR5 expansion. The DR3 notebooks remain part of the public project history rather than being treated as archived material.
