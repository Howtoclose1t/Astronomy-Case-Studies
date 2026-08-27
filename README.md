# Astronomical Data Processing and Machine Learning Case Studies

This repository presents a collection of personal case studies in modern astronomical data processing and machine learning. It covers end-to-end workflows involving data from JWST, the Square Kilometre Array (SKA), JADES, Euclid, and LIGO/Virgo, with methods ranging from conventional machine learning and deep learning to foundation models.

> [!IMPORTANT]
> This repository documents personal learning exercises and code-reproduction work. Its purpose is to develop practical familiarity with astronomical data-analysis pipelines and the use of machine learning and foundation models as supporting tools in astronomy. The results should not be interpreted as original research findings or as general evaluations of any particular algorithm. Large FITS files, model weights, and runtime caches are not included; refer to the instructions for each task to obtain the required data.

## Project Overview

| No. | Case Study | Data or Platform | Core Methods |
| ---: | --- | --- | --- |
| **01** | **JWST/JADES Image Processing** | JWST/NIRCam F200W | Background estimation, source extraction, completeness testing through Sérsic-profile injection, and U-Net segmentation |
| **02** | **Classical SKA Radio-Source Detection** | SKA SDC1 simulated radio data | WCS coordinates, k-sigma clipping, threshold detection, and catalogue cross-matching |
| **03** | **YOLO Radio-Source Detection** | SKA SDC1 | Image tiling, `asinh` stretching, bounding-box encoding, YOLOv8, and CIANNA |
| **04** | **JADES Catalogue Matching** | JADES DR3/DR4/DR5 | NIRSpec–NIRCam cross-matching, positional-tolerance matching, and quality control |
| **05** | **JADES Photometric-Redshift Estimation** | JADES and EAZY | XGBoost, TabPFN, gated residual hybrid modelling, and TreeSHAP |
| **06** | **DINGO-T1 Gravitational-Wave Inference** | LIGO/Virgo event GW190701_203306 | Neural posterior estimation and the DINGO-T1 Transformer |
| **07** | **Euclid Q1 Data Analysis** | ESA Euclid Archive and DESI DR1 | Cutout pipeline, gravitational-lensing arc analysis, and photometric-versus-spectroscopic redshift validation |

## Repository Structure

```text
.
├── 01_JWST_Image_Processing/       # JWST F200W image cutouts, background subtraction, and completeness tests
├── 02_SKA_Radio_Source_Detection/  # SKA SDC1 radio-continuum statistics and classical source detection
├── 03_YOLO_Radio_Astronomy/        # Radio-image tiling and object detection with YOLO/CIANNA
├── 04_JADES_Catalog_Matching/      # Multi-stage JADES spectroscopic–photometric catalogue matching
├── 05_JADES_PhotoZ/
│   ├── 05A_JADES_PhotoZ_ML/        # XGBoost gated-residual correction of EAZY photometric redshifts
│   └── 05B_JADES_Foundation/       # TabPFN residual correction and ablation experiments
├── 06_DINGO_T1_Case_Study/         # Gravitational-wave posterior inference with a pretrained DINGO-T1 model
└── 07_Euclid_Q1_Case_Study/        # Euclid Q1 queries, NGC 6505 strong-lens analysis, and DESI validation
```

## Case Studies and Results

### 01. JWST/JADES Image Processing

#### Image analysis and completeness tests

In a JADES F200W cutout, source-detection thresholds of $1.5\sigma$, $2\sigma$, $3\sigma$, and $5\sigma$ yielded 5,080, 2,207, 1,232, and 775 detections, respectively. Injection–recovery experiments produced the following results:

- **Point sources:** recovery rates were 12%, 62%, 84%, and 100% at peak amplitudes of $3\sigma$, $5\sigma$, $6\sigma$, and $8\sigma$, respectively.
- **Sérsic-profile extended sources:** recovery rates were 32%, 66%, and 100% at $2.5\sigma$, $3\sigma$, and $4\sigma$, respectively. These measurements quantify the selection effects affecting faint, extended objects under the adopted detection settings.

#### U-Net pixel-level segmentation

After eight training epochs, the highest validation intersection over union (IoU) was 0.9437. The final segmentation output contained 2,004 candidate objects.

### 02. Classical SKA Radio-Source Detection

- A $32{,}768 \times 32{,}768$ pixel SKA SDC1 simulated radio image containing 46,307 injected ground-truth sources was processed.
- Detection at a $3\sigma$ threshold produced 10,975 candidates: 10,415 matched sources, 560 unmatched detections, and 35,892 injected faint sources that were not recovered.
- Under this experimental configuration, classical threshold detection achieved **94.9% precision**, **22.5% recall**, and an **F1 score of 36.4%**.

### 03. YOLO Radio-Source Detection

- **Data pipeline:** the large FITS image was divided into tiles, robust normalisation and an `asinh` contrast stretch were applied to compress its dynamic range, and celestial coordinates were converted into YOLO-format bounding boxes.
- **YOLOv8 evaluation:** in the recorded small-scale experiment, the model achieved **78.7% precision**, **16.2% recall**, an **F1 score of 26.9%**, and **mAP@0.5 of 0.113**. This case study establishes an end-to-end experimental workflow for deep-learning-based object detection in radio-astronomy images.

### 04. JADES Catalogue Matching

- **Two-stage matching:** sources were first matched by official identifiers. Records without an identifier match then fell back to nearest-neighbour sky-coordinate matching within $0.2''$. Duplicate observations were resolved before applying redshift-quality, reduction-problem, non-stellar-source, and valid-filter criteria.
- **DR3 experiment:** the resulting machine-learning-ready dataset contained 846 secure, non-stellar sources with measurements in at least five valid photometric filters.
- **DR4/DR5 extension:** from 5,190 DR4 spectroscopic records, 4,946 were matched by official DR5 identifiers and 18 by positional fallback; 131 remained unmatched. After duplicate resolution, there were 4,964 unique matches, of which 3,171 passed the final modelling criteria and were used in Case Study 05.

### 05. JADES Photometric Redshifts and Machine Learning

The experiments used the 3,171 objects produced in Case Study 04, with a fixed split of 2,219 training objects, 476 validation objects, and 476 test objects. The objective was to assess whether machine-learning models could improve the EAZY template-fitting redshifts under this specific experimental setup.

#### 05A. XGBoost + EAZY

- A **gated residual hybrid model** was developed in which machine learning determines whether an EAZY estimate should be corrected and, if so, the size of that correction.
- Compared with photometry-only XGBoost, which produced an MAE of 0.4213 and 97 catastrophic outliers, the gated-residual model reduced the test-set MAE to 0.1357. It also reduced the number of catastrophic outliers from 20 for EAZY to 16.

#### 05B. TabPFN Foundation Model + EAZY

- TabPFN v3 was used to learn the EAZY residual in logarithmic redshift, $r = \log(1+z_{\mathrm{spec}}) - \log(1+z_{\mathrm{EAZY}})$.
- The table reports the five frozen methods evaluated on the same 476-source test set. The EAZY and XGBoost predictions were reused from Case Study 05A rather than retrained.

| Method | Normalised Bias | Median Normalised Absolute Error | $\sigma_{\mathrm{NMAD}}$ | Redshift MAE | Catastrophic Outliers | Outlier Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EAZY DR5 `z_a` | 0.010433 | 0.017061 | 0.021029 | 0.170490 | 20 | 4.20% |
| XGBoost: photometry only | -0.002838 | 0.058638 | 0.087048 | 0.421305 | 97 | 20.38% |
| XGBoost: gated EAZY residual | 0.003361 | 0.014293 | 0.022499 | 0.135674 | 16 | 3.36% |
| TabPFN v3: photometry only | 0.001318 | 0.013875 | 0.020546 | 0.145296 | 25 | 5.25% |
| **TabPFN v3: EAZY residual** | **0.000218** | **0.011210** | **0.016599** | **0.094889** | **15** | **3.15%** |

Within this dataset and fixed split, the TabPFN–EAZY residual hybrid produced the strongest overall results. Relative to EAZY, it reduced the median normalised absolute error by 34.3%, $\sigma_{\mathrm{NMAD}}$ by 21.1%, redshift MAE by 44.3%, and the number of catastrophic outliers from 20 to 15. Broader claims would require evaluation across additional datasets and configurations.

### 06. DINGO-T1 Gravitational-Wave Inference

- The DINGO-T1 workflow was applied to the real gravitational-wave event `GW190701_203306` using neural posterior estimation.
- Posterior distributions were compared for the H1, H1–L1, and H1–L1–V1 detector networks. In this exercise, combining detectors substantially narrowed the posterior distributions of several physical parameters, including mass, inclination, and luminosity distance. The case study demonstrates the practical use of a Transformer-based model for rapid gravitational-wave inference.

### 07. Euclid Q1 Data Analysis

- **Data retrieval:** `astroquery` was used to retrieve VIS and NISP cutouts of the NGC 6505 strong-lensing system from the ESA Euclid Science Archive.
- **Lensing-arc measurement:** after subtracting the foreground galaxy's light profile, the measured Einstein-ring radius was approximately $2.65''$, consistent with the reference value of about $2.5''$ used in the accompanying analysis.
- **External validation:** 83,824 Euclid photometric-redshift sources were cross-matched with DESI DR1 spectroscopic-redshift targets. A $0.5''$ matching radius yielded 885 unique matches, of which 884 had valid redshifts for evaluation. This sample had a normalised bias of -0.0132, $\sigma_{\mathrm{NMAD}} = 0.0408$, and a catastrophic-outlier rate of 6.67%. In the faintest $H$-band bin, $22.13 < H \leq 24.49$, the outlier rate increased to 19.6%.

## Quick Start

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Obtain the data

Large FITS files, generated catalogues, model weights, and runtime products are excluded from the repository. Prepare only the data required for the case study you intend to run.

| Case Study | Required Preparation |
| --- | --- |
| **01** | Download the F200W imaging used by the workflow from [JADES at MAST](https://archive.stsci.edu/hlsp/jades) and place it under `01_JWST_Image_Processing/data/`. The CEERS catalogue notebook performs its own MAST query. |
| **02** | From [SKA Science Data Challenge 1](https://www.skao.int/en/464/ska-science-data-challenge-1), place `SKAMid_B2_1000h_v3.fits` in `02_SKA_Radio_Source_Detection/data/raw/` and `TrainingSet_B2_v2.txt` in `02_SKA_Radio_Source_Detection/data/catalog/`. |
| **03** | Use the B1 download and preparation cells in `03_YOLO_Radio_Astronomy/notebooks/03_yolo_sdc1_exploration.ipynb`. |
| **04** | Follow `04_JADES_Catalog_Matching/DATA_SOURCES.md` for the exact DR3, DR4, and DR5 catalogue filenames and locations. |
| **05** | Run Case Study 04 first. Then run 05A before 05B, because 05B reuses the frozen features, split assignments, feature state, and test predictions generated by 05A. |
| **06** | The notebook downloads the DINGO-T1 model from [Zenodo record 17726076](https://zenodo.org/records/17726076) and retrieves the event configuration files. Do not commit model files, authentication material, or generated event products. |
| **07** | Ensure that the [Euclid Science Archive](https://eas.esac.esa.int/sas/) is accessible. The numbered notebooks query the archive and download the required cutouts. |

### 3. Create a task-specific environment

There is no root-level environment that covers every case study. For Cases 01, 02, 04, 05A, and 07, enter the relevant directory, create a virtual environment, and install its requirements file:

```bash
cd <task-directory>
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Case Study 03 uses its Conda environment:

```bash
conda env create -f 03_YOLO_Radio_Astronomy/environment.yml
conda activate yolo-radio-astronomy
```

Case Study 05B uses a separate Python 3.11 environment with TabPFN 8.4.0:

```bash
conda env create -f 05_JADES_PhotoZ/05B_JADES_Foundation/environment.yml
conda activate jades_fm
```

TabPFN requires authorised model access; keep the authentication token in the local environment and never commit it. The recorded 05B run used a CUDA-capable GPU.

For the local DINGO-T1 workflow in Case Study 06:

```bash
cd 06_DINGO_T1_Case_Study
git clone --branch dingo-t1 https://github.com/dingo-gw/dingo.git
python -m pip install -e "./dingo"
python -m pip install jupyterlab
```

The DINGO notebook also contains a Google Colab installation path.

### 4. Run the notebooks

Start Jupyter from the relevant case-study directory so that local paths and `src/` imports resolve correctly. Within each workflow, run numbered notebooks in filename order. The main dependency chain is:

```text
04_JADES_Catalog_Matching
└── 05_JADES_PhotoZ/05A_JADES_PhotoZ_ML
    └── 05_JADES_PhotoZ/05B_JADES_Foundation
```

The other case studies can be run independently. Case Study 06 starts from `06_DINGO_T1_Case_Study/baseline/tutorial_inference_with_DINGO-T1.ipynb`; Case Study 07 uses the five numbered notebooks under `07_Euclid_Q1_Case_Study/workspace/notebooks/`.

## Summary and Reflections

Together, these seven case studies form an end-to-end learning workflow spanning raw FITS-image handling, coordinate-based catalogue matching, signal detection, high-dimensional feature engineering, and the use of modern machine-learning and foundation models to refine physical-parameter estimates. A recurring theme is the importance of uncertainty and selection effects in astronomical data modelling. The reported metrics document the behaviour of the reproduced workflows under their stated assumptions; they are not intended as general benchmarks.
