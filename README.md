# <div align="center">

# 

# \# 💧 SDWA Drinking Water Quality — ML Project

# 

# \[!\[Course](https://img.shields.io/badge/Course-AAI--540-blue?style=flat-square)](https://www.sandiego.edu)

# \[!\[University](https://img.shields.io/badge/University-San%20Diego-navy?style=flat-square\&logo=graduation-cap)](https://www.sandiego.edu)

# \[!\[Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square\&logo=python\&logoColor=white)](https://www.python.org)

# \[!\[AWS](https://img.shields.io/badge/AWS-SageMaker%20Studio-FF9900?style=flat-square\&logo=amazonaws\&logoColor=white)](https://aws.amazon.com/sagemaker)

# \[!\[Data](https://img.shields.io/badge/Data-EPA%20SDWA-green?style=flat-square)](https://echo.epa.gov/tools/data-downloads/sdwa-download-summary)

# \[!\[License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

# 

# \*\*Predicting drinking water violations across U.S. public water systems using EPA compliance data and machine learning.\*\*

# 

# \*AAI-540 Team Project · University of San Diego · Spring 2026\*

# 

# </div>

# 

# \---

# 

# \## 📋 Table of Contents

# 

# | # | Section |

# |---|---------|

# | 1 | \[Project Overview](#1--project-overview) |

# | 2 | \[Repository Structure](#2--repository-structure) |

# | 3 | \[S3 Bucket Structure](#3--s3-bucket-structure) |

# | 4 | \[SDWA Dataset Schema](#4--sdwa-dataset-schema) |

# | 5 | \[Setup: Connect to Shared S3 from SageMaker Studio](#5--setup-connect-to-shared-s3-from-sagemaker-studio) |

# | 6 | \[Reproducibility Workflow](#6--reproducibility-workflow) |

# | 7 | \[Team Conventions](#7--team-conventions) |

# | 8 | \[References](#8--references) |

# 

# \---

# 

# \## 1 · Project Overview

# 

# This project applies machine learning to the \*\*EPA Safe Drinking Water Act (SDWA)\*\* national compliance database to predict drinking water violations at public water systems (PWS) across the United States.

# 

# \### 🎯 Goals

# 

# | Goal | Description |

# |------|-------------|

# | \*\*Prediction\*\* | Predict which public water systems are at high risk of future health-based violations |

# | \*\*Classification\*\* | Identify violation severity and type from system characteristics and historical patterns |

# | \*\*Explainability\*\* | Surface key risk factors for water system non-compliance to support regulatory action |

# 

# \### 🧩 Problem Statement

# 

# Millions of Americans are served by water systems that have a history of violations under the Safe Drinking Water Act. Using EPA's publicly available compliance data, we build ML models to proactively identify at-risk systems — enabling regulators and public health officials to prioritize inspections and interventions before violations occur.

# 

# \### 🤖 ML Tasks

# 

# \- \*\*Binary Classification\*\* — Will a given water system incur a health-based violation in the next reporting period?

# \- \*\*Multi-class Classification\*\* — What category of violation (`MCL`, `TT`, `MON`, `RPT`) is most likely?

# \- \*\*Clustering / EDA\*\* — Identify geographic or demographic clusters of chronic non-compliance.

# 

# \---

# 

# \## 2 · Repository Structure

# 

# ```text

# aai540-sdwa/

# │

# ├── README.md                     # This file

# ├── .gitignore

# ├── requirements.txt              # Python dependencies

# │

# ├── data/

# │   ├── raw/                      # Local copies of raw CSVs (gitignored)

# │   └── processed/                # Cleaned/feature-engineered data (gitignored)

# │

# ├── notebooks/

# │   ├── 00\_data\_ingestion.ipynb   # Load data from S3, initial inspection

# │   ├── 01\_eda.ipynb              # Exploratory data analysis

# │   ├── 02\_preprocessing.ipynb    # Cleaning, encoding, feature engineering

# │   ├── 03\_modeling.ipynb         # Model training and evaluation

# │   └── 04\_explainability.ipynb   # SHAP / feature importance analysis

# │

# ├── src/

# │   ├── \_\_init\_\_.py

# │   ├── data\_loader.py            # S3 data loading utilities

# │   ├── preprocessing.py          # Feature engineering pipeline

# │   ├── model.py                  # Model definitions and training loops

# │   └── evaluate.py               # Metrics, confusion matrix, ROC

# │

# ├── models/

# │   └── .gitkeep                  # Saved model artifacts (gitignored except placeholder)

# │

# └── reports/

# &#x20;   ├── figures/                  # Charts and visualizations

# &#x20;   └── final\_report.pdf          # Final deliverable

# ```

# 

# \---

# 

# \## 3 · S3 Bucket Structure

# 

# > \[!IMPORTANT]

# > All raw SDWA data files are stored in the shared team S3 bucket. \*\*Do not re-upload raw data\*\* — always read directly from S3.

# 

# ```text

# s3://aai540-sdwa-team/

# │

# ├── raw/

# │   ├── SDWA\_SITES.csv

# │   ├── SDWA\_PUB\_WATER\_SYSTEMS.csv

# │   ├── SDWA\_VIOLATIONS\_ENFORCEMENT.csv

# │   ├── SDWA\_FACILITIES.csv

# │   ├── SDWA\_SERVICE\_AREAS.csv

# │   ├── SDWA\_LCR\_SAMPLES.csv

# │   ├── SDWA\_GEOGRAPHIC\_AREAS.csv

# │   └── SDWA\_PN\_VIOLATION\_ASSOC.csv

# │

# ├── processed/

# │   ├── features\_merged.parquet   # Joined + engineered feature table

# │   ├── train.parquet

# │   ├── val.parquet

# │   └── test.parquet

# │

# ├── models/

# │   ├── xgboost\_v1.pkl

# │   └── random\_forest\_v1.pkl

# │

# └── reports/

# &#x20;   └── figures/

# ```

# 

# | Property | Value |

# |----------|-------|

# | \*\*Bucket Name\*\* | `aai540-sdwa-team` |

# | \*\*Region\*\* | `us-east-1` |

# | \*\*Access\*\* | AWS Academy lab credentials (see \[Section 5](#5--setup-connect-to-shared-s3-from-sagemaker-studio)) |

# 

# \---

# 

# \## 4 · SDWA Dataset Schema

# 

# All datasets originate from the \[EPA SDWA national download](https://echo.epa.gov/tools/data-downloads/sdwa-download-summary). Column descriptions follow the official EPA data dictionary. Click any table name below to expand its schema.

# 

# \---

# 

# <details>

# <summary><strong>4.1 \&nbsp;SDWA\_SITES</strong> — Geographic and ownership metadata for each regulated site</summary>

# 

# <br>

# 

# | Column | Type | Description |

# |--------|------|-------------|

# | `PWSID` | `str` | \*\*Primary key\*\* — Public Water System ID (e.g., `NM3510123`) |

# | `SITE\_ID` | `str` | Unique site identifier within the PWS |

# | `SITE\_NAME` | `str` | Name of the physical site |

# | `SITE\_TYPE\_CODE` | `str` | Type of site (e.g., `CC` = Consecutive Connection) |

# | `STATE\_CODE` | `str` | Two-letter state abbreviation |

# | `ZIP\_CODE` | `str` | ZIP code of the site |

# | `TRIBAL\_CODE` | `str` | Tribal nation code if applicable \*(nullable)\* |

# | `LATITUDE` | `float` | Site latitude (decimal degrees) |

# | `LONGITUDE` | `float` | Site longitude (decimal degrees) |

# | `CONGRESSIONAL\_DISTRICT\_NO` | `str` | Congressional district number |

# | `COUNTY\_SERVED` | `str` | County in which the site is located |

# | `CITY\_SERVED` | `str` | City in which the site is located |

# 

# </details>

# 

# \---

# 

# <details>

# <summary><strong>4.2 \&nbsp;SDWA\_PUB\_WATER\_SYSTEMS</strong> — Core registry of all public water systems (central entity)</summary>

# 

# <br>

# 

# | Column | Type | Description |

# |--------|------|-------------|

# | `PWSID` | `str` | \*\*Primary key\*\* — Public Water System ID |

# | `PWS\_NAME` | `str` | Name of the water system |

# | `PRIMACY\_AGENCY\_CODE` | `str` | State/territory code of the primacy agency |

# | `EPA\_REGION` | `str` | EPA region number (`01`–`10`) |

# | `PWS\_TYPE\_CODE` | `str` | `CWS`=Community · `NTNCWS`=Non-Transient Non-Community · `TNCWS`=Transient Non-Community |

# | `OWNER\_TYPE\_CODE` | `str` | `F`=Federal · `S`=State · `L`=Local · `M`=Public/Private Mix · `P`=Private · `N`=Native American |

# | `POPULATION\_SERVED\_COUNT` | `int` | Estimated number of people served |

# | `SERVICE\_CONNECTIONS\_COUNT` | `int` | Number of service connections |

# | `ACTIVITY\_CODE` | `str` | `A`=Active · `I`=Inactive · `N`=Inactive (Non-NTNC) |

# | `PWS\_ACTIVITY\_EFF\_DATE` | `date` | Effective date of current activity status |

# | `PRIMARY\_SOURCE\_CODE` | `str` | `GW`=Groundwater · `SW`=Surface Water · `GU`=Groundwater Under Influence |

# | `GW\_SW\_CODE` | `str` | Simplified: `GW` or `SW` |

# | `OUTSTANDING\_PERFORMER` | `str` | `Y` if system has outstanding performance designation |

# | `OUTSTANDING\_PERF\_BEGIN\_DATE` | `date` | Date of outstanding performer designation |

# | `ORG\_NAME` | `str` | Organization/operator name |

# | `ADMIN\_NAME` | `str` | Administrator contact name |

# | `EMAIL\_ADDR` | `str` | Contact email \*(may be redacted)\* |

# | `PHONE\_NUMBER` | `str` | Contact phone |

# | `ADDRESS\_LINE1` | `str` | Mailing address |

# | `CITY\_NAME` | `str` | City |

# | `STATE\_CODE` | `str` | State |

# | `ZIP\_CODE` | `str` | ZIP code |

# | `COUNTRY\_CODE` | `str` | Country (typically `US`) |

# | `CITIES\_SERVED` | `str` | Pipe-delimited list of cities served |

# | `COUNTIES\_SERVED` | `str` | Pipe-delimited list of counties served |

# 

# </details>

# 

# \---

# 

# <details>

# <summary><strong>4.3 \&nbsp;SDWA\_VIOLATIONS\_ENFORCEMENT</strong> — All recorded violations and enforcement actions \&nbsp;🎯 <em>Primary target variable source</em></summary>

# 

# <br>

# 

# | Column | Type | Description |

# |--------|------|-------------|

# | `PWSID` | `str` | \*\*Foreign key\*\* → `SDWA\_PUB\_WATER\_SYSTEMS.PWSID` |

# | `VIOLATION\_ID` | `str` | Unique violation record ID |

# | `FACILITY\_ID` | `str` | Facility where violation occurred \*(nullable)\* |

# | `POPULATION\_SERVED\_COUNT` | `int` | Population served at time of violation |

# | `NPM\_CANDIDATE` | `str` | `Y` if violation is a National Priority candidate |

# | `PRIMACY\_AGENCY\_CODE` | `str` | State/territory of primacy |

# | `EPA\_REGION` | `str` | EPA region |

# | `VIOLATION\_CODE` | `str` | EPA violation code (see SDWA code table) |

# | `VIOLATION\_CATEGORY\_CODE` | `str` | `MCL`=Max Contaminant Level · `MR`=Monitoring/Reporting · `TT`=Treatment Technique · `PN`=Public Notification · `OTHER` |

# | `IS\_HEALTH\_BASED\_IND` | `str` | ⭐ `Y` if health-based — \*\*this is the model target label\*\* |

# | `CONTAMINANT\_CODE` | `str` | EPA contaminant code |

# | `CONTAMINANT\_NAME` | `str` | Human-readable name (e.g., `Lead`, `Nitrate`, `Total Coliform`) |

# | `COMPLIANCE\_PERIOD\_BEGIN\_DATE` | `date` | Start of compliance monitoring period |

# | `COMPLIANCE\_PERIOD\_END\_DATE` | `date` | End of compliance monitoring period |

# | `VIOLATION\_BEGIN\_DATE` | `date` | Date violation began |

# | `VIOLATION\_END\_DATE` | `date` | Date violation ended \*(nullable if ongoing)\* |

# | `RULE\_CODE` | `str` | Applicable rule code (e.g., `110`=TCR · `400`=Lead \& Copper) |

# | `RULE\_GROUP\_CODE` | `str` | Broader rule grouping |

# | `RULE\_FAMILY\_CODE` | `str` | Rule family (e.g., `100`=Microbials · `200`=Chemicals) |

# | `SEVERITY\_IND\_CNT` | `int` | Count-based severity indicator |

# | `PUBLIC\_NOTIFICATION\_TIER` | `int` | PN tier: `1`=most severe → `3`=least \*(nullable)\* |

# | `CALCULATED\_RTC\_DATE` | `date` | Calculated return-to-compliance date |

# | `RTC\_DATE` | `date` | Actual return-to-compliance date \*(nullable if unresolved)\* |

# | `ENFORCEMENT\_ID` | `str` | Associated enforcement action ID \*(nullable)\* |

# | `ENF\_ACTION\_TYPE\_CODE` | `str` | Type of enforcement action taken |

# | `ENF\_ACTION\_DATE` | `date` | Date of enforcement action |

# 

# </details>

# 

# \---

# 

# <details>

# <summary><strong>4.4 \&nbsp;SDWA\_FACILITIES</strong> — Physical infrastructure details for facilities within each water system</summary>

# 

# <br>

# 

# | Column | Type | Description |

# |--------|------|-------------|

# | `PWSID` | `str` | \*\*Foreign key\*\* → `SDWA\_PUB\_WATER\_SYSTEMS.PWSID` |

# | `FACILITY\_ID` | `str` | Unique facility ID within the PWS |

# | `FACILITY\_NAME` | `str` | Name of the facility |

# | `STATE\_FACILITY\_ID` | `str` | State-assigned facility identifier |

# | `FACILITY\_TYPE\_CODE` | `str` | `CC`=Consecutive Connection · `DS`=Distribution System · `IN`=Intake · `WL`=Well · `TP`=Treatment Plant |

# | `WATER\_TYPE\_CODE` | `str` | Type of water handled (`GW`, `SW`, `GU`, etc.) |

# | `AVAILABILITY\_CODE` | `str` | `E`=Emergency · `I`=Interim · `P`=Permanent · `S`=Seasonal |

# | `SELLER\_TREATMENT\_CODE` | `str` | Treatment provided by seller (for purchased water) |

# | `ACTIVITY\_CODE` | `str` | `A`=Active · `I`=Inactive |

# | `SUBMISSION\_STATUS\_CODE` | `str` | `A`=Approved · `P`=Pending |

# | `IS\_SOURCE\_IND` | `str` | `Y` if this facility is a source |

# | `IS\_SOURCE\_TREATED\_IND` | `str` | `Y` if source water is treated |

# 

# </details>

# 

# \---

# 

# <details>

# <summary><strong>4.5 \&nbsp;SDWA\_SERVICE\_AREAS</strong> — Geographic service area data for water systems</summary>

# 

# <br>

# 

# | Column | Type | Description |

# |--------|------|-------------|

# | `PWSID` | `str` | \*\*Foreign key\*\* → `SDWA\_PUB\_WATER\_SYSTEMS.PWSID` |

# | `SERVICE\_AREA\_TYPE\_CODE` | `str` | Type of area served (e.g., `RETAIL`, `WHOLESALE`) |

# | `IS\_PRIMARY\_SERVICE\_AREA\_CODE` | `str` | `Y` if this is the primary service area |

# | `PRIMACY\_AGENCY\_CODE` | `str` | Primacy agency state code |

# 

# </details>

# 

# \---

# 

# <details>

# <summary><strong>4.6 \&nbsp;SDWA\_LCR\_SAMPLES</strong> — Lead and Copper Rule sample results (critical for health-based features)</summary>

# 

# <br>

# 

# | Column | Type | Description |

# |--------|------|-------------|

# | `PWSID` | `str` | \*\*Foreign key\*\* → `SDWA\_PUB\_WATER\_SYSTEMS.PWSID` |

# | `SAMPLE\_ID` | `str` | Unique sample identifier |

# | `SAMPLING\_START\_DATE` | `date` | Date sample collection started |

# | `SAMPLING\_END\_DATE` | `date` | Date sample collection ended |

# | `RECONCILIATION\_ID` | `str` | Reconciliation record ID |

# | `PRIMACY\_AGENCY\_CODE` | `str` | State code |

# | `SAR\_ID` | `str` | Sample analyte result ID |

# | `CONTAMINANT\_CODE` | `str` | `PB90`=Lead 90th percentile · `CU90`=Copper 90th percentile |

# | `RESULT\_SIGN\_CODE` | `str` | Result qualifier: `=`, `<`, or `>` |

# | `SAMPLE\_MEASURE` | `float` | Measured concentration value |

# | `UNIT\_OF\_MEASURE` | `str` | Unit (typically `mg/L` or `ppb`) |

# 

# </details>

# 

# \---

# 

# <details>

# <summary><strong>4.7 \&nbsp;SDWA\_GEOGRAPHIC\_AREAS</strong> — Maps water systems to geographic/political boundaries</summary>

# 

# <br>

# 

# | Column | Type | Description |

# |--------|------|-------------|

# | `PWSID` | `str` | \*\*Foreign key\*\* → `SDWA\_PUB\_WATER\_SYSTEMS.PWSID` |

# | `GEO\_ID` | `str` | Unique geographic area ID |

# | `PRIMACY\_AGENCY\_CODE` | `str` | State code |

# | `CITY\_SERVED` | `str` | City served by the system |

# | `COUNTY\_SERVED` | `str` | County served |

# | `STATE\_SERVED` | `str` | State served |

# | `ZIP\_CODE\_SERVED` | `str` | ZIP code of the served area |

# | `ANSI\_ENTITY\_CODE` | `str` | ANSI geographic entity code |

# | `TRIBAL\_CODE` | `str` | Tribal nation code \*(nullable)\* |

# 

# </details>

# 

# \---

# 

# <details>

# <summary><strong>4.8 \&nbsp;SDWA\_PN\_VIOLATION\_ASSOC</strong> — Associates public notification records with originating violations</summary>

# 

# <br>

# 

# | Column | Type | Description |

# |--------|------|-------------|

# | `PWSID` | `str` | \*\*Foreign key\*\* → `SDWA\_PUB\_WATER\_SYSTEMS.PWSID` |

# | `PN\_VIOLATION\_ID` | `str` | Public notification violation ID |

# | `RELATED\_VIOLATION\_ID` | `str` | Original underlying violation ID |

# | `PRIMACY\_AGENCY\_CODE` | `str` | State code |

# 

# </details>

# 

# \---

# 

# \## 5 · Setup: Connect to Shared S3 from SageMaker Studio

# 

# \### 5.1 Prerequisites

# 

# \- ✅ Active AWS Academy Learner Lab session (lab must be \*\*Started\*\* — green dot)

# \- ✅ Academy lab credentials refreshed (they expire with each session)

# 

# \---

# 

# \### 5.2 Configure AWS Credentials

# 

# > \[!WARNING]

# > AWS Academy \*\*does not support persistent credential storage.\*\* You must set credentials at the start of every lab session. \*\*Never commit credentials to Git.\*\*

# 

# \*\*Option A — Environment Variables\*\* \*(recommended for notebooks)\*

# 

# Open a terminal in SageMaker Studio via `File → New → Terminal`, then run:

# 

# ```bash

# export AWS\_ACCESS\_KEY\_ID="your\_access\_key\_id"

# export AWS\_SECRET\_ACCESS\_KEY="your\_secret\_access\_key"

# export AWS\_SESSION\_TOKEN="your\_session\_token"

# export AWS\_DEFAULT\_REGION="us-east-1"

# ```

# 

# \*\*Option B — AWS CLI\*\*

# 

# ```bash

# aws configure

# \# Enter your Academy key, secret, session token, and region: us-east-1

# ```

# 

# \*\*Option C — Credentials File\*\*

# 

# Paste directly into `\~/.aws/credentials`:

# 

# ```ini

# \[default]

# aws\_access\_key\_id     = YOUR\_KEY

# aws\_secret\_access\_key = YOUR\_SECRET

# aws\_session\_token     = YOUR\_TOKEN

# ```

# 

# \---

# 

# \### 5.3 Verify Bucket Access

# 

# ```bash

# aws s3 ls s3://aai540-sdwa-team/

# ```

# 

# Expected output lists the `raw/`, `processed/`, `models/`, and `reports/` prefixes.

# 

# \---

# 

# \### 5.4 Load Data in a Notebook

# 

# Install dependencies (first time only):

# 

# ```bash

# pip install -r requirements.txt

# ```

# 

# \*\*Load a single SDWA table from S3:\*\*

# 

# ```python

# import boto3

# import pandas as pd

# from io import StringIO

# 

# BUCKET = "aai540-sdwa-team"

# REGION = "us-east-1"

# 

# s3 = boto3.client("s3", region\_name=REGION)

# 

# def load\_sdwa\_table(table\_name: str) -> pd.DataFrame:

# &#x20;   """Load a raw SDWA CSV table from the shared S3 bucket."""

# &#x20;   key = f"raw/{table\_name}.csv"

# &#x20;   obj = s3.get\_object(Bucket=BUCKET, Key=key)

# &#x20;   df  = pd.read\_csv(StringIO(obj\["Body"].read().decode("utf-8")), low\_memory=False)

# &#x20;   print(f"Loaded {table\_name}: {df.shape\[0]:,} rows × {df.shape\[1]} columns")

# &#x20;   return df

# 

# \# Example usage

# pws   = load\_sdwa\_table("SDWA\_PUB\_WATER\_SYSTEMS")

# viols = load\_sdwa\_table("SDWA\_VIOLATIONS\_ENFORCEMENT")

# sites = load\_sdwa\_table("SDWA\_SITES")

# ```

# 

# \*\*Load processed Parquet files\*\* \*(faster — preferred for modeling):\*

# 

# ```python

# import pandas as pd

# 

# s3\_path = "s3://aai540-sdwa-team/processed/features\_merged.parquet"

# df = pd.read\_parquet(s3\_path, storage\_options={"anon": False})

# ```

# 

# > \[!NOTE]

# > `pyarrow` and `s3fs` are required for Parquet reads — both included in `requirements.txt`.

# 

# \---

# 

# \## 6 · Reproducibility Workflow

# 

# \### 6.1 Execution Order

# 

# Run notebooks \*\*strictly in numerical order:\*\*

# 

# ```text

# 00\_data\_ingestion.ipynb   →  Confirms S3 access, profiles raw data

# 01\_eda.ipynb              →  Visual exploration, missing value analysis

# 02\_preprocessing.ipynb    →  Cleaning, joins, encoding, feature store

# 03\_modeling.ipynb         →  Train/val/test split, model training, tuning

# 04\_explainability.ipynb   →  SHAP values, feature importance, error analysis

# ```

# 

# \---

# 

# \### 6.2 Random Seeds

# 

# Set at the \*\*top of every notebook\*\* that involves randomness:

# 

# ```python

# RANDOM\_SEED = 42

# 

# import numpy as np

# import random

# 

# np.random.seed(RANDOM\_SEED)

# random.seed(RANDOM\_SEED)

# ```

# 

# > \[!TIP]

# > For scikit-learn and XGBoost, always pass `random\_state=RANDOM\_SEED` to every estimator.

# 

# \---

# 

# \### 6.3 Data Versioning

# 

# > \[!CAUTION]

# > Raw files in `s3://aai540-sdwa-team/raw/` are \*\*read-only.\*\* Never overwrite them.

# 

# When making breaking changes to processed outputs, version by date prefix:

# 

# ```text

# s3://aai540-sdwa-team/processed/2026-05-25/features\_merged.parquet

# ```

# 

# The `processed/` root always points to the latest stable version used in `03\_modeling.ipynb`.

# 

# \---

# 

# \### 6.4 Model Tracking

# 

# Copy this cell into every modeling run to log experiments consistently:

# 

# ```python

# import json

# 

# experiment = {

# &#x20;   "model":       "XGBoostClassifier",

# &#x20;   "version":     "v1",

# &#x20;   "date":        "2026-05-25",

# &#x20;   "author":      "your\_name",

# &#x20;   "features":    list(X\_train.columns),

# &#x20;   "hyperparams": model.get\_params(),

# &#x20;   "val\_roc\_auc": val\_roc\_auc,

# &#x20;   "val\_f1":      val\_f1,

# &#x20;   "notes":       "Baseline run, no class weighting"

# }

# 

# print(json.dumps(experiment, indent=2))

# ```

# 

# Save trained models to S3:

# 

# ```python

# import pickle, boto3

# 

# model\_key = "models/xgboost\_v1.pkl"

# s3.put\_object(Bucket=BUCKET, Key=model\_key, Body=pickle.dumps(model))

# print(f"Model saved → s3://{BUCKET}/{model\_key}")

# ```

# 

# \---

# 

# \### 6.5 Git Hygiene

# 

# ```bash

# \# Before every commit — confirm no credentials or data files are staged

# git status

# 

# git add notebooks/ src/

# git commit -m "feat: add preprocessing pipeline with LCR join"

# git push origin main

# ```

# 

# \*\*`.gitignore` must include:\*\*

# 

# ```gitignore

# \# Data (never commit raw or processed data)

# data/raw/

# data/processed/

# \*.csv

# \*.parquet

# 

# \# Models

# models/

# \*.pkl

# 

# \# Secrets \& credentials

# .env

# .aws/

# 

# \# Python artifacts

# \_\_pycache\_\_/

# .ipynb\_checkpoints/

# ```

# 

# \---

# 

# \## 7 · Team Conventions

# 

# | Convention | Rule |

# |------------|------|

# | \*\*Branch naming\*\* | `feat/<short-description>` · `fix/<short-description>` · `exp/<experiment-name>` |

# | \*\*Commit style\*\* | Conventional Commits: `feat:` · `fix:` · `docs:` · `exp:` · `refactor:` |

# | \*\*Notebook cells\*\* | Clear all outputs before committing — `Cell → All Output → Clear` |

# | \*\*Column names\*\* | Always uppercase to match EPA schema (e.g., `PWSID`, not `pwsid`) |

# | \*\*Date fields\*\* | Parse with `pd.to\_datetime(..., errors='coerce')` — many fields contain nulls |

# | \*\*S3 writes\*\* | All writes go under `processed/` or `models/` — never overwrite `raw/` |

# | \*\*Secrets\*\* | Environment variables only — never hardcode credentials |

# 

# \---

# 

# \## 8 · References

# 

# | Resource | Link |

# |----------|------|

# | EPA SDWA Datasets Download | \[echo.epa.gov](https://echo.epa.gov/tools/data-downloads/sdwa-download-summary) |

# | EPA SDWA Data Dictionary (PDF) | \[SDWA\_FileDesc\_2024.pdf](https://echo.epa.gov/system/files/SDWA\_FileDesc\_2024.pdf) |

# | Safe Drinking Water Act — 40 CFR 141–143 | \[ecfr.gov](https://www.ecfr.gov/current/title-40/chapter-I/subchapter-D) |

# | EPA Lead and Copper Rule | \[epa.gov](https://www.epa.gov/ground-water-and-drinking-water/lead-and-copper-rule) |

# | AWS boto3 S3 Documentation | \[docs.aws.amazon.com](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html) |

# | SageMaker Studio User Guide | \[docs.aws.amazon.com](https://docs.aws.amazon.com/sagemaker/latest/dg/studio.html) |

# 

# \---

# 

# <div align="center">

# 

# \*\*AAI-540 · University of San Diego · Spring 2026\*\*

# 

# \*Built with EPA open data and AWS Academy infrastructure\*

# 

# </div>



