# UpKeep

UpKeep is an early-warning system for wind turbines. It uses turbine sensor readings to predict what component temperatures should normally be and identify unusual behaviour.

The system aims to detect potential problems with components such as gearbox and generator bearings before a fault occurs. It uses green, yellow and red health statuses and provides explanations for warnings.

## Project goals

- Predict normal component temperatures using wind speed, power output and ambient temperature.
- Detect unusual temperature behaviour using prediction residuals.
- Identify abnormal patterns using an anomaly detection model.
- Measure the relationship between high-gust periods and component wear.
- Display turbine health and warnings through a Streamlit dashboard.

## Technology

- Python 3.11
- pandas and pyarrow
- LightGBM
- scikit-learn
- joblib
- Streamlit and Plotly
- python-dotenv
- pytest

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/wcs-ai-2026/UpKeep.git
cd UpKeep
```

### 2. Create a virtual environment

```bash
python3.11 -m venv .venv
```

### 3. Activate the virtual environment

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the temporary stubs

The project currently includes temporary implementations in `src/upkeep/stubs.py`. These allow team members to build and test other parts of the system before all the real components are ready.

To check that the stubs can be imported, run:

```bash
python -c "from src.upkeep.stubs import make_features, NormalModel, explain, score_reading; print('UpKeep stubs imported successfully')"
```

The stubs return sample data and fixed model predictions. They are not the final predictive models.

## Dataset

UpKeep uses the CARE to Compare wind turbine dataset.

The dataset is downloaded separately from Zenodo. Place the downloaded raw files in:

```text
data/raw/
```

The raw dataset must not be committed to GitHub.

The team also uses a smaller seed sample in `data/seed/` so development can begin without downloading the full dataset.
### How to download the dataset

1. Open the [CARE to Compare dataset on Zenodo](https://zenodo.org/records/15846963).
2. Find `CARE_To_Compare.zip` under the Files section.
3. Download the ZIP file (approximately 5.5 GB).
4. Extract the downloaded files into `data/raw/`.

Do not upload the raw dataset to GitHub.

## Project structure

```text
UpKeep/
├── README.md
├── CONTRACTS.md
├── SENSORS.md
├── FEATURES.md
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/
│   ├── seed/
│   ├── processed/
│   └── live/
├── src/
│   └── upkeep/
│       ├── stubs.py
│       ├── data/
│       ├── features/
│       ├── models/
│       ├── health/
│       ├── explain/
│       └── stream/
├── predictions/
├── eval/
├── app/
└── tests/
```

## Development guidelines

- Create a separate Git branch for each Jira ticket.
- Do not push directly to `main`.
- Submit changes through pull requests.
- Keep the `main` branch working.
- Never commit raw datasets, saved models or real `.env` files.
- Use the shared contracts when connecting different parts of the system.

## Important notes

UpKeep provides early warnings to support maintenance decisions. It does not control turbines or replace human maintenance decisions.