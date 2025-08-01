# Run Instructions for Energy Analysis Project

This guide provides the project file structure and step-by-step commands to set up and run the `project1-energy-analysis` project using `uv` and launch the Streamlit dashboard.

## Project File Structure
```
project1-energy-analysis/
├── .pytest_cache/                # Cache for pytest
├── .venv/                        # Virtual environment
├── build/                        # Build artifacts
├── config/                       # Configuration files
├── Dashboard/
│   └── main_dashboard.py         # Streamlit dashboard script
├── data/
│   ├── cache/                   # Cached data
│   ├── logs/                    # Data logs
│   ├── processed/               # Processed data (e.g., daily_*.csv, latest_historical.csv)
│   └── raw/                     # Raw NOAA/EIA data
├── logs/                        # Pipeline logs (e.g., pipeline_*.log)
├── notebooks/                   # Jupyter notebooks
├── src/
│   ├── logs/                    # Source logs
│   ├── us_weather_energy_analysis.egg-info/  # Package metadata
│   ├── visualization/
│   │   ├── __pycache__/         # Compiled Python files
│   │   ├── dashboard_config.py   # Dashboard config
│   │   ├── data_manager.py      # Data utilities
│   │   ├── pipeline_manager.py  # Pipeline utilities
│   │   ├── table_display.py     # Table visualization
│   │   └── visualization.py     # Visualization logic
│   ├── __pycache__/             # Compiled Python files
│   ├── analysis.py              # Data analysis (thresholds, correlations)
│   ├── config.py               # Project config (cities, API keys)
│   ├── data_fetcher.py         # NOAA/EIA data fetching
│   ├── data_processor.py       # Data processing
│   └── pipeline.py             # Data pipeline
├── tests/                       # Unit tests
├── __pycache__/                 # Compiled Python files
├── .env                         # API keys
├── .gitignore                   # Git ignore
├── AI_USAGE.md                  # AI usage doc
├── pyproject.toml               # Project metadata, dependencies
├── README.md                    # Project doc
├── requirements.txt             # Legacy dependencies
├── uv.lock                      # uv dependency lock
└── video_link.md                # Video link
```

## Prerequisites
1. **Install `uv`**:
   ```powershell
   Invoke-WebRequest -Uri https://astral.sh/uv/install.ps1 | Invoke-Expression
   uv --version
   ```
2. **Install Python**: Ensure Python 3.8+ (`python --version`). Download: https://www.python.org/downloads/
3. **API Keys**: Add to `.env`:
   ```
   NOAA_API_KEY=your_noaa_key
   EIA_API_KEY=your_eia_key
   ```
   Request keys: NOAA (https://www.ncdc.noaa.gov/cdo-web/token), EIA (https://www.eia.gov/opendata/register.cfm).

## Setup and Run
1. **Navigate to Project**:
   ```powershell
   cd C:\Users\HP\Desktop\BYU-pathway\pioneeracademy\project1-energy-analysis
   dir
   ```
   Confirm: `pyproject.toml`, `uv.lock`, `src/`, `Dashboard/`.

2. **Set Up Virtual Environment**:
   ```powershell
   uv venv
   .\.venv\Scripts\Activate.ps1
   uv sync
   ```
   Verify: `(us-weather-energy-analysis)` in prompt.

3. **Run Data Pipeline**:
   ```powershell
   cd src
   uv run python pipeline.py
   ```
   Outputs: `data/processed/daily_*.csv`, `latest_historical.csv`.
   Logs: `type ..\logs\pipeline_*.log`.

4. **Launch Streamlit Dashboard**:
   ```powershell
   cd ..\Dashboard
   uv run streamlit run main_dashboard.py
   ```
   Open: `http://localhost:8501`.

## Dashboard Features
- **Time Series**: Select city. View temperature (solid blue, left axis), energy (dotted red, right axis, red/green/yellow markers).
- **Geographic Map**: Cities with red (high), green (low), yellow (neutral), grey (missing) markers.
- **Sidebar**: Adjust date range, cities, seasonal thresholds, percentiles.

## Troubleshooting
- **Pipeline Errors**:
  - Check: `type logs\pipeline_*.log`.
  - Verify `.env` keys.
  - Ensure `data/raw/` has data.
- **Dashboard Issues**:
  - Confirm: `data/processed/latest_historical.csv`.
  - Check terminal for Streamlit errors.
- **No Markers/Map Colors**:
  - Check logs for “Insufficient energy data”.
  - Update `analysis.py`, `visualization.py` (ask lead).

## Visual Debugging
- **VS Code**:
  ```powershell
  code .
  ```
  Open: `data/processed/*.csv`, `logs/pipeline_*.log`.
- **Jupyter**:
  ```powershell
  cd notebooks
  uv run jupyter notebook
  ```
  Open: `http://localhost:8888`.

## Notes
- Run `uv run python pipeline.py --days 270` for longer data (if supported).
- Check `README.md` for details.
- Contact lead for API keys or errors.