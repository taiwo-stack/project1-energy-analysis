US Weather + Energy Analysis Pipeline
Overview
This project provides a production-grade pipeline for analyzing correlations between weather and energy consumption across five US cities (New York, Chicago, Houston, Phoenix, Seattle). It fetches daily data from NOAA (weather) and EIA (energy) APIs, performs rigorous data quality checks, and presents insights via an advanced Streamlit dashboard. Designed for energy companies, it supports demand forecasting to minimize costs and prevent outages.
Features

Data Pipeline: Fetches and processes daily and historical (90-day) data with exponential backoff for reliability.
Quality Assurance: Checks for missing data, outliers, and freshness, with detailed reports.
Advanced Dashboard:
Interactive US map with energy usage and temperature.
Dual-axis time series with weekend shading and unified hover.
Correlation scatter plot with regression line and statistical annotations.
Heatmap of energy usage by temperature and day of week.
KPI cards for average temperature, energy, and data freshness.
Data export functionality.



Setup

Install Dependencies:uv sync


Register for APIs:
NOAA: https://www.ncdc.noaa.gov/cdo-web/token
EIA: https://www.eia.gov/opendata/register.php
Set environment variables:export NOAA_API_TOKEN="your-noaa-token"
export EIA_API_TOKEN="your-eia-token"

On Windows:setx NOAA_API_TOKEN "your-noaa-token"
setx EIA_API_TOKEN "your-eia-token"




Directory Structure:project1-energy-analysis/
├── README.md
├── AI_USAGE.md
├── pyproject.toml
├── .gitignore
├── config/
│   └── config.yaml
├── src/
│   ├── config.py
│   ├── data_fetcher.py
│   ├── data_processor.py
│   ├── analysis.py
│   └── pipeline.py
├── dashboards/
│   └── app.py
├── logs/
│   └── pipeline.log
├── data/
│   ├── raw/
│   └── processed/
│       └── quality/
├── notebooks/
│   └── exploration.ipynb
├── tests/
│   └── test_pipeline.py
└── video_link.md



Usage

Run Pipeline:uv run python src/pipeline.py


Fetches daily and historical data, performs quality checks, and saves to data/processed/.


View Dashboard:uv run streamlit run dashboards/app.py


Access at http://localhost:8501.


Schedule Daily Runs (cron):0 2 * * * /path/to/uv run python /path/to/project/src/pipeline.py


Run Tests:uv run pytest tests/test_pipeline.py



Data Sources

NOAA: https://www.ncei.noaa.gov/cdo-web/api/v2
EIA: https://api.eia.gov/v2/electricity/
Backup: NOAA datasets, EIA data browser

Quality Checks

Missing values: >10% flagged.
Outliers: Temperatures >130°F or <-50°F, negative energy consumption.
Freshness: Data older than 24 hours flagged.

Dashboard Highlights

Geographic Overview: Mapbox-powered map with dynamic sizing and color scales.
Time Series: Dual-axis chart with weekend shading and unified hover for insights.
Correlation: Scatter plot with regression line and R²/correlation metrics.
Heatmap: Energy usage patterns by temperature and day of week.
Export: Download filtered data as CSV.

Troubleshooting

Check logs/pipeline.log for errors.
Verify environment variables with echo $NOAA_API_TOKEN.
Ensure API keys are valid and quotas are not exceeded.
