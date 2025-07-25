AI Usage
Tools Used

Grok 3 (xAI): Code generation, debugging, prompt refinement, and documentation.

Effective Prompts

"Generate a Python pipeline with exponential backoff for NOAA and EIA APIs, including logging with loguru."
"Create an advanced Streamlit dashboard with Plotly, including a mapbox geographic map, dual-axis time series with weekend shading, correlation scatter with regression, and heatmap."
"Write a production-ready README.md with cron scheduling and environment variable instructions."
"Add KPI cards and data export functionality to a Streamlit dashboard."

Mistakes Fixed

Issue: Incorrect NOAA API endpoint (/v1 instead of /v2).
Fix: Updated to /v2 after checking NOAA documentation.
Lesson: Always verify API endpoints against official sources.


Issue: Missing hover data in dashboard visualizations.
Fix: Added hover_data and hovermode='x unified' in Plotly charts.
Lesson: Specify interactive features in prompts.


Issue: Incomplete error handling in data fetching.
Fix: Added try-except blocks and backoff retries.
Lesson: Include error handling requirements in prompts.



Time Saved

Approximately 8-10 hours on pipeline development, dashboard design, and documentation.
