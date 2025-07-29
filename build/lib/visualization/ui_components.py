"""UI components and layout management for the dashboard."""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
from loguru import logger


class UIComponents:
    """Handles all UI components and layout for the dashboard."""
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="US Weather and Energy Dashboard",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def render_main_header(self):
        """Render the main dashboard header."""
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0; margin-bottom: 2rem; 
                    background: linear-gradient(90deg, #1f77b4, #ff7f0e); 
                    border-radius: 10px; color: white;'>
            <h1 style='margin: 0; font-size: 2.5rem;'>🏙️ US Weather & Energy Dashboard</h1>
            <p style='margin: 0; font-size: 1.1rem; opacity: 0.9;'>Real-time analysis of energy consumption patterns across major US cities</p>
        </div>
        """, unsafe_allow_html=True)
    
    def setup_sidebar_controls(self, config) -> Dict[str, Any]:
        """Setup sidebar controls and return their values."""
        with st.sidebar:
            st.markdown("# 🎛️ Dashboard Controls")
            st.markdown("---")
            
            # Date Selection Section
            st.markdown("### 📅 Analysis Period")
            date_preset = st.selectbox(
                "Quick Select",
                options=["Last 7 days", "Last 14 days", "Last 30 days", "Last 60 days", "Last 90 days", "Custom Range"],
                index=2,
                help="Choose your analysis time period"
            )
            
            # Calculate date range
            date_range = self._calculate_date_range(date_preset, config.max_fetch_days if hasattr(config, 'max_fetch_days') else 90)
            
            st.markdown("---")
            
            # City Selection Section
            st.markdown("### 🏙️ City Selection")
            selected_cities = self._setup_city_selection(config)
            
            st.markdown("---")
            
            # Display Options Section
            st.markdown("### ⚙️ Display Options")
            show_correlations = st.checkbox("📊 Show Correlations", value=True)
            show_last_day_change = st.checkbox("📈 Last Day Change", value=True)
            show_quality = st.checkbox("🔍 Data Quality", value=False)
            show_historical_table = st.checkbox("📋 Historical Data Table", value=True, 
                                               help="Show detailed historical data table")
            
            st.markdown("---")
            
            # Advanced Settings Section
            st.markdown("### 🔧 Advanced Settings")
            lookback_days = st.slider("Historical Baseline (days)", 14, 90, 30)
            recent_days = st.slider("Recent Period (days)", 1, 14, 7)
            
            st.markdown("---")
            
            # Info Section
            st.markdown("### ℹ️ About")
            st.info("This dashboard analyzes energy consumption patterns across major US cities, showing correlations with weather data and usage trends.")
            
            # Current selections summary
            self._show_selection_summary(selected_cities, date_range)
        
        return {
            'date_range': date_range,
            'selected_cities': selected_cities,
            'show_correlations': show_correlations,
            'show_last_day_change': show_last_day_change,
            'show_quality': show_quality,
            'show_historical_table': show_historical_table,
            'lookback_days': lookback_days,
            'recent_days': recent_days
        }
    
    def _calculate_date_range(self, date_preset: str, max_fetch_days: int):
        """Calculate date range based on preset selection."""
        end_date = datetime.now().date() - timedelta(days=2)
        
        if date_preset == "Last 7 days":
            start_date = end_date - timedelta(days=7)
            return (start_date, end_date)
        elif date_preset == "Last 14 days":
            start_date = end_date - timedelta(days=14)
            return (start_date, end_date)
        elif date_preset == "Last 30 days":
            start_date = end_date - timedelta(days=30)
            return (start_date, end_date)
        elif date_preset == "Last 60 days":
            start_date = end_date - timedelta(days=60)
            return (start_date, end_date)
        elif date_preset == "Last 90 days":
            start_date = end_date - timedelta(days=90)
            return (start_date, end_date)
        else:  # Custom Range
            default_start = end_date - timedelta(days=30)
            return st.date_input(
                "Custom Date Range",
                value=(default_start, end_date),
                min_value=end_date - timedelta(days=max_fetch_days),
                max_value=end_date,
                help="Select custom start and end dates"
            )
    
    def _setup_city_selection(self, config):
        """Setup city selection controls."""
        all_cities = [city.name for city in config.cities]
        
        # Quick selection buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎯 Select All", use_container_width=True):
                st.session_state.selected_cities = all_cities
        with col2:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.selected_cities = []
        
        # Multi-select for cities
        selected_cities = st.multiselect(
            "Choose Cities",
            options=all_cities,
            default=st.session_state.get('selected_cities', all_cities[:6]),
            help="Select cities for analysis - all visualizations will include selected cities",
            key='selected_cities'
        )
        
        if not selected_cities:
            st.warning("⚠️ Please select at least one city")
        
        return selected_cities
    
    def _show_selection_summary(self, selected_cities: List[str], date_range):
        """Show current selection summary."""
        if selected_cities:
            st.markdown("### 📋 Current Selection")
            st.write(f"**Cities:** {len(selected_cities)} selected")
            if hasattr(date_range, '__len__') and len(date_range) == 2:
                days_selected = (date_range[1] - date_range[0]).days
                st.write(f"**Period:** {days_selected} days")
    
    def show_summary_metrics(self, df: pd.DataFrame, show_last_day_change: bool, calculate_last_day_change_func):
        """Show key summary metrics at the top."""
        if df.empty:
            return
        
        # Calculate basic stats
        total_cities = len(df['city'].unique())
        date_range_days = (df['date'].max() - df['date'].min()).days
        avg_energy = df['energy_demand'].mean()
        avg_temp = df['temperature_avg'].mean() if 'temperature_avg' in df.columns else 0
        
        # Show metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🏙️ Cities Analyzed",
                value=total_cities,
                help="Number of cities in current analysis"
            )
        
        with col2:
            st.metric(
                label="📅 Data Period",
                value=f"{date_range_days} days",
                help="Number of days in analysis period"
            )
        
        with col3:
            st.metric(
                label="⚡ Avg Energy",
                value=f"{avg_energy:,.0f} MWh",
                help="Average energy demand across all cities"
            )
        
        with col4:
            st.metric(
                label="🌡️ Avg Temperature",
                value=f"{avg_temp:.1f}°F",
                help="Average temperature across all cities"
            )
        
        # Show last day changes if enabled
        if show_last_day_change:
            self._show_last_day_changes(df, calculate_last_day_change_func)
    
    def _show_last_day_changes(self, df: pd.DataFrame, calculate_last_day_change_func):
        """Display last day energy changes."""
        last_day_changes = calculate_last_day_change_func(df)
        if last_day_changes:
            st.markdown("### 📈 Recent Energy Changes")
            
            sorted_changes = sorted(
                last_day_changes.items(), 
                key=lambda x: abs(x[1]['pct_change']), 
                reverse=True
            )

            change_cols = st.columns(min(5, len(sorted_changes)))
            for idx, (city, change_data) in enumerate(sorted_changes[:5]):
                with change_cols[idx]:
                    pct_change = change_data['pct_change']
                    
                    days_gap_text = f" ({change_data['days_between']}d)" if change_data['days_between'] > 1 else ""
                    delta_text = f"{pct_change:.1f}%" if pct_change >= 0 else f"{pct_change:.1f}%"
                    
                    st.metric(
                        f"{city}{days_gap_text}",
                        f"{change_data['latest_usage']:,.0f} MWh",
                        delta=delta_text,
                        delta_color="inverse",
                        help=f"Change from {change_data['previous_date']} to {change_data['latest_date']}"
                    )

            # Expandable section for all changes
            if len(last_day_changes) > 4:
                with st.expander(f"📋 View All {len(last_day_changes)} Cities' Changes"):
                    change_df = pd.DataFrame([
                        {
                            'City': city,
                            'Latest (MWh)': f"{data['latest_usage']:,.0f}",
                            'Previous (MWh)': f"{data['previous_usage']:,.0f}",
                            'Change (%)': f"{data['pct_change']:+.1f}%",
                            'Latest Date': data['latest_date'],
                            'Previous Date': data['previous_date']
                        }
                        for city, data in sorted_changes
                    ])
                    st.dataframe(change_df, use_container_width=True, hide_index=True)
    
    def show_data_quality(self, df: pd.DataFrame, generate_quality_report_func, check_data_freshness_func):
        """Display enhanced data quality report."""
        with st.expander("🔍 Data Quality Report"):
            quality_report = generate_quality_report_func(df)
            summary = quality_report['summary']
            
            # Check data freshness
            freshness_info = check_data_freshness_func(df)
            
            # Quality metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Records", summary['total_rows'])
            with col2:
                st.metric("❌ Missing Values", summary['missing_values'])
            with col3:
                st.metric("⚠️ Outliers", summary['outliers'])
            with col4:
                # Data freshness metric with color coding
                if freshness_info['warning_level'] == 'good':
                    st.metric("🟢 Data Freshness", "Fresh", delta=f"{freshness_info['days_old']} days old")
                elif freshness_info['warning_level'] == 'warning':
                    st.metric("🟡 Data Freshness", "Stale", delta=f"{freshness_info['days_old']} days old")
                else:
                    st.metric("🔴 Data Freshness", "Very Stale", delta=f"{freshness_info['days_old']} days old")
            
            # Show detailed quality information
            self._show_quality_details(quality_report, freshness_info)
    
    def _show_quality_details(self, quality_report: Dict, freshness_info: Dict):
        """Show detailed quality information."""
        # Freshness details
        st.markdown("#### 📅 Data Freshness Analysis")
        
        if freshness_info['warning_level'] == 'good':
            st.success(f"✅ **{freshness_info['status']}**")
        elif freshness_info['warning_level'] == 'warning':
            st.warning(f"⚠️ **{freshness_info['status']}**")
        else:
            st.error(f"🚨 **{freshness_info['status']}**")
        
        # Additional freshness details
        if freshness_info['latest_date']:
            col_a, col_b = st.columns(2)
            with col_a:
                st.info(f"**Latest Data:** {freshness_info['latest_date']}")
            with col_b:
                st.info(f"**Current Date:** {datetime.now().strftime('%Y-%m-%d')}")
        
        # Issues found
        st.markdown("#### 🚨 Quality Issues")
        all_issues = []
        
        # Add freshness issues
        if not freshness_info['is_fresh']:
            if freshness_info['days_old'] and freshness_info['days_old'] > 7:
                all_issues.append(f"⏱️ Data is {freshness_info['days_old']} days old - consider refreshing data sources")
            elif freshness_info['days_old'] and freshness_info['days_old'] > 3:
                all_issues.append(f"⏱️ Data is {freshness_info['days_old']} days old - may impact real-time insights")
        
        # Add existing quality issues
        if quality_report['issues']:
            all_issues.extend(quality_report['issues'])
        
        if all_issues:
            for issue in all_issues:
                st.write(f"• {issue}")
        else:
            st.success("✅ No data quality issues detected!")
        
        # Recommendations based on freshness
        if freshness_info['days_old'] and freshness_info['days_old'] > 1:
            st.markdown("#### 💡 Insights")
            if freshness_info['days_old'] > 7:
                st.write("• 📞 **Action:** Check data pipeline and API connections")
            elif freshness_info['days_old'] > 3:
                st.write("• 📈 **Impact:** Recent trends may not be fully captured")
            else:
                st.write("• 🔄 **Stale reason:** Data is 2 days delayed to wait for data availability before fetching")
    
    def show_historical_data_table(self, df: pd.DataFrame):
        """Display enhanced historical data table with collapsible interface."""
        if df.empty:
            return
        
        # Prepare data for table
        table_data = self._prepare_historical_table_data(df)
        
        if table_data.empty:
            st.warning("⚠️ No historical data available for table display")
            return
        
        # Custom CSS for the table
        self._inject_table_css()
        
        # Create expandable section
        with st.expander("📋 **Historical Data Explorer** - Detailed Records & Analysis", expanded=False):
            self._render_table_container(table_data)
            self._render_table_controls_and_data(table_data)
    
    def _prepare_historical_table_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and format data for the historical table display."""
        try:
            if df.empty:
                return pd.DataFrame()
            
            display_df = df.copy()
            display_df = display_df.sort_values(['date', 'city'], ascending=[False, True])
            
            formatted_data = []
            for _, row in display_df.iterrows():
                formatted_row = {
                    '📅 Date': row['date'].strftime('%Y-%m-%d'),
                    '🏙️ City': row['city'],
                    '🌡️ Temp (°F)': f"{row.get('temperature_avg', row.get('temperature_max', 0)):.1f}",
                    '⚡ Energy (MWh)': f"{row['energy_demand']:,.0f}",
                    '📊 Day': row['day_of_week'],
                    '📈 Weekend': '✅ Yes' if row.get('is_weekend', False) else '❌ No',
                }
                
                # Add temperature range if available
                if 'temperature_min' in row and 'temperature_max' in row:
                    formatted_row['🌡️ Min/Max'] = f"{row['temperature_min']:.1f} / {row['temperature_max']:.1f}"
                
                # Add weather description if available
                if 'weather_description' in row and pd.notna(row['weather_description']):
                    formatted_row['🌤️ Weather'] = row['weather_description'].title()
                
                formatted_data.append(formatted_row)
            
            return pd.DataFrame(formatted_data)
            
        except Exception as e:
            logger.error(f"Failed to prepare historical table data: {str(e)}")
            return pd.DataFrame()
    
    def _inject_table_css(self):
        """Inject custom CSS for table styling."""
        st.markdown("""
        <style>
        .historical-table-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        .table-header {
            color: white;
            text-align: center;
            margin-bottom: 15px;
            font-size: 1.5rem;
            font-weight: bold;
        }
        .table-stats {
            display: flex;
            justify-content: space-around;
            margin-bottom: 15px;
            color: white;
            font-size: 0.9rem;
        }
        .stat-item {
            text-align: center;
            padding: 8px 15px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            backdrop-filter: blur(10px);
        }
        .dataframe {
            font-size: 0.85rem !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_table_container(self, table_data: pd.DataFrame):
        """Render the table container with statistics."""
        st.markdown('<div class="historical-table-container">', unsafe_allow_html=True)
        st.markdown('<div class="table-header">📊 Historical Weather & Energy Data</div>', unsafe_allow_html=True)
        
        # Summary statistics
        total_records = len(table_data)
        date_range_str = f"{table_data['📅 Date'].iloc[-1]} to {table_data['📅 Date'].iloc[0]}"
        cities_count = len(table_data['🏙️ City'].unique())
        
        st.markdown(f"""
        <div class="table-stats">
            <div class="stat-item">
                <div><strong>📊 Total Records</strong></div>
                <div>{total_records:,}</div>
            </div>
            <div class="stat-item">
                <div><strong>🏙️ Cities</strong></div>
                <div>{cities_count}</div>
            </div>
            <div class="stat-item">
                <div><strong>📅 Date Range</strong></div>
                <div>{date_range_str}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_table_controls_and_data(self, table_data: pd.DataFrame):
        """Render table controls and data display."""
        # Add search and filter options
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_term = st.text_input(
                "🔍 Search cities or weather conditions:",
                placeholder="Type to filter records...",
                help="Search by city name or weather description"
            )
        
        with col2:
            sort_by = st.selectbox(
                "📊 Sort by:",
                options=['📅 Date', '🏙️ City', '⚡ Energy (MWh)', '🌡️ Temp (°F)'],
                index=0,
                help="Choose sorting column"
            )
        
        with col3:
            sort_order = st.selectbox(
                "🔄 Order:",
                options=['Descending', 'Ascending'],
                index=0,
                help="Sort order"
            )
        
        # Apply filters and display table
        filtered_table = self._apply_table_filters(table_data, search_term, sort_by, sort_order)
        self._display_filtered_table(filtered_table, table_data)
    
    def _apply_table_filters(self, table_data: pd.DataFrame, search_term: str, sort_by: str, sort_order: str) -> pd.DataFrame:
        """Apply search and sort filters to table data."""
        filtered_table = table_data.copy()
        
        # Apply search filter
        if search_term:
            mask = (
                filtered_table['🏙️ City'].str.contains(search_term, case=False, na=False) |
                (filtered_table.get('🌤️ Weather', pd.Series(dtype='object')).str.contains(search_term, case=False, na=False))
            )
            filtered_table = filtered_table[mask]
        
        # Apply sorting
        if sort_by in filtered_table.columns:
            ascending = (sort_order == 'Ascending')
            if sort_by in ['⚡ Energy (MWh)', '🌡️ Temp (°F)']:
                sort_values = pd.to_numeric(filtered_table[sort_by].str.replace(',', ''), errors='coerce')
                filtered_table = filtered_table.iloc[sort_values.sort_values(ascending=ascending).index]
            else:
                filtered_table = filtered_table.sort_values(sort_by, ascending=ascending)
        
        return filtered_table
    
    def _display_filtered_table(self, filtered_table: pd.DataFrame, original_table: pd.DataFrame):
        """Display the filtered table with controls."""
        # Display record count after filtering
        if len(filtered_table) != len(original_table):
            st.info(f"📊 Showing {len(filtered_table):,} of {len(original_table):,} records (filtered)")
        
        # Display the table
        if not filtered_table.empty:
            st.markdown("### 📋 Data Table")
            st.markdown("*Scroll within the table to view all records*")
            
            st.dataframe(
                filtered_table,
                use_container_width=True,
                hide_index=True,
                height=400,
                column_config={
                    '📅 Date': st.column_config.TextColumn('📅 Date', width='medium'),
                    '🏙️ City': st.column_config.TextColumn('🏙️ City', width='medium'),
                    '🌡️ Temp (°F)': st.column_config.TextColumn('🌡️ Temp (°F)', width='small'),
                    '⚡ Energy (MWh)': st.column_config.TextColumn('⚡ Energy (MWh)', width='medium'),
                    '📊 Day': st.column_config.TextColumn('📊 Day', width='small'),
                    '📈 Weekend': st.column_config.TextColumn('📈 Weekend', width='small'),
                }
            )
            
            # Add download and stats options
            self._add_table_download_options(filtered_table)
        else:
            st.warning("⚠️ No records match your search criteria. Try adjusting your filters.")
    
    def _add_table_download_options(self, filtered_table: pd.DataFrame):
        """Add download options and quick stats for the table."""
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            # Export to CSV
            csv_data = filtered_table.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"historical_energy_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download filtered data as CSV file"
            )
        
        with col2:
            # Quick stats toggle
            if st.button("📊 Quick Stats", help="Show summary statistics"):
                st.session_state.show_quick_stats = not st.session_state.get('show_quick_stats', False)
        
        with col3:
            st.markdown(f"*📊 Table showing {len(filtered_table):,} records with 400px height for scrolling*")
        
        # Show quick stats if toggled
        if st.session_state.get('show_quick_stats', False):
            self._show_table_quick_stats(filtered_table)
    
    def _show_table_quick_stats(self, filtered_table: pd.DataFrame):
        """Show quick statistics for the filtered table."""
        st.markdown("#### 📊 Quick Statistics")
        
        # Extract numeric values for statistics
        energy_values = pd.to_numeric(filtered_table['⚡ Energy (MWh)'].str.replace(',', ''), errors='coerce').dropna()
        temp_values = pd.to_numeric(filtered_table['🌡️ Temp (°F)'], errors='coerce').dropna()
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            avg_energy = energy_values.mean() if len(energy_values) > 0 else 0
            st.metric("⚡ Avg Energy", f"{avg_energy:,.0f} MWh")
        with stat_col2:
            avg_temp = temp_values.mean() if len(temp_values) > 0 else 0
            st.metric("🌡️ Avg Temp", f"{avg_temp:.1f}°F")
        with stat_col3:
            max_energy = energy_values.max() if len(energy_values) > 0 else 0
            st.metric("📈 Max Energy", f"{max_energy:,.0f} MWh")
        with stat_col4:
            max_temp = temp_values.max() if len(temp_values) > 0 else 0
            st.metric("🌡️ Max Temp", f"{max_temp:.1f}°F")