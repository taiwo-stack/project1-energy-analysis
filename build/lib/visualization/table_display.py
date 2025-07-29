"""Historical data table display component."""
import pandas as pd
import streamlit as st
from datetime import datetime
from loguru import logger


class HistoricalTableDisplay:
    """Handles the display of historical data tables."""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
    
    def show_historical_data_table(self, df: pd.DataFrame, show_historical_table: bool):
        """Display enhanced historical data table with collapsible interface."""
        if not show_historical_table or df.empty:
            return
        
        # Prepare data for table
        table_data = self.data_manager.prepare_historical_table_data(df)
        
        if table_data.empty:
            st.warning("⚠️ No historical data available for table display")
            return
        
        # Create collapsible section with custom styling
        st.markdown("---")
        
        # Custom CSS for the table
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
        
        # Create expandable section
        with st.expander("📋 **Historical Data Explorer** - Detailed Records & Analysis", expanded=False):
            # Add container with custom styling
            st.markdown('<div class="historical-table-container">', unsafe_allow_html=True)
            
            # Table header
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
            
            # Apply search filter
            filtered_table = table_data.copy()
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
                    # Convert to numeric for proper sorting
                    sort_values = pd.to_numeric(filtered_table[sort_by].str.replace(',', ''), errors='coerce')
                    filtered_table = filtered_table.iloc[sort_values.sort_values(ascending=ascending).index]
                else:
                    filtered_table = filtered_table.sort_values(sort_by, ascending=ascending)
            
            # Display record count after filtering
            if len(filtered_table) != len(table_data):
                st.info(f"📊 Showing {len(filtered_table):,} of {len(table_data):,} records (filtered)")
            
            # Display the table with fixed height and scrolling
            if not filtered_table.empty:
                # Configure table display options
                st.markdown("### 📋 Data Table")
                st.markdown("*Scroll within the table to view all records*")
                
                # Display table with pagination-like behavior
                st.dataframe(
                    filtered_table,
                    use_container_width=True,
                    hide_index=True,
                    height=400,  # Fixed height for scrolling (approximately 14 rows)
                    column_config={
                        '📅 Date': st.column_config.TextColumn('📅 Date', width='medium'),
                        '🏙️ City': st.column_config.TextColumn('🏙️ City', width='medium'),
                        '🌡️ Temp (°F)': st.column_config.TextColumn('🌡️ Temp (°F)', width='small'),
                        '⚡ Energy (MWh)': st.column_config.TextColumn('⚡ Energy (MWh)', width='medium'),
                        '📊 Day': st.column_config.TextColumn('📊 Day', width='small'),
                        '📈 Weekend': st.column_config.TextColumn('📈 Weekend', width='small'),
                    }
                )
                
                # Add download option
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
                    st.markdown("#### 📊 Quick Statistics")
                    
                    # Extract numeric values for statistics - handle NaN values properly
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
            
            else:
                st.warning("⚠️ No records match your search criteria. Try adjusting your filters.")


class SummaryMetrics:
    """Handles summary metrics display."""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
    
    def show_summary_metrics(self, df: pd.DataFrame, show_last_day_change: bool):
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
            last_day_changes = self.data_manager.calculate_last_day_change(df)
            if last_day_changes:
                st.markdown("### 📈 Recent Energy Changes")
                
                # Show top 5 changes
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