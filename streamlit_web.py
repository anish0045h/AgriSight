import streamlit as st
import pandas as pd
import sys
import io
import matplotlib.pyplot as plt
import calendar
import numpy as np
#import joblib
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
#from xgboost import XGBRegressor
#from lightgbm import LGBMRegressor
import sqlite3

st.set_page_config(layout="wide")
st.title("Agmarknet Price Report")

# Function to load crop options (cached)
@st.cache_data
def get_crop_options():
    con = sqlite3.connect("my_database.db")
    query = "SELECT DISTINCT source_file FROM uploaded_data"
    crop_options = pd.read_sql_query(query, con)["source_file"].tolist()
    con.close()
    return crop_options

# Function to load crop data (cached)
@st.cache_data
def load_crop_data(source_file):
    con = sqlite3.connect("my_database.db")
    query = f"SELECT * FROM uploaded_data WHERE source_file = '{source_file}'"
    df = pd.read_sql_query(query, con)
    con.close()
    return df

# Get crop options
crop_options = get_crop_options()

# Clean names for display
crop_display_names = [
     name.replace("_dataset.xls", "").replace(".xls", "").replace("_", " ").capitalize()
    for name in crop_options
]

# Dropdown for crop selection
selected_crop_display = st.selectbox("Choose a crop:", crop_display_names)
selected_crop_file = crop_options[crop_display_names.index(selected_crop_display)]

# Fetch dataset (cached)
#df = load_crop_data(selected_crop_file)

# Display dataset
#st.subheader(f"Showing data for: {selected_crop_display}")
#st.dataframe(df.head(8))

# Add some spacing
#st.write("")

with st.spinner(f"Loading {selected_crop_display} data..."):
    try:
        # **FIXED: Load data using the database function**
        df = load_crop_data(selected_crop_file)

        if df.empty:
            st.error("Failed to load data or the dataset is empty.")
            st.stop()
        # Rename price columns to concise names
        try:
            df.rename(columns={
                'Min Price (Rs./Quintal)': 'Min_Price',
                'Max Price (Rs./Quintal)': 'Max_Price',
                'Modal Price (Rs./Quintal)': 'Modal_Price',
            }, inplace=True)
        except Exception:
            pass
        # Parse last column as Date (day-first) and sort
        try:
            df['Date'] = pd.to_datetime(df.iloc[:, -1], dayfirst=True, errors='coerce')
            df = df.sort_values(by='Date')
            # Date-derived features
            if 'Date' in df.columns:
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
                df['Quarter'] = df['Date'].dt.quarter
                df['Day'] = df['Date'].dt.day
                df['day_of_week'] = df['Date'].dt.dayofweek
                df['week_of_year'] = df['Date'].dt.isocalendar().week
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        except Exception:
            pass
        st.success("File loaded successfully ✅")
        st.write("### Dataset Preview")
        st.dataframe(df.head())

        # Interactive Filters
        st.write("---")
        st.subheader("Data Filters")
        st.write("Use these filters to explore specific data subsets:")

        # Create columns for filters
        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            # District filter
            if 'District Name' in df.columns:
                all_districts = sorted(df['District Name'].dropna().unique())
                selected_districts = st.multiselect(
                    "Filter by District:",
                    all_districts,
                    default=[],
                    help="Select districts to include in analysis (leave empty to show all)"
                )
            else:
                selected_districts = []
                st.info("District column not found")

            # Market filter
            market_col = 'Market Name' if 'Market Name' in df.columns else (
                'Market' if 'Market' in df.columns else None)
            if market_col is not None:
                all_markets = sorted(df[market_col].dropna().unique())
                selected_markets = st.multiselect(
                    "Filter by Market:",
                    all_markets,
                    default=[],
                    help="Select markets to include in analysis (leave empty to show all)"
                )
            else:
                selected_markets = []
                st.info("Market column not found")

        with filter_col2:
            # Variety filter
            if 'Variety' in df.columns:
                all_varieties = sorted(df['Variety'].dropna().unique())
                selected_varieties = st.multiselect(
                    "Filter by Variety:",
                    all_varieties,
                    default=[],
                    help="Select varieties to include in analysis (leave empty to show all)"
                )
            else:
                selected_varieties = []
                st.info("Variety column not found")

            # Grade filter
            if 'Grade' in df.columns:
                all_grades = sorted(df['Grade'].dropna().unique())
                selected_grades = st.multiselect(
                    "Filter by Grade:",
                    all_grades,
                    default=[],
                    help="Select grades to include in analysis (leave empty to show all)"
                )
            else:
                selected_grades = []
                st.info("Grade column not found")

        # Date range filter
        if 'Date' in df.columns:
            # Drop NaT values before getting min/max
            valid_dates = df['Date'].dropna()
            if not valid_dates.empty:
                min_date_val = valid_dates.min().date()
                max_date_val = valid_dates.max().date()
                st.write("**Date Range Filter:**")
                date_col1, date_col2 = st.columns(2)
                with date_col1:
                    min_date = st.date_input(
                        "Start Date:",
                        value=min_date_val,
                        min_value=min_date_val,
                        max_value=max_date_val
                    )
                with date_col2:
                    max_date = st.date_input(
                        "End Date:",
                        value=max_date_val,
                        min_value=min_date_val,
                        max_value=max_date_val
                    )
            else:
                min_date = None
                max_date = None
                st.info("No valid dates found in dataset.")
        else:
            min_date = None
            max_date = None

        # Apply filters
        df_filtered = df.copy()
        filter_applied = False

        if selected_districts and len(selected_districts) > 0:
            df_filtered = df_filtered[df_filtered['District Name'].isin(selected_districts)]
            filter_applied = True

        if selected_markets and len(selected_markets) > 0:
            df_filtered = df_filtered[df_filtered[market_col].isin(selected_markets)]
            filter_applied = True

        if selected_varieties and len(selected_varieties) > 0:
            df_filtered = df_filtered[df_filtered['Variety'].isin(selected_varieties)]
            filter_applied = True

        if selected_grades and len(selected_grades) > 0:
            df_filtered = df_filtered[df_filtered['Grade'].isin(selected_grades)]
            filter_applied = True

        if min_date and max_date:
            df_filtered = df_filtered[
                (df_filtered['Date'].dt.date >= min_date) &
                (df_filtered['Date'].dt.date <= max_date)
                ]
            filter_applied = True

        # Show filter results
        if filter_applied:
            st.success(f"✅ Filters applied! Showing {len(df_filtered)} records out of {len(df)} total records")
            st.write(f"**Filtered Dataset Preview:**")
            st.dataframe(df_filtered.head())
        else:
            st.info("ℹ️ No filters applied - showing all data")
            df_filtered = df

        # Clear filters button
        if st.button("Clear All Filters"):
            st.rerun()

        # Summary metrics
        st.write("---")
        st.subheader("Summary")
        # Metrics order: Variety, Grade, District, Market
        col1, col2 = st.columns(2)
        with col1:
            if 'Variety' in df_filtered.columns:
                st.metric(label="Total Varieties", value=int(df_filtered['Variety'].nunique()))
            else:
                st.metric(label="Total Varieties", value="N/A")
        with col2:
            if 'Grade' in df_filtered.columns:
                st.metric(label="Total Grades", value=int(df_filtered['Grade'].nunique()))
            else:
                st.metric(label="Total Grades", value="N/A")

        col3, col4 = st.columns(2)
        with col3:
            if 'District Name' in df_filtered.columns:
                st.metric(label="Total Districts", value=int(df_filtered['District Name'].nunique()))
            else:
                st.metric(label="Total Districts", value="N/A")
        with col4:
            market_col = 'Market Name' if 'Market Name' in df_filtered.columns else (
                'Market' if 'Market' in df_filtered.columns else None)
            if market_col is not None:
                st.metric(label="Total Markets", value=int(df_filtered[market_col].nunique()))
            else:
                st.metric(label="Total Markets", value="N/A")

        # Expanders: Variety, Grade, District, Market (with scrollbars)
        with st.expander("Show unique Variety names"):
            if 'Variety' in df_filtered.columns:
                varieties = sorted(
                    [v for v in df_filtered['Variety'].dropna().astype(str).unique() if v and v != 'nan'])
                var_df = pd.DataFrame({"Variety": varieties})
                _rows = len(var_df)
                _row_h, _hdr_h, _pad, _max_h = 28, 38, 16, 400
                _h = min(_max_h, _hdr_h + _row_h * max(_rows, 1) + _pad)
                st.dataframe(var_df, use_container_width=True, height=_h)
            else:
                st.write("Variety column not found")

        with st.expander("Show unique Grade names"):
            if 'Grade' in df_filtered.columns:
                grades = sorted([g for g in df_filtered['Grade'].dropna().astype(str).unique() if g and g != 'nan'])
                st.dataframe(pd.DataFrame({"Grade": grades}), height=200, use_container_width=True)
            else:
                st.write("Grade column not found")

        with st.expander("Show all Districts"):
            if 'District Name' in df_filtered.columns:
                districts = sorted(
                    [d for d in df_filtered['District Name'].dropna().astype(str).unique() if d and d != 'nan'])
                dist_df = pd.DataFrame({"District Name": districts})
                _rows = len(dist_df)
                _row_h, _hdr_h, _pad, _max_h = 28, 38, 16, 400
                _h = min(_max_h, _hdr_h + _row_h * max(_rows, 1) + _pad)
                st.dataframe(dist_df, use_container_width=True, height=_h)
            else:
                st.write("District column not found")

        with st.expander("Show unique Market names"):
            market_col = 'Market Name' if 'Market Name' in df_filtered.columns else (
                'Market' if 'Market' in df.columns else None)
            if market_col is not None:
                markets = sorted([m for m in df_filtered[market_col].dropna().astype(str).unique() if m and m != 'nan'])
                st.dataframe(pd.DataFrame({"Market": markets}), height=200, use_container_width=True)
            else:
                st.write("Market column not found")

        # Date coverage details
        if 'Date' in df_filtered.columns:
            st.write(f"Date range: {df_filtered['Date'].min().date()} → {df_filtered['Date'].max().date()}")
            if 'Year' in df_filtered.columns:
                st.write("Records per year:")
                counts = df_filtered['Year'].value_counts().sort_index()
                for y, c in counts.items():
                    st.write(f"- {int(y)}: {int(c)} records")
            else:
                st.write("Records per year: Year not derived")

        # Crop-specific insights
        st.write("---")
        st.subheader("🌾 Crop-Specific Insights")

        if selected_crop_display == "Tomato":
            st.write("🍅 **Tomato Market Analysis:**")
            if 'Modal_Price' in df_filtered.columns:
                avg_price = df_filtered['Modal_Price'].mean()
                price_volatility = df_filtered['Modal_Price'].std()
                st.write(f"• **Average Price:** ₹{avg_price:,.2f} per quintal")
                st.write(f"• **Price Volatility:** ₹{price_volatility:,.2f} (standard deviation)")

                if 'Month' in df_filtered.columns:
                    monthly_avg = df_filtered.groupby('Month')['Modal_Price'].mean()
                    best_month = monthly_avg.idxmax()
                    worst_month = monthly_avg.idxmin()
                    st.write(f"• **Best Month:** {calendar.month_name[best_month]} (₹{monthly_avg[best_month]:,.2f})")
                    st.write(
                        f"• **Worst Month:** {calendar.month_name[worst_month]} (₹{monthly_avg[worst_month]:,.2f})")

                if 'District Name' in df_filtered.columns:
                    top_district = df_filtered.groupby('District Name')['Modal_Price'].mean().idxmax()
                    st.write(f"• **Highest Price District:** {top_district}")

            st.write(
                "💡 **Tomato Insights:** Tomatoes are highly seasonal with significant price fluctuations. Prices typically peak during monsoon and winter months due to supply constraints.")

        elif selected_crop_display == "Coconut":
            st.write("🥥 **Coconut Market Analysis:**")
            if 'Modal_Price' in df_filtered.columns:
                avg_price = df_filtered['Modal_Price'].mean()
                price_volatility = df_filtered['Modal_Price'].std()
                st.write(f"• **Average Price:** ₹{avg_price:,.2f} per quintal")
                st.write(f"• **Price Volatility:** ₹{price_volatility:,.2f} (standard deviation)")

                if 'Month' in df_filtered.columns:
                    monthly_avg = df_filtered.groupby('Month')['Modal_Price'].mean()
                    best_month = monthly_avg.idxmax()
                    worst_month = monthly_avg.idxmin()
                    st.write(f"• **Best Month:** {calendar.month_name[best_month]} (₹{monthly_avg[best_month]:,.2f})")
                    st.write(
                        f"• **Worst Month:** {calendar.month_name[worst_month]} (₹{monthly_avg[worst_month]:,.2f})")

                if 'District Name' in df_filtered.columns:
                    top_district = df_filtered.groupby('District Name')['Modal_Price'].mean().idxmax()
                    st.write(f"• **Highest Price District:** {top_district}")

            st.write(
                "💡 **Coconut Insights:** Coconuts have relatively stable prices throughout the year with moderate seasonal variations. Coastal regions typically show higher prices due to quality and demand.")

        elif selected_crop_display == "Rice":
            st.write("🍚 **Rice Market Analysis:**")
            if 'Modal_Price' in df_filtered.columns:
                avg_price = df_filtered['Modal_Price'].mean()
                price_volatility = df_filtered['Modal_Price'].std()
                st.write(f"• **Average Price:** ₹{avg_price:,.2f} per quintal")
                st.write(f"• **Price Volatility:** ₹{price_volatility:,.2f} (standard deviation)")

                if 'Month' in df_filtered.columns:
                    monthly_avg = df_filtered.groupby('Month')['Modal_Price'].mean()
                    best_month = monthly_avg.idxmax()
                    worst_month = monthly_avg.idxmin()
                    st.write(f"• **Best Month:** {calendar.month_name[best_month]} (₹{monthly_avg[best_month]:,.2f})")
                    st.write(
                        f"• **Worst Month:** {calendar.month_name[worst_month]} (₹{monthly_avg[worst_month]:,.2f})")

                if 'District Name' in df_filtered.columns:
                    top_district = df_filtered.groupby('District Name')['Modal_Price'].mean().idxmax()
                    st.write(f"• **Highest Price District:** {top_district}")

            st.write(
                "💡 **Rice Insights:** Rice prices are influenced by harvest seasons and government policies. Prices often dip during harvest months and rise during lean periods.")

        elif selected_crop_display == "Banana":
            st.write("🍌 **Banana Market Analysis:**")
            if 'Modal_Price' in df_filtered.columns:
                avg_price = df_filtered['Modal_Price'].mean()
                price_volatility = df_filtered['Modal_Price'].std()
                st.write(f"• **Average Price:** ₹{avg_price:,.2f} per quintal")
                st.write(f"• **Price Volatility:** ₹{price_volatility:,.2f} (standard deviation)")

                if 'Month' in df_filtered.columns:
                    monthly_avg = df_filtered.groupby('Month')['Modal_Price'].mean()
                    best_month = monthly_avg.idxmax()
                    worst_month = monthly_avg.idxmin()
                    st.write(f"• **Best Month:** {calendar.month_name[best_month]} (₹{monthly_avg[best_month]:,.2f})")
                    st.write(
                        f"• **Worst Month:** {calendar.month_name[worst_month]} (₹{monthly_avg[worst_month]:,.2f})")

                if 'District Name' in df_filtered.columns:
                    top_district = df_filtered.groupby('District Name')['Modal_Price'].mean().idxmax()
                    st.write(f"• **Highest Price District:** {top_district}")

            st.write(
                "💡 **Banana Insights:** Bananas have year-round availability but prices vary based on seasonal demand and supply. Prices typically peak during festivals and dip during peak harvest seasons.")

        elif selected_crop_display == "Onion":
            st.write("🧅 **Onion Market Analysis:**")
            if 'Modal_Price' in df_filtered.columns:
                avg_price = df_filtered['Modal_Price'].mean()
                price_volatility = df_filtered['Modal_Price'].std()
                st.write(f"• **Average Price:** ₹{avg_price:,.2f} per quintal")
                st.write(f"• **Price Volatility:** ₹{price_volatility:,.2f} (standard deviation)")

                if 'Month' in df_filtered.columns:
                    monthly_avg = df_filtered.groupby('Month')['Modal_Price'].mean()
                    best_month = monthly_avg.idxmax()
                    worst_month = monthly_avg.idxmin()
                    st.write(f"• **Best Month:** {calendar.month_name[best_month]} (₹{monthly_avg[best_month]:,.2f})")
                    st.write(
                        f"• **Worst Month:** {calendar.month_name[worst_month]} (₹{monthly_avg[worst_month]:,.2f})")

                if 'District Name' in df_filtered.columns:
                    top_district = df_filtered.groupby('District Name')['Modal_Price'].mean().idxmax()
                    st.write(f"• **Highest Price District:** {top_district}")

            st.write(
                "💡 **Onion Insights:** Onions are known for high price volatility. Prices can spike dramatically during supply shortages and monsoon months, making them a critical commodity for price monitoring.")

        # Market trend analysis
        if 'Date' in df_filtered.columns and 'Modal_Price' in df_filtered.columns:
            st.write("---")
            st.subheader("📈 Market Trend Analysis")

            # Optional custom date range for trend analysis
            trend_col1, trend_col2 = st.columns(2)
            with trend_col1:
                use_custom_range = st.checkbox("Use custom date range for trend analysis", value=False)
            if use_custom_range:
                tcol1, tcol2 = st.columns(2)
                with tcol1:
                    trend_start = st.date_input(
                        "Trend start date:",
                        value=df_filtered['Date'].min().date(),
                        min_value=df_filtered['Date'].min().date(),
                        max_value=df_filtered['Date'].max().date(),
                        key="trend_start",
                    )
                with tcol2:
                    trend_end = st.date_input(
                        "Trend end date:",
                        value=df_filtered['Date'].max().date(),
                        min_value=df_filtered['Date'].min().date(),
                        max_value=df_filtered['Date'].max().date(),
                        key="trend_end",
                    )
                df_trend = df_filtered[
                    (df_filtered['Date'].dt.date >= trend_start) & (df_filtered['Date'].dt.date <= trend_end)].copy()
            else:
                df_trend = df_filtered.copy()

            # Calculate price trends
            df_sorted = df_trend.sort_values('Date')
            if len(df_sorted) > 1:
                first_date = df_sorted['Date'].iloc[0].date()
                last_date = df_sorted['Date'].iloc[-1].date()
                first_price = df_sorted['Modal_Price'].iloc[0]
                last_price = df_sorted['Modal_Price'].iloc[-1]
                price_change = last_price - first_price
                price_change_pct = (price_change / first_price * 100) if first_price != 0 else 0

                st.write(
                    f"• **Price Trend:** {'📈 Increasing' if price_change > 0 else '📉 Decreasing' if price_change < 0 else '➡️ Stable'}")
                st.write(
                    f"• **Total Change (from {first_date} to {last_date}):** ₹{price_change:,.2f} ({price_change_pct:+.2f}%)")
                st.write(f"• **Starting Price ({first_date}):** ₹{first_price:,.2f}")
                st.write(f"• **Current Price ({last_date}):** ₹{last_price:,.2f}")

                # Seasonal pattern detection
                if 'Month' in df_trend.columns:
                    seasonal_pattern = df_trend.groupby('Month')['Modal_Price'].mean()
                    if len(seasonal_pattern) >= 3:
                        st.write("• **Seasonal Pattern:** Available (check monthly charts below)")
                    else:
                        st.write("• **Seasonal Pattern:** Insufficient data for analysis")
            else:
                st.info("Not enough data points in the selected range to compute a trend.")

        # Market Segmentation Analysis (removed at user request)

        # (describe tables removed per request)

        # (info output removed per request)

        # --- Visualizations ---
        st.write("---")
        st.subheader("Visualizations")
        show_viz = st.checkbox("Show all visualizations", value=False)
        show_summaries = st.checkbox("Show chart summaries", value=True) if show_viz else False

        # 1) Average Modal Price per Year (Line)
        if show_viz and 'Year' in df_filtered.columns and 'Modal_Price' in df_filtered.columns:
            yearly_avg = df_filtered.groupby('Year')['Modal_Price'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(yearly_avg['Year'], yearly_avg['Modal_Price'], marker='o')
            ax.set_title("Average Modal Price per Year (Rs./Quintal)")
            ax.set_xlabel("Year")
            ax.set_ylabel("Price (Rs./Quintal)")
            ax.grid(True)
            st.pyplot(fig)
            if show_summaries and len(yearly_avg) > 0:
                idx_max = yearly_avg['Modal_Price'].idxmax()
                idx_min = yearly_avg['Modal_Price'].idxmin()
                best_year = int(yearly_avg.loc[idx_max, 'Year'])
                best_price = float(yearly_avg.loc[idx_max, 'Modal_Price'])
                worst_year = int(yearly_avg.loc[idx_min, 'Year'])
                worst_price = float(yearly_avg.loc[idx_min, 'Modal_Price'])
                overall_avg = float(pd.to_numeric(df_filtered['Modal_Price'], errors='coerce').mean())
                first_val = float(yearly_avg['Modal_Price'].iloc[0])
                last_val = float(yearly_avg['Modal_Price'].iloc[-1])
                pct_change = ((last_val - first_val) / first_val * 100.0) if first_val else float('nan')
                st.write(
                    f"The highest average price was in {best_year} ({best_price:,.2f}). "
                    f"The lowest was in {worst_year} ({worst_price:,.2f}). "
                    f"Overall average price: {overall_avg:,.2f}. "
                    f"Change from first to last year: {pct_change:,.2f}%"
                )
        elif show_viz:
            st.info("Need 'Year' and 'Modal_Price' columns for yearly average chart.")

        # 2) Boxplot of Modal Price by Month
        if show_viz and 'Month' in df_filtered.columns and 'Modal_Price' in df_filtered.columns:
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            try:
                df_filtered.boxplot(column='Modal_Price', by='Month', ax=ax2)
                ax2.set_title("Distribution of Modal Price by Month")
                plt.suptitle("")
                ax2.set_xlabel("Month")
                ax2.set_ylabel("Modal Price (Rs./Quintal)")
                plt.setp(ax2.get_xticklabels(), rotation=0)
                st.pyplot(fig2)
            except Exception:
                st.info("Unable to create month-wise boxplot.")
            if show_summaries:
                # Compute monthly stats for narrative: median (level) and IQR/std (stability)
                monthly_groups = df_filtered.groupby('Month')['Modal_Price']
                med = monthly_groups.median()
                q75 = monthly_groups.quantile(0.75)
                q25 = monthly_groups.quantile(0.25)
                iqr = (q75 - q25).rename('IQR')
                std = monthly_groups.std().rename('STD')
                month_stats = (
                    pd.concat([med.rename('median'), iqr, std], axis=1)
                    .dropna()
                    .sort_index()
                )
                if len(month_stats) > 0:
                    # Lowest and most stable: prioritize low median, then low IQR
                    lowest_stable = month_stats.sort_values(by=['median', 'IQR'], ascending=[True, True]).head(4)
                    # Highest and volatile: prioritize high median, then high IQR
                    highest_volatile = month_stats.sort_values(by=['median', 'IQR'], ascending=[False, False]).head(3)


                    def months_to_names(idx):
                        return [calendar.month_name[int(m)] if str(m).isdigit() else str(m) for m in idx]


                    low_names = months_to_names(lowest_stable.index)
                    high_names = months_to_names(highest_volatile.index)

                    st.write(
                        f"Prices are generally lowest and most stable in {', '.join(low_names)} (predictable months). "
                        f"In {', '.join(high_names)}, prices are usually higher and more volatile."
                    )
        elif show_viz:
            st.info("Need 'Month' and 'Modal_Price' columns for monthly boxplot.")

        # 3) Average Modal Price by Variety (Bar)
        if show_viz and 'Variety' in df_filtered.columns and 'Modal_Price' in df_filtered.columns:
            variety_avg = df_filtered.groupby('Variety')['Modal_Price'].mean().sort_values(ascending=False)
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            variety_avg.plot(kind='bar', color='orange', ax=ax3)
            ax3.set_title("Average Modal Price by Variety")
            ax3.set_xlabel("Variety")
            ax3.set_ylabel("Average Modal Price (Rs./Quintal)")
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig3)
            if show_summaries and len(variety_avg) > 0:
                all_var = variety_avg.round(2)
                st.write(f"Average price by variety ({len(all_var)} varieties):")
                var_df = all_var.to_frame(name='Avg Price')
                _rows = len(var_df)
                _row_h, _hdr_h, _pad, _max_h = 28, 38, 16, 600
                _h = min(_max_h, _hdr_h + _row_h * max(_rows, 1) + _pad)
                st.dataframe(var_df, use_container_width=True, height=_h)
        elif show_viz:
            st.info("Need 'Variety' and 'Modal_Price' columns for variety average chart.")

        # 4) Grade distribution (Pie)
        if show_viz and 'Grade' in df_filtered.columns:
            grade_counts = df_filtered['Grade'].value_counts()
            if len(grade_counts) > 0:
                fig4, ax4 = plt.subplots(figsize=(5, 5))
                ax4.pie(grade_counts, labels=grade_counts.index, autopct='%1.1f%%', startangle=140)
                ax4.set_title("Distribution of Grades")
                st.pyplot(fig4)
                if show_summaries:
                    top_grade = grade_counts.idxmax()
                    share = grade_counts.max() / grade_counts.sum() * 100.0
                    st.write(f"Top grade: {top_grade} ({share:,.1f}% of records)")
                # Percentages table
                grade_pct = (grade_counts / grade_counts.sum() * 100.0).round(2)
                grade_df = grade_pct.rename("Percent").reset_index().rename(columns={"index": "Grade"}).sort_values(
                    by="Percent", ascending=False)
                _rows = len(grade_df)
                _row_h, _hdr_h, _pad, _max_h = 28, 38, 16, 600
                _h = min(_max_h, _hdr_h + _row_h * max(_rows, 1) + _pad)
                st.dataframe(
                    grade_df,
                    use_container_width=True,
                    height=_h,
                )
        elif show_viz:
            st.info("'Grade' column not found for grade distribution pie chart.")

        # 5) Variety distribution
        if show_viz and 'Variety' in df_filtered.columns:
            variety_counts = df_filtered['Variety'].value_counts()
            if len(variety_counts) > 0:
                fig5, ax5 = plt.subplots(figsize=(6, 6))
                ax5.pie(variety_counts, labels=variety_counts.index, autopct='%1.1f%%', startangle=140)
                ax5.set_title("Distribution of Varieties")
                st.pyplot(fig5)
                if show_summaries:
                    st.write(
                        f"Total varieties: {len(variety_counts):,}; Top variety share: {variety_counts.max() / variety_counts.sum() * 100.0:,.1f}%")
                # Percentages table
                variety_pct = (variety_counts / variety_counts.sum() * 100.0).round(2)
                variety_df = variety_pct.rename("Percent").reset_index().rename(
                    columns={"index": "Variety"}).sort_values(by="Percent", ascending=False)
                _rows = len(variety_df)
                _row_h, _hdr_h, _pad, _max_h = 28, 38, 16, 600
                _h = min(_max_h, _hdr_h + _row_h * max(_rows, 1) + _pad)
                st.dataframe(
                    variety_df,
                    use_container_width=True,
                    height=_h,
                )
        elif show_viz:
            st.info("'Variety' column not found for variety distribution pie chart.")

        # 7) Price distribution histogram
        if show_viz and 'Modal_Price' in df_filtered.columns:
            s = pd.to_numeric(df_filtered['Modal_Price'], errors='coerce').dropna()
            if len(s) > 0:
                figH, axH = plt.subplots(figsize=(8, 4))
                axH.hist(s, bins=40, color='skyblue', edgecolor='black')
                axH.set_title("Price Distribution of Modal Price")
                axH.set_xlabel("Modal Price (Rs./Quintal)")
                axH.set_ylabel("Count / Frequency")
                axH.grid(axis='y', linestyle='--', alpha=0.7)
                axH.set_xlim(0, 15000)
                st.pyplot(figH)

                # Narrative: find the bin with the most observations
                try:
                    counts, bin_edges = np.histogram(s, bins=40, range=(0, 15000))
                except Exception:
                    import numpy as _np

                    counts, bin_edges = _np.histogram(s, bins=40)
                max_idx = counts.argmax()
                low = float(bin_edges[max_idx])
                high = float(bin_edges[max_idx + 1])
                st.write(f"Most prices fall in the range ₹{low:,.0f}–₹{high:,.0f}.")
        elif show_viz:
            st.info("'Modal_Price' column not found for price distribution chart.")

        # 6) Average Modal Price by District (Bar)
        if show_viz and 'District Name' in df_filtered.columns and 'Modal_Price' in df_filtered.columns:
            district_avg = df_filtered.groupby('District Name')['Modal_Price'].mean().sort_values(ascending=False)
            if len(district_avg) > 0:
                figD, axD = plt.subplots(figsize=(10, 4))
                district_avg.plot(kind='bar', color='skyblue', edgecolor='black', ax=axD)
                axD.set_title("Average Modal Price by District")
                axD.set_xlabel("District Name")
                axD.set_ylabel("Average Modal Price (Rs./Quintal)")
                plt.setp(axD.get_xticklabels(), rotation=45, ha='right')
                axD.grid(axis='y', linestyle='--', alpha=0.7)
                st.pyplot(figD)
                if show_summaries and len(district_avg) >= 1:
                    top3 = district_avg.head(3).round(2)
                    bot3 = district_avg.tail(3).round(2)
                    top_text = ", ".join([f"{idx} ({val:,.2f})" for idx, val in top3.items()])
                    bot_text = ", ".join([f"{idx} ({val:,.2f})" for idx, val in bot3.items()])
                    st.write(
                        f"Most expensive districts: {top_text}. "
                        f"Least expensive districts: {bot_text}."
                    )
        elif show_viz:
            st.info("Need 'District Name' and 'Modal_Price' columns for district average chart.")

        # --- Summary Insights ---
        st.write("---")
        st.subheader("Summary Insights")

        # Yearly trend summary
        if 'Year' in df_filtered.columns and 'Modal_Price' in df_filtered.columns and len(df_filtered) > 0:
            yearly_avg_sum = df_filtered.groupby('Year')['Modal_Price'].mean().reset_index().sort_values('Year')
            if len(yearly_avg_sum) > 0:
                start_year = int(yearly_avg_sum['Year'].iloc[0])
                end_year = int(yearly_avg_sum['Year'].iloc[-1])
                start_avg = float(yearly_avg_sum['Modal_Price'].iloc[0])
                end_avg = float(yearly_avg_sum['Modal_Price'].iloc[-1])
                growth_pct = ((end_avg - start_avg) / start_avg * 100.0) if start_avg else float('nan')
                overall_avg_price = float(pd.to_numeric(df_filtered['Modal_Price'], errors='coerce').mean())
                high_idx = yearly_avg_sum['Modal_Price'].idxmax()
                low_idx = yearly_avg_sum['Modal_Price'].idxmin()
                high_year = int(yearly_avg_sum.loc[high_idx, 'Year'])
                high_price = float(yearly_avg_sum.loc[high_idx, 'Modal_Price'])
                low_year = int(yearly_avg_sum.loc[low_idx, 'Year'])
                low_price = float(yearly_avg_sum.loc[low_idx, 'Modal_Price'])
                st.write(
                    f"From {start_year} to {end_year}, the average price trend shows a change, with the highest average price recorded in {high_year} (₹{high_price:,.2f}) and the lowest in {low_year} (₹{low_price:,.2f}). "
                    f"This reflects an overall change of {growth_pct:,.2f}% across the period, with the overall average price standing at ₹{overall_avg_price:,.2f}."
                )

        # Seasonal trends summary
        if 'Month' in df_filtered.columns and 'Modal_Price' in df_filtered.columns:
            monthly_groups_sum = df_filtered.groupby('Month')['Modal_Price']
            med_sum = monthly_groups_sum.median()
            q75_sum = monthly_groups_sum.quantile(0.75)
            q25_sum = monthly_groups_sum.quantile(0.25)
            iqr_sum = (q75_sum - q25_sum).rename('IQR')
            month_stats_sum = (
                pd.concat([med_sum.rename('median'), iqr_sum], axis=1)
                .dropna()
                .sort_index()
            )
            if len(month_stats_sum) > 0:
                lowest_stable_sum = month_stats_sum.sort_values(by=['median', 'IQR'], ascending=[True, True]).head(4)
                highest_volatile_sum = month_stats_sum.sort_values(by=['median', 'IQR'], ascending=[False, False]).head(
                    3)


                def months_to_names_sum(idx):
                    return [calendar.month_name[int(m)] if str(m).isdigit() else str(m) for m in idx]


                low_names_sum = months_to_names_sum(lowest_stable_sum.index)
                high_names_sum = months_to_names_sum(highest_volatile_sum.index)
                st.write(
                    f"Seasonal trends highlight that prices are generally lowest and most stable during {', '.join(low_names_sum)} (predictable months). "
                    f"In contrast, {', '.join(high_names_sum)} tend to show higher and more volatile prices, indicating stronger market fluctuations."
                )

        # Grade dominance and price concentration
        if 'Modal_Price' in df_filtered.columns:
            # Price concentration range using histogram peak
            prices = pd.to_numeric(df_filtered['Modal_Price'], errors='coerce').dropna()
            common_range_text = None
            if len(prices) > 0:
                try:
                    cnts, bins = np.histogram(prices, bins=40, range=(float(prices.min()), float(prices.max())))
                except Exception:
                    cnts, bins = np.histogram(prices, bins=40)
                max_idx_c = cnts.argmax()
                low_c = float(bins[max_idx_c])
                high_c = float(bins[max_idx_c + 1])
                common_range_text = f"₹{low_c:,.0f}–₹{high_c:,.0f}"

        if 'Grade' in df_filtered.columns:
            grade_counts_sum = df_filtered['Grade'].value_counts()
            if len(grade_counts_sum) > 0:
                top_grade_name = grade_counts_sum.idxmax()
                top_grade_share = grade_counts_sum.max() / grade_counts_sum.sum() * 100.0
                if common_range_text:
                    st.write(
                        f"When analyzing by grade, the {top_grade_name} grade dominates the market ({top_grade_share:,.1f}% of records), "
                        f"with most prices concentrated within the {common_range_text} range."
                    )
                else:
                    st.write(
                        f"When analyzing by grade, the {top_grade_name} grade dominates the market ({top_grade_share:,.1f}% of records)."
                    )

        # District-wise extremes
        if 'District Name' in df_filtered.columns and 'Modal_Price' in df_filtered.columns:
            district_avg_sum = df_filtered.groupby('District Name')['Modal_Price'].mean().round(2)
            if len(district_avg_sum) > 0:
                top3_sum = district_avg_sum.sort_values(ascending=False).head(3)
                bot3_sum = district_avg_sum.sort_values(ascending=True).head(3)
                top_text = ", ".join([f"{idx} (₹{val:,.2f})" for idx, val in top3_sum.items()])
                bot_text = ", ".join([f"{idx} (₹{val:,.2f})" for idx, val in bot3_sum.items()])
                st.write(
                    f"District-wise, {top_text} stand out as the most expensive regions. "
                    f"On the other hand, the least expensive markets include {bot_text}."
                )




    except Exception as e:
        st.error(f"Error loading file: {e}")