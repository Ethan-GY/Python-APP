import os
import subprocess
import sys

def install_from_requirements():
    requirements_file = 'requirements.txt'
    if os.path.exists(requirements_file):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            print("Suceess")
        except subprocess.CalledProcessError as e:
            print(f"Failure: {e}")

install_from_requirements()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

def load_and_preprocess_data():
    """Load and preprocess the energy consumption dataset"""
    try:
        df = pd.read_csv('World Energy Consumption New.csv')
        
        # Display dataset info
        st.sidebar.write(f"Dataset shape: {df.shape}")
        st.sidebar.write(f"Countries: {df['country'].nunique()}")
        st.sidebar.write(f"Years: {df['year'].min()} - {df['year'].max()}")
        
        # Filter countries with sufficient data
        countries_with_gdp = df.groupby('country')['gdp'].count()
        valid_countries = countries_with_gdp[countries_with_gdp >= 5].index.tolist()
        
        # Filter dataset
        df_filtered = df[df['country'].isin(valid_countries)].copy()
        
        # Convert data types
        numeric_columns = ['gdp', 'population', 'fossil_share_energy', 'renewables_share_energy', 
                         'coal_share_energy', 'gas_share_energy', 'oil_share_energy',
                         'coal_consumption', 'gas_consumption', 'oil_consumption',
                         'renewables_consumption', 'primary_energy_consumption',
                         'hydro_consumption', 'solar_consumption', 'wind_consumption',
                         'nuclear_consumption', 'biofuel_consumption']
        
        for col in numeric_columns:
            if col in df_filtered.columns:
                df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
        
        df_filtered['year'] = pd.to_numeric(df_filtered['year'], errors='coerce')
        
        # Calculate additional metrics
        df_filtered['fossil_share'] = pd.to_numeric(df_filtered['fossil_share_energy'], errors='coerce')
        df_filtered['renewables_share'] = pd.to_numeric(df_filtered['renewables_share_energy'], errors='coerce')
        df_filtered['non_fossil_share'] = 100 - df_filtered['fossil_share']
        
        # Calculate total energy consumption by sector if not available
        if 'primary_energy_consumption' not in df_filtered.columns:
            energy_columns = ['coal_consumption', 'gas_consumption', 'oil_consumption', 
                            'renewables_consumption', 'nuclear_consumption']
            df_filtered['primary_energy_consumption'] = df_filtered[energy_columns].sum(axis=1, skipna=True)
        
        # Calculate energy intensity (energy consumption per GDP)
        df_filtered['energy_intensity'] = df_filtered['primary_energy_consumption'] / (df_filtered['gdp'] / 1e9)  # GDP in billions
        
        return df_filtered
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def create_gdp_energy_sector_analysis(df):
    """Create comprehensive GDP vs energy consumption by sector analysis"""
    st.header("📊 GDP vs Energy Consumption by Sector")
    
    # Get countries with good data coverage
    country_coverage = df.groupby('country').agg({
        'gdp': 'count',
        'primary_energy_consumption': 'count',
        'coal_consumption': 'count',
        'gas_consumption': 'count',
        'oil_consumption': 'count'
    }).mean(axis=1)
    
    top_countries = country_coverage.nlargest(20).index.tolist()
    
    # Create multiple visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overall Trends", 
        "Sector Analysis", 
        "Country Comparison", 
        "Correlation Analysis"
    ])
    
    with tab1:
        create_overall_trends(df, top_countries)
    
    with tab2:
        create_sector_analysis(df, top_countries)
    
    with tab3:
        create_country_comparison(df, top_countries)
    
    with tab4:
        create_correlation_analysis(df)

def create_overall_trends(df, countries):
    """Create overall GDP and energy consumption trends"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. GDP vs Total Energy Consumption
    ax = axes[0, 0]
    for country in countries[:8]:
        country_data = df[df['country'] == country].dropna(subset=['gdp', 'primary_energy_consumption'])
        if len(country_data) > 5:
            # Normalize for comparison
            gdp_norm = (country_data['gdp'] / country_data['gdp'].max()) * 100
            energy_norm = (country_data['primary_energy_consumption'] / country_data['primary_energy_consumption'].max()) * 100
            
            ax.plot(country_data['year'], gdp_norm, 
                   label=f'{country} - GDP', linewidth=2, alpha=0.8)
            ax.plot(country_data['year'], energy_norm, 
                   label=f'{country} - Energy', linewidth=2, linestyle='--', alpha=0.8)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Normalized Value (%)')
    ax.set_title('GDP vs Total Energy Consumption (Normalized)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. Energy Intensity Trends
    ax = axes[0, 1]
    for country in countries[:8]:
        country_data = df[df['country'] == country].dropna(subset=['gdp', 'primary_energy_consumption'])
        if len(country_data) > 5:
            if 'energy_intensity' not in country_data.columns:
                country_data['energy_intensity'] = country_data['primary_energy_consumption'] / (country_data['gdp'] / 1e9)
            
            ax.plot(country_data['year'], country_data['energy_intensity'], 
                   marker='o', markersize=3, linewidth=2, label=country)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Energy Intensity (Energy/GDP)')
    ax.set_title('Energy Intensity Trends')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. Sector Consumption Composition (Latest Year)
    ax = axes[1, 0]
    latest_year = df['year'].max()
    latest_data = df[df['year'] == latest_year].dropna(
        subset=['coal_consumption', 'gas_consumption', 'oil_consumption', 'renewables_consumption']
    )
    
    if not latest_data.empty:
        # Select top countries by GDP
        latest_data = latest_data.nlargest(8, 'gdp')
        
        sectors = ['Coal', 'Gas', 'Oil', 'Renewables']
        consumption_data = []
        
        for country in latest_data['country']:
            country_row = latest_data[latest_data['country'] == country].iloc[0]
            row_data = [country]
            total_energy = 0
            
            for sector in sectors:
                consumption = country_row.get(f'{sector.lower()}_consumption', 0)
                if pd.notna(consumption):
                    row_data.append(consumption)
                    total_energy += consumption
                else:
                    row_data.append(0)
            
            # Calculate percentages
            if total_energy > 0:
                percentages = [x/total_energy*100 for x in row_data[1:]]
                consumption_data.append([country] + percentages)
        
        if consumption_data:
            consumption_df = pd.DataFrame(consumption_data, columns=['Country'] + sectors)
            consumption_df.set_index('Country', inplace=True)
            
            consumption_df.plot(kind='bar', stacked=True, ax=ax, alpha=0.8)
            ax.set_xlabel('Country')
            ax.set_ylabel('Energy Consumption Share (%)')
            ax.set_title(f'Energy Consumption by Sector ({latest_year})')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)
    
    # 4. GDP Growth vs Energy Consumption Growth
    ax = axes[1, 1]
    growth_data = []
    
    for country in countries[:15]:
        country_data = df[df['country'] == country].dropna(subset=['gdp', 'primary_energy_consumption'])
        if len(country_data) > 10:
            country_data = country_data.sort_values('year')
            
            # Calculate compound annual growth rates
            gdp_cagr = (country_data['gdp'].iloc[-1] / country_data['gdp'].iloc[0]) ** (1/len(country_data)) - 1
            energy_cagr = (country_data['primary_energy_consumption'].iloc[-1] / country_data['primary_energy_consumption'].iloc[0]) ** (1/len(country_data)) - 1
            
            growth_data.append({
                'country': country,
                'gdp_growth': gdp_cagr * 100,
                'energy_growth': energy_cagr * 100
            })
    
    if growth_data:
        growth_df = pd.DataFrame(growth_data)
        
        ax.scatter(growth_df['gdp_growth'], growth_df['energy_growth'], alpha=0.6, s=60)
        
        # Add country labels
        for _, row in growth_df.iterrows():
            ax.annotate(row['country'], 
                       (row['gdp_growth'], row['energy_growth']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add trend line
        if len(growth_df) > 2:
            z = np.polyfit(growth_df['gdp_growth'], growth_df['energy_growth'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(growth_df['gdp_growth'].min(), growth_df['gdp_growth'].max(), 100)
            ax.plot(x_range, p(x_range), 'r--', alpha=0.8, label='Trend')
        
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel('GDP Annual Growth Rate (%)')
        ax.set_ylabel('Energy Consumption Annual Growth Rate (%)')
        ax.set_title('GDP Growth vs Energy Consumption Growth')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

def create_sector_analysis(df, countries):
    """Create detailed sector-by-sector analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define sectors to analyze
    sectors = [
        ('coal_consumption', 'Coal', 'red'),
        ('gas_consumption', 'Natural Gas', 'blue'), 
        ('oil_consumption', 'Oil', 'green'),
        ('renewables_consumption', 'Renewables', 'orange')
    ]
    
    # 1. Sector Consumption Trends
    ax = axes[0, 0]
    for sector_col, sector_name, color in sectors:
        # Calculate global total for this sector
        sector_totals = df.groupby('year')[sector_col].sum().reset_index()
        if len(sector_totals) > 5:
            ax.plot(sector_totals['year'], sector_totals[sector_col], 
                   color=color, linewidth=2, label=sector_name)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Consumption')
    ax.set_title('Global Energy Consumption by Sector')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Sector Share Evolution
    ax = axes[0, 1]
    
    # Calculate total energy by year
    total_energy_by_year = df.groupby('year')[[
        'coal_consumption', 'gas_consumption', 'oil_consumption', 'renewables_consumption'
    ]].sum()
    
    # Calculate shares
    for sector_col, sector_name, color in sectors:
        if sector_col in total_energy_by_year.columns:
            sector_share = (total_energy_by_year[sector_col] / total_energy_by_year.sum(axis=1)) * 100
            ax.plot(sector_share.index, sector_share.values, 
                   color=color, linewidth=2, label=sector_name)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Market Share (%)')
    ax.set_title('Energy Sector Market Shares Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. GDP vs Sector Consumption Elasticity
    ax = axes[1, 0]
    elasticity_data = []
    
    for country in countries[:10]:
        country_data = df[df['country'] == country].dropna(subset=['gdp', 'coal_consumption', 'gas_consumption', 'oil_consumption'])
        if len(country_data) > 8:
            # Calculate elasticities for each sector
            for sector_col, sector_name, _ in sectors[:3]:  # Fossil fuels only
                if sector_col in country_data.columns:
                    gdp_growth = country_data['gdp'].pct_change().mean() * 100
                    sector_growth = country_data[sector_col].pct_change().mean() * 100
                    
                    if gdp_growth != 0:
                        elasticity = sector_growth / gdp_growth
                        elasticity_data.append({
                            'country': country,
                            'sector': sector_name,
                            'elasticity': elasticity
                        })
    
    if elasticity_data:
        elasticity_df = pd.DataFrame(elasticity_data)
        
        # Pivot for grouped bar chart
        pivot_df = elasticity_df.pivot(index='country', columns='sector', values='elasticity')
        
        if not pivot_df.empty:
            pivot_df.plot(kind='bar', ax=ax, alpha=0.8)
            ax.set_xlabel('Country')
            ax.set_ylabel('Elasticity (Sector Growth / GDP Growth)')
            ax.set_title('Energy Consumption Elasticity by Sector')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Unit Elasticity')
            plt.xticks(rotation=45)
    
    # 4. Sector Transition Patterns
    ax = axes[1, 1]
    
    # Analyze how countries transition between energy sources
    transition_data = []
    
    for country in countries[:8]:
        country_data = df[df['country'] == country].dropna(
            subset=['coal_consumption', 'gas_consumption', 'oil_consumption', 'renewables_consumption']
        )
        if len(country_data) > 10:
            early_period = country_data[country_data['year'] <= country_data['year'].median()]
            late_period = country_data[country_data['year'] > country_data['year'].median()]
            
            if len(early_period) > 0 and len(late_period) > 0:
                early_mix = early_period[['coal_consumption', 'gas_consumption', 'oil_consumption', 'renewables_consumption']].mean()
                late_mix = late_period[['coal_consumption', 'gas_consumption', 'oil_consumption', 'renewables_consumption']].mean()
                
                # Calculate transition: reduction in coal, increase in gas/renewables
                coal_change = (late_mix['coal_consumption'] - early_mix['coal_consumption']) / early_mix.sum() * 100
                renewable_change = (late_mix['renewables_consumption'] - early_mix['renewables_consumption']) / early_mix.sum() * 100
                
                transition_data.append({
                    'country': country,
                    'coal_change': coal_change,
                    'renewable_change': renewable_change
                })
    
    if transition_data:
        transition_df = pd.DataFrame(transition_data)
        
        ax.scatter(transition_df['coal_change'], transition_df['renewable_change'], alpha=0.6, s=60)
        
        for _, row in transition_df.iterrows():
            ax.annotate(row['country'], 
                       (row['coal_change'], row['renewable_change']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel('Change in Coal Share (%)')
        ax.set_ylabel('Change in Renewable Share (%)')
        ax.set_title('Energy Transition Patterns: Coal vs Renewables')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

def create_country_comparison(df, countries):
    """Create country-level comparison of GDP and energy sectors"""
    
    # Let user select countries to compare
    selected_countries = st.multiselect(
        "Select countries to compare:",
        countries,
        default=countries[:3] if len(countries) >= 3 else countries
    )
    
    if not selected_countries:
        st.info("Please select at least one country for comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. GDP Comparison
    ax = axes[0, 0]
    for country in selected_countries:
        country_data = df[df['country'] == country].dropna(subset=['gdp'])
        if len(country_data) > 0:
            ax.plot(country_data['year'], country_data['gdp'] / 1e9, 
                   linewidth=2, label=country, marker='o', markersize=3)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('GDP (Billions USD)')
    ax.set_title('GDP Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Total Energy Consumption Comparison
    ax = axes[0, 1]
    for country in selected_countries:
        country_data = df[df['country'] == country].dropna(subset=['primary_energy_consumption'])
        if len(country_data) > 0:
            ax.plot(country_data['year'], country_data['primary_energy_consumption'], 
                   linewidth=2, label=country, marker='s', markersize=3)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Energy Consumption')
    ax.set_title('Total Energy Consumption Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Energy Mix Radar Chart (Latest Year)
    ax = axes[1, 0]
    latest_year = df['year'].max()
    
    radar_data = []
    sectors = ['Coal', 'Gas', 'Oil', 'Renewables']
    
    for country in selected_countries:
        country_data = df[(df['country'] == country) & (df['year'] == latest_year)]
        if not country_data.empty:
            row = country_data.iloc[0]
            mix_data = [country]
            total_energy = 0
            
            for sector in sectors:
                consumption = row.get(f'{sector.lower()}_consumption', 0)
                if pd.notna(consumption):
                    mix_data.append(consumption)
                    total_energy += consumption
                else:
                    mix_data.append(0)
            
            # Calculate percentages
            if total_energy > 0:
                percentages = [x/total_energy*100 for x in mix_data[1:]]
                radar_data.append([country] + percentages)
    
    if radar_data:
        radar_df = pd.DataFrame(radar_data, columns=['Country'] + sectors)
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(sectors), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, (_, row) in enumerate(radar_df.iterrows()):
            values = row[sectors].tolist()
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, linewidth=2, label=row['Country'])
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(sectors)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Share (%)')
        ax.set_title(f'Energy Mix Comparison ({latest_year})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    # 4. Sector Growth Rates
    ax = axes[1, 1]
    growth_data = []
    
    for country in selected_countries:
        country_data = df[df['country'] == country].dropna(
            subset=['coal_consumption', 'gas_consumption', 'oil_consumption', 'renewables_consumption']
        )
        if len(country_data) > 5:
            country_data = country_data.sort_values('year')
            
            for sector_col, sector_name, _ in [
                ('coal_consumption', 'Coal', 'red'),
                ('gas_consumption', 'Gas', 'blue'),
                ('oil_consumption', 'Oil', 'green'),
                ('renewables_consumption', 'Renewables', 'orange')
            ]:
                if sector_col in country_data.columns:
                    start_val = country_data[sector_col].iloc[0]
                    end_val = country_data[sector_col].iloc[-1]
                    years = country_data['year'].iloc[-1] - country_data['year'].iloc[0]
                    
                    if years > 0 and start_val > 0:
                        cagr = (end_val / start_val) ** (1/years) - 1
                        growth_data.append({
                            'country': country,
                            'sector': sector_name,
                            'growth_rate': cagr * 100
                        })
    
    if growth_data:
        growth_df = pd.DataFrame(growth_data)
        
        # Pivot for grouped bar chart
        pivot_df = growth_df.pivot(index='country', columns='sector', values='growth_rate')
        
        if not pivot_df.empty:
            pivot_df.plot(kind='bar', ax=ax, alpha=0.8)
            ax.set_xlabel('Country')
            ax.set_ylabel('Annual Growth Rate (%)')
            ax.set_title('Sector Consumption Growth Rates')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.xticks(rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)

def create_correlation_analysis(df):
    """Create correlation analysis between GDP and energy sectors"""
    
    # Calculate correlations
    correlation_data = []
    sectors = ['coal_consumption', 'gas_consumption', 'oil_consumption', 'renewables_consumption']
    
    for country in df['country'].unique():
        country_data = df[df['country'] == country].dropna(subset=['gdp'] + sectors)
        if len(country_data) > 10:
            for sector in sectors:
                corr = country_data['gdp'].corr(country_data[sector])
                if not pd.isna(corr):
                    correlation_data.append({
                        'country': country,
                        'sector': sector.replace('_consumption', '').title(),
                        'correlation': corr
                    })
    
    if not correlation_data:
        st.info("Insufficient data for correlation analysis")
        return
    
    correlation_df = pd.DataFrame(correlation_data)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Correlation heatmap by country and sector
    ax = axes[0]
    pivot_corr = correlation_df.pivot(index='country', columns='sector', values='correlation')
    
    # Select countries with most data
    if len(pivot_corr) > 15:
        pivot_corr = pivot_corr.iloc[:15]
    
    if not pivot_corr.empty:
        im = ax.imshow(pivot_corr.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(pivot_corr.columns)))
        ax.set_yticks(range(len(pivot_corr.index)))
        ax.set_xticklabels(pivot_corr.columns, rotation=45)
        ax.set_yticklabels(pivot_corr.index)
        
        # Add correlation values
        for i in range(len(pivot_corr.index)):
            for j in range(len(pivot_corr.columns)):
                ax.text(j, i, f'{pivot_corr.iloc[i, j]:.2f}', 
                       ha='center', va='center', fontsize=8)
        
        ax.set_title('GDP-Energy Sector Correlations by Country')
        plt.colorbar(im, ax=ax)
    
    # 2. Average correlation by sector
    ax = axes[1]
    sector_avg_corr = correlation_df.groupby('sector')['correlation'].mean().sort_values()
    
    bars = ax.barh(range(len(sector_avg_corr)), sector_avg_corr.values,
                  color=['red' if x>0 else 'blue' for x in sector_avg_corr.values])
    
    ax.set_yticks(range(len(sector_avg_corr)))
    ax.set_yticklabels(sector_avg_corr.index)
    ax.set_xlabel('Average Correlation Coefficient')
    ax.set_title('Average GDP Correlation by Energy Sector')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for bar, corr in zip(bars, sector_avg_corr.values):
        ax.text(bar.get_width() + (0.02 if bar.get_width() >=0 else -0.05), 
               bar.get_y() + bar.get_height()/2,
               f'{corr:.3f}', ha='left' if bar.get_width() >=0 else 'right', 
               va='center', fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display insights
    st.subheader("Correlation Insights")
    
    highest_corr_sector = sector_avg_corr.idxmax()
    lowest_corr_sector = sector_avg_corr.idxmin()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Highest Correlation Sector",
            highest_corr_sector,
            f"{sector_avg_corr[highest_corr_sector]:.3f}"
        )
    
    with col2:
        st.metric(
            "Lowest Correlation Sector", 
            lowest_corr_sector,
            f"{sector_avg_corr[lowest_corr_sector]:.3f}"
        )

def main():
    st.set_page_config(page_title="GDP & Energy Sector Analysis", page_icon="🌍", layout="wide")
    
    st.title("🌍 GDP & Energy Consumption by Sector Analysis")
    st.markdown("""
    Comprehensive analysis of the relationship between economic development (GDP) and energy consumption 
    across different sectors (coal, gas, oil, renewables).
    """)
    
    # Load data
    with st.spinner('Loading and processing data...'):
        df = load_and_preprocess_data()
    
    if df.empty:
        st.error("No data loaded. Please check the data file.")
        return
    
    # Main analysis navigation
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        [
            "GDP vs Energy Sectors", 
            "Country Comparison", 
            "Sector Transition", 
            "Full Report"
        ]
    )
    
    if analysis_type == "GDP vs Energy Sectors":
        create_gdp_energy_sector_analysis(df)
    
    elif analysis_type == "Country Comparison":
        st.header("🏛️ Country-Level Energy Sector Comparison")
        countries = sorted(df['country'].unique())
        selected_countries = st.multiselect(
            "Select countries:",
            countries,
            default=countries[:4] if len(countries) >= 4 else countries
        )
        
        if selected_countries:
            create_country_comparison(df, selected_countries)
    
    elif analysis_type == "Sector Transition":
        st.header("🔄 Energy Sector Transition Analysis")
        create_sector_analysis(df, sorted(df['country'].unique())[:20])
    
    else:  # Full Report
        st.header("📋 Comprehensive Analysis Report")
        
        # Executive Summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Key Sector Insights")
            st.markdown("""
            **Energy-GDP Relationships:**
            - Different sectors show varying correlations with economic growth
            - Renewable energy adoption patterns vary by development stage
            - Sector transitions follow distinct economic pathways
            
            **Global Trends:**
            - Shift from coal to gas and renewables in developed economies
            - Oil consumption remains stable in many economies
            - Renewable growth accelerates with economic development
            """)
        
        with col2:
            st.subheader("Policy Implications")
            st.markdown("""
            **Sector-Specific Strategies:**
            - Targeted renewable energy incentives
            - Coal phase-out roadmaps
            - Natural gas as transition fuel
            - Oil efficiency improvements
            - Cross-sector integration policies
            """)
        
        # Display comprehensive analysis
        create_gdp_energy_sector_analysis(df)
    
    # Data summary
    with st.expander("Data Summary"):
        st.write(f"**Dataset Overview:**")
        st.write(f"- Countries: {df['country'].nunique()}")
        st.write(f"- Time Period: {df['year'].min()} - {df['year'].max()}")
        st.write(f"- Total Observations: {len(df)}")
        
        # Sector data availability
        sectors = ['coal_consumption', 'gas_consumption', 'oil_consumption', 'renewables_consumption']
        sector_stats = []
        
        for sector in sectors:
            available = df[sector].notna().sum()
            sector_stats.append({
                'Sector': sector.replace('_consumption', '').title(),
                'Available Data Points': available,
                'Coverage (%)': f"{(available/len(df)*100):.1f}%"
            })
        
        st.table(pd.DataFrame(sector_stats))

if __name__ == "__main__":
    main()