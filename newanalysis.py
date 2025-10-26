import os
import subprocess
import sys

def install_from_requirements():
    requirements_file = 'requirements.txt'
    if os.path.exists(requirements_file):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            print("Success")
        except subprocess.CalledProcessError as e:
            print(f"Failure: {e}")

install_from_requirements()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
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
        
        # Filter countries with sufficient data
        countries_with_data = df.groupby('country')['gdp'].count()
        valid_countries = countries_with_data[countries_with_data >= 10].index.tolist()
        df_filtered = df[df['country'].isin(valid_countries)].copy()
        
        # Convert data types
        numeric_columns = ['gdp', 'population', 'fossil_share_energy', 'renewables_share_energy']
        for col in numeric_columns:
            if col in df_filtered.columns:
                df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
        
        df_filtered['year'] = pd.to_numeric(df_filtered['year'], errors='coerce')
        df_filtered['fossil_share'] = pd.to_numeric(df_filtered['fossil_share_energy'], errors='coerce')
        df_filtered['renewables_share'] = pd.to_numeric(df_filtered['renewables_share_energy'], errors='coerce')
        
        return df_filtered
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def create_key_trends(df):
    """Create essential trend analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Get top countries by data availability
    country_counts = df.groupby('country').size().sort_values(ascending=False)
    top_countries = country_counts.head(10).index.tolist()
    
    # 1. GDP vs Fossil Fuel Share
    ax = axes[0, 0]
    for country in top_countries[:6]:
        country_data = df[df['country'] == country].dropna(subset=['gdp', 'fossil_share'])
        if len(country_data) > 5:
            ax.plot(country_data['year'], country_data['fossil_share'], 
                   linewidth=2, label=country, alpha=0.8)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Fossil Fuel Share (%)')
    ax.set_title('Fossil Fuel Dependency Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Renewable Energy Adoption
    ax = axes[0, 1]
    for country in top_countries[:6]:
        country_data = df[df['country'] == country].dropna(subset=['year', 'renewables_share'])
        if len(country_data) > 5:
            ax.plot(country_data['year'], country_data['renewables_share'], 
                   linewidth=2, label=country)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Renewable Energy Share (%)')
    ax.set_title('Renewable Energy Adoption')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Correlation Analysis
    ax = axes[1, 0]
    correlations = []
    countries = []
    
    for country in df['country'].unique():
        country_data = df[df['country'] == country].dropna(subset=['gdp', 'fossil_share'])
        if len(country_data) > 8:
            corr = country_data['gdp'].corr(country_data['fossil_share'])
            if not pd.isna(corr):
                correlations.append(corr)
                countries.append(country)
    
    # Get top 10 by correlation strength
    sorted_data = sorted(zip(correlations, countries), key=lambda x: abs(x[0]), reverse=True)
    correlations, countries = zip(*sorted_data[:10])
    
    bars = ax.barh(countries, correlations, 
                  color=['red' if x>0 else 'blue' for x in correlations])
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Correlation Coefficient')
    ax.set_title('GDP vs Fossil Fuel Correlation')
    
    # 4. Latest Year Energy Mix
    ax = axes[1, 1]
    latest_year = df['year'].max()
    latest_data = df[df['year'] == latest_year].dropna(
        subset=['fossil_share_energy', 'renewables_share_energy']
    ).nlargest(8, 'gdp')
    
    if not latest_data.empty:
        x = range(len(latest_data))
        width = 0.35
        
        ax.bar(x, latest_data['fossil_share'], width, label='Fossil', alpha=0.8)
        ax.bar([i + width for i in x], latest_data['renewables_share'], width, label='Renewable', alpha=0.8)
        
        ax.set_xlabel('Country')
        ax.set_ylabel('Energy Share (%)')
        ax.set_title(f'Energy Mix - {latest_year}')
        ax.set_xticks([i + width/2 for i in x])
        ax.set_xticklabels(latest_data['country'], rotation=45)
        ax.legend()
    
    plt.tight_layout()
    return fig

def create_country_comparison(df, selected_countries):
    """Create country comparison charts"""
    if not selected_countries:
        return None
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Fossil fuel comparison
    ax = axes[0]
    for country in selected_countries:
        country_data = df[df['country'] == country]
        if not country_data.empty:
            ax.plot(country_data['year'], country_data['fossil_share'], 
                   label=country, linewidth=2)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Fossil Fuel Share (%)')
    ax.set_title('Fossil Fuel Dependency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Renewable energy comparison
    ax = axes[1]
    for country in selected_countries:
        country_data = df[df['country'] == country]
        if not country_data.empty:
            ax.plot(country_data['year'], country_data['renewables_share'], 
                   label=country, linewidth=2)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Renewable Energy Share (%)')
    ax.set_title('Renewable Energy Adoption')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    st.set_page_config(page_title="GDP & Energy Analysis", page_icon="🌍", layout="wide")
    
    st.title("🌍 GDP & Energy Structure Analysis")
    st.markdown("Analysis of economic development and energy transformation relationships")
    
    # Load data
    with st.spinner('Loading data...'):
        df = load_and_preprocess_data()
    
    if df.empty:
        st.error("No data loaded. Please check the data file.")
        return
    
    # Sidebar
    st.sidebar.header("Analysis Options")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis",
        ["Key Trends", "Country Comparison"]
    )
    
    # Country selection
    available_countries = sorted(df['country'].unique())
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        available_countries,
        default=available_countries[:3] if available_countries else []
    )
    
    # Year range
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=int(df['year'].min()),
        max_value=int(df['year'].max()),
        value=(1990, int(df['year'].max()))
    )
    
    # Filter data
    filtered_df = df[
        (df['country'].isin(selected_countries)) & 
        (df['year'] >= year_range[0]) & 
        (df['year'] <= year_range[1])
    ]
    
    # Display summary
    if st.sidebar.checkbox("Show Data Summary"):
        st.subheader("Data Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Countries", len(filtered_df['country'].unique()))
        with col2:
            st.metric("Data Points", len(filtered_df))
        st.dataframe(filtered_df.head(50))
    
    # Main analysis
    if analysis_type == "Key Trends":
        st.header("Key Energy Trends")
        fig = create_key_trends(filtered_df)
        st.pyplot(fig)
    
    else:  # Country Comparison
        st.header("Country Comparison")
        if selected_countries:
            fig = create_country_comparison(filtered_df, selected_countries)
            if fig:
                st.pyplot(fig)
            
            # Country statistics
            st.subheader("Country Statistics")
            stats_data = []
            for country in selected_countries:
                country_data = filtered_df[filtered_df['country'] == country].dropna(
                    subset=['gdp', 'fossil_share', 'renewables_share']
                )
                if len(country_data) > 0:
                    latest = country_data[country_data['year'] == country_data['year'].max()].iloc[0]
                    stats_data.append({
                        'Country': country,
                        'GDP (B)': f"${latest['gdp']/1e9:.0f}" if pd.notna(latest['gdp']) else 'N/A',
                        'Fossil %': f"{latest['fossil_share']:.1f}%",
                        'Renewable %': f"{latest['renewables_share']:.1f}%"
                    })
            
            if stats_data:
                st.table(pd.DataFrame(stats_data))

if __name__ == "__main__":
    main()