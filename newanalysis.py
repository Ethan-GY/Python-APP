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
        
        # Filter countries with sufficient GDP data
        countries_with_gdp = df.groupby('country')['gdp'].count()
        valid_countries = countries_with_gdp[countries_with_gdp >= 10].index.tolist()
        
        # Also include countries with good energy data even if GDP is limited
        energy_countries = df.groupby('country')['fossil_share_energy'].count()
        energy_valid = energy_countries[energy_countries >= 15].index.tolist()
        
        # Combine and get unique countries
        all_valid_countries = list(set(valid_countries + energy_valid))
        
        # Filter dataset
        df_filtered = df[df['country'].isin(all_valid_countries)].copy()
        
        # Convert data types
        numeric_columns = ['gdp', 'population', 'fossil_share_energy', 'renewables_share_energy', 
                         'coal_share_energy', 'gas_share_energy', 'oil_share_energy',
                         'coal_consumption', 'gas_consumption', 'oil_consumption',
                         'renewables_consumption', 'primary_energy_consumption']
        
        for col in numeric_columns:
            if col in df_filtered.columns:
                df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
        
        df_filtered['year'] = pd.to_numeric(df_filtered['year'], errors='coerce')
        
        # Calculate additional metrics
        df_filtered['fossil_share'] = pd.to_numeric(df_filtered['fossil_share_energy'], errors='coerce')
        df_filtered['renewables_share'] = pd.to_numeric(df_filtered['renewables_share_energy'], errors='coerce')
        df_filtered['non_fossil_share'] = 100 - df_filtered['fossil_share']
        
        # Calculate energy intensity (energy consumption per GDP)
        df_filtered['energy_intensity'] = df_filtered['primary_energy_consumption'] / (df_filtered['gdp'] / 1e9)  # GDP in billions
        
        return df_filtered
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def create_comprehensive_trends(df):
    """Create comprehensive trend analysis with multiple countries"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Get top countries by data availability
    country_counts = df.groupby('country').size().sort_values(ascending=False)
    top_countries = country_counts.head(15).index.tolist()
    
    # 1. GDP Growth vs Fossil Fuel Share
    ax = axes[0, 0]
    for country in top_countries[:8]:  # Limit to 8 for readability
        country_data = df[df['country'] == country].dropna(subset=['gdp', 'fossil_share'])
        if len(country_data) > 5:
            # Normalize GDP for better comparison
            min_gdp = country_data['gdp'].min()
            max_gdp = country_data['gdp'].max()
            if max_gdp > min_gdp:
                normalized_gdp = (country_data['gdp'] - min_gdp) / (max_gdp - min_gdp) * 100
                ax.plot(country_data['year'], normalized_gdp, 
                       label=f'{country} - GDP', linewidth=2, alpha=0.8)
                ax.plot(country_data['year'], country_data['fossil_share'], 
                       label=f'{country} - Fossil %', linewidth=2, linestyle='--', alpha=0.8)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Normalized Values (%)')
    ax.set_title('GDP Growth vs Fossil Fuel Share (Normalized)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. Energy Transition Scatter Plot
    ax = axes[0, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_countries)))
    
    for i, country in enumerate(top_countries):
        country_data = df[df['country'] == country].dropna(subset=['gdp', 'fossil_share'])
        if len(country_data) > 3:
            x = country_data['gdp'] / 1e9  # Convert to billions
            y = country_data['fossil_share']
            ax.scatter(x, y, color=colors[i], label=country, alpha=0.7, s=50)
            
            # Add trend line
            if len(country_data) > 5:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), color=colors[i], alpha=0.5, linewidth=1)
    
    ax.set_xlabel('GDP (Billions USD)')
    ax.set_ylabel('Fossil Fuel Share (%)')
    ax.set_title('GDP vs Fossil Fuel Share with Trend Lines')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. Renewable Energy Adoption
    ax = axes[1, 0]
    for country in top_countries[:10]:
        country_data = df[df['country'] == country].dropna(subset=['year', 'renewables_share'])
        if len(country_data) > 5:
            ax.plot(country_data['year'], country_data['renewables_share'], 
                   marker='o', markersize=3, linewidth=2, label=country)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Renewable Energy Share (%)')
    ax.set_title('Renewable Energy Adoption Over Time')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. Current Energy Mix (latest year)
    ax = axes[1, 1]
    latest_year = df['year'].max()
    latest_data = df[df['year'] == latest_year].dropna(
        subset=['coal_share_energy', 'gas_share_energy', 'oil_share_energy', 'renewables_share_energy']
    )
    
    if not latest_data.empty:
        # Select top 10 countries by GDP
        latest_data = latest_data.nlargest(10, 'gdp')
        
        categories = ['Coal', 'Gas', 'Oil', 'Renewables']
        bottom = np.zeros(len(latest_data))
        
        for i, category in enumerate(categories):
            share_col = f'{category.lower()}_share_energy'
            if share_col in latest_data.columns:
                values = latest_data[share_col].fillna(0).values
                ax.bar(latest_data['country'], values, bottom=bottom, 
                      label=category, alpha=0.8)
                bottom += values
        
        ax.set_xlabel('Country')
        ax.set_ylabel('Energy Share (%)')
        ax.set_title(f'Energy Mix by Country ({latest_year})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

def create_correlation_analysis(df):
    """Create correlation and statistical analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. GDP vs Energy Metrics Correlation Matrix
    ax = axes[0, 0]
    correlation_data = df[['gdp', 'fossil_share', 'renewables_share', 
                         'coal_share_energy', 'gas_share_energy', 'oil_share_energy']].corr()
    
    im = ax.imshow(correlation_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation_data.columns)))
    ax.set_yticks(range(len(correlation_data.columns)))
    ax.set_xticklabels([col.replace('_', ' ').title() for col in correlation_data.columns], rotation=45)
    ax.set_yticklabels([col.replace('_', ' ').title() for col in correlation_data.columns])
    
    # Add correlation values
    for i in range(len(correlation_data.columns)):
        for j in range(len(correlation_data.columns)):
            ax.text(j, i, f'{correlation_data.iloc[i, j]:.2f}', 
                   ha='center', va='center', fontsize=8)
    
    ax.set_title('Correlation Matrix: GDP vs Energy Metrics')
    plt.colorbar(im, ax=ax)
    
    # 2. Country-level GDP-Energy Correlations
    ax = axes[0, 1]
    correlations = []
    countries = []
    
    for country in df['country'].unique():
        country_data = df[df['country'] == country].dropna(subset=['gdp', 'fossil_share'])
        if len(country_data) > 10:  # Require minimum data points
            corr = country_data['gdp'].corr(country_data['fossil_share'])
            if not pd.isna(corr):
                correlations.append(corr)
                countries.append(country)
    
    # Sort by correlation strength
    sorted_data = sorted(zip(correlations, countries), key=lambda x: abs(x[0]), reverse=True)
    correlations, countries = zip(*sorted_data[:15])  # Top 15
    
    bars = ax.barh(countries, correlations, 
                  color=['red' if x>0 else 'blue' for x in correlations])
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Correlation Coefficient')
    ax.set_title('GDP vs Fossil Fuel Share Correlation by Country')
    
    # Add value labels
    for bar, corr in zip(bars, correlations):
        width = bar.get_width()
        ax.text(width + (0.01 if width >=0 else -0.03), bar.get_y() + bar.get_height()/2,
               f'{corr:.3f}', ha='left' if width >=0 else 'right', va='center', fontsize=8)
    
    # 3. Energy Intensity Analysis
    ax = axes[1, 0]
    for country in list(countries)[:8]:  # Use countries from previous analysis
        country_data = df[df['country'] == country].dropna(subset=['gdp', 'primary_energy_consumption'])
        if len(country_data) > 5:
            # Calculate energy intensity if not already calculated
            if 'energy_intensity' not in country_data.columns:
                country_data['energy_intensity'] = country_data['primary_energy_consumption'] / (country_data['gdp'] / 1e9)
            
            ax.plot(country_data['year'], country_data['energy_intensity'], 
                   marker='s', markersize=3, linewidth=2, label=country)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Energy Intensity (Energy/GDP)')
    ax.set_title('Energy Intensity Trends')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. Development Stage Analysis
    ax = axes[1, 1]
    latest_data = df[df['year'] == df['year'].max()].dropna(subset=['gdp', 'renewables_share'])
    
    if not latest_data.empty:
        # Use log scale for GDP to better show distribution
        ax.scatter(np.log10(latest_data['gdp']), latest_data['renewables_share'], 
                 alpha=0.6, s=50)
        
        # Add country labels for outliers or interesting cases
        for _, row in latest_data.nlargest(5, 'renewables_share').iterrows():
            ax.annotate(row['country'], 
                       (np.log10(row['gdp']), row['renewables_share']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add trend line
        x_log = np.log10(latest_data['gdp'].dropna())
        y = latest_data['renewables_share'].dropna()
        if len(x_log) > 2:
            z = np.polyfit(x_log, y, 1)
            p = np.poly1d(z)
            x_range = np.linspace(x_log.min(), x_log.max(), 100)
            ax.plot(x_range, p(x_range), 'r--', alpha=0.8, label='Trend')
        
        ax.set_xlabel('Log(GDP)')
        ax.set_ylabel('Renewable Energy Share (%)')
        ax.set_title('Economic Development vs Renewable Energy (Latest Year)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_advanced_analysis(df):
    """Create advanced statistical and predictive analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Energy Transition Pathways
    ax = axes[0, 0]
    
    # Classify countries by development stage (simplified)
    latest_gdp = df[df['year'] == df['year'].max()][['country', 'gdp']].dropna()
    latest_gdp = latest_gdp.groupby('country')['gdp'].mean()
    
    high_income = latest_gdp[latest_gdp > latest_gdp.quantile(0.75)].index.tolist()
    low_income = latest_gdp[latest_gdp < latest_gdp.quantile(0.25)].index.tolist()
    
    for i, (countries, label) in enumerate([(high_income, 'High Income'), 
                                          (low_income, 'Low Income')]):
        for country in countries[:5]:  # Show 5 from each group
            country_data = df[df['country'] == country].dropna(subset=['year', 'fossil_share'])
            if len(country_data) > 10:
                ax.plot(country_data['year'], country_data['fossil_share'], 
                       color='red' if i == 0 else 'blue', 
                       alpha=0.6, linewidth=2, label=f'{country} ({label})')
    
    # Add average lines
    years = sorted(df['year'].unique())
    avg_fossil = [df[df['year'] == y]['fossil_share'].mean() for y in years]
    ax.plot(years, avg_fossil, 'k--', linewidth=3, label='Global Average', alpha=0.8)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Fossil Fuel Share (%)')
    ax.set_title('Energy Transition Pathways by Development Stage')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. Decoupling Analysis: GDP vs Energy Consumption
    ax = axes[0, 1]
    decoupling_data = []
    
    for country in df['country'].unique():
        country_data = df[df['country'] == country].dropna(subset=['gdp', 'primary_energy_consumption'])
        if len(country_data) > 10:
            # Calculate growth rates
            country_data = country_data.sort_values('year')
            gdp_growth = country_data['gdp'].pct_change().mean() * 100
            energy_growth = country_data['primary_energy_consumption'].pct_change().mean() * 100
            
            decoupling_data.append({
                'country': country,
                'gdp_growth': gdp_growth,
                'energy_growth': energy_growth,
                'decoupling': gdp_growth - energy_growth
            })
    
    if decoupling_data:
        decoupling_df = pd.DataFrame(decoupling_data)
        decoupling_df = decoupling_df.dropna().nlargest(15, 'decoupling')
        
        y_pos = np.arange(len(decoupling_df))
        bars = ax.barh(y_pos, decoupling_df['decoupling'])
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(decoupling_df['country'])
        ax.set_xlabel('Decoupling Index (GDP Growth - Energy Growth)')
        ax.set_title('Top 15 Countries by Economic-Energy Decoupling')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # Add value labels
        for bar, value in zip(bars, decoupling_df['decoupling']):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{value:.1f}%', ha='left', va='center', fontsize=8)
    
    # 3. Renewable Energy Growth Rates
    ax = axes[1, 0]
    growth_data = []
    
    for country in df['country'].unique():
        country_data = df[df['country'] == country].dropna(subset=['year', 'renewables_share'])
        if len(country_data) > 10:
            country_data = country_data.sort_values('year')
            start_share = country_data['renewables_share'].iloc[0]
            end_share = country_data['renewables_share'].iloc[-1]
            years = country_data['year'].iloc[-1] - country_data['year'].iloc[0]
            
            if years > 0 and start_share > 0:
                annual_growth = (end_share / start_share) ** (1/years) - 1
                growth_data.append({
                    'country': country,
                    'annual_growth': annual_growth * 100,
                    'start_share': start_share,
                    'end_share': end_share
                })
    
    if growth_data:
        growth_df = pd.DataFrame(growth_data)
        growth_df = growth_df.nlargest(10, 'annual_growth')
        
        bars = ax.barh(growth_df['country'], growth_df['annual_growth'])
        ax.set_xlabel('Annual Renewable Growth Rate (%)')
        ax.set_title('Top 10 Countries by Renewable Energy Growth Rate')
        
        # Add value labels
        for bar, growth in zip(bars, growth_df['annual_growth']):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{growth:.1f}%', ha='left', va='center', fontsize=8)
    
    # 4. Future Projection based on current trends
    ax = axes[1, 1]
    
    # Simple projection: linear trend of renewable adoption
    global_renewable = df.groupby('year')['renewables_share'].mean().reset_index()
    
    if len(global_renewable) > 5:
        # Fit linear trend
        x = global_renewable['year'].values
        y = global_renewable['renewables_share'].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Project to 2030 and 2050
        future_years = np.array([2025, 2030, 2035, 2040, 2045, 2050])
        future_share = slope * future_years + intercept
        
        # Plot historical and projection
        ax.plot(x, y, 'bo-', label='Historical', linewidth=2)
        ax.plot(future_years, future_share, 'ro--', label='Projection', linewidth=2)
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Global Renewable Energy Share (%)')
        ax.set_title(f'Renewable Energy Projection (R² = {r_value**2:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add projection values
        for year, share in zip(future_years, future_share):
            ax.annotate(f'{share:.1f}%', (year, share), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    return fig

def main():
    st.set_page_config(page_title="GDP & Energy Analysis", page_icon="🌍", layout="wide")
    
    st.title("🌍 Comprehensive GDP & Energy Structure Analysis")
    st.markdown("""
    This analysis explores the relationship between economic development (GDP) and energy structure transformation,
    using data from multiple countries over several decades.
    """)
    
    # Load data
    with st.spinner('Loading and processing data...'):
        df = load_and_preprocess_data()
    
    if df.empty:
        st.error("No data loaded. Please check the data file.")
        return
    
    # Sidebar
    st.sidebar.header("Analysis Options")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Comprehensive Trends", "Correlation Analysis", "Advanced Analysis", "Country Comparison", "Full Report"]
    )
    
    # Country selection for detailed analysis
    available_countries = sorted(df['country'].unique())
    selected_countries = st.sidebar.multiselect(
        "Select Countries for Detailed Analysis",
        available_countries,
        default=available_countries[:5] if len(available_countries) >= 5 else available_countries
    )
    
    # Year range selection
    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=int(df['year'].min()),
        max_value=int(df['year'].max()),
        value=(1990, int(df['year'].max()))
    )
    
    # Filter data based on selections
    filtered_df = df[
        (df['country'].isin(selected_countries)) & 
        (df['year'] >= year_range[0]) & 
        (df['year'] <= year_range[1])
    ]
    
    # Display data summary
    if st.sidebar.checkbox("Show Data Summary"):
        st.subheader("Data Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Countries in Analysis", len(filtered_df['country'].unique()))
        with col2:
            st.metric("Years Covered", f"{filtered_df['year'].min()} - {filtered_df['year'].max()}")
        with col3:
            st.metric("Total Data Points", len(filtered_df))
        
        st.dataframe(filtered_df.head(100))
    
    # Main analysis based on selection
    if analysis_type == "Comprehensive Trends":
        st.header("Comprehensive Energy Trends Analysis")
        fig1 = create_comprehensive_trends(filtered_df)
        st.pyplot(fig1)
        
        st.markdown("""
        **Key Insights:**
        - Economic growth patterns vary significantly across countries
        - Fossil fuel dependency shows different trajectories
        - Renewable energy adoption is accelerating globally
        - Energy mix diversity differs by economic development stage
        """)
    
    elif analysis_type == "Correlation Analysis":
        st.header("Statistical Correlation Analysis")
        fig2 = create_correlation_analysis(filtered_df)
        st.pyplot(fig2)
        
        st.markdown("""
        **Statistical Findings:**
        - GDP shows varying correlations with energy metrics across countries
        - Energy intensity trends reveal efficiency improvements
        - Development level influences renewable energy adoption
        - Correlation patterns indicate structural economic changes
        """)
    
    elif analysis_type == "Advanced Analysis":
        st.header("Advanced Statistical Analysis")
        fig3 = create_advanced_analysis(filtered_df)
        st.pyplot(fig3)
        
        st.markdown("""
        **Advanced Insights:**
        - Clear divergence in energy transition pathways by income level
        - Economic-energy decoupling indicates sustainable development
        - Renewable growth rates highlight leaders in energy transition
        - Projections suggest continued shift toward clean energy
        """)
    
    elif analysis_type == "Country Comparison":
        st.header("Country-Specific Analysis")
        
        if selected_countries:
            # Create country comparison dashboard
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Energy Transition Comparison")
                
                # Fossil fuel share comparison
                fig, ax = plt.subplots(figsize=(10, 6))
                for country in selected_countries:
                    country_data = filtered_df[filtered_df['country'] == country]
                    if not country_data.empty:
                        ax.plot(country_data['year'], country_data['fossil_share'], 
                               label=country, linewidth=2)
                
                ax.set_xlabel('Year')
                ax.set_ylabel('Fossil Fuel Share (%)')
                ax.set_title('Fossil Fuel Dependency Comparison')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                st.subheader("Renewable Energy Progress")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                for country in selected_countries:
                    country_data = filtered_df[filtered_df['country'] == country]
                    if not country_data.empty:
                        ax.plot(country_data['year'], country_data['renewables_share'], 
                               label=country, linewidth=2)
                
                ax.set_xlabel('Year')
                ax.set_ylabel('Renewable Energy Share (%)')
                ax.set_title('Renewable Energy Adoption Comparison')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            # Country statistics table
            st.subheader("Country Performance Metrics")
            country_stats = []
            
            for country in selected_countries:
                country_data = filtered_df[filtered_df['country'] == country].dropna(
                    subset=['gdp', 'fossil_share', 'renewables_share']
                )
                if len(country_data) > 2:
                    latest = country_data[country_data['year'] == country_data['year'].max()].iloc[0]
                    stats = {
                        'Country': country,
                        'Latest GDP (B)': f"${latest['gdp']/1e9:.0f}" if pd.notna(latest['gdp']) else 'N/A',
                        'Fossil Share': f"{latest['fossil_share']:.1f}%" if pd.notna(latest['fossil_share']) else 'N/A',
                        'Renewable Share': f"{latest['renewables_share']:.1f}%" if pd.notna(latest['renewables_share']) else 'N/A',
                        'Data Years': len(country_data)
                    }
                    country_stats.append(stats)
            
            if country_stats:
                st.table(pd.DataFrame(country_stats))
    
    else:  # Full Report
        st.header("Comprehensive Analysis Report")
        
        # Executive Summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Key Findings")
            st.markdown("""
            **Global Energy Transition is Underway:**
            - Renewable energy share is increasing across most economies
            - Fossil fuel dependency shows declining trends in developed countries
            - Economic growth increasingly decoupled from energy consumption
            
            **Development Patterns:**
            - High-income countries lead in renewable adoption
            - Emerging economies show varied transition pathways
            - Energy intensity improvements are widespread
            """)
        
        with col2:
            st.subheader("🎯 Policy Implications")
            st.markdown("""
            **Strategic Recommendations:**
            - Accelerate renewable energy infrastructure investment
            - Implement carbon pricing mechanisms
            - Promote energy efficiency standards
            - Support clean technology innovation
            - Foster international cooperation on climate goals
            """)
        
        # Display all analysis charts
        st.subheader("Comprehensive Analysis Charts")
        
        fig1 = create_comprehensive_trends(filtered_df)
        st.pyplot(fig1)
        
        fig2 = create_correlation_analysis(filtered_df)
        st.pyplot(fig2)
        
        fig3 = create_advanced_analysis(filtered_df)
        st.pyplot(fig3)
    
    # Conclusion and recommendations
    st.markdown("---")
    st.subheader("📈 Overall Conclusions")
    
    conclusion_col1, conclusion_col2, conclusion_col3 = st.columns(3)
    
    with conclusion_col1:
        st.info("""
        **Economic Transformation**
        GDP growth enables energy structure optimization,
        driving transition from fossil fuels to clean energy sources
        """)
    
    with conclusion_col2:
        st.success("""
        **Environmental Progress**
        Renewable energy expansion reduces environmental impact,
        making sustainable development achievable
        """)
    
    with conclusion_col3:
        st.warning("""
        **Transition Challenges**
        Balancing economic, energy, and environmental objectives
        requires coordinated policy and technological innovation
        """)
    
    # Data sources and methodology
    with st.expander("Data Sources and Methodology"):
        st.markdown("""
        **Data Sources:**
        - World Energy Consumption dataset
        - Multiple countries over several decades
        - GDP, energy consumption, and energy mix data
        
        **Methodology:**
        - Time series analysis of energy structure evolution
        - Correlation analysis between economic and energy metrics
        - Comparative analysis across development stages
        - Statistical trend analysis and projections
        
        **Limitations:**
        - Data availability varies by country and year
        - GDP figures in nominal USD
        - Energy share calculations based on primary energy
        """)

if __name__ == "__main__":
    main()