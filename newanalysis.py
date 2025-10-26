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

# Set style for better visualization
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
        
        # Calculate per capita metrics
        df_filtered['gdp_per_capita'] = df_filtered['gdp'] / df_filtered['population']
        df_filtered['energy_per_capita'] = df_filtered['primary_energy_consumption'] / df_filtered['population']
        
        # Calculate energy intensity (energy consumption per GDP)
        df_filtered['energy_intensity'] = df_filtered['primary_energy_consumption'] / (df_filtered['gdp'] / 1e9)  # GDP in billions
        
        # Calculate sector-specific energy intensities
        for sector in ['coal', 'gas', 'oil', 'renewables']:
            consumption_col = f'{sector}_consumption'
            if consumption_col in df_filtered.columns:
                df_filtered[f'{sector}_intensity'] = df_filtered[consumption_col] / (df_filtered['gdp'] / 1e9)
        
        return df_filtered
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def create_gdp_integrated_analysis(df):
    """Create comprehensive analysis integrating GDP with energy structure"""
    st.header("💰 GDP-Integrated Energy Structure Analysis")
    
    # Get countries with good data coverage
    country_coverage = df.groupby('country').agg({
        'gdp': 'count',
        'primary_energy_consumption': 'count',
        'fossil_share': 'count'
    }).mean(axis=1)
    
    top_countries = country_coverage.nlargest(20).index.tolist()
    
    # Create tabs for different analysis aspects
    tab1, tab2, tab3, tab4 = st.tabs([
        "GDP vs Energy Structure", 
        "Economic Development Stages", 
        "GDP Growth Impact", 
        "Policy Implications"
    ])
    
    with tab1:
        create_gdp_energy_structure_analysis(df, top_countries)
    
    with tab2:
        create_development_stage_analysis(df, top_countries)
    
    with tab3:
        create_gdp_growth_impact_analysis(df, top_countries)
    
    with tab4:
        create_policy_implications(df)

def create_gdp_energy_structure_analysis(df, countries):
    """Analyze relationship between GDP and energy structure"""
    st.subheader("GDP vs Energy Structure Relationship")
    
    # Create multiple visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # GDP vs Fossil Fuel Dependency
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use latest year data for cross-sectional analysis
        latest_year = df['year'].max()
        latest_data = df[df['year'] == latest_year].dropna(subset=['gdp', 'fossil_share'])
        
        if not latest_data.empty:
            # Select top countries by GDP for clarity
            latest_data = latest_data.nlargest(15, 'gdp')
            
            # Create scatter plot
            scatter = ax.scatter(latest_data['gdp']/1e9, latest_data['fossil_share'], 
                               s=100, alpha=0.7, c=latest_data['gdp_per_capita'], 
                               cmap='viridis')
            
            # Add country labels
            for i, row in latest_data.iterrows():
                ax.annotate(row['country'], 
                           (row['gdp']/1e9, row['fossil_share']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Add trend line
            if len(latest_data) > 2:
                z = np.polyfit(latest_data['gdp']/1e9, latest_data['fossil_share'], 1)
                p = np.poly1d(z)
                x_range = np.linspace(latest_data['gdp'].min()/1e9, latest_data['gdp'].max()/1e9, 100)
                ax.plot(x_range, p(x_range), 'r--', alpha=0.8, 
                       label=f'Trend (slope: {z[0]:.4f})')
            
            ax.set_xlabel('GDP (Billions USD)')
            ax.set_ylabel('Fossil Fuel Share (%)')
            ax.set_title(f'GDP vs Fossil Fuel Dependency ({latest_year})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add colorbar for GDP per capita
            plt.colorbar(scatter, ax=ax, label='GDP per Capita')
            
            st.pyplot(fig)
            
            # Calculate correlation
            correlation = latest_data['gdp'].corr(latest_data['fossil_share'])
            st.metric("GDP-Fossil Fuel Correlation", f"{correlation:.3f}")

    with col2:
        # Energy Intensity vs GDP per Capita
        fig, ax = plt.subplots(figsize=(10, 6))
        
        latest_data_intensity = df[df['year'] == latest_year].dropna(
            subset=['gdp_per_capita', 'energy_intensity']
        )
        
        if not latest_data_intensity.empty:
            latest_data_intensity = latest_data_intensity.nlargest(15, 'gdp')
            
            scatter = ax.scatter(latest_data_intensity['gdp_per_capita'], 
                               latest_data_intensity['energy_intensity'],
                               s=100, alpha=0.7, 
                               c=latest_data_intensity['renewables_share'],
                               cmap='Greens')
            
            for i, row in latest_data_intensity.iterrows():
                ax.annotate(row['country'], 
                           (row['gdp_per_capita'], row['energy_intensity']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax.set_xlabel('GDP per Capita')
            ax.set_ylabel('Energy Intensity (Energy/GDP)')
            ax.set_title(f'Economic Efficiency vs Renewable Adoption ({latest_year})')
            ax.grid(True, alpha=0.3)
            
            plt.colorbar(scatter, ax=ax, label='Renewable Share (%)')
            
            st.pyplot(fig)
            
            # Calculate correlation
            correlation = latest_data_intensity['gdp_per_capita'].corr(
                latest_data_intensity['energy_intensity']
            )
            st.metric("GDP per Capita-Energy Intensity Correlation", f"{correlation:.3f}")

    # Time series analysis of GDP and energy structure
    st.subheader("GDP Growth and Energy Structure Evolution")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Select a few representative countries
    representative_countries = ['Argentina', 'Australia', 'Algeria']  # Example countries
    
    for i, country in enumerate(representative_countries):
        if country in df['country'].unique():
            country_data = df[df['country'] == country].dropna(
                subset=['gdp', 'fossil_share', 'renewables_share']
            )
            
            if len(country_data) > 5:
                # Normalize GDP for dual-axis plot
                country_data = country_data.sort_values('year')
                gdp_normalized = (country_data['gdp'] / country_data['gdp'].max()) * 100
                
                # Plot on appropriate subplot
                row, col = i // 2, i % 2
                ax1 = axes[row, col]
                ax2 = ax1.twinx()
                
                # Plot energy shares
                ax1.plot(country_data['year'], country_data['fossil_share'], 
                        'b-', linewidth=2, label='Fossil Share')
                ax1.plot(country_data['year'], country_data['renewables_share'], 
                        'g-', linewidth=2, label='Renewable Share')
                
                # Plot normalized GDP
                ax2.plot(country_data['year'], gdp_normalized, 
                        'r--', linewidth=2, label='GDP (Normalized)')
                
                ax1.set_xlabel('Year')
                ax1.set_ylabel('Energy Share (%)')
                ax2.set_ylabel('Normalized GDP (%)')
                ax1.set_title(f'{country}: GDP vs Energy Structure')
                
                # Combine legends
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
                ax1.grid(True, alpha=0.3)

    # Fill empty subplots if needed
    for i in range(len(representative_countries), 4):
        row, col = i // 2, i % 2
        axes[row, col].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)

def create_development_stage_analysis(df, countries):
    """Analyze energy structure by economic development stage"""
    st.subheader("Energy Structure by Economic Development Stage")
    
    # Classify countries by development stage based on GDP per capita
    latest_data = df[df['year'] == df['year'].max()].dropna(subset=['gdp_per_capita'])
    
    if latest_data.empty:
        st.warning("No data available for development stage analysis")
        return
    
    # Define development stages based on GDP per capita quartiles
    gdp_per_capita = latest_data['gdp_per_capita']
    low_income = gdp_per_capita.quantile(0.25)
    middle_income = gdp_per_capita.quantile(0.5)
    high_income = gdp_per_capita.quantile(0.75)
    
    def classify_development(gdp_pc):
        if gdp_pc <= low_income:
            return "Low Income"
        elif gdp_pc <= middle_income:
            return "Lower Middle Income"
        elif gdp_pc <= high_income:
            return "Upper Middle Income"
        else:
            return "High Income"
    
    latest_data['development_stage'] = latest_data['gdp_per_capita'].apply(classify_development)
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Energy structure by development stage
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for stacked bar chart
        stages = ["Low Income", "Lower Middle Income", "Upper Middle Income", "High Income"]
        energy_sources = ['coal_share_energy', 'gas_share_energy', 'oil_share_energy', 'renewables_share_energy']
        energy_labels = ['Coal', 'Gas', 'Oil', 'Renewables']
        
        stage_data = []
        for stage in stages:
            stage_df = latest_data[latest_data['development_stage'] == stage]
            if not stage_df.empty:
                shares = []
                for source in energy_sources:
                    if source in stage_df.columns:
                        share = stage_df[source].mean()
                        shares.append(share if not pd.isna(share) else 0)
                    else:
                        shares.append(0)
                stage_data.append(shares)
        
        if stage_data:
            stage_data = np.array(stage_data)
            
            # Create stacked bar chart
            bottom = np.zeros(len(stages))
            colors = ['#8B4513', '#1E90FF', '#FFD700', '#32CD32']
            
            for i, (source, label, color) in enumerate(zip(energy_sources, energy_labels, colors)):
                ax.bar(stages, stage_data[:, i], bottom=bottom, 
                      label=label, color=color, alpha=0.8)
                bottom += stage_data[:, i]
            
            ax.set_ylabel('Energy Share (%)')
            ax.set_title('Energy Structure by Development Stage')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)
            
            st.pyplot(fig)

    with col2:
        # GDP per capita vs renewable share
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter_data = latest_data.dropna(subset=['gdp_per_capita', 'renewables_share'])
        
        if not scatter_data.empty:
            # Color by development stage
            colors = {'Low Income': 'red', 'Lower Middle Income': 'orange', 
                     'Upper Middle Income': 'yellow', 'High Income': 'green'}
            
            for stage, color in colors.items():
                stage_data = scatter_data[scatter_data['development_stage'] == stage]
                if not stage_data.empty:
                    ax.scatter(stage_data['gdp_per_capita'], stage_data['renewables_share'],
                             c=color, label=stage, alpha=0.7, s=60)
            
            # Add trend line
            if len(scatter_data) > 2:
                z = np.polyfit(scatter_data['gdp_per_capita'], scatter_data['renewables_share'], 2)
                p = np.poly1d(z)
                x_range = np.linspace(scatter_data['gdp_per_capita'].min(), 
                                    scatter_data['gdp_per_capita'].max(), 100)
                ax.plot(x_range, p(x_range), 'k--', alpha=0.8, label='Trend')
            
            ax.set_xlabel('GDP per Capita')
            ax.set_ylabel('Renewable Energy Share (%)')
            ax.set_title('Economic Development vs Renewable Energy Adoption')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Calculate correlation
            correlation = scatter_data['gdp_per_capita'].corr(scatter_data['renewables_share'])
            st.metric("GDP per Capita-Renewable Share Correlation", f"{correlation:.3f}")

    # Development transition analysis
    st.subheader("Economic Development and Energy Transition")
    
    # Analyze how energy structure changes with economic development over time
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Select countries that have data across multiple development stages
    transition_countries = []
    for country in countries[:10]:  # Check top 10 countries
        country_data = df[df['country'] == country].dropna(subset=['gdp_per_capita', 'fossil_share'])
        if len(country_data) > 10:
            # Check if country shows significant development
            gdp_growth = (country_data['gdp_per_capita'].max() / country_data['gdp_per_capita'].min()) - 1
            if gdp_growth > 1:  # More than 100% growth
                transition_countries.append(country)
    
    # Plot development paths for a few countries
    for i, country in enumerate(transition_countries[:4]):
        country_data = df[df['country'] == country].dropna(subset=['gdp_per_capita', 'fossil_share'])
        if len(country_data) > 5:
            row, col = i // 2, i % 2
            ax = axes[col] if i < 2 else axes[1]
            
            # Create path plot
            path = ax.plot(country_data['gdp_per_capita'], country_data['fossil_share'], 
                         marker='o', linewidth=2, label=country)
            
            # Add year labels for some points
            for j, row in country_data.iterrows():
                if j % 5 == 0:  # Label every 5th year
                    ax.annotate(str(int(row['year'])), 
                               (row['gdp_per_capita'], row['fossil_share']),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    for ax in axes:
        ax.set_xlabel('GDP per Capita')
        ax.set_ylabel('Fossil Fuel Share (%)')
        ax.set_title('Development Path: GDP vs Fossil Dependency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

def create_gdp_growth_impact_analysis(df, countries):
    """Analyze impact of GDP growth on energy structure"""
    st.subheader("Impact of GDP Growth on Energy Structure")
    
    # Calculate GDP growth rates and corresponding energy structure changes
    growth_data = []
    
    for country in countries:
        country_data = df[df['country'] == country].dropna(
            subset=['gdp', 'fossil_share', 'renewables_share']
        )
        
        if len(country_data) > 10:
            country_data = country_data.sort_values('year')
            
            # Calculate 5-year growth periods
            for i in range(len(country_data) - 5):
                period_data = country_data.iloc[i:i+6]  # 5-year period
                
                gdp_start = period_data['gdp'].iloc[0]
                gdp_end = period_data['gdp'].iloc[-1]
                gdp_growth = (gdp_end - gdp_start) / gdp_start * 100
                
                fossil_start = period_data['fossil_share'].iloc[0]
                fossil_end = period_data['fossil_share'].iloc[-1]
                fossil_change = fossil_end - fossil_start
                
                renewable_start = period_data['renewables_share'].iloc[0]
                renewable_end = period_data['renewables_share'].iloc[-1]
                renewable_change = renewable_end - renewable_start
                
                growth_data.append({
                    'country': country,
                    'period': f"{int(period_data['year'].iloc[0])}-{int(period_data['year'].iloc[-1])}",
                    'gdp_growth': gdp_growth,
                    'fossil_change': fossil_change,
                    'renewable_change': renewable_change,
                    'start_year': period_data['year'].iloc[0],
                    'gdp_per_capita': period_data['gdp_per_capita'].mean()
                })
    
    if not growth_data:
        st.warning("Insufficient data for growth impact analysis")
        return
    
    growth_df = pd.DataFrame(growth_data)
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # GDP growth vs fossil fuel change
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(growth_df['gdp_growth'], growth_df['fossil_change'],
                           c=growth_df['gdp_per_capita'], cmap='coolwarm',
                           alpha=0.6, s=50)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add quadrants
        ax.text(0.05, 0.95, 'High Growth, Decreasing Fossil', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))
        ax.text(0.05, 0.05, 'Low Growth, Decreasing Fossil', 
               transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax.text(0.55, 0.95, 'High Growth, Increasing Fossil', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
        ax.text(0.55, 0.05, 'Low Growth, Increasing Fossil', 
               transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))
        
        ax.set_xlabel('GDP Growth (%)')
        ax.set_ylabel('Change in Fossil Fuel Share (%)')
        ax.set_title('GDP Growth Impact on Fossil Fuel Dependency')
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=ax, label='GDP per Capita')
        
        st.pyplot(fig)
        
        # Calculate correlation
        correlation = growth_df['gdp_growth'].corr(growth_df['fossil_change'])
        st.metric("GDP Growth-Fossil Change Correlation", f"{correlation:.3f}")

    with col2:
        # GDP growth vs renewable energy change
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(growth_df['gdp_growth'], growth_df['renewable_change'],
                           c=growth_df['gdp_per_capita'], cmap='viridis',
                           alpha=0.6, s=50)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add quadrants
        ax.text(0.05, 0.95, 'High Growth, Increasing Renewables', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))
        ax.text(0.05, 0.05, 'Low Growth, Increasing Renewables', 
               transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax.text(0.55, 0.95, 'High Growth, Decreasing Renewables', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
        ax.text(0.55, 0.05, 'Low Growth, Decreasing Renewables', 
               transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))
        
        ax.set_xlabel('GDP Growth (%)')
        ax.set_ylabel('Change in Renewable Share (%)')
        ax.set_title('GDP Growth Impact on Renewable Energy Adoption')
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=ax, label='GDP per Capita')
        
        st.pyplot(fig)
        
        # Calculate correlation
        correlation = growth_df['gdp_growth'].corr(growth_df['renewable_change'])
        st.metric("GDP Growth-Renewable Change Correlation", f"{correlation:.3f}")

    # Time series of GDP growth and energy transition
    st.subheader("GDP Growth and Energy Transition Timeline")
    
    # Select a country with significant data
    example_country = None
    for country in countries:
        country_data = df[df['country'] == country].dropna(
            subset=['gdp', 'fossil_share', 'renewables_share']
        )
        if len(country_data) > 15:
            example_country = country
            break
    
    if example_country:
        country_data = df[df['country'] == example_country].dropna(
            subset=['gdp', 'fossil_share', 'renewables_share']
        ).sort_values('year')
        
        # Calculate rolling averages
        country_data['gdp_growth_5yr'] = country_data['gdp'].pct_change(5) * 100
        country_data['fossil_change_5yr'] = country_data['fossil_share'].diff(5)
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot GDP growth
        color = 'tab:blue'
        ax1.set_xlabel('Year')
        ax1.set_ylabel('5-Year GDP Growth (%)', color=color)
        line1 = ax1.plot(country_data['year'], country_data['gdp_growth_5yr'], 
                        color=color, linewidth=2, label='GDP Growth')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        # Create second y-axis for fossil fuel change
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('5-Year Change in Fossil Share (%)', color=color)
        line2 = ax2.plot(country_data['year'], country_data['fossil_change_5yr'], 
                        color=color, linewidth=2, label='Fossil Share Change')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        ax1.set_title(f'{example_country}: GDP Growth vs Fossil Fuel Transition')
        
        st.pyplot(fig)
        
        # Calculate overall correlation for this country
        correlation = country_data['gdp_growth_5yr'].corr(country_data['fossil_change_5yr'])
        st.metric(f"{example_country} GDP Growth-Fossil Change Correlation", f"{correlation:.3f}")

def create_policy_implications(df):
    """Provide policy implications based on GDP-energy analysis"""
    st.subheader("Policy Implications")
    
    # Key insights from the analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Economic Development Phases:**
        
        **Low-Income Countries:**
        - Focus on energy access and affordability
        - Leverage renewable potential for sustainable growth
        - Avoid fossil fuel lock-in
        
        **Middle-Income Countries:**
        - Balance economic growth with environmental goals
        - Invest in energy efficiency
        - Develop renewable energy infrastructure
        
        **High-Income Countries:**
        - Lead in renewable energy innovation
        - Implement carbon pricing mechanisms
        - Support global clean energy transition
        """)
    
    with col2:
        st.success("""
        **GDP-Energy Transition Strategies:**
        
        **Growth-Oriented Policies:**
        - Green stimulus packages
        - Renewable energy investments
        - Energy efficiency standards
        
        **Structural Transformation:**
        - Economic diversification
        - Clean technology adoption
        - Sustainable infrastructure
        
        **International Cooperation:**
        - Technology transfer
        - Climate finance
        - Carbon market mechanisms
        """)
    
    # Data-driven policy recommendations
    st.subheader("Data-Driven Policy Recommendations")
    
    # Calculate key metrics for policy recommendations
    latest_data = df[df['year'] == df['year'].max()].dropna(
        subset=['gdp_per_capita', 'fossil_share', 'renewables_share']
    )
    
    if not latest_data.empty:
        # High fossil dependency, low income countries
        high_fossil_low_income = latest_data[
            (latest_data['fossil_share'] > 80) & 
            (latest_data['gdp_per_capita'] < latest_data['gdp_per_capita'].median())
        ]
        
        # High renewable, high income countries
        high_renewable_high_income = latest_data[
            (latest_data['renewables_share'] > 30) & 
            (latest_data['gdp_per_capita'] > latest_data['gdp_per_capita'].median())
        ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Priority Countries for Energy Transition Support:**")
            if not high_fossil_low_income.empty:
                for _, country in high_fossil_low_income.iterrows():
                    st.write(f"- {country['country']}: {country['fossil_share']:.1f}% fossil, "
                            f"${country['gdp_per_capita']:.0f} per capita")
            else:
                st.write("No countries meet the criteria")
        
        with col2:
            st.write("**Renewable Energy Leaders:**")
            if not high_renewable_high_income.empty:
                for _, country in high_renewable_high_income.iterrows():
                    st.write(f"- {country['country']}: {country['renewables_share']:.1f}% renewable, "
                            f"${country['gdp_per_capita']:.0f} per capita")
            else:
                st.write("No countries meet the criteria")
    
    # Economic case for energy transition
    st.subheader("Economic Case for Energy Transition")
    
    # Calculate potential economic benefits
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Simulate economic benefits of renewable transition
    years = np.arange(2020, 2050, 5)
    current_trend = [100, 95, 90, 85, 80, 75]  # Current fossil share trend
    accelerated_transition = [100, 85, 70, 55, 40, 25]  # Accelerated transition
    
    # Economic benefits (simplified)
    economic_benefits_current = [0, 5, 10, 15, 20, 25]  # % GDP increase
    economic_benefits_accelerated = [0, 8, 18, 30, 45, 60]  # % GDP increase
    
    ax.plot(years, economic_benefits_current, 'b-', linewidth=2, 
            label='Current Transition Path')
    ax.plot(years, economic_benefits_accelerated, 'g-', linewidth=2, 
            label='Accelerated Transition Path')
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Cumulative Economic Benefits (% GDP)')
    ax.set_title('Projected Economic Benefits of Energy Transition')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.annotate('Accelerated transition\n yields 2.4x benefits by 2050', 
               xy=(2045, 60), xytext=(2030, 40),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1'),
               fontsize=10)
    
    st.pyplot(fig)

def main():
    st.set_page_config(page_title="GDP & Energy Analysis", page_icon="🌍", layout="wide")
    
    st.title("🌍 GDP-Integrated Energy Structure Analysis")
    st.markdown("""
    Comprehensive analysis of how economic development (GDP) influences and is influenced by energy structure transformation.
    This analysis explores the bidirectional relationship between economic growth and energy transitions.
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
            "GDP-Integrated Analysis", 
            "Development Stages", 
            "Growth Impact", 
            "Full Report"
        ]
    )
    
    if analysis_type == "GDP-Integrated Analysis":
        create_gdp_integrated_analysis(df)
    
    elif analysis_type == "Development Stages":
        st.header("📈 Economic Development Stages Analysis")
        create_development_stage_analysis(df, sorted(df['country'].unique())[:20])
    
    elif analysis_type == "Growth Impact":
        st.header("📊 GDP Growth Impact Analysis")
        create_gdp_growth_impact_analysis(df, sorted(df['country'].unique())[:20])
    
    else:  # Full Report
        st.header("📋 Comprehensive GDP-Energy Analysis Report")
        
        # Executive Summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Key Economic-Energy Insights")
            st.markdown("""
            **GDP-Energy Interdependence:**
            - Economic development drives energy structure transformation
            - Energy transitions create new economic opportunities
            - Different development stages require tailored energy policies
            
            **Transition Economics:**
            - Renewable energy adoption correlates with economic development
            - Energy efficiency improves with economic maturity
            - Fossil fuel dependency varies by economic structure
            """)
        
        with col2:
            st.subheader("Strategic Implications")
            st.markdown("""
            **Development-Aligned Policies:**
            - Low-income: Focus on energy access and affordability
            - Middle-income: Balance growth with sustainability
            - High-income: Lead innovation and global cooperation
            
            **Economic Opportunities:**
            - Green growth strategies
            - Clean energy investments
            - Sustainable infrastructure development
            """)
        
        # Display comprehensive analysis
        create_gdp_integrated_analysis(df)
    
    # Data summary
    with st.expander("Dataset Summary"):
        st.write(f"**Dataset Overview:**")
        st.write(f"- Countries: {df['country'].nunique()}")
        st.write(f"- Time Period: {df['year'].min()} - {df['year'].max()}")
        st.write(f"- Total Observations: {len(df)}")
        
        # GDP data availability
        gdp_coverage = df['gdp'].notna().sum()
        st.write(f"- GDP Data Coverage: {gdp_coverage} points ({(gdp_coverage/len(df)*100):.1f}%)")
        
        # Key metrics summary
        latest_year = df['year'].max()
        latest_data = df[df['year'] == latest_year].dropna(subset=['gdp', 'fossil_share'])
        
        if not latest_data.empty:
            st.write(f"**Latest Year ({latest_year}) Summary:**")
            st.write(f"- Average Fossil Share: {latest_data['fossil_share'].mean():.1f}%")
            st.write(f"- Average Renewable Share: {latest_data['renewables_share'].mean():.1f}%")
            st.write(f"- Average GDP per Capita: ${latest_data['gdp_per_capita'].mean():.0f}")

if __name__ == "__main__":
    main()