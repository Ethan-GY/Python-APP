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

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_and_preprocess_data():
    """加载和预处理数据"""
    df = pd.read_csv('World Energy Consumption New.csv')
    
    # 选择有GDP数据的国家（排除地区聚合数据）
    countries_with_gdp = df[df['gdp'].notna()]['country'].unique()
    
    # 过滤出主要国家数据
    main_countries = ['Argentina', 'Australia', 'Austria', 'Algeria']
    df_filtered = df[df['country'].isin(main_countries)].copy()
    
    # 转换数据类型
    df_filtered['gdp'] = pd.to_numeric(df_filtered['gdp'], errors='coerce')
    df_filtered['year'] = pd.to_numeric(df_filtered['year'], errors='coerce')
    
    # 计算能源结构指标
    df_filtered['fossil_share'] = pd.to_numeric(df_filtered['fossil_share_energy'], errors='coerce')
    df_filtered['renewables_share'] = pd.to_numeric(df_filtered['renewables_share_energy'], errors='coerce')
    df_filtered['coal_share'] = pd.to_numeric(df_filtered['coal_share_energy'], errors='coerce')
    df_filtered['gas_share'] = pd.to_numeric(df_filtered['gas_share_energy'], errors='coerce')
    df_filtered['oil_share'] = pd.to_numeric(df_filtered['oil_share_energy'], errors='coerce')
    
    return df_filtered

def create_gdp_energy_trends(df):
    """创建GDP与能源结构的趋势图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 按国家分组
    for country in df['country'].unique():
        country_data = df[df['country'] == country].dropna(subset=['gdp', 'fossil_share'])
        
        if len(country_data) > 5:  # 确保有足够的数据点
            # GDP vs 化石能源占比
            ax1.plot(country_data['year'], country_data['fossil_share'], 
                   marker='o', linewidth=2, label=country)
            
            # GDP vs 可再生能源占比
            ax2.plot(country_data['year'], country_data['renewables_share'], 
                   marker='s', linewidth=2, label=country)
    
    ax1.set_xlabel('年份')
    ax1.set_ylabel('化石能源占比 (%)')
    ax1.set_title('GDP增长与化石能源占比趋势')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('年份')
    ax2.set_ylabel('可再生能源占比 (%)')
    ax2.set_title('GDP增长与可再生能源占比趋势')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 散点图：GDP vs 能源结构
    colors = plt.cm.Set3(np.linspace(0, 1, len(df['country'].unique())))
    
    for i, country in enumerate(df['country'].unique()):
        country_data = df[df['country'] == country].dropna(subset=['gdp', 'fossil_share'])
        if len(country_data) > 3:
            ax3.scatter(country_data['gdp']/1e9, country_data['fossil_share'], 
                      color=colors[i], label=country, alpha=0.7, s=60)
    
    ax3.set_xlabel('GDP (十亿美元)')
    ax3.set_ylabel('化石能源占比 (%)')
    ax3.set_title('GDP与化石能源占比关系')
    ax3.legend()
    
    # 能源结构组成图（最新年份）
    latest_year = df['year'].max()
    latest_data = df[df['year'] == latest_year].dropna(subset=['coal_share', 'gas_share', 'oil_share', 'renewables_share'])
    
    if not latest_data.empty:
        countries = latest_data['country'].values
        coal_share = latest_data['coal_share'].values
        gas_share = latest_data['gas_share'].values
        oil_share = latest_data['oil_share'].values
        renew_share = latest_data['renewables_share'].values
        
        width = 0.2
        x = np.arange(len(countries))
        
        ax4.bar(x - 1.5*width, coal_share, width, label='煤炭', alpha=0.8)
        ax4.bar(x - 0.5*width, gas_share, width, label='天然气', alpha=0.8)
        ax4.bar(x + 0.5*width, oil_share, width, label='石油', alpha=0.8)
        ax4.bar(x + 1.5*width, renew_share, width, label='可再生能源', alpha=0.8)
        
        ax4.set_xlabel('国家')
        ax4.set_ylabel('能源占比 (%)')
        ax4.set_title(f'{latest_year}年各国能源结构组成')
        ax4.set_xticks(x)
        ax4.set_xticklabels(countries, rotation=45)
        ax4.legend()
    
    plt.tight_layout()
    return fig

def create_energy_transition_analysis(df):
    """创建能源转型分析图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 各国能源转型路径
    for country in df['country'].unique():
        country_data = df[df['country'] == country].dropna(subset=['fossil_share', 'renewables_share', 'gdp'])
        if len(country_data) > 5:
            # 计算GDP增长率
            country_data = country_data.sort_values('year')
            country_data['gdp_growth'] = country_data['gdp'].pct_change() * 100
            
            ax1.plot(country_data['year'], country_data['fossil_share'], 
                   label=f'{country}-化石', linewidth=2, alpha=0.8)
            ax1.plot(country_data['year'], country_data['renewables_share'], 
                   label=f'{country}-可再生', linewidth=2, linestyle='--', alpha=0.8)
    
    ax1.set_xlabel('年份')
    ax1.set_ylabel('能源占比 (%)')
    ax1.set_title('各国能源结构转型路径')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. GDP与能源结构的散点拟合图
    for country in df['country'].unique():
        country_data = df[df['country'] == country].dropna(subset=['gdp', 'fossil_share'])
        if len(country_data) > 3:
            x = country_data['gdp'] / 1e9  # 转换为十亿美元
            y = country_data['fossil_share']
            
            # 线性拟合
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            line = slope * x + intercept
            
            ax2.scatter(x, y, alpha=0.6, s=50, label=country)
            ax2.plot(x, line, alpha=0.8, linewidth=2, 
                   label=f'{country}拟合 (R²={r_value**2:.3f})')
    
    ax2.set_xlabel('GDP (十亿美元)')
    ax2.set_ylabel('化石能源占比 (%)')
    ax2.set_title('GDP与化石能源占比的线性拟合')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 能源结构变化的热力图
    pivot_data = df.pivot_table(index='country', columns='year', 
                              values='fossil_share', aggfunc='mean')
    
    if not pivot_data.empty:
        im = ax3.imshow(pivot_data.values, cmap='RdYlBu_r', aspect='auto')
        ax3.set_xticks(range(len(pivot_data.columns)))
        ax3.set_xticklabels([str(int(col)) for col in pivot_data.columns], rotation=45)
        ax3.set_yticks(range(len(pivot_data.index)))
        ax3.set_yticklabels(pivot_data.index)
        ax3.set_title('各国化石能源占比变化热力图')
        plt.colorbar(im, ax=ax3, label='化石能源占比 (%)')
    
    # 4. 相关系数分析
    correlations = []
    countries = []
    for country in df['country'].unique():
        country_data = df[df['country'] == country].dropna(subset=['gdp', 'fossil_share'])
        if len(country_data) > 5:
            corr = country_data['gdp'].corr(country_data['fossil_share'])
            correlations.append(corr)
            countries.append(country)
    
    if correlations:
        bars = ax4.bar(countries, correlations, color=['red' if x>0 else 'blue' for x in correlations])
        ax4.set_xlabel('国家')
        ax4.set_ylabel('相关系数')
        ax4.set_title('GDP与化石能源占比的相关系数')
        ax4.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                   f'{corr:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def create_environmental_impact_analysis(df):
    """创建环境影响分析"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 检查是否有温室气体排放数据
    has_ghg = 'greenhouse_gas_emissions' in df.columns and df['greenhouse_gas_emissions'].notna().any()
    
    if has_ghg:
        # 1. 温室气体排放趋势
        for country in df['country'].unique():
            country_data = df[df['country'] == country].dropna(subset=['greenhouse_gas_emissions'])
            if len(country_data) > 3:
                ax1.plot(country_data['year'], country_data['greenhouse_gas_emissions'], 
                       marker='o', linewidth=2, label=country)
        
        ax1.set_xlabel('年份')
        ax1.set_ylabel('温室气体排放量')
        ax1.set_title('温室气体排放趋势')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. 能源强度分析（如有碳强度数据）
    if 'carbon_intensity_elec' in df.columns:
        for country in df['country'].unique():
            country_data = df[df['country'] == country].dropna(subset=['carbon_intensity_elec', 'gdp'])
            if len(country_data) > 3:
                ax2.plot(country_data['year'], country_data['carbon_intensity_elec'], 
                       marker='s', linewidth=2, label=country)
        
        ax2.set_xlabel('年份')
        ax2.set_ylabel('碳强度')
        ax2.set_title('电力碳强度趋势')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. GDP增长与环境指标的散点图
    for country in df['country'].unique():
        country_data = df[df['country'] == country].dropna(subset=['gdp', 'fossil_share'])
        if len(country_data) > 3:
            # 计算变化率
            country_data = country_data.sort_values('year')
            country_data['gdp_growth'] = country_data['gdp'].pct_change() * 100
            country_data['fossil_change'] = country_data['fossil_share'].diff()
            
            valid_data = country_data.dropna(subset=['gdp_growth', 'fossil_change'])
            if len(valid_data) > 2:
                ax3.scatter(valid_data['gdp_growth'], valid_data['fossil_change'], 
                          label=country, alpha=0.7, s=60)
    
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('GDP增长率 (%)')
    ax3.set_ylabel('化石能源占比变化 (%)')
    ax3.set_title('GDP增长与化石能源占比变化关系')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 可持续发展路径分析
    latest_data = df[df['year'] == df['year'].max()].dropna(subset=['gdp', 'renewables_share'])
    if not latest_data.empty:
        ax4.scatter(latest_data['gdp']/1e9, latest_data['renewables_share'], 
                  s=100, alpha=0.7)
        
        for i, row in latest_data.iterrows():
            ax4.annotate(row['country'], (row['gdp']/1e9, row['renewables_share']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax4.set_xlabel('GDP (十亿美元)')
        ax4.set_ylabel('可再生能源占比 (%)')
        ax4.set_title('当前各国经济水平与可再生能源发展')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    st.set_page_config(page_title="GDP与能源环境分析", page_icon="🌍", layout="wide")
    
    st.title("🌍 GDP增长对能源结构与环境影响分析")
    st.markdown("""
    本分析基于全球能源消费数据，研究GDP增长如何影响能源结构转型和环境影响。
    通过多种可视化方法展示经济发展与能源环境的关系。
    """)
    
    # 加载数据
    df = load_and_preprocess_data()
    
    # 侧边栏
    st.sidebar.header("分析选项")
    analysis_type = st.sidebar.selectbox(
        "选择分析类型",
        ["综合趋势分析", "能源转型分析", "环境影响分析", "完整报告"]
    )
    
    # 数据显示
    if st.sidebar.checkbox("显示原始数据"):
        st.subheader("数据概览")
        st.dataframe(df.head(100))
        
        st.subheader("数据统计")
        st.write(df.describe())
    
    # 主分析区域
    if analysis_type == "综合趋势分析":
        st.header("GDP与能源结构综合趋势")
        fig1 = create_gdp_energy_trends(df)
        st.pyplot(fig1)
        
        st.markdown("""
        **主要观察:**
        - GDP增长通常伴随着能源结构的转型
        - 发达国家往往有更低化石能源占比和更高可再生能源占比
        - 不同国家的能源转型路径存在显著差异
        """)
    
    elif analysis_type == "能源转型分析":
        st.header("能源转型深度分析")
        fig2 = create_energy_transition_analysis(df)
        st.pyplot(fig2)
        
        st.markdown("""
        **关键发现:**
        - GDP与化石能源占比的相关性因国家发展阶段而异
        - 能源转型路径显示从化石燃料向可再生能源的转变
        - 热力图清晰展示各国能源结构的时间演变
        """)
    
    elif analysis_type == "环境影响分析":
        st.header("环境影响评估")
        fig3 = create_environmental_impact_analysis(df)
        st.pyplot(fig3)
        
        st.markdown("""
        **环境影响洞察:**
        - 经济增长与环境保护需要平衡
        - 可再生能源发展是降低环境影响的关键
        - 不同经济增长模式对环境的影响差异显著
        """)
    
    else:  # 完整报告
        st.header("完整分析报告")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("核心结论")
            st.markdown("""
            📊 **主要发现:**
            
            1. **经济发展与能源转型密切相关**
               - 高收入国家通常有更清洁的能源结构
               - GDP增长为能源转型提供资金支持
            
            2. **能源结构决定环境影响**
               - 化石能源主导的结构导致更高碳排放
               - 可再生能源占比提升改善环境表现
            
            3. **转型路径多样化**
               - 不同国家根据资源禀赋选择不同路径
               - 政策引导在转型中起关键作用
            """)
        
        with col2:
            st.subheader("政策建议")
            st.markdown("""
            🎯 **战略方向:**
            
            • **促进清洁能源投资**
              - 鼓励可再生能源技术研发
              - 建立绿色金融体系
            
            • **优化能源结构**
              - 逐步淘汰高污染能源
              - 发展多元化能源供应
            
            • **加强国际合作**
              - 共享最佳实践
              - 协调气候变化应对
            """)
        
        # 显示所有图表
        st.subheader("综合分析图表")
        fig1 = create_gdp_energy_trends(df)
        st.pyplot(fig1)
        
        fig2 = create_energy_transition_analysis(df)
        st.pyplot(fig2)
        
        fig3 = create_environmental_impact_analysis(df)
        st.pyplot(fig3)
    
    # 结论部分
    st.markdown("---")
    st.subheader("📈 总体结论")
    
    conclusion_col1, conclusion_col2, conclusion_col3 = st.columns(3)
    
    with conclusion_col1:
        st.info("""
        **经济发展驱动转型**
        GDP增长为能源结构优化提供经济基础，
        推动从传统化石能源向清洁能源转型
        """)
    
    with conclusion_col2:
        st.success("""
        **环境改善可期**
        随着可再生能源占比提升，
        经济增长与环境影响的脱钩成为可能
        """)
    
    with conclusion_col3:
        st.warning("""
        **挑战与机遇并存**
        转型过程需要平衡经济、能源、环境目标，
        政策引导和技术创新至关重要
        """)

if __name__ == "__main__":
    main()