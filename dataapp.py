import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import streamlit as st

@st.cache_data
def load_and_clean_tmdb():
    df = pd.read_csv('tmdb_5000_movies.csv')
    
    # 转换 release_date 为 datetime
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year
    
    # 过滤无效预算/票房
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)].copy()
    
    # 计算 ROI（投资回报率），限制极端值
    df['roi'] = np.where(df['budget'] > 0, df['revenue'] / df['budget'], np.nan)
    df = df[df['roi'] <= 100]  # 剔除 ROI > 100 的异常点（如《女巫布莱尔》）
    
    # 解析 genres 字段
    def parse_genres(x):
        try:
            return [g['name'] for g in ast.literal_eval(x)]
        except:
            return []
    
    df['genres_list'] = df['genres'].apply(parse_genres)
    
    return df

df = load_and_clean_tmdb()

st.sidebar.header("🔍 Filters")

# 年份滑块
min_year = int(df['release_year'].min())
max_year = int(df['release_year'].max())
year_range = st.sidebar.slider("Select Year Range", min_year, max_year, (1990, 2015))

# 类型多选
all_genres = set(g for sublist in df['genres_list'] for g in sublist)
selected_genres = st.sidebar.multiselect(
    "Select Genres",
    sorted(all_genres),
    default=["Drama", "Action", "Comedy"]
)

# 筛选数据
df_filtered = df[
    (df['release_year'] >= year_range[0]) &
    (df['release_year'] <= year_range[1])
]

if selected_genres:
    df_filtered = df_filtered[
        df_filtered['genres_list'].apply(lambda x: any(g in selected_genres for g in x))
    ]

st.header("1. Budget vs Revenue (Matplotlib)")

fig, ax = plt.subplots(figsize=(10, 6))

# 使用 numpy 处理颜色映射
roi = df_filtered['roi'].values
sc = ax.scatter(
    df_filtered['budget'],
    df_filtered['revenue'],
    c=roi,
    cmap='viridis',
    alpha=0.7,
    s=30
)

ax.set_xlabel('Budget (USD)', fontsize=12)
ax.set_ylabel('Revenue (USD)', fontsize=12)
ax.set_title('Budget vs Revenue (Color = ROI)', fontsize=14)
ax.set_xscale('log')
ax.set_yscale('log')

# 添加 colorbar
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('ROI (Revenue / Budget)', rotation=270, labelpad=15)

st.pyplot(fig)

# 展开 genres
df_exploded = df_filtered.explode('genres_list').rename(columns={'genres_list': 'genre'})

# 聚合统计（使用 pandas + numpy）
genre_stats = df_exploded.groupby('genre').agg(
    avg_rating=('vote_average', 'mean'),
    movie_count=('title', 'size'),
    total_revenue=('revenue', 'sum')
).reset_index()

# 过滤低频类型
genre_stats = genre_stats[genre_stats['movie_count'] >= 5]

st.header("2. Average Rating by Genre")

genre_stats_sorted = genre_stats.sort_values('avg_rating')

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(
    genre_stats_sorted['genre'],
    genre_stats_sorted['avg_rating'],
    color='steelblue'
)

ax.set_xlabel('Average Rating', fontsize=12)
ax.set_title('Average IMDb-style Rating by Genre (Min 5 Movies)', fontsize=14)
ax.set_xlim(0, 10)

# 在柱子末端标注数值
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.05, bar.get_y() + bar.get_height()/2,
            f'{width:.2f}', va='center', fontsize=10)

st.pyplot(fig)

st.header("3. Genre Bubble Chart: Revenue vs Rating")

# 归一化气泡大小（避免过大）
sizes = np.sqrt(genre_stats['movie_count']) * 20  # sqrt 缓解数量差异

fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(
    genre_stats['total_revenue'],
    genre_stats['avg_rating'],
    s=sizes,
    alpha=0.6,
    c=genre_stats.index,  # 不同颜色区分类型
    cmap='tab20'
)

ax.set_xlabel('Total Revenue (USD)', fontsize=12)
ax.set_ylabel('Average Rating', fontsize=12)
ax.set_title('Genre Comparison: Bubble Size = √(Movie Count)', fontsize=14)
ax.set_xscale('log')

# 添加图例说明气泡含义
for i, row in genre_stats.iterrows():
    ax.annotate(row['genre'], 
                (row['total_revenue'], row['avg_rating']),
                fontsize=8, alpha=0.8)

st.pyplot(fig)