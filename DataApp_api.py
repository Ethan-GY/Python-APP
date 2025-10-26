import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import streamlit as st
import requests
import json

@st.cache_data
def load_and_clean_tmdb():
    df = pd.read_csv('tmdb_5000_movies.csv')
    
    # Convert release_date to datetime
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year
    
    # Filter invalid budget/revenue
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)].copy()
    
    # Calculate ROI (Return on Investment), limit extreme values
    df['roi'] = np.where(df['budget'] > 0, df['revenue'] / df['budget'], np.nan)
    df = df[df['roi'] <= 100]  # Remove outliers with ROI > 100 (e.g., The Blair Witch Project)
    
    # Parse genres field
    def parse_genres(x):
        try:
            return [g['name'] for g in ast.literal_eval(x)]
        except:
            return []
    
    df['genres_list'] = df['genres'].apply(parse_genres)
    
    return df

def analyze_data_with_api(df_filtered, genre_stats, analysis_type="overall"):
    """
    Call QWEN API to analyze movie data
    """
    api_key = "sk-bb0301c0ab834446b534fd3e6074622a"
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"  # QWEN API endpoint
    
    # Prepare data summary for API
    data_summary = {
        "total_movies": len(df_filtered),
        "time_period": f"{df_filtered['release_year'].min()}-{df_filtered['release_year'].max()}",
        "avg_budget": df_filtered['budget'].mean(),
        "avg_revenue": df_filtered['revenue'].mean(),
        "avg_roi": df_filtered['roi'].mean(),
        "avg_rating": df_filtered['vote_average'].mean(),
        "top_genres": genre_stats.nlargest(5, 'movie_count')[['genre', 'movie_count', 'avg_rating']].to_dict('records')
    }
    
    # Different prompt templates based on analysis type
    prompts = {
        "overall": f"""
        Analyze this movie dataset and provide key insights:
        
        Dataset Overview:
        - Period: {data_summary['time_period']}
        - Total Movies: {data_summary['total_movies']:,}
        - Average Budget: ${data_summary['avg_budget']:,.0f}
        - Average Revenue: ${data_summary['avg_revenue']:,.0f}
        - Average ROI: {data_summary['avg_roi']:.2f}x
        - Average Rating: {data_summary['avg_rating']:.2f}/10
        
        Top Genres by Movie Count:
        {chr(10).join([f"- {g['genre']}: {g['movie_count']} movies, {g['avg_rating']:.2f} avg rating" for g in data_summary['top_genres']])}
        
        Please provide:
        1. Key trends and patterns in the data
        2. Most profitable genres and their characteristics
        3. Relationship between budget, revenue, and ratings
        4. Business recommendations for film producers
        5. Any surprising findings or outliers
        
        Provide the analysis in clear, structured English.
        """,
        
        "financial": f"""
        Perform financial analysis on this movie dataset:
        
        Financial Metrics:
        - Average ROI: {data_summary['avg_roi']:.2f}x
        - Average Budget: ${data_summary['avg_budget']:,.0f}
        - Average Revenue: ${data_summary['avg_revenue']:,.0f}
        - Period: {data_summary['time_period']}
        
        Genre Performance:
        {chr(10).join([f"- {g['genre']}: ROI calculation based on available data" for g in data_summary['top_genres']])}
        
        Analyze:
        1. Which genres provide best ROI?
        2. Optimal budget ranges for maximum returns
        3. Risk vs reward patterns
        4. Investment recommendations
        """,
        
        "genre_comparison": f"""
        Compare movie genres performance:
        
        Genre Statistics:
        {chr(10).join([f"- {g['genre']}: {g['movie_count']} movies, Rating: {g['avg_rating']:.2f}" for g in data_summary['top_genres']])}
        
        Analyze:
        1. Critical vs commercial success by genre
        2. Genre popularity trends
        3. Audience preferences
        4. Genre evolution over time
        """
    }
    
    prompt = prompts.get(analysis_type, prompts["overall"])
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "qwen-turbo",  # Adjust based on available QWEN models
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a data analyst specializing in film industry analytics. Provide insightful, data-driven analysis based on the movie dataset provided. Focus on practical business insights and clear explanations."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
        },
        "parameters": {
            "max_tokens": 2000,
            "temperature": 0.7
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        # Extract the analysis text from QWEN response
        if 'output' in result and 'text' in result['output']:
            return result['output']['text']
        elif 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            return "Analysis completed. Key insights: [API response format unexpected]"
            
    except requests.exceptions.RequestException as e:
        return f"API call failed: {str(e)}"
    except Exception as e:
        return f"Error processing API response: {str(e)}"

df = load_and_clean_tmdb()

st.sidebar.header("🔍 Filters")

# Year range slider
min_year = int(df['release_year'].min())
max_year = int(df['release_year'].max())
year_range = st.sidebar.slider("Select Year Range", min_year, max_year, (1990, 2015))

# Genre multi-select
all_genres = set(g for sublist in df['genres_list'] for g in sublist)
selected_genres = st.sidebar.multiselect(
    "Select Genres",
    sorted(all_genres),
    default=["Drama", "Action", "Comedy"]
)

# Rating filter slider
rating_range = st.sidebar.slider("Filter by Rating", 0.0, 10.0, (6.0, 8.0))

# Filter data
df_filtered = df[
    (df['release_year'] >= year_range[0]) &
    (df['release_year'] <= year_range[1]) &
    (df['vote_average'] >= rating_range[0]) & 
    (df['vote_average'] <= rating_range[1])
]

if selected_genres:
    df_filtered = df_filtered[
        df_filtered['genres_list'].apply(lambda x: any(g in selected_genres for g in x))
    ]

st.header("1. Budget vs Revenue Analysis")

fig, ax = plt.subplots(figsize=(10, 6))

# Use numpy for color mapping
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

# Add colorbar
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('ROI (Revenue / Budget)', rotation=270, labelpad=15)

st.pyplot(fig)

# Expand genres for analysis
df_exploded = df_filtered.explode('genres_list').rename(columns={'genres_list': 'genre'})

# Aggregate statistics using pandas + numpy
genre_stats = df_exploded.groupby('genre').agg(
    avg_rating=('vote_average', 'mean'),
    movie_count=('title', 'size'),
    total_revenue=('revenue', 'sum'),
    avg_budget=('budget', 'mean'),
    avg_roi=('roi', 'mean')
).reset_index()

# Filter low-frequency genres
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
ax.set_title('Average IMDb-style Rating by Genre (Minimum 5 Movies)', fontsize=14)
ax.set_xlim(0, 10)

# Add value labels at the end of bars
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.05, bar.get_y() + bar.get_height()/2,
            f'{width:.2f}', va='center', fontsize=10)

st.pyplot(fig)

st.header("3. Genre Bubble Chart: Revenue vs Rating")

# Normalize bubble sizes (avoid overly large bubbles)
sizes = np.sqrt(genre_stats['movie_count']) * 20  # sqrt to mitigate count differences

fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(
    genre_stats['total_revenue'],
    genre_stats['avg_rating'],
    s=sizes,
    alpha=0.6,
    c=genre_stats.index,  # Different colors for different genres
    cmap='tab20'
)

ax.set_xlabel('Total Revenue (USD)', fontsize=12)
ax.set_ylabel('Average Rating', fontsize=12)
ax.set_title('Genre Comparison: Bubble Size = √(Movie Count)', fontsize=14)
ax.set_xscale('log')

# Add genre labels
for i, row in genre_stats.iterrows():
    ax.annotate(row['genre'], 
                (row['total_revenue'], row['avg_rating']),
                fontsize=8, alpha=0.8)

st.pyplot(fig)

# API Analysis Section
st.header("4. AI-Powered Data Analysis")

analysis_type = st.selectbox(
    "Select Analysis Type",
    ["overall", "financial", "genre_comparison"],
    help="Choose the type of analysis you want to perform on the filtered data"
)

if st.button("Generate AI Analysis"):
    with st.spinner("Analyzing data with QWEN API..."):
        analysis_result = analyze_data_with_api(df_filtered, genre_stats, analysis_type)
        
        st.subheader("AI Analysis Results")
        st.write(analysis_result)

# Data Summary
st.header("5. Dataset Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Movies", f"{len(df_filtered):,}")

with col2:
    st.metric("Average Rating", f"{df_filtered['vote_average'].mean():.2f}")

with col3:
    st.metric("Average ROI", f"{df_filtered['roi'].mean():.2f}x")

with col4:
    st.metric("Time Period", f"{year_range[0]}-{year_range[1]}")