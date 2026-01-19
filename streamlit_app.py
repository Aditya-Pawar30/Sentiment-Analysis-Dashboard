import streamlit as st # pyright: ignore[reportMissingImports]
import plotly.express as px # pyright: ignore[reportMissingImports]
import plotly.graph_objects as go # pyright: ignore[reportMissingImports]
from sentiment_analyzer import SentimentAnalyzer
from database import DatabaseManager
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Real-Time Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# Initialize components
@st.cache_resource
def load_analyzer():
    return SentimentAnalyzer()

@st.cache_resource
def load_database():
    return DatabaseManager()

analyzer = load_analyzer()
db = load_database()

# Header
st.title("📊 Real-Time Sentiment Analysis Dashboard")
st.markdown("**Analyze text sentiment using AI-powered transformer models**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.info("**Model:** DistilBERT (SST-2)")
    
    st.header("📖 About")
    st.write("""
    This dashboard uses a fine-tuned DistilBERT model 
    to analyze sentiment in real-time.
    """)

# Main content - Create tabs
tab1, tab2, tab3 = st.tabs(["🔍 Analyze", "📈 Analytics", "📝 History"])

with tab1:
    st.header("Analyze Text Sentiment")
    
    text_input = st.text_area(
        "Enter text to analyze:",
        placeholder="Type your text here... (e.g., 'I love this product!')",
        height=150
    )
    
    if st.button("🚀 Analyze", type="primary"):
        if text_input:
            with st.spinner("Analyzing sentiment..."):
                result = analyzer.analyze(text_input)
                db.save_sentiment(result)
                
                # Display result
                sentiment_emoji = "😊" if result['sentiment'] == 'positive' else "😞"
                sentiment_color = "green" if result['sentiment'] == 'positive' else "red"
                
                st.markdown("### Analysis Result")
                st.markdown(f"**Sentiment:** :{sentiment_color}[{result['sentiment'].upper()} {sentiment_emoji}]")
                st.markdown(f"**Confidence:** {result['confidence']:.2%}")
                
                # Score breakdown
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Positive Score", f"{result['scores']['positive']:.2%}")
                with col2:
                    st.metric("Negative Score", f"{result['scores']['negative']:.2%}")
        else:
            st.warning("Please enter some text to analyze!")

with tab2:
    st.header("📈 Analytics Dashboard")
    
    # Get statistics
    stats = db.get_statistics()
    
    # KPI Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Analyzed", stats['total_count'])
    with col2:
        st.metric("Positive Rate", f"{stats['positive_rate']:.1f}%")
    with col3:
        st.metric("Negative Rate", f"{stats['negative_rate']:.1f}%")
    
    if stats['total_count'] > 0:
        # Get recent records for visualization
        df = db.get_recent_records(limit=100)
        
        # Sentiment distribution pie chart
        sentiment_counts = df['sentiment'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color=sentiment_counts.index,
            color_discrete_map={'positive': '#00CC96', 'negative': '#EF553B'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available yet. Start analyzing some text!")

with tab3:
    st.header("📝 Analysis History")
    
    df = db.get_recent_records(limit=50)
    
    if not df.empty:
        st.write(f"Showing last {len(df)} analyses")
        
        for idx, row in df.iterrows():
            sentiment_emoji = "😊" if row['sentiment'] == 'positive' else "😞"
            
            with st.expander(f"{sentiment_emoji} {row['text'][:80]}..."):
                st.markdown(f"**Sentiment:** {row['sentiment'].upper()}")
                st.markdown(f"**Confidence:** {row['confidence']:.2%}")
                st.markdown(f"**Time:** {row['timestamp']}")
    else:
        st.info("No analysis history yet!")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & Transformers")