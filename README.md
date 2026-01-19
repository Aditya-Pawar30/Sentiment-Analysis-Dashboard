# Real-Time Sentiment Analysis Dashboard

A Streamlit-based web application for real-time sentiment analysis using AI-powered transformer models.

## Features

- **Real-time Sentiment Analysis**: Analyze text sentiment using DistilBERT model fine-tuned on SST-2 dataset
- **Interactive Dashboard**: Clean, modern UI with multiple tabs for analysis, analytics, and history
- **Data Persistence**: SQLite database to store analysis history and statistics
- **Visual Analytics**: Pie charts and metrics for sentiment distribution
- **Confidence Scores**: Detailed breakdown of positive/negative confidence scores

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirement.txt
   ```

   Or if using Python module:
   ```bash
   python -m pip install -r requirement.txt
   ```

## Usage

1. **Run the Streamlit application**:
   ```bash
   streamlit run streamlit_app.py
   ```

   Or if streamlit command is not in PATH:
   ```bash
   python -m streamlit run streamlit_app.py
   ```

2. **Open your browser** to the URL displayed (usually http://localhost:8501)

3. **Start analyzing text**:
   - Go to the "🔍 Analyze" tab
   - Enter text in the text area
   - Click "🚀 Analyze" to get sentiment results

## Project Structure

- `streamlit_app.py`: Main Streamlit application with UI and tabs
- `sentiment_analyzer.py`: Sentiment analysis logic using Transformers
- `database.py`: Database management with SQLAlchemy
- `requirement.txt`: Python dependencies
- `sentiment_analysis.db`: SQLite database (created automatically)

## Model Information

- **Model**: DistilBERT base uncased fine-tuned on SST-2
- **Task**: Binary sentiment classification (positive/negative)
- **Framework**: Hugging Face Transformers + PyTorch

## Dependencies

- streamlit: Web app framework
- transformers: Hugging Face transformers library
- torch: PyTorch deep learning framework
- pandas: Data manipulation
- plotly: Interactive charts
- sqlalchemy: Database ORM

## Troubleshooting

If you encounter issues:

1. **Module not found**: Ensure all dependencies are installed
2. **Streamlit command not found**: Use `python -m streamlit run streamlit_app.py`
3. **Model download issues**: Check internet connection for first run
4. **Database issues**: Delete `sentiment_analysis.db` to reset

## License

Built with ❤️ using open-source libraries.
