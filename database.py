from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import pandas as pd

Base = declarative_base()

class SentimentRecord(Base):
    __tablename__ = 'sentiment_records'
    
    id = Column(Integer, primary_key=True)
    text = Column(String, nullable=False)
    sentiment = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    positive_score = Column(Float)
    negative_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.now)

class DatabaseManager:
    def __init__(self, db_path="sentiment_analysis.db"):
        """Initialize database connection"""
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def save_sentiment(self, result):
        """Save sentiment analysis result to database"""
        record = SentimentRecord(
            text=result['text'],
            sentiment=result['sentiment'],
            confidence=result['confidence'],
            positive_score=result['scores']['positive'],
            negative_score=result['scores']['negative'],
            timestamp=result['timestamp']
        )
        self.session.add(record)
        self.session.commit()
    
    def get_recent_records(self, limit=100):
        """Get recent sentiment records"""
        records = self.session.query(SentimentRecord)\
            .order_by(SentimentRecord.timestamp.desc())\
            .limit(limit)\
            .all()
        
        return pd.DataFrame([{
            'text': r.text,
            'sentiment': r.sentiment,
            'confidence': r.confidence,
            'positive_score': r.positive_score,
            'negative_score': r.negative_score,
            'timestamp': r.timestamp
        } for r in records])
    
    def get_statistics(self):
        """Get overall statistics"""
        records = self.session.query(SentimentRecord).all()
        
        if not records:
            return {
                'total_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'positive_rate': 0,
                'negative_rate': 0
            }
        
        total = len(records)
        positive = sum(1 for r in records if r.sentiment == 'positive')
        negative = sum(1 for r in records if r.sentiment == 'negative')
        
        return {
            'total_count': total,
            'positive_count': positive,
            'negative_count': negative,
            'positive_rate': (positive / total * 100) if total > 0 else 0,
            'negative_rate': (negative / total * 100) if total > 0 else 0
        }