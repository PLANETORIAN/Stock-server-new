import os
import sys
import logging
from typing import List
import pandas as pd
from stock_recommender import StockRecommender

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_tickers(file_path: str) -> List[str]:
    """
    Load tickers from a file
    """
    try:
        with open(file_path, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
        return tickers
    except Exception as e:
        logger.error(f"Error loading tickers from {file_path}: {str(e)}")
        return []

def main():
    # Initialize recommender
    recommender = StockRecommender()
    
    # Load tickers from file
    tickers_file = "ml_model/tickers.txt"
    if not os.path.exists(tickers_file):
        logger.error(f"Tickers file not found: {tickers_file}")
        return
    
    tickers = load_tickers(tickers_file)
    if not tickers:
        logger.error("No tickers loaded")
        return
    
    logger.info(f"Loaded {len(tickers)} tickers")
    
    # Train the model
    logger.info("Training recommendation model...")
    success = recommender.train(tickers)
    
    if not success:
        logger.error("Failed to train model")
        return
    
    # Test recommendations
    logger.info("\nTesting recommendations...")
    
    # Test similar stock recommendations
    test_ticker = tickers[0]
    logger.info(f"\nSimilar stocks to {test_ticker}:")
    similar_stocks = recommender.recommend(test_ticker, n_recommendations=5)
    for stock in similar_stocks:
        logger.info(f"\nTicker: {stock['ticker']}")
        logger.info(f"Name: {stock['name']}")
        logger.info(f"Sector: {stock['sector']}")
        logger.info(f"Similarity Score: {stock['similarity_score']:.3f}")
        logger.info("Metrics:")
        for metric, value in stock['metrics'].items():
            logger.info(f"  {metric}: {value:.3f}")
    
    # Test metric-based recommendations
    logger.info("\nStocks matching target metrics:")
    target_metrics = {
        'beta': 1.2,
        'sharpe_ratio': 1.5,
        'max_drawdown': -0.2
    }
    metric_based = recommender.recommend_by_metrics(target_metrics, n_recommendations=5)
    for stock in metric_based:
        logger.info(f"\nTicker: {stock['ticker']}")
        logger.info(f"Name: {stock['name']}")
        logger.info(f"Sector: {stock['sector']}")
        logger.info(f"Similarity Score: {stock['similarity_score']:.3f}")
        logger.info("Metrics:")
        for metric, value in stock['metrics'].items():
            logger.info(f"  {metric}: {value:.3f}")
    
    # Test sector recommendations
    logger.info("\nTop performing stocks by sector:")
    sectors = recommender.stock_data['sector'].unique()
    for sector in sectors[:3]:  # Test first 3 sectors
        logger.info(f"\nTop stocks in {sector}:")
        sector_stocks = recommender.get_sector_recommendations(sector, n_recommendations=3)
        for stock in sector_stocks:
            logger.info(f"\nTicker: {stock['ticker']}")
            logger.info(f"Name: {stock['name']}")
            logger.info("Metrics:")
            for metric, value in stock['metrics'].items():
                logger.info(f"  {metric}: {value:.3f}")

if __name__ == "__main__":
    main() 