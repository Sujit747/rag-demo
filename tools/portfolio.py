from langchain_core.tools import BaseTool
from models.database import DatabaseManager
from models.data_models import Portfolio
from datetime import datetime
import yfinance as yf
from typing import Optional

class PortfolioAnalysisTool(BaseTool):
    name: str = "portfolio_analysis"
    description: str = "Analyze user's portfolio performance"
    
    _db_manager: Optional[DatabaseManager] = None
    
    @classmethod
    def set_db_manager(cls, db_manager: DatabaseManager):
        cls._db_manager = db_manager
    
    def _run(self, user_id: str) -> str:
        try:
            portfolio = self._db_manager.get_portfolio(user_id)
            if not portfolio:
                default_holdings = {
                    'RELIANCE.NS': 10,
                    'TCS.NS': 5,
                    'HDFCBANK.NS': 8,
                    'INFY.NS': 12
                }
                default_portfolio = Portfolio(
                    user_id=user_id,
                    holdings=default_holdings,
                    total_value=0.0,
                    last_updated=datetime.now()
                )
                self._db_manager.save_portfolio(default_portfolio)
                portfolio = default_portfolio
            portfolio_data = portfolio.holdings
            
            total_value = 0
            performance_summary = []
            
            for symbol, quantity in portfolio_data.items():
                stock = yf.Ticker(symbol)
                hist = stock.history(period="1mo")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    month_ago_price = hist['Close'].iloc[0]
                    value = current_price * quantity
                    total_value += value
                    
                    monthly_return = ((current_price - month_ago_price) / month_ago_price) * 100
                    
                    performance_summary.append({
                        'symbol': symbol,
                        'quantity': quantity,
                        'current_value': value,
                        'monthly_return': monthly_return
                    })
            
            analysis = f"""
            Portfolio Analysis for User {user_id}:
            
            Total Portfolio Value: ₹{total_value:.2f}
            
            Holdings Performance (1 Month):
            """
            
            for holding in performance_summary:
                analysis += f"\n• {holding['symbol']}: ₹{holding['current_value']:.2f} ({holding['monthly_return']:+.2f}%)"
            
            avg_return = sum(h['monthly_return'] for h in performance_summary) / len(performance_summary)
            
            if avg_return > 5:
                analysis += "\n\n✅ Your portfolio is performing well! Consider rebalancing if any single stock exceeds 30% allocation."
            elif avg_return > 0:
                analysis += "\n\n📈 Moderate performance. Consider diversifying into different sectors."
            else:
                analysis += "\n\n⚠️ Portfolio is underperforming. Consider reviewing your stock selection and risk management."
            
            return analysis
            
        except Exception as e:
            return f"Error analyzing portfolio: {str(e)}"
    
    async def _arun(self, user_id: str) -> str:
        return self._run(user_id)