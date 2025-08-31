# Financial Assistant with LangChain, LangGraph & LlamaIndex - FAISS VERSION
# 2-Day Demo Implementation - UPDATED WITH FAISS
import streamlit.components.v1 as components
import os
import streamlit as st
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import json
from typing import Dict, List, Any, Optional
import sqlite3
from dataclasses import dataclass, asdict
import asyncio
from pathlib import Path
from typing_extensions import TypedDict
from typing import List, Dict, Any
# FIXED IMPORTS - Updated to use the correct packages
from langchain_openai import ChatOpenAI  # FIXED: Updated import
from langchain.agents import AgentExecutor
from langchain.tools import BaseTool
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.base import BaseCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()
# LangGraph imports for workflow orchestration
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

# LlamaIndex with FAISS imports
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.faiss import FaissVectorStore  # CHANGED: Using FAISS instead of Chroma
from llama_index.embeddings.openai import OpenAIEmbedding
import faiss  # ADDED: FAISS import
import numpy as np  # ADDED: For FAISS operations
import pickle  # ADDED: For metadata persistence
from langchain_core.tools import BaseTool
from datetime import datetime
import yfinance as yf
import logging
from typing import Optional


# ============================================================================
# 1. DATA MODELS & STORAGE
# ============================================================================

@dataclass
class UserProfile:
    user_id: str
    risk_tolerance: str  # Conservative, Moderate, Aggressive
    investment_goals: List[str]
    monthly_sip_amount: float
    preferred_sectors: List[str]
    created_at: datetime
    last_interaction: datetime

@dataclass
class Portfolio:
    user_id: str
    holdings: Dict[str, float]  # symbol -> quantity
    total_value: float
    last_updated: datetime

@dataclass
class FinancialGoal:
    goal_id: str
    user_id: str
    goal_type: str  # SIP, Tax Planning, Emergency Fund
    target_amount: float
    current_amount: float
    deadline: datetime
    status: str

class DatabaseManager:
    def __init__(self, db_path: str = "financial_assistant.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                risk_tolerance TEXT,
                investment_goals TEXT,
                monthly_sip_amount REAL,
                preferred_sectors TEXT,
                created_at TIMESTAMP,
                last_interaction TIMESTAMP
            )
        ''')
        
        # Portfolio table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolios (
                user_id TEXT PRIMARY KEY,
                holdings TEXT,
                total_value REAL,
                last_updated TIMESTAMP
            )
        ''')
        
        # Financial goals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS financial_goals (
                goal_id TEXT PRIMARY KEY,
                user_id TEXT,
                goal_type TEXT,
                target_amount REAL,
                current_amount REAL,
                deadline TIMESTAMP,
                status TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_user_profile(self, profile: UserProfile):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO user_profiles VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            profile.user_id,
            profile.risk_tolerance,
            json.dumps(profile.investment_goals),
            profile.monthly_sip_amount,
            json.dumps(profile.preferred_sectors),
            profile.created_at,
            profile.last_interaction
        ))
        conn.commit()
        conn.close()
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM user_profiles WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return UserProfile(
                user_id=row[0],
                risk_tolerance=row[1],
                investment_goals=json.loads(row[2]),
                monthly_sip_amount=row[3],
                preferred_sectors=json.loads(row[4]),
                created_at=row[5],
                last_interaction=row[6]
            )
        return None

    def save_portfolio(self, portfolio: Portfolio):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO portfolios VALUES (?, ?, ?, ?)
        ''', (
            portfolio.user_id,
            json.dumps(portfolio.holdings),
            portfolio.total_value,
            portfolio.last_updated
        ))
        conn.commit()
        conn.close()

    def get_portfolio(self, user_id: str) -> Optional[Portfolio]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM portfolios WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return Portfolio(
                user_id=row[0],
                holdings=json.loads(row[1]),
                total_value=row[2],
                last_updated=row[3]
            )
        return None
    
    def save_financial_goals(self, goals: List[FinancialGoal]):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for goal in goals:
            cursor.execute('''
                INSERT OR REPLACE INTO financial_goals VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                goal.goal_id,
                goal.user_id,
                goal.goal_type,
                goal.target_amount,
                goal.current_amount,
                goal.deadline,
                goal.status
            ))
        conn.commit()
        conn.close()

    def get_financial_goals(self, user_id: str) -> List[FinancialGoal]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM financial_goals WHERE user_id = ?', (user_id,))
        rows = cursor.fetchall()
        conn.close()
        return [FinancialGoal(*row) for row in rows]

# ============================================================================
# 2. FINANCIAL DATA TOOLS (LangChain Tools)
# ============================================================================

class StockPriceTool(BaseTool):
    name: str = "stock_price_lookup"
    description: str = "Get current stock price and basic info for Indian/US stocks"
    
    def _run(self, symbol: str) -> str:
        try:
            # Stock symbol mapping for common Indian stocks
            symbol_mapping = {
                'RELIANCE': 'RELIANCE.NS',
                'TCS': 'TCS.NS',
                'HDFC': 'HDFCBANK.NS',
                'HDFCBANK': 'HDFCBANK.NS',
                'INFY': 'INFY.NS',
                'INFOSYS': 'INFY.NS',
                'ICICIBANK': 'ICICIBANK.NS',
                'ICICI': 'ICICIBANK.NS',
                'SBIN': 'SBIN.NS',
                'SBI': 'SBIN.NS',
                'ITC': 'ITC.NS',
                'WIPRO': 'WIPRO.NS',
                'LT': 'LT.NS',
                'LARSEN': 'LT.NS',
                'HCLTECH': 'HCLTECH.NS',
                'HCL': 'HCLTECH.NS',
                'BAJFINANCE': 'BAJFINANCE.NS',
                'BAJAJ': 'BAJFINANCE.NS',
                'MARUTI': 'MARUTI.NS',
                'ASIANPAINT': 'ASIANPAINT.NS',
                'ASIAN': 'ASIANPAINT.NS'
            }
            
            # Clean and normalize symbol
            original_symbol = symbol.upper().strip()
            
            # Check if it's in our mapping
            if original_symbol in symbol_mapping:
                symbol = symbol_mapping[original_symbol]
            # If already has .NS or .BO, use as is
            elif original_symbol.endswith(('.NS', '.BO')):
                symbol = original_symbol
            # For other symbols, try with .NS first
            elif len(original_symbol) <= 10 and original_symbol.isalpha():
                symbol = original_symbol + '.NS'
            else:
                symbol = original_symbol
            
            print(f"Looking up symbol: {symbol}")  # Debug print
            
            stock = yf.Ticker(symbol)
            
            # Try different periods to get data
            hist = None
            for period in ["1d", "5d", "1mo"]:
                try:
                    hist = stock.history(period=period)
                    if not hist.empty:
                        break
                except:
                    continue
            
            if hist is None or hist.empty:
                # Try without .NS for international stocks
                if symbol.endswith('.NS'):
                    symbol_without_ns = symbol.replace('.NS', '')
                    stock = yf.Ticker(symbol_without_ns)
                    hist = stock.history(period="1d")
                
                if hist is None or hist.empty:
                    return f"❌ Could not find data for {original_symbol}. Please check the symbol name."
            
            # Get stock info
            try:
                info = stock.info
            except:
                info = {}
            
            current_price = hist['Close'].iloc[-1]
            previous_close = hist['Open'].iloc[0] if len(hist) > 1 else current_price
            change = current_price - previous_close
            change_pct = (change / previous_close) * 100 if previous_close != 0 else 0
            
            # Format currency based on exchange
            currency = "₹" if symbol.endswith(('.NS', '.BO')) else "$"
            
            return f"""
📈 **{info.get('longName', symbol)} ({symbol})**

💰 **Current Price**: {currency}{current_price:.2f}
📊 **Change**: {currency}{change:.2f} ({change_pct:+.2f}%)
🏢 **Company**: {info.get('shortName', info.get('longName', 'N/A'))}
🏭 **Sector**: {info.get('sector', 'N/A')}
📅 **Last Updated**: {hist.index[-1].strftime('%Y-%m-%d %H:%M')}

💡 **Tip**: Stock data from Yahoo Finance
            """
        except Exception as e:
            return f"❌ Error fetching data for {symbol}: {str(e)}\n\n💡 Try using common names like: RELIANCE, TCS, HDFC, INFY, SBI"
    
    async def _arun(self, symbol: str) -> str:
        return self._run(symbol)

class PortfolioAnalysisTool(BaseTool):
    name: str = "portfolio_analysis"
    description: str = "Analyze user's portfolio performance"
    
    # Use class attribute instead of instance attribute
    _db_manager: Optional[DatabaseManager] = None
    
    @classmethod
    def set_db_manager(cls, db_manager: DatabaseManager):
        cls._db_manager = db_manager
    
    def _run(self, user_id: str) -> str:
        try:
            # Get portfolio from database (mock data for demo)
            portfolio = self._db_manager.get_portfolio(user_id)
            if not portfolio:
                # Create a default portfolio if none exists (for demo)
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
            
            # Generate analysis
            analysis = f"""
            Portfolio Analysis for User {user_id}:
            
            Total Portfolio Value: ₹{total_value:.2f}
            
            Holdings Performance (1 Month):
            """
            
            for holding in performance_summary:
                analysis += f"\n• {holding['symbol']}: ₹{holding['current_value']:.2f} ({holding['monthly_return']:+.2f}%)"
            
            # Add recommendations
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

class SIPReminderTool(BaseTool):
    name: str = "sip_reminder_check"
    description: str = "Check for upcoming SIP payments"
    
    # Use class attribute instead of instance attribute
    _db_manager: Optional[DatabaseManager] = None
    
    @classmethod
    def set_db_manager(cls, db_manager: DatabaseManager):
        cls._db_manager = db_manager
    
    def _run(self, user_id: str) -> str:
        try:
            # Mock SIP data - in real implementation, fetch from database
            goals = self._db_manager.get_financial_goals(user_id)
            sip_data = [
                {'fund_name': goal.goal_type, 'amount': goal.target_amount, 'due_date': goal.deadline.strftime('%dth every month')}
                for goal in goals if goal.goal_type == 'SIP'
            ]
            if not sip_data:
                return "No SIP goals set up. Consider adding some!"
            # Rest of the code remains the same
            
            today = datetime.now()
            reminders = []
            
            # Check which SIPs are due soon
            for sip in sip_data:
                if '15th' in sip['due_date'] and today.day >= 13:
                    reminders.append(f"🔔 SIP Due: {sip['fund_name']} - ₹{sip['amount']} on 15th")
                elif '10th' in sip['due_date'] and today.day >= 8:
                    reminders.append(f"🔔 SIP Due: {sip['fund_name']} - ₹{sip['amount']} on 10th")
                elif '25th' in sip['due_date'] and today.day >= 23:
                    reminders.append(f"🔔 SIP Due: {sip['fund_name']} - ₹{sip['amount']} on 25th")
            
            if reminders:
                return "Your SIP Reminders:\n" + "\n".join(reminders)
            else:
                return "No SIP payments due in the next few days. All good! 👍"
                
        except Exception as e:
            return f"Error checking SIP reminders: {str(e)}"
    
    async def _arun(self, user_id: str) -> str:
        return self._run(user_id)



class AddPortfolioSharesTool(BaseTool):
    name: str = "add_portfolio_shares"
    description: str = "Add shares to user's portfolio"
    
    _db_manager: Optional[DatabaseManager] = None
    
    @classmethod
    def set_db_manager(cls, db_manager: DatabaseManager):
        cls._db_manager = db_manager
    
    def _run(self, user_id: str, symbol: str, quantity: float) -> str:
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        
        try:
            logger.debug(f"AddPortfolioSharesTool: user_id={user_id}, symbol={symbol}, quantity={quantity}")
            # Define symbol mapping at method level
            symbol_mapping = {
                'RELIANCE': 'RELIANCE.NS', 'TCS': 'TCS.NS', 'HDFC': 'HDFCBANK.NS',
                'HDFCBANK': 'HDFCBANK.NS', 'INFY': 'INFY.NS', 'INFOSYS': 'INFY.NS',
                'ICICIBANK': 'ICICIBANK.NS', 'ICICI': 'ICICIBANK.NS', 'SBIN': 'SBIN.NS',
                'SBI': 'SBIN.NS', 'ITC': 'ITC.NS', 'WIPRO': 'WIPRO.NS', 'LT': 'LT.NS',
                'LARSEN': 'LT.NS', 'HCLTECH': 'HCLTECH.NS', 'HCL': 'HCLTECH.NS',
                'BAJFINANCE': 'BAJFINANCE.NS', 'BAJAJ': 'BAJFINANCE.NS',
                'MARUTI': 'MARUTI.NS', 'ASIANPAINT': 'ASIANPAINT.NS', 'ASIAN': 'ASIANPAINT.NS'
            }
            # Clean and normalize symbol
            symbol = symbol.upper().strip()
            logger.debug(f"Original symbol: {symbol}")
            if symbol in symbol_mapping:
                symbol = symbol_mapping[symbol]
            elif not symbol.endswith(('.NS', '.BO')):
                symbol = symbol + '.NS'
            logger.debug(f"Normalized symbol: {symbol}")
            
            # Verify stock data
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1d")
            logger.debug(f"yfinance history for {symbol}: {'non-empty' if not hist.empty else 'empty'}")
            if hist.empty:
                # Retry without .NS for international stocks
                if symbol.endswith('.NS'):
                    symbol = symbol.replace('.NS', '')
                    stock = yf.Ticker(symbol)
                    hist = stock.history(period="1d")
                    logger.debug(f"Retried yfinance history for {symbol}: {'non-empty' if not hist.empty else 'empty'}")
                if hist.empty:
                    return f"❌ Invalid stock symbol: {symbol}. Try RELIANCE, TCS, etc."
            
            # Get or create portfolio
            portfolio = self._db_manager.get_portfolio(user_id)
            logger.debug(f"Portfolio retrieved: {portfolio}")
            if not portfolio:
                portfolio = Portfolio(
                    user_id=user_id,
                    holdings={},
                    total_value=0.0,
                    last_updated=datetime.now()
                )
                logger.debug("Created new portfolio")
            
            # Update holdings
            portfolio.holdings[symbol] = portfolio.holdings.get(symbol, 0) + quantity
            current_price = hist['Close'].iloc[-1]
            portfolio.total_value = sum(
                yf.Ticker(s).history(period="1d")['Close'].iloc[-1] * q
                for s, q in portfolio.holdings.items()
                if not yf.Ticker(s).history(period="1d").empty
            )
            portfolio.last_updated = datetime.now()
            logger.debug(f"Updated holdings: {portfolio.holdings}, total_value: {portfolio.total_value}")
            self._db_manager.save_portfolio(portfolio)
            logger.debug("Portfolio saved to DB")
            
            return f"✅ Added {quantity} shares of {symbol} to your portfolio. Current value: ₹{portfolio.total_value:.2f}"
        except Exception as e:
            logger.error(f"Error in AddPortfolioSharesTool: {str(e)}", exc_info=True)
            return f"❌ Error adding shares: {str(e)}. Please check the symbol or try again later."
    
    async def _arun(self, user_id: str, symbol: str, quantity: float) -> str:
        return self._run(user_id, symbol, quantity)

class SetSIPReminderTool(BaseTool):
    name: str = "set_sip_reminder"
    description: str = "Set a new SIP reminder for the user"
    
    _db_manager: Optional[DatabaseManager] = None
    
    @classmethod
    def set_db_manager(cls, db_manager: DatabaseManager):
        cls._db_manager = db_manager
    
    def _run(self, user_id: str, fund_name: str, amount: float, due_day: int) -> str:
        try:
            if not (1 <= due_day <= 31):
                return "❌ Due day must be between 1 and 31."
            if amount <= 0:
                return "❌ Amount must be positive."
            
            # Create a unique goal_id
            goal_id = f"{user_id}_sip_{fund_name}_{datetime.now().isoformat()}"
            goal = FinancialGoal(
                goal_id=goal_id,
                user_id=user_id,
                goal_type="SIP",
                target_amount=amount,
                current_amount=0.0,
                deadline=datetime.now().replace(day=due_day),
                status="Active"
            )
            self._db_manager.save_financial_goals([goal])
            return f"✅ Set SIP reminder: ₹{amount} for {fund_name} on day {due_day} of each month."
        except Exception as e:
            return f"❌ Error setting SIP reminder: {str(e)}"
    
    async def _arun(self, user_id: str, fund_name: str, amount: float, due_day: int) -> str:
        return self._run(user_id, fund_name, amount, due_day)

# ============================================================================
# 3. LANGGRAPH WORKFLOW STATE & NODES
# ============================================================================

class AgentState(TypedDict):
    messages: List[str]
    user_id: str
    current_task: str
    context: Dict[str, Any]
    tool_results: Dict[str, str]
    final_response: str

class FinancialWorkflowGraph:
    def __init__(self, tools: List[BaseTool], llm, db_manager: DatabaseManager):
        self.tools = {tool.name: tool for tool in tools}
        self.llm = llm
        self.db_manager = db_manager
        self.tool_node = ToolNode(tools)
        self.workflow = StateGraph(AgentState)
        self._build_graph()
    
    def _build_graph(self):
        self.workflow.add_node("classify_intent", self.classify_intent)
        self.workflow.add_node("fetch_user_context", self.fetch_user_context)
        self.workflow.add_node("execute_financial_task", self.execute_financial_task)
        self.workflow.add_node("generate_response", self.generate_response)
        self.workflow.add_node("update_memory", self.update_memory)
        self.workflow.add_edge("classify_intent", "fetch_user_context")
        self.workflow.add_edge("fetch_user_context", "execute_financial_task")
        self.workflow.add_edge("execute_financial_task", "generate_response")
        self.workflow.add_edge("generate_response", "update_memory")
        self.workflow.set_entry_point("classify_intent")
        self.app = self.workflow.compile()
    
    def classify_intent(self, state: AgentState) -> AgentState:
        last_message = state["messages"][-1] if state["messages"] else ""
        intent_prompt = f"""
        Classify the user's financial intent from this message: "{last_message}"
        Possible intents:
        1. stock_lookup - User wants stock price or info
        2. portfolio_analysis - User wants portfolio review
        3. sip_reminder - User wants SIP reminders
        4. add_portfolio_shares - User wants to add shares to portfolio
        5. set_sip_reminder - User wants to set a new SIP reminder
        6. general_advice - General financial advice
        7. goal_tracking - Track financial goals
        Return only the intent name.
        """
        response = self.llm.invoke(intent_prompt)
        intent = response.content.strip().lower()
        valid_intents = ["stock_lookup", "portfolio_analysis", "sip_reminder", "add_portfolio_shares", "set_sip_reminder", "general_advice", "goal_tracking"]
        if intent not in valid_intents:
            intent = "general_advice"
        return {
            **state,
            "current_task": intent,
            "context": {**state["context"], "intent": intent}
        }
    
    def fetch_user_context(self, state: AgentState) -> AgentState:
        profile = self.db_manager.get_user_profile(state["user_id"])
        if profile:
            user_profile = asdict(profile)
        else:
            default_profile = UserProfile(
                user_id=state["user_id"],
                risk_tolerance="Moderate",
                investment_goals=["Retirement", "Wealth Creation"],
                monthly_sip_amount=10000,
                preferred_sectors=["Technology", "Banking"],
                created_at=datetime.now(),
                last_interaction=datetime.now()
            )
            self.db_manager.save_user_profile(default_profile)
            user_profile = asdict(default_profile)
        return {
            **state,
            "context": {**state["context"], "user_profile": user_profile}
        }
    
    def execute_financial_task(self, state: AgentState) -> AgentState:
        intent = state["current_task"]
        last_message = state["messages"][-1] if state["messages"] else ""
        tool_results = {}
        
        # FIXED: Define symbol_mapping at the top of the method scope
        symbol_mapping = {
            'RELIANCE': 'RELIANCE.NS', 'TCS': 'TCS.NS', 'HDFC': 'HDFCBANK.NS',
            'HDFCBANK': 'HDFCBANK.NS', 'INFY': 'INFY.NS', 'INFOSYS': 'INFY.NS',
            'ICICIBANK': 'ICICIBANK.NS', 'ICICI': 'ICICIBANK.NS', 'SBIN': 'SBIN.NS',
            'SBI': 'SBIN.NS', 'ITC': 'ITC.NS', 'WIPRO': 'WIPRO.NS', 'LT': 'LT.NS',
            'LARSEN': 'LT.NS', 'HCLTECH': 'HCLTECH.NS', 'HCL': 'HCLTECH.NS',
            'BAJFINANCE': 'BAJFINANCE.NS', 'BAJAJ': 'BAJFINANCE.NS',
            'MARUTI': 'MARUTI.NS', 'ASIANPAINT': 'ASIANPAINT.NS', 'ASIAN': 'ASIANPAINT.NS'
        }
        
        try:
            if intent == "stock_lookup":
                # Clean and collect candidate words
                words = [''.join(c for c in word if c.isalnum()).upper() for word in last_message.split()]
                candidates = []
                for word in words:
                    if word in symbol_mapping or word.endswith(('.NS', '.BO')) or (len(word) <= 10 and word.isalpha()):
                        candidates.append(word)
                
                # Prioritize: mapping > .NS/.BO > last alpha word
                symbol = None
                for word in candidates:
                    if word in symbol_mapping:
                        symbol = word
                        break
                    elif word.endswith(('.NS', '.BO')):
                        symbol = word
                        break
                if not symbol and candidates:
                    symbol = candidates[-1]  # Take last candidate as fallback
                
                if symbol:
                    result = self.tools["stock_price_lookup"]._run(symbol)
                    tool_results["stock_data"] = result
                else:
                    tool_results["stock_data"] = "Please specify a valid stock symbol (e.g., RELIANCE, TCS)."
            
            elif intent == "portfolio_analysis":
                result = self.tools["portfolio_analysis"]._run(state["user_id"])
                tool_results["portfolio_analysis"] = result
            
            elif intent == "sip_reminder":
                result = self.tools["sip_reminder_check"]._run(state["user_id"])
                tool_results["sip_reminders"] = result
            
            elif intent == "add_portfolio_shares":
                # Clean and collect candidate words
                words = [''.join(c for c in word if c.isalnum()).upper() for word in last_message.split()]
                symbol = None
                quantity = None
                candidates = []
                for word in words:
                    if word in symbol_mapping or word.endswith(('.NS', '.BO')) or (len(word) <= 10 and word.isalpha()):
                        candidates.append(word)
                    try:
                        qty = float(word)
                        if qty > 0:
                            quantity = qty
                    except:
                        pass
                
                # Prioritize symbol: mapping > .NS/.BO > last alpha word
                for word in candidates:
                    if word in symbol_mapping:
                        symbol = word
                        break
                    elif word.endswith(('.NS', '.BO')):
                        symbol = word
                        break
                if not symbol and candidates:
                    symbol = candidates[-1]
                
                if symbol and quantity:
                    result = self.tools["add_portfolio_shares"]._run(state["user_id"], symbol, quantity)
                    tool_results["add_shares"] = result
                else:
                    tool_results["add_shares"] = "Please specify a valid stock symbol and quantity (e.g., 'Add 10 RELIANCE shares')."
            
            elif intent == "set_sip_reminder":
                # Existing logic is robust; keep unchanged
                import re
                amount_match = re.search(r'₹?\s*(\d+\.?\d*)', last_message)
                day_match = re.search(r'(\d{1,2})(?:th|st|nd|rd)', last_message)
                fund_name = None
                for word in last_message.split():
                    if word.upper() not in ['SET', 'SIP', 'FOR', 'ON', 'REMINDER'] and not word.isdigit() and not word.startswith('₹'):
                        fund_name = word if not fund_name else f"{fund_name} {word}"
                amount = float(amount_match.group(1)) if amount_match else None
                due_day = int(day_match.group(1)) if day_match else None
                if fund_name and amount and due_day:
                    result = self.tools["set_sip_reminder"]._run(state["user_id"], fund_name, amount, due_day)
                    tool_results["set_sip"] = result
                else:
                    tool_results["set_sip"] = "Please specify fund name, amount, and due day (e.g., 'Set ₹5000 SIP for HDFC Fund on 15th')."
            
            else:
                tool_results["general"] = "I can help with stock lookups, portfolio analysis, SIP reminders, adding shares, or setting SIPs. What would you like to do?"
        
        except Exception as e:
            tool_results["error"] = f"Error executing task: {str(e)}"
        
        return {
            **state,
            "tool_results": {**state["tool_results"], **tool_results}
        }
    
    def generate_response(self, state: AgentState) -> AgentState:
        user_profile = state["context"].get("user_profile", {})
        tool_results = state["tool_results"]
        last_message = state["messages"][-1] if state["messages"] else ""
        response_prompt = f"""
        You are a helpful financial assistant. Generate a personalized response based on:
        User Message: "{last_message}"
        User Profile: Risk Tolerance: {user_profile.get('risk_tolerance', 'Unknown')}, Goals: {user_profile.get('investment_goals', [])}
        Tool Results: {json.dumps(tool_results, indent=2)}
        For portfolio analysis or adding shares, include the full tool output before adding advice. For SIPs, confirm the action or list reminders. Keep it conversational and under 200 words.
        """
        response = self.llm.invoke(response_prompt)
        return {
            **state,
            "final_response": response.content
        }
    
    def update_memory(self, state: AgentState) -> AgentState:
        profile = self.db_manager.get_user_profile(state["user_id"])
        if profile:
            profile.last_interaction = datetime.now()
            self.db_manager.save_user_profile(profile)
        return state


# ============================================================================
# 4. LLAMAINDEX MEMORY & CONTEXT MANAGEMENT WITH FAISS
# ============================================================================

class LlamaIndexMemoryManagerFAISS:
    """
    Updated memory manager using FAISS instead of ChromaDB for vector storage
    """
    def __init__(self, persist_dir: str = "./memory_storage", openai_api_key: str = None, dimension: int = 1536):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)
        self.dimension = dimension  # OpenAI text-embedding-ada-002 dimension
        
        # FAISS index paths
        self.faiss_index_path = self.persist_dir / "faiss_index.bin"
        self.metadata_path = self.persist_dir / "metadata.pkl"
        
        # Initialize FAISS index
        self._initialize_faiss_index()
        
        # Set OpenAI embedding model with API key
        self.embed_model = OpenAIEmbedding(
            model="text-embedding-ada-002",
            api_key=openai_api_key
        )
        
        # Create FAISS vector store
        self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Initialize index with FAISS vector store
        self.index = VectorStoreIndex(
            [],
            storage_context=self.storage_context,
            embed_model=self.embed_model
        )
        
        # Chat memory buffer
        self.chat_memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
        
        print(f"✅ FAISS Memory Manager initialized with dimension {self.dimension}")
    
    def _initialize_faiss_index(self):
        """Initialize or load FAISS index"""
        try:
            if self.faiss_index_path.exists():
                # Load existing FAISS index
                self.faiss_index = faiss.read_index(str(self.faiss_index_path))
                print(f"📂 Loaded existing FAISS index with {self.faiss_index.ntotal} vectors")
            else:
                # Create new FAISS index (using IndexFlatIP for cosine similarity)
                self.faiss_index = faiss.IndexFlatIP(self.dimension)
                print("🆕 Created new FAISS index")
                
            # Load metadata
            if self.metadata_path.exists():
                with open(self.metadata_path, 'rb') as f:
                    self.metadata_store = pickle.load(f)
                print(f"📂 Loaded {len(self.metadata_store)} metadata entries")
            else:
                self.metadata_store = {}
                print("🆕 Created new metadata store")
                
        except Exception as e:
            print(f"⚠️ Error loading FAISS index, creating new one: {str(e)}")
            self.faiss_index = faiss.IndexFlatIP(self.dimension)
            self.metadata_store = {}
    
    def _save_faiss_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.faiss_index, str(self.faiss_index_path))
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata_store, f)
                
            print(f"💾 Saved FAISS index with {self.faiss_index.ntotal} vectors")
        except Exception as e:
            print(f"❌ Error saving FAISS index: {str(e)}")
    
    def add_interaction(self, user_id: str, user_message: str, assistant_response: str):
        """Add a user interaction to long-term memory using FAISS"""
        try:
            # Clean up old interactions first
            self.cleanup_old_interactions(user_id)
            
            # Create document text
            interaction_text = f"User: {user_message}\nAssistant: {assistant_response}"
            
            # Generate embedding
            embedding = self.embed_model.get_text_embedding(interaction_text)
            embedding_array = np.array([embedding], dtype=np.float32)
            
            # Normalize for cosine similarity (FAISS IndexFlatIP expects normalized vectors)
            faiss.normalize_L2(embedding_array)
            
            # Add to FAISS index
            self.faiss_index.add(embedding_array)
            
            # Store metadata
            doc_id = f"interaction_{user_id}_{datetime.now().isoformat()}_{self.faiss_index.ntotal-1}"
            self.metadata_store[self.faiss_index.ntotal-1] = {
                "doc_id": doc_id,
                "user_id": user_id,
                "text": interaction_text,
                "timestamp": datetime.now().isoformat(),
                "interaction_type": "conversation"
            }
            
            # Save to disk
            self._save_faiss_index()
            
            print(f"➕ Added interaction to FAISS memory: {len(interaction_text)} chars")
            
        except Exception as e:
            print(f"❌ Error adding interaction to FAISS memory: {str(e)}")
    
    def add_financial_data(self, user_id: str, data_type: str, data: str):
        """Add financial data to FAISS memory"""
        try:
            # Generate embedding
            embedding = self.embed_model.get_text_embedding(data)
            embedding_array = np.array([embedding], dtype=np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embedding_array)
            
            # Add to FAISS index
            self.faiss_index.add(embedding_array)
            
            # Store metadata
            doc_id = f"data_{user_id}_{data_type}_{datetime.now().isoformat()}_{self.faiss_index.ntotal-1}"
            self.metadata_store[self.faiss_index.ntotal-1] = {
                "doc_id": doc_id,
                "user_id": user_id,
                "text": data,
                "data_type": data_type,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to disk
            self._save_faiss_index()
            
            print(f"➕ Added financial data to FAISS memory: {data_type}")
            
        except Exception as e:
            print(f"❌ Error adding financial data to FAISS memory: {str(e)}")
    
    def query_memory(self, user_id: str, query: str, top_k: int = 3) -> str:
        """Query user's interaction history using FAISS similarity search"""
        try:
            if self.faiss_index.ntotal == 0:
                return "No previous interactions found."
            
            # Generate query embedding
            query_embedding = self.embed_model.get_text_embedding(query)
            query_array = np.array([query_embedding], dtype=np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_array)
            
            # Search FAISS index
            similarities, indices = self.faiss_index.search(query_array, min(top_k * 2, self.faiss_index.ntotal))
            
            # Filter by user_id and collect results
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx in self.metadata_store:
                    metadata = self.metadata_store[idx]
                    if metadata.get("user_id") == user_id and similarity > 0.1:  # Similarity threshold
                        results.append({
                            "text": metadata["text"],
                            "similarity": float(similarity),
                            "timestamp": metadata["timestamp"]
                        })
            
            # Sort by similarity and take top_k
            results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:top_k]
            
            if not results:
                return "No relevant previous interactions found."
            
            # Format response
            response_parts = []
            for i, result in enumerate(results, 1):
                response_parts.append(f"{i}. {result['text'][:200]}... (Similarity: {result['similarity']:.3f})")
            
            return "Previous relevant interactions:\n" + "\n\n".join(response_parts)
            
        except Exception as e:
            print(f"❌ Error querying FAISS memory: {str(e)}")
            return "Error retrieving previous interactions."
    
    def get_user_context(self, user_id: str) -> str:
        """Get recent context for a user using FAISS"""
        try:
            return self.query_memory(user_id, "recent interactions and preferences", top_k=5)
        except Exception as e:
            print(f"❌ Error getting user context: {str(e)}")
            return "No previous context available."

    def cleanup_old_interactions(self, user_id: str, days_threshold: int = 30):
        """Clean up old interactions from FAISS index"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_threshold)).isoformat()
            
            # Find indices to remove
            indices_to_remove = []
            for idx, metadata in self.metadata_store.items():
                if (metadata.get("user_id") == user_id and 
                    metadata.get("timestamp", "") < cutoff_date):
                    indices_to_remove.append(idx)
            
            # Remove old metadata (FAISS doesn't support direct removal, so we mark for cleanup)
            for idx in indices_to_remove:
                if idx in self.metadata_store:
                    del self.metadata_store[idx]
            
            if indices_to_remove:
                print(f"🧹 Marked {len(indices_to_remove)} old interactions for cleanup")
                self._save_faiss_index()
            
            # Note: For production, you might want to periodically rebuild the FAISS index 
            # to actually remove old vectors, but for demo purposes, we just remove metadata
            
        except Exception as e:
            print(f"⚠️ Cleanup error: {str(e)}")  # Silent fail for demo
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the FAISS memory store"""
        return {
            "total_vectors": self.faiss_index.ntotal,
            "total_metadata": len(self.metadata_store),
            "dimension": self.dimension,
            "index_type": type(self.faiss_index).__name__
        }

# ============================================================================
# 5. MAIN FINANCIAL ASSISTANT CLASS (UPDATED FOR FAISS)
# ============================================================================

class FinancialAssistant:
    def __init__(self):
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        self.db_manager = DatabaseManager()
        # CHANGED: Using FAISS-based memory manager
        self.memory_manager = LlamaIndexMemoryManagerFAISS(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-3.5-turbo",
            api_key=openai_api_key
        )
        
        # Initialize tools
        portfolio_tool = PortfolioAnalysisTool()
        PortfolioAnalysisTool.set_db_manager(self.db_manager)
        sip_tool = SIPReminderTool()
        SIPReminderTool.set_db_manager(self.db_manager)
        add_shares_tool = AddPortfolioSharesTool()
        AddPortfolioSharesTool.set_db_manager(self.db_manager)
        set_sip_tool = SetSIPReminderTool()
        SetSIPReminderTool.set_db_manager(self.db_manager)
        
        self.tools = [
            StockPriceTool(),
            portfolio_tool,
            sip_tool,
            add_shares_tool,
            set_sip_tool
        ]
        
        self.workflow_graph = FinancialWorkflowGraph(self.tools, self.llm, self.db_manager)
        print("✅ Financial Assistant initialized with LangChain + LangGraph + LlamaIndex + FAISS")
    
    async def process_message(self, user_id: str, message: str) -> str:
        initial_state = {
            "messages": [message],
            "user_id": user_id,
            "current_task": "",
            "context": {},
            "tool_results": {},
            "final_response": ""
        }
        try:
            final_state = await self.workflow_graph.app.ainvoke(initial_state)
            response = final_state["final_response"]
            self.memory_manager.add_interaction(user_id, message, response)
            return response
        except Exception as e:
            error_response = f"I encountered an error: {str(e)}. Please try again."
            self.memory_manager.add_interaction(user_id, message, error_response)
            return error_response
    
    def get_user_insights(self, user_id: str) -> str:
        return self.memory_manager.get_user_context(user_id)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get FAISS memory statistics"""
        return self.memory_manager.get_stats()

# ============================================================================
# 6. STREAMLIT DEMO INTERFACE (UPDATED FOR FAISS)
# ============================================================================
def display_portfolio_chart(holdings, values):
    chart_config = {
        "type": "bar",
        "data": {
            "labels": list(holdings.keys()),
            "datasets": [{
                "label": "Portfolio Holdings Value (₹)",
                "data": values,
                "backgroundColor": ["#4CAF50", "#2196F3", "#FFC107", "#FF5722"],
                "borderColor": ["#388E3C", "#1976D2", "#FFA000", "#D81B60"],
                "borderWidth": 1
            }]
        },
        "options": {
            "scales": {
                "y": {"beginAtZero": True, "title": {"display": True, "text": "Value (₹)"}},
                "x": {"title": {"display": True, "text": "Stocks"}}
            },
            "plugins": {"title": {"display": True, "text": "Portfolio Holdings"}}
        }
    }
    components.html(f"""
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <canvas id="portfolioChart" width="400" height="200"></canvas>
        <script>
            const ctx = document.getElementById('portfolioChart').getContext('2d');
            new Chart(ctx, {json.dumps(chart_config)});
        </script>
    """, height=300)

def main():
    # Configure logging
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Output to terminal
            logging.FileHandler('financial_assistant.log')  # Save to file
        ]
    )
    logger = logging.getLogger(__name__)
    
    st.set_page_config(
        page_title="Financial Assistant Demo - FAISS Edition",
        page_icon="💰",
        layout="wide"
    )
    st.title("🤖 Financial Assistant Demo - FAISS Edition")
    st.subheader("Powered by LangChain + LangGraph + LlamaIndex + FAISS")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        user_id = st.text_input(
            "User ID",
            value="demo_user_123",
            help="Unique identifier for user sessions"
        )
        st.markdown("---")
        st.markdown("### 🔧 Framework Stack")
        st.markdown("- **LangChain**: Agent & tool orchestration")
        st.markdown("- **LangGraph**: Multi-step workflow management")
        st.markdown("- **LlamaIndex**: Context & memory management")
        st.markdown("- **FAISS**: High-performance vector similarity search")
        st.markdown("- **yFinance**: Real financial data")
        st.markdown("---")
        st.markdown("### 🎯 Demo Features")
        st.markdown("- Stock price lookup")
        st.markdown("- Portfolio analysis")
        st.markdown("- SIP reminders")
        st.markdown("- Add portfolio shares")
        st.markdown("- Set SIP reminders")
        st.markdown("- Long-term memory with FAISS")
        st.markdown("---")
        
        # FAISS Memory Statistics
        if 'assistant' in st.session_state:
            st.markdown("### 📊 FAISS Memory Stats")
            try:
                stats = st.session_state.assistant.get_memory_stats()
                st.metric("Total Vectors", stats["total_vectors"])
                st.metric("Metadata Entries", stats["total_metadata"])
                st.metric("Vector Dimension", stats["dimension"])
                st.text(f"Index Type: {stats['index_type']}")
            except Exception as e:
                st.error(f"Error loading stats: {str(e)}")
        
        st.markdown("---")
        st.markdown("### 📜 Debug Log")
        if os.path.exists('financial_assistant.log'):
            with open('financial_assistant.log', 'r') as f:
                st.text_area("Recent Logs", f.read(), height=200)
    
    if 'assistant' not in st.session_state:
        try:
            logger.info("Initializing Financial Assistant with FAISS")
            with st.spinner("Initializing FAISS vector database..."):
                st.session_state.assistant = FinancialAssistant()
            st.success("✅ Financial Assistant with FAISS ready!")
        except Exception as e:
            logger.error(f"Error initializing assistant: {str(e)}", exc_info=True)
            st.error(f"❌ Error initializing assistant: {str(e)}")
            return
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt := st.chat_input("Ask me about stocks, portfolio, SIPs, or financial advice..."):
        logger.info(f"Processing user prompt: {prompt}")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Processing your request with FAISS memory..."):
                try:
                    response = asyncio.run(
                        st.session_state.assistant.process_message(user_id, prompt)
                    )
                    logger.debug(f"Assistant response: {response}")
                    st.write(response)
                    if "portfolio" in response.lower() or "added" in response.lower():
                        portfolio = st.session_state.assistant.db_manager.get_portfolio(user_id)
                        if portfolio:
                            logger.debug(f"Portfolio for chart: {portfolio.holdings}")
                            values = []
                            valid_holdings = {}
                            for symbol, quantity in portfolio.holdings.items():
                                try:
                                    stock = yf.Ticker(symbol)
                                    hist = stock.history(period="1d")
                                    if not hist.empty:
                                        value = hist['Close'].iloc[-1] * quantity
                                        values.append(value)
                                        valid_holdings[symbol] = quantity
                                        logger.debug(f"Chart data for {symbol}: value={value}")
                                    else:
                                        values.append(0.0)
                                        logger.warning(f"No data for {symbol}")
                                        st.warning(f"No data for {symbol}")
                                except Exception as e:
                                    values.append(0.0)
                                    logger.error(f"Error fetching data for {symbol}: {str(e)}")
                                    st.warning(f"Error fetching data for {symbol}: {str(e)}")
                            if any(v > 0 for v in values):
                                display_portfolio_chart(valid_holdings, values)
                            else:
                                logger.error("No valid stock data available for chart")
                                st.error("No valid stock data available for chart.")
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    logger.error(f"Error processing prompt: {error_msg}", exc_info=True)
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    st.markdown("---")
    st.markdown("### 💡 Try These Examples:")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("📈 Get RELIANCE price"):
            example_query = "What's the current price of RELIANCE stock?"
            st.session_state.messages.append({"role": "user", "content": example_query})
            st.rerun()
    with col2:
        if st.button("📊 Analyze portfolio"):
            example_query = "Can you analyze my portfolio performance?"
            st.session_state.messages.append({"role": "user", "content": example_query})
            st.rerun()
    with col3:
        if st.button("🔔 Check SIPs"):
            example_query = "Do I have any SIP payments due?"
            st.session_state.messages.append({"role": "user", "content": example_query})
            st.rerun()
    with col4:
        if st.button("➕ Add shares"):
            example_query = "Add 10 RELIANCE shares to my portfolio"
            st.session_state.messages.append({"role": "user", "content": example_query})
            st.rerun()
    with col5:
        if st.button("⏰ Set SIP"):
            example_query = "Set ₹5000 SIP for HDFC Fund on 15th"
            st.session_state.messages.append({"role": "user", "content": example_query})
            st.rerun()
    
    # FAISS-specific features
    st.markdown("---")
    st.markdown("### 🧠 FAISS Memory Features:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Query Memory"):
            example_query = "What did we discuss about my investment preferences?"
            st.session_state.messages.append({"role": "user", "content": example_query})
            st.rerun()
    with col2:
        if st.button("📈 Memory Context"):
            if 'assistant' in st.session_state:
                context = st.session_state.assistant.get_user_insights(user_id)
                st.info(f"User Context:\n{context}")
            else:
                st.warning("Assistant not initialized yet")


if __name__ == "__main__":
    main()