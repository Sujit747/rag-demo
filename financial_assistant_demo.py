# Financial Assistant with LangChain, LangGraph & LlamaIndex
# 2-Day Demo Implementation - FIXED VERSION
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

# LlamaIndex for advanced memory and context management
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb

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
        
        # Create the workflow graph
        self.workflow = StateGraph(AgentState)
        self._build_graph()
    
    def _build_graph(self):
        # Add nodes
        self.workflow.add_node("classify_intent", self.classify_intent)
        self.workflow.add_node("fetch_user_context", self.fetch_user_context)
        self.workflow.add_node("execute_financial_task", self.execute_financial_task)
        self.workflow.add_node("generate_response", self.generate_response)
        self.workflow.add_node("update_memory", self.update_memory)
        
        # Add edges (workflow flow)
        self.workflow.add_edge("classify_intent", "fetch_user_context")
        self.workflow.add_edge("fetch_user_context", "execute_financial_task")
        self.workflow.add_edge("execute_financial_task", "generate_response")
        self.workflow.add_edge("generate_response", "update_memory")
        
        # Set entry point
        self.workflow.set_entry_point("classify_intent")
        
        # Compile the graph
        self.app = self.workflow.compile()
    
    def classify_intent(self, state: AgentState) -> AgentState:
        """Classify user intent - what financial task they want to perform"""
        last_message = state["messages"][-1] if state["messages"] else ""
        
        intent_prompt = f"""
        Classify the user's financial intent from this message: "{last_message}"
        
        Possible intents:
        1. stock_lookup - User wants stock price or info
        2. portfolio_analysis - User wants portfolio review
        3. sip_reminder - User wants SIP reminders
        4. general_advice - General financial advice
        5. goal_tracking - Track financial goals
        
        Return only the intent name.
        """
        
        response = self.llm.invoke(intent_prompt)
        intent = response.content.strip().lower()
        valid_intents = ["stock_lookup", "portfolio_analysis", "sip_reminder", "general_advice", "goal_tracking"]
        if intent not in valid_intents:
            intent = "general_advice"
        return {
            **state,
            "current_task": intent,
            "context": {**state["context"], "intent": intent}
        }
    
    def fetch_user_context(self, state: AgentState) -> AgentState:
        """Fetch user profile and context from memory"""
        profile = self.db_manager.get_user_profile(state["user_id"])
        
        if profile:
            user_profile = asdict(profile)
        else:
            # Create default profile for demo
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
        
        # FIXED: Return updated state dict
        return {
            **state,
            "context": {**state["context"], "user_profile": user_profile}
        }
    
    def execute_financial_task(self, state: AgentState) -> AgentState:
        """Execute the appropriate financial task based on intent"""
        intent = state["current_task"]
        last_message = state["messages"][-1] if state["messages"] else ""
        tool_results = {}
        
        try:
            if intent == "stock_lookup":
                # Define valid stock symbols from symbol_mapping
                symbol_mapping = {
                    'RELIANCE': 'RELIANCE.NS', 'TCS': 'TCS.NS', 'HDFC': 'HDFCBANK.NS',
                    'HDFCBANK': 'HDFCBANK.NS', 'INFY': 'INFY.NS', 'INFOSYS': 'INFY.NS',
                    'ICICIBANK': 'ICICIBANK.NS', 'ICICI': 'ICICIBANK.NS', 'SBIN': 'SBIN.NS',
                    'SBI': 'SBIN.NS', 'ITC': 'ITC.NS', 'WIPRO': 'WIPRO.NS', 'LT': 'LT.NS',
                    'LARSEN': 'LT.NS', 'HCLTECH': 'HCLTECH.NS', 'HCL': 'HCLTECH.NS',
                    'BAJFINANCE': 'BAJFINANCE.NS', 'BAJAJ': 'BAJFINANCE.NS', 
                    'MARUTI': 'MARUTI.NS', 'ASIANPAINT': 'ASIANPAINT.NS', 'ASIAN': 'ASIANPAINT.NS'
                }
                words = last_message.upper().split()
                symbol = None
                # Prioritize symbols in mapping or valid ticker format
                for word in words:
                    if word in symbol_mapping or word.endswith(('.NS', '.BO')) or (len(word) <= 10 and word.isalpha()):
                        symbol = word
                        break
                if not symbol:
                    # Fallback: Look for any word that could be a ticker
                    for word in words:
                        if len(word) <= 10 and (word.isalpha() or '.' in word):
                            symbol = word
                            break
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
                
            else:
                tool_results["general"] = "I can help with stock lookups, portfolio analysis, and SIP reminders. What would you like to know?"
                
        except Exception as e:
            tool_results["error"] = f"Error executing task: {str(e)}"
        
        return {
            **state,
            "tool_results": {**state["tool_results"], **tool_results}
        }
    
    def generate_response(self, state: AgentState) -> AgentState:
        """Generate final response using LLM with context"""
        user_profile = state["context"].get("user_profile", {})
        tool_results = state["tool_results"]
        last_message = state["messages"][-1] if state["messages"] else ""
        
        response_prompt = f"""
        You are a helpful financial assistant. Generate a personalized response based on:
        
        User Message: "{last_message}"
        User Profile: Risk Tolerance: {user_profile.get('risk_tolerance', 'Unknown')}, Goals: {user_profile.get('investment_goals', [])}
        
        Tool Results: {json.dumps(tool_results, indent=2)}
        
        For portfolio analysis, include the full tool output (stock details, total value, performance) before adding advice. For other queries, provide a concise, conversational response. Keep it under 200 words.
        """
        
        response = self.llm.invoke(response_prompt)
        
        return {
            **state,
            "final_response": response.content
        }
    
    def update_memory(self, state: AgentState) -> AgentState:
        """Update user interaction memory"""
        # Update last interaction time
        profile = self.db_manager.get_user_profile(state["user_id"])
        if profile:
            profile.last_interaction = datetime.now()
            self.db_manager.save_user_profile(profile)
        
        # FIXED: Return state dict unchanged
        return state


# ============================================================================
# 4. LLAMAINDEX MEMORY & CONTEXT MANAGEMENT
# ============================================================================

class LlamaIndexMemoryManager:
    def __init__(self, persist_dir: str = "./memory_storage", openai_api_key: str = None):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)
        
        # Initialize ChromaDB for vector storage
        self.chroma_client = chromadb.PersistentClient(path=str(self.persist_dir / "chroma_db"))
        self.chroma_collection = self.chroma_client.get_or_create_collection("financial_memory")
        
        # Create vector store
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Set OpenAI embedding model with API key
        self.embed_model = OpenAIEmbedding(
            model="text-embedding-ada-002",
            api_key=openai_api_key
        )
        
        # Initialize empty index with explicit embed_model
        self.index = VectorStoreIndex(
            [],
            storage_context=self.storage_context,
            embed_model=self.embed_model
        )
        
        # Chat memory buffer
        self.chat_memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
    
    def add_interaction(self, user_id: str, user_message: str, assistant_response: str):
        """Add a user interaction to long-term memory"""
        self.cleanup_old_interactions(user_id)
        interaction_doc = Document(
            text=f"User: {user_message}\nAssistant: {assistant_response}",
            metadata={
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "interaction_type": "conversation"
            }
        )
        
        self.index.insert(interaction_doc)
    
    def add_financial_data(self, user_id: str, data_type: str, data: str):
        """Add financial data to memory"""
        data_doc = Document(
            text=data,
            metadata={
                "user_id": user_id,
                "data_type": data_type,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        self.index.insert(data_doc)
    
    def query_memory(self, user_id: str, query: str, top_k: int = 3) -> str:
        """Query user's interaction history"""
        query_engine = self.index.as_query_engine(
            similarity_top_k=top_k,
            filters={"user_id": user_id}
        )
        
        response = query_engine.query(query)
        return str(response)
    
    def get_user_context(self, user_id: str) -> str:
        """Get recent context for a user"""
        try:
            return self.query_memory(user_id, "recent interactions and preferences", top_k=5)
        except:
            return "No previous context available."

    def cleanup_old_interactions(self, user_id: str, days_threshold: int = 30):
        cutoff_date = (datetime.now() - timedelta(days=days_threshold)).isoformat()
        try:
            documents = self.chroma_collection.get(where={"user_id": user_id})
            for doc_id, metadata in zip(documents.get("ids", []), documents.get("metadatas", [])):
                if metadata and metadata.get("timestamp", "") < cutoff_date:
                    self.chroma_collection.delete(ids=[doc_id])
        except Exception as e:
            print(f"Cleanup error: {str(e)}")  # Silent fail for demo

# ============================================================================
# 5. MAIN FINANCIAL ASSISTANT CLASS
# ============================================================================

class FinancialAssistant:
    def __init__(self):
        # Initialize core components
        openai_api_key = os.getenv('OPENAI_API_KEY')

        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")

        self.db_manager = DatabaseManager()
        self.memory_manager = LlamaIndexMemoryManager(openai_api_key=openai_api_key)
        
        # Initialize LLM
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
        
        self.tools = [
            StockPriceTool(),
            portfolio_tool,
            sip_tool
        ]
        
        # Initialize LangGraph workflow
        self.workflow_graph = FinancialWorkflowGraph(self.tools, self.llm, self.db_manager)
        
        print("✅ Financial Assistant initialized with LangChain + LangGraph + LlamaIndex")
    
    async def process_message(self, user_id: str, message: str) -> str:
        """Process user message through the complete workflow"""
        # FIXED: Create initial state as dict
        initial_state = {
            "messages": [message],
            "user_id": user_id,
            "current_task": "",
            "context": {},
            "tool_results": {},
            "final_response": ""
        }
        
        try:
            # Run through LangGraph workflow
            final_state = await self.workflow_graph.app.ainvoke(initial_state)
            
            # Get response
            response = final_state["final_response"]
            
            # Update LlamaIndex memory
            self.memory_manager.add_interaction(user_id, message, response)
            
            return response
            
        except Exception as e:
            error_response = f"I encountered an error: {str(e)}. Please try again."
            self.memory_manager.add_interaction(user_id, message, error_response)
            return error_response
    
    def get_user_insights(self, user_id: str) -> str:
        """Get insights about user from memory"""
        return self.memory_manager.get_user_context(user_id)

# ============================================================================
# 6. STREAMLIT DEMO INTERFACE
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
    st.set_page_config(
        page_title="Financial Assistant Demo",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("🤖 Financial Assistant Demo")
    st.subheader("Powered by LangChain + LangGraph + LlamaIndex")    
    # Sidebar for configuration
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
        st.markdown("- **yFinance**: Real financial data")
        
        st.markdown("---")
        st.markdown("### 🎯 Demo Features")
        st.markdown("- Stock price lookup")
        st.markdown("- Portfolio analysis")
        st.markdown("- SIP reminders")
        st.markdown("- Long-term memory")
        st.markdown("- Multi-step workflows")
    
    # # Main interface
    # if not openai_key:
    #     st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
    #     return
    
    # Initialize assistant
    if 'assistant' not in st.session_state:
        try:
            st.session_state.assistant = FinancialAssistant()
            st.success("✅ Financial Assistant ready!")
        except Exception as e:
            st.error(f"❌ Error initializing assistant: {str(e)}")
            return
    
    # Chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # User input
    # User input
    if prompt := st.chat_input("Ask me about stocks, portfolio, SIPs, or financial advice..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Processing your request..."):
                try:
                    # Process through async workflow
                    response = asyncio.run(
                        st.session_state.assistant.process_message(user_id, prompt)
                    )
                    st.write(response)
                    # Check if response is portfolio-related (case-insensitive)
                    if "portfolio" in response.lower():
                        portfolio = st.session_state.assistant.db_manager.get_portfolio(user_id)
                        if portfolio:
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
                                    else:
                                        values.append(0.0)  # Fallback
                                        st.warning(f"No data for {symbol}")
                                except Exception as e:
                                    values.append(0.0)
                                    st.warning(f"Error fetching data for {symbol}: {str(e)}")
                            if any(v > 0 for v in values):
                                display_portfolio_chart(valid_holdings, values)
                            else:
                                st.error("No valid stock data available for chart.")
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    # Example queries section
    st.markdown("---")
    st.markdown("### 💡 Try These Examples:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 Get RELIANCE stock price"):
            example_query = "What's the current price of RELIANCE stock?"
            st.session_state.messages.append({"role": "user", "content": example_query})
            st.rerun()
    
    with col2:
        if st.button("📊 Analyze my portfolio"):
            example_query = "Can you analyze my portfolio performance?"
            st.session_state.messages.append({"role": "user", "content": example_query})
            st.rerun()
    
    with col3:
        if st.button("🔔 Check SIP reminders"):
            example_query = "Do I have any SIP payments due?"
            st.session_state.messages.append({"role": "user", "content": example_query})
            st.rerun()
    

if __name__ == "__main__":
    main()

