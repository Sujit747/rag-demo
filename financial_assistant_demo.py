# Financial Assistant with LangChain, LangGraph & LlamaIndex
# 2-Day Demo Implementation

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
from llama_index.embeddings.openai import OpenAIEmbedding
# Core imports for our framework stack
# from langchain.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.tools import BaseTool
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.base import BaseCallbackHandler
    
# LangGraph imports for workflow orchestration
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

# LlamaIndex for advanced memory and context management
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.chroma import ChromaVectorStore
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

# ============================================================================
# 2. FINANCIAL DATA TOOLS (LangChain Tools)
# ============================================================================

class StockPriceTool(BaseTool):
   
    name: str = "stock_price_lookup"
    description: str = "Get current stock price and basic info for Indian/US stocks"
    
    def _run(self, symbol: str) -> str:
        try:
            # Add .NS for Indian stocks if not present
            if not symbol.endswith(('.NS', '.BO')) and len(symbol) <= 10:
                symbol = symbol + '.NS'
            
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1d")
            info = stock.info
            
            if hist.empty:
                return f"Could not find data for {symbol}"
            
            current_price = hist['Close'].iloc[-1]
            change = hist['Close'].iloc[-1] - hist['Open'].iloc[-1]
            change_pct = (change / hist['Open'].iloc[-1]) * 100
            
            return f"""
            Stock: {symbol}
            Current Price: ₹{current_price:.2f}
            Change: ₹{change:.2f} ({change_pct:.2f}%)
            Company: {info.get('longName', 'N/A')}
            Sector: {info.get('sector', 'N/A')}
            """
        except Exception as e:
            return f"Error fetching data for {symbol}: {str(e)}"
    
    async def _arun(self, symbol: str) -> str:
        return self._run(symbol)

from langchain.tools import BaseTool
from typing import Optional

class PortfolioAnalysisTool(BaseTool):
    name: str = "portfolio_analysis"
    description: str = "Analyze user's portfolio performance"
    
    def __init__(self, db_manager: DatabaseManager, **kwargs):
        super().__init__(**kwargs)  # Pass any additional kwargs to BaseTool
        self.db_manager = db_manager
    
    def _run(self, user_id: str) -> str:
        try:
            # Get portfolio from database (mock data for demo)
            portfolio_data = {
                'RELIANCE.NS': 10,
                'TCS.NS': 5,
                'HDFCBANK.NS': 8,
                'INFY.NS': 12
            }
            
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
    name: str = "sip_reminder"
    description: str = "Check for upcoming SIP payments"
    
    def __init__(self, db_manager: DatabaseManager, **kwargs):
        super().__init__(**kwargs)
        self.db_manager = db_manager
    
    def _run(self, user_id: str) -> str:
        try:
            # Mock SIP data - in real implementation, fetch from database
            sip_data = [
                {'fund_name': 'HDFC Top 100 Fund', 'amount': 5000, 'due_date': '15th every month'},
                {'fund_name': 'SBI Blue Chip Fund', 'amount': 3000, 'due_date': '10th every month'},
                {'fund_name': 'Axis Long Term Equity Fund', 'amount': 2000, 'due_date': '25th every month'}
            ]
            
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

class AgentState:
    def __init__(self):
        self.messages: List[str] = []
        self.user_id: str = ""
        self.current_task: str = ""
        self.context: Dict[str, Any] = {}
        self.tool_results: Dict[str, str] = {}
        self.final_response: str = ""

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
        last_message = state.messages[-1] if state.messages else ""
        
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
        
        intent = self.llm.predict(intent_prompt).strip().lower()
        state.current_task = intent
        state.context['intent'] = intent
        
        return state
    
    def fetch_user_context(self, state: AgentState) -> AgentState:
        """Fetch user profile and context from memory"""
        profile = self.db_manager.get_user_profile(state.user_id)
        
        if profile:
            state.context['user_profile'] = asdict(profile)
        else:
            # Create default profile for demo
            default_profile = UserProfile(
                user_id=state.user_id,
                risk_tolerance="Moderate",
                investment_goals=["Retirement", "Wealth Creation"],
                monthly_sip_amount=10000,
                preferred_sectors=["Technology", "Banking"],
                created_at=datetime.now(),
                last_interaction=datetime.now()
            )
            self.db_manager.save_user_profile(default_profile)
            state.context['user_profile'] = asdict(default_profile)
        
        return state
    
    def execute_financial_task(self, state: AgentState) -> AgentState:
        """Execute the appropriate financial task based on intent"""
        intent = state.current_task
        last_message = state.messages[-1] if state.messages else ""
        
        try:
            if intent == "stock_lookup":
                # Extract stock symbol from message
                words = last_message.upper().split()
                symbol = None
                for word in words:
                    if len(word) <= 10 and (word.isalpha() or '.' in word):
                        symbol = word
                        break
                
                if symbol:
                    result = self.tools["stock_price_lookup"]._run(symbol)
                    state.tool_results["stock_data"] = result
                else:
                    state.tool_results["stock_data"] = "Please specify a stock symbol."
                    
            elif intent == "portfolio_analysis":
                result = self.tools["portfolio_analysis"]._run(state.user_id)
                state.tool_results["portfolio_analysis"] = result
                
            elif intent == "sip_reminder":
                result = self.tools["sip_reminder_check"]._run(state.user_id)
                state.tool_results["sip_reminders"] = result
                
            else:
                state.tool_results["general"] = "I can help with stock lookups, portfolio analysis, and SIP reminders. What would you like to know?"
                
        except Exception as e:
            state.tool_results["error"] = f"Error executing task: {str(e)}"
        
        return state
    
    def generate_response(self, state: AgentState) -> AgentState:
        """Generate final response using LLM with context"""
        user_profile = state.context.get('user_profile', {})
        tool_results = state.tool_results
        last_message = state.messages[-1] if state.messages else ""
        
        response_prompt = f"""
        You are a helpful financial assistant. Generate a personalized response based on:
        
        User Message: "{last_message}"
        User Profile: Risk Tolerance: {user_profile.get('risk_tolerance', 'Unknown')}, Goals: {user_profile.get('investment_goals', [])}
        
        Tool Results: {json.dumps(tool_results, indent=2)}
        
        Provide a helpful, personalized response. Include actionable advice when appropriate.
        Keep it conversational and under 200 words.
        """
        
        response = self.llm.predict(response_prompt)
        state.final_response = response
        
        return state
    
    def update_memory(self, state: AgentState) -> AgentState:
        """Update user interaction memory"""
        # Update last interaction time
        profile = self.db_manager.get_user_profile(state.user_id)
        if profile:
            profile.last_interaction = datetime.now()
            self.db_manager.save_user_profile(profile)
        
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

# ============================================================================
# 5. MAIN FINANCIAL ASSISTANT CLASS
# ============================================================================

class FinancialAssistant:
    def __init__(self, openai_api_key: str):
        # Initialize core components
        self.db_manager = DatabaseManager()
        self.memory_manager = LlamaIndexMemoryManager()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            api_key=openai_api_key
        )
        
        # Initialize tools
        self.tools = [
            StockPriceTool(),
            PortfolioAnalysisTool(self.db_manager),
            SIPReminderTool(self.db_manager)
        ]
        
        # Initialize LangGraph workflow
        self.workflow_graph = FinancialWorkflowGraph(self.tools, self.llm, self.db_manager)
        
        print("✅ Financial Assistant initialized with LangChain + LangGraph + LlamaIndex")
    
    async def process_message(self, user_id: str, message: str) -> str:
        """Process user message through the complete workflow"""
        # Create initial state
        state = AgentState()
        state.user_id = user_id
        state.messages = [message]
        
        try:
            # Run through LangGraph workflow
            final_state = await self.workflow_graph.app.ainvoke(state)
            
            # Get response
            response = final_state.final_response
            
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
        
        openai_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            help="Enter your OpenAI API key"
        )
        
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
    
    # Main interface
    if not openai_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
        return
    
    # Initialize assistant
    if 'assistant' not in st.session_state:
        try:
            st.session_state.assistant = FinancialAssistant(openai_key)
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
    
    # Technical details expander
    with st.expander("🔍 Technical Implementation Details"):
        st.markdown("""
        ### Architecture Overview:
        
        1. **LangChain Integration**:
           - Custom tools for financial data (StockPriceTool, PortfolioAnalysisTool, SIPReminderTool)
           - Agent-based architecture with memory management
           - Structured prompts for consistent responses
        
        2. **LangGraph Workflow**:
           - Multi-step workflow: Intent Classification → Context Fetching → Task Execution → Response Generation
           - State management across workflow nodes
           - Error handling and recovery mechanisms
        
        3. **LlamaIndex Memory**:
           - Vector-based storage using ChromaDB
           - Persistent conversation memory across sessions
           - Context-aware query processing
        
        4. **Data Integration**:
           - Real-time stock data via yFinance API
           - SQLite for structured data storage
           - JSON serialization for complex data types
        
        5. **Demo Capabilities**:
           - Multi-turn conversations with context retention
           - Personalized responses based on user profiles
           - Real financial data integration
           - Scalable architecture for production deployment
        """)

if __name__ == "__main__":
    main()

# ============================================================================
# 7. REQUIREMENTS.TXT
# ============================================================================

# Requirements for the demo:
"""
streamlit>=1.28.0
langchain>=0.0.300
langgraph>=0.0.40
llama-index>=0.8.0
llama-index-vector-stores-chroma>=0.1.0
openai>=0.28.0
yfinance>=0.2.0
pandas>=1.5.0
chromadb>=0.4.0
sqlite3
asyncio
python-dateutil
"""

# ============================================================================
# 8. SETUP INSTRUCTIONS FOR 2-DAY DEMO
# ============================================================================

"""
DAY 1 SETUP & DEMO:
1. Install requirements: pip install -r requirements.txt
2. Run: streamlit run financial_assistant_demo.py
3. Demo LangChain tools and LangGraph workflows
4. Show agent orchestration and multi-step processing

DAY 2 ENHANCEMENTS:
1. Add more sophisticated LlamaIndex memory queries
2. Implement user preference learning
3. Add financial goal tracking
4. Demo persistent memory across sessions

PRESENTATION TALKING POINTS:

LangChain Features:
- Custom tool creation for domain-specific tasks
- Agent orchestration with memory
- Structured prompt templates
- Error handling and recovery

LangGraph Features:
- Visual workflow representation
- State management across nodes
- Conditional routing based on intent
- Async processing capabilities

LlamaIndex Features:
- Vector-based memory storage
- Context-aware information retrieval
- Persistent cross-session memory
- Semantic search capabilities

Real-world Applications:
- Personal finance management
- Investment advisory systems
- Automated financial reporting
- Risk assessment and compliance

Scalability Considerations:
- Database optimization for large user bases
- Caching strategies for frequent queries
- API rate limiting and error handling
- Security and privacy considerations
"""