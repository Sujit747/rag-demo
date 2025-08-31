from typing_extensions import TypedDict
from typing import List, Dict, Any
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.tools import BaseTool
from models.database import DatabaseManager
from models.data_models import UserProfile
from dataclasses import asdict
from datetime import datetime
import json
import re

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
                words = [''.join(c for c in word if c.isalnum()).upper() for word in last_message.split()]
                candidates = []
                for word in words:
                    if word in symbol_mapping or word.endswith(('.NS', '.BO')) or (len(word) <= 10 and word.isalpha()):
                        candidates.append(word)
                
                symbol = None
                for word in candidates:
                    if word in symbol_mapping:
                        symbol = word
                        break
                    elif word.endswith(('.NS', '.BO')):
                        symbol = word
                        break
                if not symbol and candidates:
                    symbol = candidates[-1]
                
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