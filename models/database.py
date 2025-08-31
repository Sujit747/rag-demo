from supabase import create_client, Client
from typing import Optional, List
from models.data_models import UserProfile, Portfolio, FinancialGoal
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class DatabaseManager:
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL or SUPABASE_KEY not found in .env file")
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
    
    def save_user_profile(self, profile: UserProfile):
        data = {
            "user_id": profile.user_id,
            "risk_tolerance": profile.risk_tolerance,
            "investment_goals": profile.investment_goals,
            "monthly_sip_amount": profile.monthly_sip_amount,
            "preferred_sectors": profile.preferred_sectors,
            "created_at": profile.created_at.isoformat(),  # Convert to string
            "last_interaction": profile.last_interaction.isoformat()  # Convert to string
        }
        self.supabase.table("user_profiles").upsert(data).execute()
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        response = self.supabase.table("user_profiles").select("*").eq("user_id", user_id).execute()
        if response.data:
            row = response.data[0]
            return UserProfile(
                user_id=row["user_id"],
                risk_tolerance=row["risk_tolerance"],
                investment_goals=row["investment_goals"],
                monthly_sip_amount=row["monthly_sip_amount"],
                preferred_sectors=row["preferred_sectors"],
                created_at=datetime.fromisoformat(row["created_at"]),  # Convert back to datetime
                last_interaction=datetime.fromisoformat(row["last_interaction"])  # Convert back to datetime
            )
        # Initialize default profile if none exists
        default_profile = UserProfile(
            user_id=user_id,
            risk_tolerance="Moderate",
            investment_goals=["Retirement", "Wealth Creation"],
            monthly_sip_amount=10000.0,
            preferred_sectors=["Technology", "Banking"],
            created_at=datetime.now(),
            last_interaction=datetime.now()
        )
        self.save_user_profile(default_profile)
        return default_profile

    def save_portfolio(self, portfolio: Portfolio):
        data = {
            "user_id": portfolio.user_id,
            "holdings": portfolio.holdings,
            "total_value": portfolio.total_value,
            "last_updated": portfolio.last_updated.isoformat()  # Convert to string
        }
        self.supabase.table("portfolios").upsert(data).execute()

    def get_portfolio(self, user_id: str) -> Optional[Portfolio]:
        response = self.supabase.table("portfolios").select("*").eq("user_id", user_id).execute()
        if response.data:
            row = response.data[0]
            return Portfolio(
                user_id=row["user_id"],
                holdings=row["holdings"],
                total_value=row["total_value"],
                last_updated=datetime.fromisoformat(row["last_updated"])  # Convert back to datetime
            )
        # Initialize default portfolio if none exists
        default_portfolio = Portfolio(
            user_id=user_id,
            holdings={},
            total_value=0.0,
            last_updated=datetime.now()
        )
        self.save_portfolio(default_portfolio)
        return default_portfolio
    
    def save_financial_goals(self, goals: List[FinancialGoal]):
        for goal in goals:
            data = {
                "goal_id": goal.goal_id,
                "user_id": goal.user_id,
                "goal_type": goal.goal_type,
                "target_amount": goal.target_amount,
                "current_amount": goal.current_amount,
                "deadline": goal.deadline.isoformat(),  # Convert to string
                "status": goal.status
            }
            self.supabase.table("financial_goals").upsert(data).execute()

    def get_financial_goals(self, user_id: str) -> List[FinancialGoal]:
        response = self.supabase.table("financial_goals").select("*").eq("user_id", user_id).execute()
        return [FinancialGoal(
            goal_id=row["goal_id"],
            user_id=row["user_id"],
            goal_type=row["goal_type"],
            target_amount=row["target_amount"],
            current_amount=row["current_amount"],
            deadline=datetime.fromisoformat(row["deadline"]),  # Convert back to datetime
            status=row["status"]
        ) for row in response.data] or []  # Return empty list if no data