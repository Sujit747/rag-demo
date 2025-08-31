from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional

@dataclass
class UserProfile:
    user_id: str
    risk_tolerance: str
    investment_goals: List[str]
    monthly_sip_amount: float
    preferred_sectors: List[str]
    created_at: datetime
    last_interaction: datetime

@dataclass
class Portfolio:
    user_id: str
    holdings: Dict[str, float]
    total_value: float
    last_updated: datetime

@dataclass
class FinancialGoal:
    goal_id: str
    user_id: str
    goal_type: str
    target_amount: float
    current_amount: float
    deadline: datetime
    status: str