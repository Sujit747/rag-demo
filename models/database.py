import sqlite3
import json
from typing import Optional, List
from models.data_models import UserProfile, Portfolio, FinancialGoal
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path: str = "financial_assistant.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
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
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolios (
                user_id TEXT PRIMARY KEY,
                holdings TEXT,
                total_value REAL,
                last_updated TIMESTAMP
            )
        ''')
        
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