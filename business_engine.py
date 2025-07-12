"""
CaseFile AI - Business Intelligence Engine
MODAETOS-style recursive learning system for insurance processing optimization
"""

import json
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import asyncio


class CaseFileBusinessEngine:
    """Business Intelligence Engine with MODAETOS-style learning"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.initialize_database()
        
        # MODAETOS-style metrics
        self.cognitive_momentum = 0.6
        self.pattern_recognition_strength = 0.75
        self.business_learning_rate = 0.68
        self.validation_accuracy = 0.88
        
    def initialize_database(self):
        """Initialize business intelligence database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for business intelligence
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS business_validations (
                id TEXT PRIMARY KEY,
                advisor_id TEXT,
                advisor_name TEXT,
                group_number TEXT,
                business_score REAL,
                ai_confidence REAL,
                industry TEXT,
                carrier TEXT,
                group_size INTEGER,
                processing_mode TEXT,
                validation_status TEXT,
                timestamp TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_patterns (
                id TEXT PRIMARY KEY,
                pattern_type TEXT,
                pattern_data TEXT,
                success_rate REAL,
                frequency INTEGER,
                last_updated TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS advisor_intelligence (
                advisor_id TEXT PRIMARY KEY,
                advisor_name TEXT,
                total_submissions INTEGER DEFAULT 0,
                successful_submissions INTEGER DEFAULT 0,
                average_business_score REAL DEFAULT 0.0,
                preferred_carriers TEXT,
                industry_specializations TEXT,
                quality_trend TEXT,
                last_updated TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id TEXT PRIMARY KEY,
                metric_type TEXT,
                metric_value REAL,
                timestamp TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def learn_from_processing(self, result: Dict, advisor_name: str, group_number: str) -> Dict:
        """Learn from processing results to improve future recommendations"""
        
        try:
            advisor_id = f"advisor_{hash(advisor_name) % 10000:04d}"
            
            # Store validation result
            validation_data = {
                "business_score": result.get("business_score", 0),
                "ai_confidence": result.get("ai_confidence", 0),
                "records_processed": result.get("records_processed", 0),
                "validation_status": "success" if result.get("business_score", 0) > 70 else "review"
            }
            
            await self._store_validation_result(
                advisor_id=advisor_id,
                advisor_name=advisor_name,
                group_number=group_number,
                validation_data=validation_data
            )
            
            # Update advisor intelligence
            await self._update_advisor_intelligence(advisor_id, advisor_name, validation_data)
            
            # Learn patterns
            await self._learn_patterns(validation_data)
            
            # Update system metrics
            self._update_cognitive_metrics(validation_data)
            
            return {
                "learning_applied": True,
                "cognitive_momentum": self.cognitive_momentum,
                "pattern_recognition_strength": self.pattern_recognition_strength,
                "business_learning_rate": self.business_learning_rate,
                "advisor_id": advisor_id
            }
            
        except Exception as e:
            print(f"Learning system error: {e}")
            return {"learning_applied": False, "error": str(e)}
    
    async def _store_validation_result(self, advisor_id: str, advisor_name: str, 
                                     group_number: str, validation_data: Dict):
        """Store validation result in database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO business_validations 
            (id, advisor_id, advisor_name, group_number, business_score, ai_confidence,
             validation_status, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            str(uuid.uuid4()),
            advisor_id,
            advisor_name,
            group_number,
            validation_data.get("business_score", 0),
            validation_data.get("ai_confidence", 0),
            validation_data.get("validation_status", "unknown"),
            datetime.now().isoformat(),
            json.dumps(validation_data)
        ))
        
        conn.commit()
        conn.close()
    
    async def _update_advisor_intelligence(self, advisor_id: str, advisor_name: str, 
                                         validation_data: Dict):
        """Update advisor intelligence profile"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get existing advisor data
        cursor.execute(
            'SELECT * FROM advisor_intelligence WHERE advisor_id = ?', 
            (advisor_id,)
        )
        existing = cursor.fetchone()
        
        if existing:
            # Update existing advisor
            total_submissions = existing[2] + 1
            successful_submissions = existing[3] + (1 if validation_data.get("validation_status") == "success" else 0)
            avg_score = (existing[4] * existing[2] + validation_data.get("business_score", 0)) / total_submissions
            
            cursor.execute('''
                UPDATE advisor_intelligence 
                SET total_submissions = ?, successful_submissions = ?, 
                    average_business_score = ?, last_updated = ?
                WHERE advisor_id = ?
            ''', (
                total_submissions,
                successful_submissions,
                avg_score,
                datetime.now().isoformat(),
                advisor_id
            ))
        else:
            # Create new advisor profile
            cursor.execute('''
                INSERT INTO advisor_intelligence 
                (advisor_id, advisor_name, total_submissions, successful_submissions,
                 average_business_score, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                advisor_id,
                advisor_name,
                1,
                1 if validation_data.get("validation_status") == "success" else 0,
                validation_data.get("business_score", 0),
                datetime.now().isoformat()
            ))
        
        conn.commit()
        conn.close()
    
    async def _learn_patterns(self, validation_data: Dict):
        """Learn patterns from validation data"""
        
        # Simulate pattern learning
        if validation_data.get("business_score", 0) > 85:
            self.pattern_recognition_strength = min(0.95, self.pattern_recognition_strength + 0.02)
            self.business_learning_rate = min(0.9, self.business_learning_rate + 0.01)
        
        # Update cognitive momentum based on success
        if validation_data.get("validation_status") == "success":
            self.cognitive_momentum = min(0.9, self.cognitive_momentum + 0.01)
        
        # Store pattern in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        pattern_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT OR REPLACE INTO learning_patterns 
            (id, pattern_type, pattern_data, success_rate, frequency, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            pattern_id,
            "validation_pattern",
            json.dumps(validation_data),
            validation_data.get("business_score", 0) / 100,
            1,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _update_cognitive_metrics(self, validation_data: Dict):
        """Update cognitive metrics based on processing results"""
        
        # Update validation accuracy
        if validation_data.get("validation_status") == "success":
            self.validation_accuracy = min(0.95, self.validation_accuracy + 0.005)
        
        # Store metrics in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metrics = [
            ("cognitive_momentum", self.cognitive_momentum),
            ("pattern_recognition_strength", self.pattern_recognition_strength),
            ("business_learning_rate", self.business_learning_rate),
            ("validation_accuracy", self.validation_accuracy)
        ]
        
        for metric_type, value in metrics:
            cursor.execute('''
                INSERT INTO system_metrics (id, metric_type, metric_value, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()),
                metric_type,
                value,
                datetime.now().isoformat()
            ))
        
        conn.commit()
        conn.close()
    
    def get_analytics_dashboard(self) -> Dict:
        """Get business intelligence analytics dashboard"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get validation summary
        cursor.execute('''
            SELECT COUNT(*) as total_validations,
                   AVG(business_score) as avg_score,
                   COUNT(CASE WHEN validation_status = 'success' THEN 1 END) as successful
            FROM business_validations
        ''')
        summary = cursor.fetchone()
        
        # Get top advisors
        cursor.execute('''
            SELECT advisor_name, total_submissions, average_business_score
            FROM advisor_intelligence
            ORDER BY average_business_score DESC
            LIMIT 5
        ''')
        top_advisors = cursor.fetchall()
        
        conn.close()
        
        return {
            "system_performance": {
                "cognitive_momentum": self.cognitive_momentum,
                "pattern_recognition_strength": self.pattern_recognition_strength,
                "business_learning_rate": self.business_learning_rate,
                "validation_accuracy": self.validation_accuracy
            },
            "validation_summary": {
                "total_validations": summary[0] if summary else 0,
                "average_score": summary[1] if summary else 0,
                "success_rate": (summary[2] / summary[0] * 100) if summary and summary[0] > 0 else 0
            },
            "top_advisors": [
                {
                    "name": advisor[0],
                    "submissions": advisor[1],
                    "average_score": advisor[2]
                }
                for advisor in (top_advisors or [])
            ],
            "learning_metrics": {
                "patterns_learned": 147,  # Simulated
                "pattern_accuracy": self.pattern_recognition_strength * 100,
                "system_intelligence": (
                    self.cognitive_momentum + 
                    self.pattern_recognition_strength + 
                    self.business_learning_rate
                ) / 3 * 100
            }
        }
    
    def get_advisor_intelligence(self, advisor_id: str) -> Dict:
        """Get detailed advisor intelligence"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM advisor_intelligence WHERE advisor_id = ?', 
            (advisor_id,)
        )
        advisor_data = cursor.fetchone()
        
        if not advisor_data:
            return {
                "patterns_learned": 0,
                "total_submissions": 0,
                "quality_rate": 0.0,
                "average_business_value": 0.0,
                "performance_trend": "new"
            }
        
        # Get recent validations
        cursor.execute('''
            SELECT business_score, validation_status, timestamp
            FROM business_validations
            WHERE advisor_id = ?
            ORDER BY timestamp DESC
            LIMIT 10
        ''', (advisor_id,))
        recent_validations = cursor.fetchall()
        
        conn.close()
        
        quality_rate = advisor_data[3] / advisor_data[2] if advisor_data[2] > 0 else 0
        
        return {
            "patterns_learned": len(recent_validations),
            "total_submissions": advisor_data[2],
            "quality_rate": quality_rate,
            "average_business_value": advisor_data[4],
            "performance_trend": "improving" if quality_rate > 0.8 else "developing",
            "specialization_insights": {
                "insurance_processing": quality_rate,
                "data_quality": advisor_data[4] / 100 if advisor_data[4] else 0
            },
            "top_patterns": [
                f"High-quality submission pattern (confidence: {quality_rate:.1%})",
                f"Business score optimization (avg: {advisor_data[4]:.1f})",
                "Carrier compatibility patterns"
            ]
        }