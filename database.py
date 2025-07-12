"""
CaseFile AI - Database Module
SQLite database management for standalone operation
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


def initialize_database(db_path: str):
    """Initialize the CaseFile AI database"""
    
    # Ensure directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Users table for authentication
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            role TEXT DEFAULT 'advisor',
            company TEXT,
            created_at TEXT,
            last_login TEXT,
            preferences TEXT
        )
    ''')
    
    # File processing history
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS file_processing (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            filename TEXT,
            original_filename TEXT,
            group_number TEXT,
            advisor_name TEXT,
            carrier TEXT,
            industry TEXT,
            processing_mode TEXT,
            records_count INTEGER,
            business_score REAL,
            ai_confidence REAL,
            validation_status TEXT,
            processing_time REAL,
            created_at TEXT,
            metadata TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # AI analysis results
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ai_analysis (
            id TEXT PRIMARY KEY,
            file_processing_id TEXT,
            analysis_type TEXT,
            confidence_score REAL,
            recommendations TEXT,
            fraud_indicators TEXT,
            quality_metrics TEXT,
            created_at TEXT,
            FOREIGN KEY (file_processing_id) REFERENCES file_processing (id)
        )
    ''')
    
    # System configuration
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_config (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        )
    ''')
    
    # Chat history
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            message TEXT,
            response TEXT,
            context TEXT,
            created_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Performance metrics
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id TEXT PRIMARY KEY,
            metric_name TEXT,
            metric_value REAL,
            metric_type TEXT,
            recorded_at TEXT
        )
    ''')
    
    # Insert default configuration
    default_configs = [
        ('app_version', '1.0.0'),
        ('ai_model', 'gemma-3'),
        ('default_processing_mode', 'enhanced'),
        ('max_file_size_mb', '50'),
        ('supported_formats', 'csv,xlsx,xls'),
        ('enable_ai_features', 'true'),
        ('enable_chat', 'true')
    ]
    
    for key, value in default_configs:
        cursor.execute('''
            INSERT OR IGNORE INTO system_config (key, value, updated_at)
            VALUES (?, ?, ?)
        ''', (key, value, datetime.now().isoformat()))
    
    # Insert default admin user
    cursor.execute('''
        INSERT OR IGNORE INTO users (id, email, name, role, company, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        'admin-001',
        'admin@casefile.ai',
        'CaseFile Administrator',
        'admin',
        'CaseFile AI',
        datetime.now().isoformat()
    ))
    
    conn.commit()
    conn.close()
    
    print(f"✅ Database initialized: {db_path}")


def get_db_connection(db_path: str) -> sqlite3.Connection:
    """Get database connection"""
    return sqlite3.connect(db_path)


def store_file_processing(db_path: str, processing_data: Dict) -> str:
    """Store file processing record"""
    
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    processing_id = str(uuid.uuid4())
    
    cursor.execute('''
        INSERT INTO file_processing (
            id, user_id, filename, original_filename, group_number, advisor_name,
            carrier, industry, processing_mode, records_count, business_score,
            ai_confidence, validation_status, processing_time, created_at, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        processing_id,
        processing_data.get('user_id', 'guest'),
        processing_data.get('filename'),
        processing_data.get('original_filename'),
        processing_data.get('group_number'),
        processing_data.get('advisor_name'),
        processing_data.get('carrier'),
        processing_data.get('industry'),
        processing_data.get('processing_mode', 'standard'),
        processing_data.get('records_count', 0),
        processing_data.get('business_score', 0.0),
        processing_data.get('ai_confidence', 0.0),
        processing_data.get('validation_status', 'unknown'),
        processing_data.get('processing_time', 0.0),
        datetime.now().isoformat(),
        json.dumps(processing_data.get('metadata', {}))
    ))
    
    conn.commit()
    conn.close()
    
    return processing_id


def store_ai_analysis(db_path: str, analysis_data: Dict) -> str:
    """Store AI analysis results"""
    
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    analysis_id = str(uuid.uuid4())
    
    cursor.execute('''
        INSERT INTO ai_analysis (
            id, file_processing_id, analysis_type, confidence_score,
            recommendations, fraud_indicators, quality_metrics, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        analysis_id,
        analysis_data.get('file_processing_id'),
        analysis_data.get('analysis_type', 'document_processing'),
        analysis_data.get('confidence_score', 0.0),
        json.dumps(analysis_data.get('recommendations', [])),
        json.dumps(analysis_data.get('fraud_indicators', [])),
        json.dumps(analysis_data.get('quality_metrics', {})),
        datetime.now().isoformat()
    ))
    
    conn.commit()
    conn.close()
    
    return analysis_id


def store_chat_message(db_path: str, user_id: str, message: str, 
                      response: str, context: str = "") -> str:
    """Store chat interaction"""
    
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    chat_id = str(uuid.uuid4())
    
    cursor.execute('''
        INSERT INTO chat_history (id, user_id, message, response, context, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        chat_id,
        user_id,
        message,
        response,
        context,
        datetime.now().isoformat()
    ))
    
    conn.commit()
    conn.close()
    
    return chat_id


def get_user_stats(db_path: str, user_id: str) -> Dict:
    """Get user statistics"""
    
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # Total files processed
    cursor.execute(
        'SELECT COUNT(*) FROM file_processing WHERE user_id = ?', 
        (user_id,)
    )
    total_files = cursor.fetchone()[0]
    
    # Success rate
    cursor.execute('''
        SELECT COUNT(*) FROM file_processing 
        WHERE user_id = ? AND validation_status = 'success'
    ''', (user_id,))
    successful_files = cursor.fetchone()[0]
    
    # Average processing time
    cursor.execute('''
        SELECT AVG(processing_time) FROM file_processing 
        WHERE user_id = ? AND processing_time > 0
    ''', (user_id,))
    avg_time = cursor.fetchone()[0] or 0
    
    # Average business score
    cursor.execute('''
        SELECT AVG(business_score) FROM file_processing 
        WHERE user_id = ? AND business_score > 0
    ''', (user_id,))
    avg_score = cursor.fetchone()[0] or 0
    
    conn.close()
    
    success_rate = (successful_files / total_files * 100) if total_files > 0 else 0
    
    return {
        'total_files': total_files,
        'success_rate': success_rate,
        'avg_processing_time': avg_time,
        'avg_business_score': avg_score,
        'successful_files': successful_files
    }


def get_recent_activity(db_path: str, user_id: str = None, limit: int = 10) -> List[Dict]:
    """Get recent processing activity"""
    
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    if user_id:
        cursor.execute('''
            SELECT filename, advisor_name, carrier, validation_status, 
                   business_score, created_at
            FROM file_processing 
            WHERE user_id = ?
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (user_id, limit))
    else:
        cursor.execute('''
            SELECT filename, advisor_name, carrier, validation_status, 
                   business_score, created_at
            FROM file_processing 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
    
    results = cursor.fetchall()
    conn.close()
    
    return [
        {
            'filename': row[0],
            'advisor_name': row[1],
            'carrier': row[2],
            'validation_status': row[3],
            'business_score': row[4],
            'created_at': row[5]
        }
        for row in results
    ]


def get_system_stats(db_path: str) -> Dict:
    """Get system-wide statistics"""
    
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # Total processing stats
    cursor.execute('SELECT COUNT(*) FROM file_processing')
    total_files = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM users')
    total_users = cursor.fetchone()[0]
    
    cursor.execute('''
        SELECT COUNT(*) FROM file_processing 
        WHERE validation_status = 'success'
    ''')
    successful_files = cursor.fetchone()[0]
    
    cursor.execute('SELECT AVG(business_score) FROM file_processing')
    avg_business_score = cursor.fetchone()[0] or 0
    
    cursor.execute('SELECT AVG(processing_time) FROM file_processing')
    avg_processing_time = cursor.fetchone()[0] or 0
    
    # Recent activity (last 24 hours)
    cursor.execute('''
        SELECT COUNT(*) FROM file_processing 
        WHERE datetime(created_at) > datetime('now', '-1 day')
    ''')
    files_today = cursor.fetchone()[0]
    
    conn.close()
    
    success_rate = (successful_files / total_files * 100) if total_files > 0 else 0
    
    return {
        'total_files': total_files,
        'total_users': total_users,
        'success_rate': success_rate,
        'avg_business_score': avg_business_score,
        'avg_processing_time': avg_processing_time,
        'files_processed_today': files_today,
        'system_health': 'excellent' if success_rate > 90 else 'good' if success_rate > 80 else 'fair'
    }


def update_system_config(db_path: str, key: str, value: str):
    """Update system configuration"""
    
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO system_config (key, value, updated_at)
        VALUES (?, ?, ?)
    ''', (key, value, datetime.now().isoformat()))
    
    conn.commit()
    conn.close()


def get_system_config(db_path: str, key: str = None) -> Dict:
    """Get system configuration"""
    
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    if key:
        cursor.execute('SELECT value FROM system_config WHERE key = ?', (key,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    else:
        cursor.execute('SELECT key, value FROM system_config')
        results = cursor.fetchall()
        conn.close()
        return {row[0]: row[1] for row in results}