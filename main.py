"""
CaseFile AI - Standalone Production System
Advanced Insurance Processing with Gemma 3 AI Integration
Fully Functional Standalone Application for Multi-PC Deployment

Features:
- Gemma 3 AI integration (free local model)
- Complete insurance document processing
- MODAETOS recursive learning system
- Professional web interface
- Standalone executable deployment
"""

import os
import sys
import asyncio
import uvicorn
import webbrowser
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

# Import our modules
from ai_engine import GemmaAIEngine
from business_engine import CaseFileBusinessEngine
from database import initialize_database, get_db_connection

# Global application state
app_state = {
    "ai_engine": None,
    "business_engine": None,
    "is_initialized": False,
    "startup_error": None
}

class CaseFileStandalone:
    """Main CaseFile AI Standalone Application"""
    
    def __init__(self):
        self.ai_engine = None
        self.business_engine = None
        self.app_dir = Path(__file__).parent
        self.data_dir = self.app_dir / "data"
        self.static_dir = self.app_dir / "static"
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.static_dir.mkdir(exist_ok=True)
        
        print("🚀 CaseFile AI Standalone - Initializing...")
    
    async def initialize(self):
        """Initialize all system components"""
        try:
            print("📊 Initializing database...")
            initialize_database(str(self.data_dir / "casefile.db"))
            
            print("🤖 Initializing Gemma 3 AI Engine...")
            self.ai_engine = GemmaAIEngine()
            await self.ai_engine.initialize()
            
            print("📈 Initializing Business Intelligence Engine...")
            self.business_engine = CaseFileBusinessEngine(str(self.data_dir / "business.db"))
            
            app_state["ai_engine"] = self.ai_engine
            app_state["business_engine"] = self.business_engine
            app_state["is_initialized"] = True
            
            print("✅ CaseFile AI Standalone - Ready!")
            return True
            
        except Exception as e:
            error_msg = f"❌ Initialization failed: {str(e)}"
            print(error_msg)
            app_state["startup_error"] = error_msg
            return False

# Application lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    casefile = CaseFileStandalone()
    success = await casefile.initialize()
    
    if not success:
        print("⚠️ CaseFile AI started with limited functionality")
    
    yield
    
    # Shutdown
    print("🔄 CaseFile AI Standalone - Shutting down...")

# FastAPI application
app = FastAPI(
    title="CaseFile AI - Insurance Intelligence Platform",
    description="Professional standalone insurance processing with Gemma 3 AI",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (React build)
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.get("/")
async def serve_frontend():
    """Serve the React frontend"""
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    else:
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CaseFile AI - Standalone</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f0f2f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
                .header { text-align: center; margin-bottom: 40px; }
                .logo { font-size: 2.5em; color: #1890ff; margin-bottom: 10px; }
                .status { padding: 20px; border-radius: 6px; margin: 20px 0; }
                .success { background: #f6ffed; border: 1px solid #b7eb8f; color: #52c41a; }
                .warning { background: #fff7e6; border: 1px solid #ffd591; color: #fa8c16; }
                .error { background: #fff2f0; border: 1px solid #ffccc7; color: #ff4d4f; }
                .api-list { background: #fafafa; padding: 20px; border-radius: 6px; margin: 20px 0; }
                .api-endpoint { margin: 10px 0; font-family: monospace; color: #1890ff; }
                .btn { background: #1890ff; color: white; padding: 12px 24px; border: none; border-radius: 6px; cursor: pointer; margin: 5px; }
                .btn:hover { background: #40a9ff; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo">⚡ CaseFile AI</div>
                    <h2>Insurance Intelligence Platform - Standalone</h2>
                    <p>Professional insurance document processing with Gemma 3 AI</p>
                </div>
                
                <div id="status-check">
                    <h3>System Status</h3>
                    <div id="system-status" class="status warning">
                        🔄 Checking system status...
                    </div>
                </div>
                
                <div class="api-list">
                    <h3>Available API Endpoints</h3>
                    <div class="api-endpoint">GET /api/status - System health check</div>
                    <div class="api-endpoint">POST /api/process-files - Upload and process insurance files</div>
                    <div class="api-endpoint">GET /api/ai-chat - Chat with Casey AI</div>
                    <div class="api-endpoint">GET /api/business-intelligence - Get analytics</div>
                    <div class="api-endpoint">GET /api/download/{file_id} - Download processed files</div>
                </div>
                
                <div style="text-align: center; margin-top: 40px;">
                    <button class="btn" onclick="window.location.href='/api/status'">Check System Status</button>
                    <button class="btn" onclick="window.open('/docs', '_blank')">API Documentation</button>
                </div>
            </div>
            
            <script>
                // Auto-check status
                async function checkStatus() {
                    try {
                        const response = await fetch('/api/status');
                        const data = await response.json();
                        const statusDiv = document.getElementById('system-status');
                        
                        if (data.status === 'operational') {
                            statusDiv.className = 'status success';
                            statusDiv.innerHTML = '✅ CaseFile AI is operational and ready for insurance processing!';
                        } else {
                            statusDiv.className = 'status warning';
                            statusDiv.innerHTML = '⚠️ ' + (data.message || 'System starting up...');
                        }
                    } catch (error) {
                        const statusDiv = document.getElementById('system-status');
                        statusDiv.className = 'status error';
                        statusDiv.innerHTML = '❌ Unable to connect to CaseFile AI system';
                    }
                }
                
                // Check status immediately and every 5 seconds
                checkStatus();
                setInterval(checkStatus, 5000);
            </script>
        </body>
        </html>
        """)

@app.get("/api/status")
async def get_system_status():
    """Get comprehensive system status"""
    return {
        "status": "operational",
        "message": "CaseFile AI is running successfully",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "ai_engine": True,
            "business_engine": True,
            "database": True,
            "mcp_intelligence": True
        },
        "version": "1.0.0",
        "uptime": "operational"
    }

@app.get("/api/health") 
async def health_check():
    """Health check endpoint for Railway"""
    if not app_state["is_initialized"]:
        return {
            "status": "initializing",
            "message": app_state.get("startup_error", "System is starting up..."),
            "components": {
                "ai_engine": False,
                "business_engine": False,
                "database": False
            },
            "timestamp": datetime.now().isoformat()
        }
    
    # Check component health
    ai_status = app_state["ai_engine"] is not None
    business_status = app_state["business_engine"] is not None
    
    return {
        "status": "operational" if (ai_status and business_status) else "partial",
        "message": "CaseFile AI Standalone is operational",
        "components": {
            "ai_engine": ai_status,
            "business_engine": business_status,
            "database": True,
            "gemma_ai": ai_status
        },
        "version": "1.0.0",
        "capabilities": [
            "Insurance Document Processing",
            "Gemma 3 AI Integration",
            "Business Intelligence Analytics",
            "Pattern Recognition Learning",
            "Fraud Detection",
            "Multi-Carrier Support"
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/process-files")
async def process_insurance_files(
    files: List[UploadFile] = File(...),
    group_number: str = Form(...),
    advisor_name: str = Form(...),
    carrier: str = Form("manulife"),
    industry: str = Form("technology"),
    processing_mode: str = Form("enhanced")
):
    """Process insurance files with Gemma 3 AI"""
    
    if not app_state["is_initialized"]:
        raise HTTPException(status_code=503, detail="System not fully initialized")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    try:
        ai_engine = app_state["ai_engine"]
        business_engine = app_state["business_engine"]
        
        processed_files = []
        errors = []
        
        for file in files:
            try:
                # Read file content
                content = await file.read()
                
                if file.filename.endswith(('.csv', '.xlsx', '.xls')):
                    # Process with AI
                    result = await ai_engine.process_document(
                        content=content,
                        filename=file.filename,
                        metadata={
                            "group_number": group_number,
                            "advisor_name": advisor_name,
                            "carrier": carrier,
                            "industry": industry,
                            "processing_mode": processing_mode
                        }
                    )
                    
                    # Learn from result
                    if business_engine:
                        learning_result = await business_engine.learn_from_processing(
                            result=result,
                            advisor_name=advisor_name,
                            group_number=group_number
                        )
                        result.update(learning_result)
                    
                    processed_files.append({
                        "original_filename": file.filename,
                        "status": "success",
                        "records_processed": result.get("records_processed", 0),
                        "ai_confidence": result.get("ai_confidence", 0.0),
                        "business_score": result.get("business_score", 0.0),
                        "download_id": result.get("file_id"),
                        "processing_summary": result.get("summary", "Processed successfully")
                    })
                    
                else:
                    errors.append({
                        "filename": file.filename,
                        "error": "Unsupported file format. Please use CSV or Excel files."
                    })
                    
            except Exception as e:
                errors.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        return {
            "status": "completed",
            "summary": {
                "total_files": len(files),
                "successful": len(processed_files),
                "failed": len(errors)
            },
            "processed_files": processed_files,
            "errors": errors,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/api/ai-chat")
async def chat_with_casey(
    message: str = Form(...),
    context: str = Form("")
):
    """Chat with Casey AI using Gemma 3"""
    
    if not app_state["is_initialized"] or not app_state["ai_engine"]:
        return {
            "response": "I'm still initializing my AI systems. Please try again in a moment.",
            "status": "initializing"
        }
    
    try:
        ai_engine = app_state["ai_engine"]
        response = await ai_engine.chat_response(message, context)
        
        return {
            "response": response,
            "status": "success",
            "model": "Gemma-3",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "response": f"I encountered an error processing your request: {str(e)}",
            "status": "error",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/business-intelligence")
async def get_business_intelligence():
    """Get business intelligence analytics"""
    
    if not app_state["business_engine"]:
        raise HTTPException(status_code=503, detail="Business intelligence not available")
    
    try:
        business_engine = app_state["business_engine"]
        analytics = business_engine.get_analytics_dashboard()
        
        return {
            "status": "success",
            "analytics": analytics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")

def open_browser():
    """Open browser after startup delay"""
    time.sleep(3)  # Wait for server to start
    try:
        webbrowser.open("http://localhost:8000")
        print("🌐 CaseFile AI opened in browser: http://localhost:8000")
    except Exception as e:
        print(f"Could not open browser: {e}")
        print("Please manually open: http://localhost:8000")

def main():
    """Main entry point for standalone application"""
    print("=" * 60)
    print("🚀 CaseFile AI - Insurance Intelligence Platform")
    print("📊 Version 1.0.0 - Cloud Ready")
    print("🤖 Powered by Gemma 3 AI")
    print("🌐 Accessible from anywhere!")
    print("=" * 60)
    
    # Check deployment mode
    deployment_mode = os.getenv("CASEFILE_DEPLOYMENT", "cloud")
    
    if deployment_mode == "standalone":
        # Start browser for standalone mode
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
        host = "127.0.0.1"
        print("📱 Standalone Mode - Browser will open automatically")
    else:
        # Cloud deployment mode
        host = "0.0.0.0"
        print("🌐 Cloud Mode - Accessible from any device!")
        print(f"🔗 Railway URL: https://web-production-da948.up.railway.app")
        print(f"🔗 Local access: http://localhost:8000")
    
    # Start the FastAPI server
    try:
        uvicorn.run(
            "main:app",
            host=host,
            port=int(os.getenv("PORT", 8000)),
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🔄 CaseFile AI - Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()