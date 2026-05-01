import subprocess
import sys
import time
import os
from pathlib import Path

def run_app():
    project_root = Path(__file__).parent.absolute()
    
    # Add src to PYTHONPATH
    env = os.environ.copy()
    src_path = str(project_root / "src")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    
    print("="*60)
    print("  PCB DEFECT DETECTION SYSTEM - STARTUP SCRIPT")
    print("="*60)
    
    # 1. Start Backend API
    print("\n🚀 Starting Backend API (FastAPI)...")
    backend_cmd = [
        sys.executable, "-m", "uvicorn", 
        "src.api.server:app", 
        "--host", "0.0.0.0", 
        "--port", "8000"
    ]
    
    backend_process = subprocess.Popen(
        backend_cmd,
        cwd=str(project_root),
        env=env
    )
    
    # Wait for backend to initialize
    print("⏳ Waiting for backend to load model...")
    time.sleep(5)
    
    # 2. Start Frontend UI
    print("\n🎨 Starting Frontend Dashboard (Streamlit)...")
    frontend_cmd = [
        sys.executable, "-m", "streamlit", 
        "run", "app.py", 
        "--server.port", "8501"
    ]
    
    frontend_process = subprocess.Popen(
        frontend_cmd,
        cwd=str(project_root),
        env=env
    )
    
    print("\n" + "!"*60)
    print("  SYSTEM READY!")
    print(f"  - Dashboard: http://localhost:8501")
    print(f"  - API:       http://localhost:8000")
    print("!"*60)
    print("\n(Press Ctrl+C in this terminal to stop both servers)")
    
    try:
        while True:
            time.sleep(1)
            if backend_process.poll() is not None:
                print("❌ Backend process stopped unexpectedly.")
                break
            if frontend_process.poll() is not None:
                print("❌ Frontend process stopped unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
        backend_process.terminate()
        frontend_process.terminate()
        print("✅ Shutdown complete.")

if __name__ == "__main__":
    run_app()
