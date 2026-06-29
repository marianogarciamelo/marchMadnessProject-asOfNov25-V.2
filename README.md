## COMP 379 Project

### Local setup

1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   ```
2. Activate it:
   - macOS/Linux: `source .venv/bin/activate`
   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
3. Install the required packages (includes pandas + pyarrow for Parquet exports):
   ```bash
   pip install -r requirements.txt
   ```
4. Build the combined dataset (writes both Parquet and CSV outputs):
   ```bash
   python marchMadness.py
   ```
