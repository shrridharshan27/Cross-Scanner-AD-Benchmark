@echo off
echo Installing ad_generalization_benchmark...
cd /d "C:\Project\ML Project"
pip install -e ad_generalization_benchmark
echo.
echo Done! Now run:
echo   python ad_generalization_benchmark/run_all.py --step test
pause
