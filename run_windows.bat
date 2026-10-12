@echo off
python -m src.run_experiment
if errorlevel 1 (
  echo Experiment failed. Review the error above.
  pause
  exit /b 1
)
echo Experiment completed. Check results\tables and results\figures.
pause
