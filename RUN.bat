@echo off
REM Chain Loss Correction - Main Simulation Runner
REM Double-click this file to run the simulation

echo ============================================================
echo Chain Loss Correction Simulation
echo Based on arXiv:2511.16632
echo ============================================================
echo.

echo Running simulation...
echo This may take a minute...
echo.

python example_simulation.py

echo.
echo ============================================================
echo Simulation Complete!
echo ============================================================
echo.
echo Results saved to: chain_loss_simulation_results.png
echo.
pause

