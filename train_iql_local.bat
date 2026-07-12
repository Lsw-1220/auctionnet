@echo off
REM ============================================
REM  IQL 本地单卡训练 (Windows)
REM ============================================

REM ---------- 配置区 ----------
set TRAIN_DATA=.\strategy_train_env\data\traffic\training_data_rlData_folder\training_data_all-rlData.csv
set STEPS=20000
set BATCH_SIZE=32
set SAVE_DIR=.\saved_model\IQL_local
set DEVICE=cuda
REM ----------------------------

echo [INFO] Start IQL training on %DEVICE%
echo [INFO] Data:  %TRAIN_DATA%
echo [INFO] Steps: %STEPS%, Batch: %BATCH_SIZE%

python strategy_train_env\main\main_iql.py ^
    --steps %STEPS% ^
    --batch_size %BATCH_SIZE% ^
    --save %SAVE_DIR% ^
    --data %TRAIN_DATA% ^
    --device %DEVICE%

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Training failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo [INFO] Training done. Model saved to %SAVE_DIR%
pause
