#!/usr/bin/env bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

rm -f ai_alphazero.zip

# 将 AlphaZeroAgent 代码打包为 ai.py
cp ai_alphazero.py ai.py

zip -r ai_alphazero.zip \
  ai.py \
  main.py \
  common.py \
  protocol.py \
  custom_utils.py \
  ai_alphazero_latest.npz \
  ../SDK \
  ../tools \
  -x "*/__pycache__/*" "*/.pytest_cache/*" "*.pyc" "*.DS_Store" "*/*.so" "*/*.dll" "*/*.dylib" "*/*.exe" "*/*.obj" "*/*.o" "*/*.a" "*/*.lib" "*/*.npz"
  
# 然后再把最新的权重模型加入压缩包
# 检查本地是否有权重
if [ -f "ai_alphazero_latest.npz" ]; then
    zip -r ai_alphazero.zip ai_alphazero_latest.npz
elif [ -f "../checkpoints/ai_alphazero_latest.npz" ]; then
    cp ../checkpoints/ai_alphazero_latest.npz .
    zip -r ai_alphazero.zip ai_alphazero_latest.npz
else
    echo "Warning: ai_alphazero_latest.npz not found! AI will fallback to heuristic only."
fi

rm -f ai.py

echo "打包完成: AI/ai_alphazero.zip"
