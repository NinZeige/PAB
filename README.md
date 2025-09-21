# 运行

1. 设置环境变量`$PABROOT`为PAB数据集所在目录
  - 换言之，`$PABROOT`目录下应该有`train/`, `test/`, `pose/`等目录
2. 安装环境
   ```
   uv sync
   ```
3. 运行训练代码
   ```
   uv run main.py
   ```
