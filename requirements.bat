@echo off
python -m pip install --upgrade pip  -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install --upgrade setuptools  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy-1.25.1+mkl-cp39-cp39-win_amd64.whl
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pause
exit