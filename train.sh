source ~/.bashrc
# pip install mujoco
# pip install 'gym==0.23.1'
pip install 'gym==0.22.0'
pip install --no-cache-dir --force-reinstall \
  "cython>=0.29,<3" 

# If you’re using EGL:
export MUJOCO_GL=egl
# or if you’re on CPU/headless:
# export MUJOCO_GL=osmesa

export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:$LD_LIBRARY_PATH"

pip install --no-cache-dir --force-reinstall 'mujoco-py<2.2,>=2.1'

python -m ducks.train 
#> train.log

# nohup python -m ducks.train > train.log 2>&1 &
# echo $! > save_pid.txt