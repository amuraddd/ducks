source ~/.bashrc
pip install -U 'mujoco-py<2.2,>=2.1'

python -m ducks.train > train.log

# nohup python -m ducks.train > train.log 2>&1 &
# echo $! > save_pid.txt