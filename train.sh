source ~/.bashrc
pip install -U 'mujoco-py<2.2,>=2.1'

nohup python -m ducks.baselines.cql_train > train.log 2>&1 &
echo $! > save_pid.txt