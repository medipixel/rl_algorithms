python run_lunarlander_continuous_v2.py --cfg-path configs/lunarlander_continuous_v2/ddpg.py --off-render --episode-num 1 --max-episode-step 1 --seed 12345
python run_lunarlander_continuous_v2.py --cfg-path configs/lunarlander_continuous_v2/sacfd.py --off-render --episode-num 1 --max-episode-step 1 --seed 12345
python run_lunarlander_continuous_v2.py --cfg-path configs/lunarlander_continuous_v2/td3.py --off-render --episode-num 1 --max-episode-step 1 --seed 12345

python run_pong_no_frameskip_v4.py --cfg-path configs/pong_no_frameskip_v4/dqn.py --off-render --episode-num 1 --max-episode-step 1 --seed 12345
python run_pong_no_frameskip_v4.py --cfg-path configs/pong_no_frameskip_v4/dqn_resnet.py --off-render --episode-num 1 --max-episode-step 1 --seed 12345

python run_lunarlander_continuous_v2.py --cfg-path configs/lunarlander_continuous_v2/ddpgfd.py --off-render --episode-num 1 --max-episode-step 1 --seed 12345