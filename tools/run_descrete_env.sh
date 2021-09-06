python run_lunarlander_v2.py --cfg-path ./configs/lunarlander_v2/dqn.yaml --off-render --log
python run_lunarlander_v2.py --cfg-path ./configs/lunarlander_v2/dqfd.yaml --off-render --log
python run_lunarlander_v2.py --cfg-path ./configs/lunarlander_v2/r2d1.yaml --off-render --log

python run_pong_no_frameskip_v4.py --cfg-path ./configs/pong_no_frameskip_v4/dqn.yaml --off-render --log
python run_pong_no_frameskip_v4.py --cfg-path ./configs/pong_no_frameskip_v4/r2d1.yaml --off-render --log
python run_pong_no_frameskip_v4.py --cfg-path configs/pong_no_frameskip_v4/apex_dqn.yaml --off-render --log

python run_pong_no_frameskip_v4.py --cfg-path ./configs/pong_no_frameskip_v4/dqn_resnet.yaml --off-render --log