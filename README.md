
#train with the first stage reward #for dynamic targets need to revise the movable of the targets in the code of simple_spread, 
python main.py --env-name simple_spread --num-good-agents 5 --num-adversaries 10 --save-dir spread_static --masking --cuda-num 0 --train

#train with second stage reward #need to revise the reward settings in the code of simple_reward, 
python main.py --env-name simple_spread --num-good-agents 5 --num-adversaries 10 --save-dir spread_static_continue --masking --continue-training --load-dir marlsave/spread_static/ep10000.pt --cuda-num 0 --train

#eval with connectivity guaranteed policy filtering 
python eval.py --masking 

#eval without connectivity guaranteed policy filtering 
python eval.py --masking --train
