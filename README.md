# RL Project  

## TD3 by Béla Umlauf  

[**Download Model Weights**](https://drive.google.com/drive/folders/126NozgxKaZQaP_XyEHWZUMdtWS82TShI?usp=sharing) 
### Gameplay Demonstration  
- **TD3 vs Weak Opponent**  
  ![Weak Opponent Gameplay](td3/results/evaluation_20250226-141914.gif) 
- **TD3 vs Strong Opponent**  
  ![Weak Opponent Gameplay](td3/results/evaluation_20250226-141922.gif)



<h1>TD-MPC2 by Jonas Klötzl</span></h1>

[**Download Model Weights**](https://drive.google.com/drive/folders/1MkM6Q-X8fcvgn3UfaFKNzriIkRT1n-3p?usp=sharing)

This codebase was used to train an TD-MPC2 agent for the Hockey Environment. Most important contribution is the `tdmpc2/envs/hockey.py` file containing the hockey environment and the wrapper. Smaller changes in `tdmpc2/envs/__init__.py`, `tdmpc2/train.py` and `tdmpc2/config.yaml`.

Run training with:
```
$ python train.py task=hockey-NORMAL steps=150000
```
