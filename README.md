# DP-SIGNSGD
code for ICASSP'21 paper "DP-SIGNSGD: WHEN EFFICIENCY MEETS PRIVACY AND ROBUSTNESS"
# How to run:
```
python federated_signsgd_main.py --model=mlp --dataset=mnist --epochs=61 --num_users 31 --local_ep 1 --delta 1e-3 --iid 0 --frac 1 --local_bs 256 --class_per_user 10 --mode SIGNSGD --momentum 0 --lr 0.005 
python federated_signsgd_main.py --model=mlp --dataset=mnist --epochs=61 --num_users 31 --local_ep 1 --delta 1e-3 --iid 0 --frac 1 --local_bs 256 --class_per_user 10 --mode DP-SIGNSGD --momentum 0 --lr 0.005 --l2_norm_clip=4 
python federated_signsgd_main.py --model=mlp --dataset=mnist --epochs=61 --num_users 31 --local_ep 1 --delta 1e-3 --iid 0 --frac 1 --local_bs 256 --class_per_user 10 --mode EF-DP-SIGNSGD --momentum 0 --lr 0.005  --l2_norm_clip=4 --error_decay 0.5
python federated_signsgd_main.py --model=mlp --dataset=mnist --epochs=61 --num_users 31 --local_ep 1 --delta 1e-3 --iid 0 --frac 1 --local_bs 256 --class_per_user 10 --mode FedAvg --momentum 0 --lr 0.005 
```
## Citing
If you have found this work to be useful, please consider citing it with the following bibtex:
```
@inproceedings{lyu2021DP-SIGNSGD,
  title={DP-SIGNSGD: WHEN EFFICIENCY MEETS PRIVACY AND ROBUSTNESS},
  author={Lyu, Lingjuan},
  booktitle={ICASSP},
  year={2021}
}
```
