# DASE7506 Mini Project 1

This is the repository for mini project 1 of DASE 7506 of HKU.

## Environment Setup

```shell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Testing the final result

```shell
python evaluate.py --checkpoint runs/baseline_relu_4800_steps/checkpoint.pt --device cpu --precision fp32 --split test
```

## Reproducing the result

```shell
python train.py --implementation student --seed 17 --eval-every 300 --run-dir runs/baseline_relu_4800_steps --device cuda --steps 4800
python evaluate.py --checkpoint runs/baseline_relu_4800_steps/checkpoint.pt --split validation --device cuda
# Freeze the final method before testing:
python evaluate.py --checkpoint runs/baseline_relu_4800_steps/checkpoint.pt --split test --device cpu --precision fp32
```
