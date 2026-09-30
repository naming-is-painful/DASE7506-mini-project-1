# DASE7506 Mini Project 1

This is the repository for mini project 1 of DASE 7506 of HKU.

## Environment Setup

Using Python 3.12.

```shell
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
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

## AI Usage Declaration 

AI was relied on heavily during the initial period of trying to understand the project’s structure. Copilot in VSCode (which I believe was GPT Luna) was used. 
Later on, discussions were made between GPT 5.4 mini to find explanations for some of the results.

## Lisencing

Code is provided by course directors and there seems to include no lisencing uhh.