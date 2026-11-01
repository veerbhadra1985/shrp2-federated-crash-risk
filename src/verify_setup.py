import torch, pandas, numpy, sklearn, lightgbm, flwr, mapie, traci, sumolib
import subprocess

print("PyTorch:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
print("Flower:", flwr.__version__)
out = subprocess.run(["sumo", "--version"], capture_output=True, text=True)
print(out.stdout.splitlines()[0])
print("All good!")