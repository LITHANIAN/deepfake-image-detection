# check_env.py
import numpy as np
import scipy
import sklearn
import torch

print(f"numpy: {np.__version__}")        # 必须显示1.21.6
print(f"scipy: {scipy.__version__}")     # 必须显示1.7.3
print(f"scikit-learn: {sklearn.__version__}")  # 必须显示1.0.2
print(f"torch: {torch.__version__}")     # 必须显示1.13.1