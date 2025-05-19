import pennylane as qml
import torch
import torch.nn as nn
from config import config
from pennylane.qnn import TorchLayer

def create_quantum_node():
    dev = qml.device("default.qubit", wires=config.N_QUBITS)
    
    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights_strong, weights_basic):
        qml.AngleEmbedding(inputs, wires=range(config.N_QUBITS))
        
        # 第一层：强纠缠层
        qml.StronglyEntanglingLayers(
            weights_strong, 
            wires=range(config.N_QUBITS)
        )
        
        # 第二层：基础纠缠层
        qml.BasicEntanglerLayers(
            weights_basic,
            wires=range(config.N_QUBITS)
        )
        
        return [qml.expval(qml.PauliZ(i)) for i in range(config.N_QUBITS)]
    
    # 分别定义两种权重形状
    weight_shapes = {
        "weights_strong": (config.QUANTUM_LAYERS, config.N_QUBITS, 3),
        "weights_basic": (config.QUANTUM_LAYERS, config.N_QUBITS)
    }
    return TorchLayer(circuit, weight_shapes)

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 经典预处理层
        self.classical = nn.Sequential(
            nn.Linear(config.PCA_COMPONENTS, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, config.N_QUBITS)
        )
        
        # 量子层
        self.quantum = create_quantum_node()
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(config.N_QUBITS, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
    def forward(self, x):
        x = self.classical(x)
        x = self.quantum(x)
        return self.output(x)