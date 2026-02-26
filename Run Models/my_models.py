import torchvision.models as models
import torch
import torch.nn as nn
def get_model(num_classes, model_name = 'resnet18', pretrained=True):
    """
    Crea un modelo ResNet18 modificado para aceptar 1 canal de entrada (en lugar de 3) 
    y ajusta la capa final para el número de clases especificado.
    """
    model_dict = {
        # ResNets (used in train_TVT.py)
        "resnet18":            models.resnet18,
        "resnet50":            models.resnet50,
        # EfficientNets (used in train_binary_species.py)
        "efficientnet_b0":     models.efficientnet_b0,
        "efficientnet_b3":     models.efficientnet_b3,
        "efficientnet_b4":     models.efficientnet_b4,
        # RegNetY (used in train_binary_species.py)
        "regnet_y_400mf":      models.regnet_y_400mf,
        "regnet_y_800mf":      models.regnet_y_800mf,
        "regnet_y_1_6gf":      models.regnet_y_1_6gf,
        # MobileNet (used in train_binary_species.py)
        "mobilenet_v3_small":  models.mobilenet_v3_small,
        "mobilenet_v3_large":  models.mobilenet_v3_large,
    }

    if model_name not in model_dict:
        raise ValueError(f"Modelo '{model_name}' no soportado.")
    
    # Paso 1: Cargar el modelo preentrenado
    model = model_dict[model_name](pretrained=pretrained)

    # Paso 2: Reemplazar la primera capa para aceptar 1 canal (no 3)
    model.conv1 = nn.Conv2d(
        in_channels=1, out_channels=64,
        kernel_size=7, stride=2, padding=3, bias=False
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, activation_fn=nn.ReLU):
        """
        Crea un MLP configurable con un número arbitrario de capas ocultas.

        Args:
            input_size (int): Tamaño de la entrada.
            hidden_sizes (list[int]): Lista con los tamaños de las capas ocultas.
            num_classes (int): Número de clases de salida.
            activation_fn (nn.Module): Clase de la función de activación (por ejemplo, nn.ReLU, nn.Sigmoid).
        """
        super(MLP, self).__init__()
        layers = [nn.Flatten()]  # Aplana la entrada

        # Primera capa (entrada -> primera capa oculta)
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(activation_fn())

        # Capas ocultas adicionales
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(activation_fn())

        # Capa de salida
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))

        # Crear el modelo secuencial
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Aplanar la entrada
        return self.model(x)

def get_MLP(num_classes, input_size, hidden_sizes, activation_fn=nn.ReLU):
    """
    Crea un MLP configurable.

    Args:
        input_size (int): Tamaño de la entrada.
        hidden_sizes (list[int]): Lista con los tamaños de las capas ocultas.
        num_classes (int): Número de clases de salida.
        activation_fn (nn.Module): Clase de la función de activación (por ejemplo, nn.ReLU, nn.Sigmoid).

    Returns:
        ConfigurableMLP: Modelo MLP configurado.
    """
    return MLP(input_size, hidden_sizes, num_classes, activation_fn)