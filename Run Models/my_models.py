import torchvision.models as models
import torch
import torch.nn as nn
def get_model(num_classes, model_name = 'resnet18', pretrained=True):
    """
    Crea un modelo ResNet18 modificado para aceptar 1 canal de entrada (en lugar de 3) 
    y ajusta la capa final para el número de clases especificado.
    """
    #Cargar diccionario de los modelos preentrenados de torchvision del json
    model_dict = {
    "alexnet": models.alexnet,
    "convnext": models.convnext,
    "convnext_base": models.convnext_base,
    "convnext_large": models.convnext_large,
    "convnext_small": models.convnext_small,
    "convnext_tiny": models.convnext_tiny,
    "densenet": models.densenet,
    "densenet121": models.densenet121,
    "densenet161": models.densenet161,
    "densenet169": models.densenet169,
    "densenet201": models.densenet201,
    "efficientnet": models.efficientnet,
    "efficientnet_b0": models.efficientnet_b0,
    "efficientnet_b1": models.efficientnet_b1,
    "efficientnet_b2": models.efficientnet_b2,
    "efficientnet_b3": models.efficientnet_b3,
    "efficientnet_b4": models.efficientnet_b4,
    "efficientnet_b5": models.efficientnet_b5,
    "efficientnet_b6": models.efficientnet_b6,
    "efficientnet_b7": models.efficientnet_b7,
    "efficientnet_v2_l": models.efficientnet_v2_l,
    "efficientnet_v2_m": models.efficientnet_v2_m,
    "efficientnet_v2_s": models.efficientnet_v2_s,
    "googlenet": models.googlenet,
    "inception": models.inception,
    "inception_v3": models.inception_v3,
    "maxvit": models.maxvit,
    "maxvit_t": models.maxvit_t,
    "mnasnet": models.mnasnet,
    "mnasnet0_5": models.mnasnet0_5,
    "mnasnet0_75": models.mnasnet0_75,
    "mnasnet1_0": models.mnasnet1_0,
    "mnasnet1_3": models.mnasnet1_3,
    "mobilenet": models.mobilenet,
    "mobilenet_v2": models.mobilenet_v2,
    "mobilenet_v3_large": models.mobilenet_v3_large,
    "mobilenet_v3_small": models.mobilenet_v3_small,
    "resnet": models.resnet,
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
    "resnext50_32x4d": models.resnext50_32x4d,
    "resnext101_32x8d": models.resnext101_32x8d,
    "resnext101_64x4d": models.resnext101_64x4d,
    "regnet": models.regnet,
    "regnet_x_400mf": models.regnet_x_400mf,
    "regnet_x_800mf": models.regnet_x_800mf,
    "regnet_x_1_6gf": models.regnet_x_1_6gf,
    "regnet_x_3_2gf": models.regnet_x_3_2gf,
    "regnet_x_8gf": models.regnet_x_8gf,
    "regnet_x_16gf": models.regnet_x_16gf,
    "regnet_x_32gf": models.regnet_x_32gf,
    "regnet_y_400mf": models.regnet_y_400mf,
    "regnet_y_800mf": models.regnet_y_800mf,
    "regnet_y_1_6gf": models.regnet_y_1_6gf,
    "regnet_y_3_2gf": models.regnet_y_3_2gf,
    "regnet_y_8gf": models.regnet_y_8gf,
    "regnet_y_16gf": models.regnet_y_16gf,
    "regnet_y_32gf": models.regnet_y_32gf,
    "regnet_y_128gf": models.regnet_y_128gf,
    "shufflenet_v2_x0_5": models.shufflenet_v2_x0_5,
    "shufflenet_v2_x1_0": models.shufflenet_v2_x1_0,
    "shufflenet_v2_x1_5": models.shufflenet_v2_x1_5,
    "shufflenet_v2_x2_0": models.shufflenet_v2_x2_0,
    "squeezenet": models.squeezenet,
    "squeezenet1_0": models.squeezenet1_0,
    "squeezenet1_1": models.squeezenet1_1,
    "swin_b": models.swin_b,
    "swin_s": models.swin_s,
    "swin_t": models.swin_t,
    "swin_v2_b": models.swin_v2_b,
    "swin_v2_s": models.swin_v2_s,
    "swin_v2_t": models.swin_v2_t,
    "swin_transformer": models.swin_transformer,
    "vgg": models.vgg,
    "vgg11": models.vgg11,
    "vgg11_bn": models.vgg11_bn,
    "vgg13": models.vgg13,
    "vgg13_bn": models.vgg13_bn,
    "vgg16": models.vgg16,
    "vgg16_bn": models.vgg16_bn,
    "vgg19": models.vgg19,
    "vgg19_bn": models.vgg19_bn,
    "vision_transformer": models.vision_transformer,
    "vit_b_16": models.vit_b_16,
    "vit_b_32": models.vit_b_32,
    "vit_h_14": models.vit_h_14,
    "vit_l_16": models.vit_l_16,
    "vit_l_32": models.vit_l_32,
    "wide_resnet50_2": models.wide_resnet50_2,
    "wide_resnet101_2": models.wide_resnet101_2
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