import torch
import torch.nn as nn
import torchvision.models as models

# Resnet Class
class Resnet():
    def __init__(
            self, 
            model_version: int, 
            layers: list,
            weights='IMAGENET1K_V1'
        ):
        """
            model_version: Version of resnet
            layers: Number of residual block per layers
            weights: Pretraied weights
        """
        if model_version==18:
            self.model = models.resnet18(weights=weights)
        elif model_version==34:
            self.model = models.resnet34(weights=weights)
        elif model_version==50:
            self.model = models.resnet50(weights=weights)
        elif model_version==101:
            self.model = models.resnet101(weights=weights)
        elif model_version==152:
            self.model = models.resnet152(weights=weights)
        else:
            raise Exception("No Version for Resnet Model Found")
        
        # Modify Number of residua block per layer
        self.modify_residual_block(layers)
        self.modify_layer()
        

    def modify_residual_block(self, layers):
        layer_ids = range(1, 5)
        num_new_blocks = layers 
        for id, num_new_block in zip(layer_ids, num_new_blocks):
            layer_name = f'layer{id}'
            layer = getattr(self.model, layer_name)
            last_block = layer[-1]
            
            # Append until = num_new_block
            while len(layer) < num_new_block:
                layer.append(last_block)
            setattr(self.model, layer_name, layer)


    def modify_layer(self):
        for i in range(2, 5):
            getattr(self.model, 'layer%d'%i)[0].conv1.stride = (2,2)
            getattr(self.model, 'layer%d'%i)[0].conv2.stride = (1,1)

    def get_model(self):
        return self.model
    

def resnet18():
    ResnetModel = Resnet(
        model_version=18,
        layers=[2, 2, 2, 2]
    )
    model_resnet = ResnetModel.get_model()
    return model_resnet



def resnet34():
    ResnetModel = Resnet(
        model_version=34,
        layers=[3, 4, 6, 3]
    )
    model_resnet = ResnetModel.get_model()
    return model_resnet



def resnet50():
    ResnetModel = Resnet(
        model_version=50,
        layers=[3, 4, 6, 3]
    )
    model_resnet = ResnetModel.get_model()
    return model_resnet


def resnet101():
    ResnetModel = Resnet(
        model_version=101,
        layers=[3, 4, 23, 3]
    )
    model_resnet = ResnetModel.get_model()
    return model_resnet


def resnet152():
    ResnetModel = Resnet(
        model_version=152,
        layers=[3, 8, 36, 3]
    )
    model_resnet = ResnetModel.get_model()
    return model_resnet


# Resnet List of version
class ResnetNetModule():
    def __init__(self, name: str):
        if '18' in name:
            self.model = resnet18()
        elif '34' in name:
            self.model = resnet34()
        elif '50' in name:
            self.model = resnet50()
        elif '101' in name:
            self.model = resnet101()
        elif '152' in name:
            self.model = resnet152()
        else:
            raise Exception('No model Version Found')
    def get_model(self):
        return self.model