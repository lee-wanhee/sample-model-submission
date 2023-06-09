from model_tools.check_submission import check_models

import functools

import torchvision.models
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images
# https://github.com/brain-score/model-tools

from transformers import AutoImageProcessor, ViTMAEModel
from PIL import Image
import requests

"""
Template module for a base model submission to brain-score
"""


def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """
    
    return ['mae_vitb16_0',  'mae_vitl16_0'] # 'videomae-vitb16-videoinput'
    # ['mae-vitb16', 'videomae', 'dino', 'clip', 'vc-1', 'vip', 'vit', 'timesformer', 'deit', 'sam', 'dpt', 'cvt']


def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    model_name = name.split('_')[0]
    if model_name == 'mae':
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit_mae/modeling_vit_mae.py
        if 'vitb' in name:
            model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
        elif 'vitl' in name:
            model = ViTMAEModel.from_pretrained("facebook/vit-mae-large")
        else:
            raise NotImplementedError(f'unknown model for getting model {name}')

        # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        # image = Image.open(requests.get(url, stream=True).raw)
        # image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        # inputs = image_processor(images=image, return_tensors="pt")
        # outputs = model(**inputs)
        # last_hidden_states = outputs.last_hidden_state

        # breakpoint()

        preprocessing = functools.partial(load_preprocess_images, image_size=224)
        wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
        wrapper.image_size = 224
        
    elif model_name == 'videomae':
        if name.split('_')[-1] == 'videoinput':
            preprocessing = functools.partial(load_preprocess_images, image_size=224)
            preprocessing = lambda x: preprocessing(x).unsqueeze(0).repeat(12, 1, 1, 1)

        else:
            raise NotImplementedError(f'unknown model for getting model {name}')

    else:
        raise NotImplementedError(f'unknown model for getting model {name}')

    return wrapper


def get_layers(name):
    """
    This method returns a list of string layer names to consider per model. The benchmarks maps brain regions to
    layers and uses this list as a set of possible layers. The lists doesn't have to contain all layers, the less the
    faster the benchmark process works. Additionally the given layers have to produce an activations vector of at least
    size 25! The layer names are delivered back to the model instance and have to be resolved in there. For a pytorch
    model, the layer name are for instance dot concatenated per module, e.g. "features.2".
    :param name: the name of the model, to return the layers for
    :return: a list of strings containing all layers, that should be considered as brain area.
    """
    # https://github.com/brain-score/candidate_models/blob/master/candidate_models/model_commitments/model_layer_def.py
    model_backbone = name.split('_')[1]
    if 'vitb' in model_backbone:
        layers = []
        # layers.append('embeddings')
        for i in range(11): # 11
            index = i + 1
            layers.append(f'encoder.layer.{index}.output')
            layers.append(f'encoder.layer.{index}.layernorm_before')
            # layers = ['encoder.layer.1.output', 'encoder.layer.1.layernorm_before', 'encoder.layer.2.output'] 
        # layers.append('encoder')
        layers.append('layernorm')
    elif 'vitl' in model_backbone:
        layers = []
        for i in range(23): # 23
            index = i + 1
            layers.append(f'encoder.layer.{index}.output')
            layers.append(f'encoder.layer.{index}.layernorm_before')
            # layers = ['encoder.layer.1.output', 'encoder.layer.1.layernorm_before', 'encoder.layer.2.output'] 
        layers.append('layernorm')
    else:
        raise NotImplementedError(f'unknown model for getting layers {name}')

    return layers


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return ''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)

"""
Notes on the error:

- 'channel_x' key error: 
# 'embeddings.patch_embeddings.projection',
https://github.com/search?q=repo%3Abrain-score%2Fmodel-tools%20channel_x&type=code

"""