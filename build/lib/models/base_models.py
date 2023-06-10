from model_tools.check_submission import check_models

import functools

import torchvision.models
from model_tools.activations.pytorch import PytorchWrapper
# from model_tools.activations.pytorch import load_preprocess_images
# https://github.com/brain-score/model-tools

# https://huggingface.co/docs/transformers/model_doc/dpt
from transformers import AutoImageProcessor, ViTMAEModel
from transformers import ViTImageProcessor, ViTModel
from transformers import AutoFeatureExtractor, ResNetModel
from transformers import AutoImageProcessor, VideoMAEModel
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import CLIPProcessor, CLIPModel
from huggingface_hub import hf_hub_download
from PIL import Image
import requests

import numpy as np
import torch

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
    ## submit version 
    # 0: layer[1, 12) / 1: layer[0, 12) / 2: original model's preprocessor
    submit_version = 2
    ## model list
    # model_list = ['mae_vitb16']
    # model_list = ['mae_vitl16']
    # model_list = ['dinov1_vits16']
    # model_list = ['dinov1_vits8']
    # model_list = ['dinov1_vitb16']
    # model_list = ['dinov1_vitb8']
    # model_list = ['dinov1_resnet-50_Ramos-Ramos']
    model_list = ['clip_vitb16']
    # model_list = ['clip_vitl14']
    # model_list = ['vit_vitb16']
    # model_list = ['vit_vitb16_in21k']
    # model_list = ['vit_vitl16']
    # model_list = ['vit_vitl16_in21k']
    # model_list = ['videomae_vitb16_videoinput']
    # model_list = ['videomae_vitl16_videoinput']
    # model_list = ['videomae_vitb16_videoinput_finetuned-kinetics']
    # model_list = ['videomae_vitb16_videoinput_finetuned-ssv2']
    # model_list = ['timesformer_vitb16_videoinput_finetuned-k400']
    # model_list = ['timesformer_vitb16_videoinput_finetuned-k600']
    # model_list = ['timesformer_vitb16_videoinput_finetuned-ssv2']
    
    return [model+'_'+str(submit_version) for model in model_list]
    
    ## Category
    # Basic: ViT, DINO, MAE, CLIP
    # Video training: VideoMAE, Timesformer
    # Convolution like: CvT, Convnext, LeViT,
    # Robotics: VC-1, VIP, R3M
    # Different training: ViTMSN, Swin Transformer, DeiT
    # Different task: SAM, DPT, DETR, Segformer, ...
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
    if model_name == 'vit':
        # https://huggingface.co/models?search=google/vit
        if 'vitb' in name:
            if 'in21k' in name:
                processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
                model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
            else:
                processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
                model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        elif 'vitl' in name:
            if 'in21k' in name:
                processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224-in21k')
                model = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k')
            else:
                processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224')
                model = ViTModel.from_pretrained('google/vit-large-patch16-224')

    elif model_name == 'mae':
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit_mae/modeling_vit_mae.py
        if 'vitb' in name:
            model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
            processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        elif 'vitl' in name:
            model = ViTMAEModel.from_pretrained("facebook/vit-mae-large")
            processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-large")
        elif 'vith' in name:
            model = ViTMAEModel.from_pretrained("facebook/vit-mae-large")
            processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-large")
        else:
            raise NotImplementedError(f'unknown model for getting model {name}')

        # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        # image = Image.open(requests.get(url, stream=True).raw)
        # image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        # inputs = image_processor(images=image, return_tensors="pt")
        # outputs = model(**inputs)
        # last_hidden_states = outputs.last_hidden_state

        # breakpoint()
    
    elif model_name == 'dinov1':
        # https://huggingface.co/facebook/dino-vitb16
        # https://huggingface.co/facebook/dino-vitb8
        # https://huggingface.co/facebook/dino-vits16
        # https://huggingface.co/facebook/dino-vits8
        # https://huggingface.co/Ramos-Ramos/dino-resnet-50

        if 'vitb16' in name:
            model = ViTModel.from_pretrained('facebook/dino-vitb16')
            processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
        elif 'vitb8' in name:
            model = ViTModel.from_pretrained('facebook/dino-vitb8')
            processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb8')
        elif 'vits16' in name:
            model = ViTModel.from_pretrained('facebook/dino-vits16')
            processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
        elif 'vits8' in name:
            model = ViTModel.from_pretrained('facebook/dino-vits8')
            processor = ViTImageProcessor.from_pretrained('facebook/dino-vits8')
        elif 'resnet-50' in name:
            processor = AutoFeatureExtractor.from_pretrained('Ramos-Ramos/dino-resnet-50')
            model = ResNetModel.from_pretrained('Ramos-Ramos/dino-resnet-50')
            # Warning: The processor in this code is a copy of the one from microsoft/resnet-50. 
            # We never verified if this image prerprocessing is the one used with DINO ResNet-50.
        else:
            raise NotImplementedError(f'unknown model for getting model {name}')

    elif model_name == 'clip':
        # https://huggingface.co/models?search=openai/clip
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        model = clip_model.vision_model
        image_text_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        def clip_processor(images=None, return_tensors="pt", image_text_processor=None):
            text = ["a photo of a cat", "a photo of a dog"]
            inputs = image_text_processor(text=text, images=images, return_tensors=return_tensors)
            inputs = {"pixel_values": inputs["pixel_values"]}
            return inputs
        processor = functools.partial(clip_processor, return_tensors="pt", image_text_processor=image_text_processor)

    elif model_name == 'videomae':
        # https://huggingface.co/models?search=mcg-nju
        def videomae_processor(images=None, return_tensors="pt", image_processor=None):
            num_frames = 16
            video = create_static_video(images, num_frames)
            inputs = image_processor([video[i] for i in range(video.shape[0])], return_tensors=return_tensors)
            return inputs
        
        ## select model and processor
        if 'vitb' in name:
            if 'finetuned-kinetics' in name:
                model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
                image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
            elif 'finetuned-ssv2' in name:
                model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2")
                image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2")
            else:
                model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
                image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        elif 'vitl' in name:
            model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-large")
            image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-large")
            
        ## select input
        if name.split('_')[2] == 'videoinput':
            processor = functools.partial(videomae_processor, return_tensors="pt", image_processor=image_processor)
        else:
            raise NotImplementedError(f'unknown model for getting model {name}')

    elif model_name == 'timesformer':
        # https://huggingface.co/models?search=facebook/timesformer
        def timesformer_processor(images=None, return_tensors="pt", image_processor=None):
            num_frames = 8
            video = create_static_video(images, num_frames, normalize_0to1=True, channel_dim=1)
            inputs = image_processor([video[i] for i in range(video.shape[0])], return_tensors=return_tensors) # list(video)
            return inputs

        ## select model and processor
        if 'vitb' in name:
            if 'finetuned-k400' in name:
                image_processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
                model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")
            elif 'finetuned-ssv2' in name:
                image_processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-ssv2")
                model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-ssv2")
            elif 'finetuned-k600' in name:
                image_processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k600")
                model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k600")
        else:
            raise NotImplementedError(f'unknown model for getting model {name}')

        ## select input
        if name.split('_')[2] == 'videoinput':
            processor = functools.partial(timesformer_processor, return_tensors="pt", image_processor=image_processor)
        else:
            raise NotImplementedError(f'unknown model for getting model {name}')

    else:
        raise NotImplementedError(f'unknown model for getting model {name}')

    preprocessing = functools.partial(load_preprocess_images, processor=processor, image_size=224)
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    wrapper.image_size = 224

    breakpoint()

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
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py#L832
    # layers.append('encoder'), layers.append('embeddings') does not work because they are tuples
    # dinov1: layers.append('pooler') does not work. It is also not trained
    model_backbone = name.split('_')[1]
    layers = []
    if 'clip_vitb' in name:
        for i in range(12):
            # layers.append(f'encoder.layers.{i}.mlp')
            # layers.append(f'encoder.layers.{i}.layer_norm1')
            layers.append(f'encoder.layers.{i}')
        layers.append('encoder')
        # layers.append('post_layernorm') --> KeyError: 'embedding'
    elif 'clip_vitl' in name:
        for i in range(24):
            layers.append(f'encoder.layers.{i}.mlp')
            layers.append(f'encoder.layers.{i}.layer_norm1')
    elif 'timesformer_vitb' in name:
        for i in range(12):
            layers.append(f'timesformer.encoder.layer.{i}.layernorm_before')
            layers.append(f'timesformer.encoder.layer.{i}.output')
        # layers.append('timesformer.layernorm')
        # layers.append('classifier')
    elif 'timesformer_vitl' in name:
        for i in range(24):
            layers.append(f'timesformer.encoder.layer.{i}.layernorm_before')
            layers.append(f'timesformer.encoder.layer.{i}.output')
        # layers.append('timesformer.layernorm')
        # layers.append('classifier')
    elif 'vits' in model_backbone:
        for i in range(12): # 12
            layers.append(f'encoder.layer.{i}.output')
        layers.append('layernorm')
    elif 'vitb' in model_backbone:
        for i in range(12): # 12
            layers.append(f'encoder.layer.{i}.output')
        layers.append('layernorm')
    elif 'vitl' in model_backbone:
        for i in range(24): # 24
            layers.append(f'encoder.layer.{i}.output')
        layers.append('layernorm')
    elif 'dinov1_resnet-50' in model_backbone:
        layers = []
        layers += ['embedder.pooler']
        layers += [f'encoder.stages.0.layers.{i}' for i in range(3)]
        layers += [f'encoder.stages.1.layers.{i}' for i in range(3)]
        layers += [f'encoder.stages.2.layers.{i}' for i in range(6)]
        layers += [f'encoder.stages.3.layers.{i}' for i in range(3)]
    else:
        raise NotImplementedError(f'unknown model for getting layers {name}')

    return layers


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return ''


def load_preprocess_images(image_filepaths, image_size, processor=None, **kwargs):
    images = load_images(image_filepaths)
    # images = [<PIL.Image.Image image mode=RGB size=400x400 at 0x7F8654B2AC10>, ...]
    images = [image.resize((image_size, image_size)) for image in images]
    if processor is not None:
        images = [processor(images=image, return_tensors="pt", **kwargs) for image in images]
        if len(images[0].keys()) != 1:
            raise NotImplementedError(f'unknown processor for getting model {processor}')
        assert list(images[0].keys())[0] == 'pixel_values'
        images = [image['pixel_values'] for image in images]
        images = torch.cat(images)
        images = images.cpu().numpy()
    else:
        images = preprocess_images(images, image_size=image_size, **kwargs)
    return images

def load_images(image_filepaths):
    return [load_image(image_filepath) for image_filepath in image_filepaths]

def load_image(image_filepath):
    with Image.open(image_filepath) as pil_image:
        if 'L' not in pil_image.mode.upper() and 'A' not in pil_image.mode.upper() \
                and 'P' not in pil_image.mode.upper():  # not binary and not alpha and not palletized
            # work around to https://github.com/python-pillow/Pillow/issues/1144,
            # see https://stackoverflow.com/a/30376272/2225200
            return pil_image.copy()
        else:  # make sure potential binary images are in RGB
            rgb_image = Image.new("RGB", pil_image.size)
            rgb_image.paste(pil_image)
            return rgb_image

def preprocess_images(images, image_size, **kwargs):
    preprocess = torchvision_preprocess_input(image_size, **kwargs)
    images = [preprocess(image) for image in images]
    images = np.concatenate(images)
    return images

def torchvision_preprocess_input(image_size, **kwargs):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        torchvision_preprocess(**kwargs),
    ])

def torchvision_preprocess(normalize_mean=(0.485, 0.456, 0.406), normalize_std=(0.229, 0.224, 0.225)):
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
        lambda img: img.unsqueeze(0)
    ])

def create_static_video(image, num_frames, normalize_0to1=False, channel_dim=3):
    '''
    Create a static video with the same image in all frames.
    Args:
        image (PIL.Image.Image): Input image.
        num_frames (int): Number of frames in the video.
    Returns:
        result (np.ndarray): np array of frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    for _ in range(num_frames):
        frame = np.array(image)
        if normalize_0to1:
            frame = frame / 255.
        if channel_dim == 1:
            frame = frame.transpose(2, 0, 1)
        frames.append(frame)
    return np.stack(frames)


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