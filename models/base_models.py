from model_tools.check_submission import check_models

import functools

import torchvision.models
from model_tools.activations.pytorch import PytorchWrapper
# from model_tools.activations.pytorch import load_preprocess_images
# https://github.com/brain-score/model-tools

from transformers import AutoImageProcessor, ViTMAEModel
from transformers import ViTImageProcessor, ViTModel
from transformers import AutoFeatureExtractor, ResNetModel
from transformers import AutoImageProcessor, VideoMAEModel
from huggingface_hub import hf_hub_download
import av
from PIL import Image
import requests

import numpy as np
import torch

"""
Template module for a base model submission to brain-score
"""

def load_preprocess_images(image_filepaths, image_size, processor=None, **kwargs):
    images = load_images(image_filepaths)
    # images = [<PIL.Image.Image image mode=RGB size=400x400 at 0x7F8654B2AC10>, ...]
    if processor is not None:
        images = [processor(images=image, return_tensors="pt") for image in images]
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

def create_static_video(image, num_frames):
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
        frames.append(frame)
    return np.stack(frames)


def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """
    # 0: layer[1, 12) / 1: layer[0, 12) / 2: original model's preprocessor
    return ['videomae_vitb16_videoinput_1']
    # return ['mae_vitb16_1',  'mae_vitl16_1'] # 'videomae-vitb16-videoinput'
    # return ['dinov1_vits16_1', 'dinov1_vits8_1', 'dinov1_vitb16_1', 'dinov1_vitb8_1'] #,  #, 'dino_resnet-50_0']
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
            processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        elif 'vitl' in name:
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
            raise NotImplementedError(f'unknown model for getting model {name}')
            feature_extractor = AutoFeatureExtractor.from_pretrained('Ramos-Ramos/dino-resnet-50')
            model = ResNetModel.from_pretrained('Ramos-Ramos/dino-resnet-50')
            inputs = feature_extractor(images=image, return_tensors="pt")
            outputs = model(**inputs)

        # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
        # image = Image.open(requests.get(url, stream=True).raw)
        # inputs = processor(images=image, return_tensors="pt")
        # outputs = model(**inputs)
        # last_hidden_states = outputs.last_hidden_state

        # breakpoint()

    elif model_name == 'videomae':
        if name.split('_')[2] == 'videoinput':
            def read_video_pyav(container, indices):
                '''
                Decode the video with PyAV decoder.
                Args:
                    container (`av.container.input.InputContainer`): PyAV container.
                    indices (`List[int]`): List of frame indices to decode.
                Returns:
                    result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
                '''
                frames = []
                container.seek(0)
                start_index = indices[0]
                end_index = indices[-1]
                for i, frame in enumerate(container.decode(video=0)):
                    if i > end_index:
                        break
                    if i >= start_index and i in indices:
                        frames.append(frame)
                return np.stack([x.to_ndarray(format="rgb24") for x in frames])
            breakpoint()
            
            file_path = hf_hub_download(
                repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
            )
            container = av.open(file_path)
            # indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
            indices = [i for i in range(16)]
            video = read_video_pyav(container, indices)
            # video = np.array with shape (16, 360, 640, 3), max value 255, dtype uint8
            image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
            inputs = image_processor(list(video), return_tensors="pt")
            breakpoint()

            def processor(images=None, return_tensors="pt"):
                image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
                # change PIL.Image to torch.Tensor
                num_frames = 16
                video = create_static_video(image, num_frames)
                inputs = image_processor(list(video), return_tensors=return_tensors)
                return images
        else:
            raise NotImplementedError(f'unknown model for getting model {name}')

    else:
        raise NotImplementedError(f'unknown model for getting model {name}')

    preprocessing = functools.partial(load_preprocess_images, processor=processor, image_size=224)
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    wrapper.image_size = 224

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
    # layers.append('encoder'), layers.append('embeddings') does not work because they are tuples
    # dinov1: layers.append('pooler') does not work. It is also not trained
    model_backbone = name.split('_')[1]
    layers = []
    if 'vits' in model_backbone:
        for i in range(12): # 12
            layers.append(f'encoder.layer.{i}.output')
            # layers.append(f'encoder.layer.{i}.layernorm_before')
        layers.append('layernorm')
    elif 'vitb' in model_backbone:
        for i in range(12): # 12
            layers.append(f'encoder.layer.{i}.output')
            # layers.append(f'encoder.layer.{i}.layernorm_before')
        layers.append('layernorm')
    elif 'vitl' in model_backbone:
        for i in range(24): # 24
            layers.append(f'encoder.layer.{i}.output')
            # layers.append(f'encoder.layer.{i}.layernorm_before')
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