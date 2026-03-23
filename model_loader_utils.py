# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
from PIL import Image
import numpy as np
from comfy.utils import common_upscale
import comfy.model_management as mm
import io as iooo
import folder_paths
from datetime import datetime
import torchaudio
cur_path = os.path.dirname(os.path.abspath(__file__))

# wrapped comfy
def  re_save_video(videos,codec,filename_prefix,format):
    from comfy_api.latest import  Types
    filepath_list=[]
    file_folder_list=[]
    for i, video in enumerate(videos):
    #video=InputImpl.VideoFromComponents(Types.VideoComponents(images=images, audio=audio, frame_rate=Fraction(fps)))
        width, height = video.get_dimensions()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix+str(i),
                folder_paths.get_output_directory(),
                width,
                height
            )

        saved_metadata = None

        file = f"{filename}_{counter:05}_.{Types.VideoContainer.get_extension(format)}"
        video_path=os.path.join(full_output_folder, file)
        filename = os.path.basename(video_path)
        folder_name, _ = os.path.splitext(filename)
        video_dir=os.path.join(full_output_folder,folder_name)
        os.makedirs(video_dir, exist_ok=True)
        video.save_to(
            os.path.join(video_dir, file),
            format=Types.VideoContainer(format),
            codec=codec,
            metadata=saved_metadata
        )
        filepath_list.append(os.path.join(video_dir, file))
        file_folder_list.append(video_dir)
    return filepath_list,file_folder_list



def audio2path(audio):
    audio_file_prefix = datetime.now().strftime("%y%m%d%H%M%S")[-6:]  
    audio_file = os.path.join(folder_paths.get_temp_directory(), f"audio_refer_temp{audio_file_prefix}.wav")
    buff = iooo.BytesIO() 
    torchaudio.save(buff, audio["waveform"].squeeze(0), audio["sample_rate"],format="wav")
    with open(audio_file, 'wb') as f:
        f.write(buff.getbuffer())
    return audio_file

def dialogue_split(dialogue):
    dialogue_lines=[line.strip() for line in dialogue.split("\n") if line.strip()]
    parsed_dialogue = []
    for line in dialogue_lines:
        parts = line.split(", ")
        entry = {}
        for part in parts:
            if ":" in part:
                key, value = part.split(":", 1)  
                entry[key.strip()] = value.strip() if key.strip() not in["start","duration"] else float(value.strip())
        parsed_dialogue.append(entry)
    return parsed_dialogue



def clear_comfyui_cache():
    cf_models=mm.loaded_models()
    try:
        for pipe in cf_models:
            pipe.unpatch_model(device_to=torch.device("cpu"))
            print(f"Unpatching models.{pipe}")
    except: pass
    mm.soft_empty_cache()
    torch.cuda.empty_cache()
    max_gpu_memory = torch.cuda.max_memory_allocated()
    print(f"After Max GPU memory allocated: {max_gpu_memory / 1000 ** 3:.2f} GB")

def gc_cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def phi2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img

def tensor2image(tensor):
    tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def tensor2pillist(tensor_in):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [tensor2image(tensor_in)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[tensor2image(i) for i in tensor_list]
    return img_list

def tensor2pillist_upscale(tensor_in,width,height):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [nomarl_upscale(tensor_in,width,height)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[nomarl_upscale(i,width,height) for i in tensor_list]
    return img_list

def tensor2list(tensor_in,width,height):
    if tensor_in is None:
        return None
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        tensor_list = [tensor_upscale(tensor_in,width,height)]
    else:
        tensor_list_ = torch.chunk(tensor_in, chunks=d1)
        tensor_list=[tensor_upscale(i,width,height) for i in tensor_list_]
    return tensor_list


def tensor_upscale(tensor, width, height):
    samples = tensor.movedim(-1, 1)
    samples = common_upscale(samples, width, height, "bilinear", "center")
    samples = samples.movedim(1, -1)
    return samples

def nomarl_upscale(img, width, height):
    samples = img.movedim(-1, 1)
    img = common_upscale(samples, width, height, "bilinear", "center")
    samples = img.movedim(1, -1)
    img = tensor2image(samples)
    return img

