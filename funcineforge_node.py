 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
from omegaconf import OmegaConf
import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io,Types
import nodes
from .model_loader_utils import clear_comfyui_cache,audio2path,re_save_video
from .infer import load_model
from .predata import pre_data_simple,funcineforge_infer
import json


MAX_SEED = np.iinfo(np.int32).max
node_cr_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

weigths_funcineforge_current_path = os.path.join(folder_paths.models_dir, "funcineforge")
if not os.path.exists(weigths_funcineforge_current_path):
    os.makedirs(weigths_funcineforge_current_path)
folder_paths.add_model_folder_path("funcineforge", weigths_funcineforge_current_path) #  funcineforge dir


class FunCineForge_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FunCineForge_SM_Model",
            display_name="FunCineForge_SM_Model",
            category="FunCineForge",
            inputs=[
                io.Combo.Input("flow",options= ["none"] + [ i for i in folder_paths.get_filename_list("funcineforge") if "flow" in i.lower() and i.endswith(".pt") ] ),
                io.Combo.Input("llm",options= ["none"] + [i for i in folder_paths.get_filename_list("funcineforge") if "llm" in i.lower() and i.endswith(".pt") ]),
                io.Combo.Input("voc", options=["none"] + [i for i in folder_paths.get_filename_list("funcineforge") if "vocoder" in i.lower() and i.endswith(".pt") ]),
                io.Combo.Input("dtype",options= ["bf16","fp16","fp32"] ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls,flow,llm,voc,dtype) -> io.NodeOutput:
        clear_comfyui_cache()
        params=OmegaConf.load(os.path.join(os.path.join(node_cr_path, "exps/decode_conf/decode.yaml")))
        params["tokenizer_conf_local"] = {'init_param_path': os.path.join(weigths_funcineforge_current_path, "Qwen2-0.5B-CosyVoice-BlankEN")}
        params["model_conf_local"] = {'init_param_path': os.path.join(weigths_funcineforge_current_path, "Qwen2-0.5B-CosyVoice-BlankEN")}
        params["face_encoder_conf_local"] = {'init_param_path': os.path.join(weigths_funcineforge_current_path, "face_recog_ir101.onnx")}
        params["mode"] = "llm"
        params["infer_dtype"] = dtype
        params["xvec_model"]=os.path.join(weigths_funcineforge_current_path, "camplus.onnx")
        #model=init_engine(weigths_funcineforge_current_path,)
        model= load_model(llm, flow, voc,params,dtype,)
        return io.NodeOutput(model)


class FunCineForge_SM_Segments(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FunCineForge_SM_Segments",
            display_name="FunCineForge_SM_Segments",
            category="FunCineForge",
            inputs=[
                io.String.Input("text",default="大王若是圣明，自然知道我张仪就是掉了脑袋，也不会把秦国的土地轻易交给楚国。",multiline=True),  
                io.String.Input("clue",default="一位中年男性角色向大王陈述立场，语气沉稳且坚定，言辞间流露出对自身忠诚的强烈自信与决心。整体情感线索是忠贞不渝的承诺和不容置疑的信念。",multiline=True), 
                io.Float.Input("start_time", default=0, min=0, max=nodes.MAX_RESOLUTION,step=0.01,display_mode=io.NumberDisplay.number),
                io.Float.Input("end_time", default=6.22, min=0, max=nodes.MAX_RESOLUTION,step=0.01,display_mode=io.NumberDisplay.number),
                io.Combo.Input("age_input",default="中年",options=["儿童", "青年", "中年", "中老年", "老年", "不确定"]),
                io.Combo.Input("gender_input",default="男",options=["男", "女", "不确定"]),
                io.Combo.Input("spk",default=1,options=[1,2,3,4,5,6,7,8,9,10]),
                io.Conditioning.Input("conds",optional=True),
                io.Audio.Input("refer_audio", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conds")
                ],
            )
    @classmethod
    def execute(cls,text,clue,start_time,end_time,age_input,gender_input,spk,conds=None,refer_audio=None) -> io.NodeOutput:
        clear_comfyui_cache()
        if refer_audio is not None:
            refer_audio=audio2path(refer_audio)
        segments={
        "start": start_time,
        "end": end_time,
        "age": age_input,
        "gender": gender_input,
        "spk": str(spk),
            }
        message = {
            "text":text,
            "clue":clue,
            "spk": str(spk),
            "refer_audio": refer_audio,
        }
        if conds is not  None :
            if conds.get("conds",None) is not None:
                segment_c=conds["conds"]
                segment_c.append(segments)
                messages=conds["messages"]
                messages.append(message)
            else:
                segment_c=[segments]
                messages=[message]
        else:
            segment_c=[segments]
            messages=[message]
        output={"conds":segment_c,"messages":messages}
        return io.NodeOutput(output)
      
class FunCineForge_SM_Predata(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FunCineForge_SM_Predata",
            display_name="FunCineForge_SM_Predata",
            category="FunCineForge",
            inputs=[
                io.Video.Input("videos"),
                io.Conditioning.Input("conds"),
                io.Combo.Input("video_type",default="独白",options=["独白", "旁白", "对话", "多人", ]),
                io.Combo.Input("format", options=Types.VideoContainer.as_input(), default="auto", tooltip="The format to save the video as."),
                io.Combo.Input("codec", options=Types.VideoCodec.as_input(), default="auto", tooltip="The codec to use for the video."),
                io.String.Input("filename_prefix", default="video/ComfyUI", tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                io.String.Output(display_name="full_report")
                ],
            )
    @classmethod
    def execute(cls,videos,conds,video_type,format,codec,filename_prefix) -> io.NodeOutput:
        conds_datas=conds["conds"]
        messages=conds["messages"]
        fps=videos.get_components().frame_rate

        clear_comfyui_cache()
        video_path,video_dir= re_save_video(videos,codec,filename_prefix,format)
        jsonl_path,jsonl_items,full_report=pre_data_simple(
            video_path, os.path.join(node_cr_path,"speaker_diarization/speaker_diarization_sample/config/diar.yaml"), 
            weigths_funcineforge_current_path, video_dir, device,
            os.path.join(node_cr_path,"speaker_diarization/speaker_diarization_sample/config/decode.yaml"), conds_datas,video_type,messages,fps,
            )
        conditioning=[jsonl_path,jsonl_items,video_path,video_dir]
        #full_report = "\n".join(full_report)
        return io.NodeOutput(conditioning,full_report)
    
class FunCineForge_SM_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FunCineForge_SM_KSampler",
            display_name="FunCineForge_SM_KSampler",
            category="FunCineForge",
            inputs=[
                io.Model.Input("model"), 
                io.String.Input("infer_dir", default="/video/ComfyUI_00014_",),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED,display_mode=io.NumberDisplay.number),
                io.Conditioning.Input("conditioning",optional=True),
            ], 
            outputs=[
                io.Audio.Output(display_name="audio"),
            ],
        )
    @classmethod
    def execute(cls, model,infer_dir,seed,conditioning=None) -> io.NodeOutput:
        if conditioning is None and infer_dir :
            jsonl_path = os.path.join(infer_dir, "input_data.jsonl")
            if not os.path.exists(jsonl_path):
                raise "Conditioning data not provided and no valid infer_dir with input_data.jsonl found at jsonl_path"
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                jsonl_items = []
                for line in f:
                    if line.strip(): 
                        jsonl_items.append(json.loads(line)) 
                        #print(jsonl_items)
            video_path = os.path.join(infer_dir, f'{os.path.basename(os.path.normpath(infer_dir))}.mp4')
            conditioning=[jsonl_path,jsonl_items,video_path,infer_dir]

        cfg={"infer_dtype":model.kwargs["infer_dtype"],"seed":seed}
        audio_list=funcineforge_infer(model, conditioning,cfg)
        #audio_list=model.inference(input=index_ds, input_len=len(index_ds),**cfg)
        #waveform_list=[]
        #for audio_list in audio_lists:
        waveform=audio_list[0].cpu().float().unsqueeze(0) if len(audio_list)==1 else torch.cat(audio_list, dim=1).cpu().float().unsqueeze(0)
        #print(waveform.shape) #torch.Size([1, 1, 195840])
            #waveform_list.append(waveform)
        #waveform=torch.cat(waveform_list, dim=-1)
        output={"waveform":waveform,"sample_rate":24000}
        return io.NodeOutput(output)


class  FunCineForge_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            FunCineForge_SM_Model,
            FunCineForge_SM_Segments,
            FunCineForge_SM_Predata,
            FunCineForge_SM_KSampler,
        ]
async def comfy_entrypoint() -> FunCineForge_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return FunCineForge_SM_Extension()
