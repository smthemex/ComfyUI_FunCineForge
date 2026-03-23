import os
from pathlib import Path
import sys
sys.path.append("local")
import argparse
from .speaker_diarization.local.cluster_and_postprocess import main as cluster_main
from .speaker_diarization.local.overlap_detection import  main as overlap_main
from .speaker_diarization.local.filter_clean_list import main as filter_main
from .speaker_diarization.local.voice_activity_detection import main as vad_main
from .speaker_diarization.local.prepare_subseg_json import main as subseg_main
from .speaker_diarization.local.extract_speech_embeddings import main as extract_main
from .speaker_diarization.local.extract_visual_embeddings import main as visual_main
from .normalize_trim import main as normalize_main
from .speech_separation.run import main as separation_main
from .video_clip.videoclipper import main as clip_main
from .clean_video import main as clean_main
from .clean_srt import main as srt_main,DEFAULT_MIN_AUDIO_SEC_FOR_TEXT_CHECK,DEFAULT_MIN_ASCII_CHARS,DEFAULT_MIN_CJK_CHARS
from .utils  import generate_jsonl_data
from .speaker_diarization.speaker_diarization_sample.run import main as simple_main
from omegaconf import OmegaConf
def clean_video(root_dir):
    args = argparse.Namespace(
        root=root_dir,
        min_sec=2.0,
        max_sec=60.0,
        workers=1,
        max_outstanding=1,
        execute=True,
        log="delete_video.log"
    )
    clean_main(args)

def clean_srt(root_dir):
    args = argparse.Namespace(
        root=root_dir,
        lang='zh',
        min_audio_sec=DEFAULT_MIN_AUDIO_SEC_FOR_TEXT_CHECK,
        min_cjk_chars=DEFAULT_MIN_CJK_CHARS,
        min_ascii_chars=DEFAULT_MIN_ASCII_CHARS,
        workers=1,
        max_outstanding=1,
        execute=True,
        delete_log="delete_srt.log"
    )

    srt_main(args)

def videoclipper(stage,file,output_dir,model_dict,sd_switch=True,device="cpu"):

    args = argparse.Namespace(
        stage=stage,
        file=file,
        output_dir=output_dir,
        device=device,
        sd_switch=sd_switch,
        lang="zh",
        skip_processed=True,
        model_dict=model_dict,
    )
    clip_main(args)


def separation_root(root_dir,config_path,model_path):
    args = argparse.Namespace(
        model_type="mel_band_roformer",
        root=root_dir,
        config_path=config_path,
        model_path=model_path,
        gpus=0,
        use_amp=False,
        max_procs=1,
    )

    separation_main(args)


def normalize_and_trim(root_dir,intro=10,outro=10):
    args = argparse.Namespace(
        root=root_dir,
        intro=intro,
        outro=outro,
        workers=1,
    )

    normalize_main(args)

def generate_data_lists(root_dir):
    """
    生成视频和音频文件列表
    
    参数:
        root_dir: 数据集根目录
        
    返回:
        无，但会在每个包含clipped子目录的目录下生成video.list和wav.list文件
    """
    root_path = Path(root_dir)
    for clipped_dir in root_path.rglob("clipped"):
        if clipped_dir.is_dir():
            parent_dir = clipped_dir.parent
            
            # 生成video.list
            video_list = sorted([str(f) for f in clipped_dir.glob("*.mp4")])
            with open(os.path.join(parent_dir, "video.list"), "w") as f:
                f.write("\n".join(video_list))
            
            # 生成wav.list
            wav_list = sorted([str(f) for f in clipped_dir.glob("*.wav")])
            with open(os.path.join(parent_dir, "wav.list"), "w") as f:
                f.write("\n".join(wav_list))
            
            print(f"Generated lists for {parent_dir}")


def overlap_detection(wav_list, output_dir, local_dir, nj=1, overlap_threshold=0.8):
    args = argparse.Namespace(
        wavs=wav_list,
        out_dir=output_dir,
        local_dir=local_dir,
        overlap_threshold=str(overlap_threshold),
        batch_size=1,
        use_gpu=False,
        workers=1,
        nj=nj,
    )

    overlap_main(args)

def filter_clean_lists(wav_list, video_list, overlap_list, clean_wav_list, clean_video_list):
    args = argparse.Namespace(
        wav_list=wav_list,
        video_list=video_list,
        overlap_list=overlap_list,
        clean_wav_list=clean_wav_list,
        clean_video_list=clean_video_list,
    )

    filter_main(args)


def voice_activity_detection(wavs_list, out_file,local_model):
    args = argparse.Namespace(
        wavs=wavs_list,
        out_file=out_file,
        local_model=local_model,
    )
    vad_main(args)

def prepare_subsegments(vad_file, out_file):
    args = argparse.Namespace(
            vad=vad_file,
            out_file=out_file,
            dur=1.5,
            shift=0.75
        )
    subseg_main(args)

def extract_speech_embeddings(pretrained_model, conf_file, subseg_json, embs_out, gpus=0, use_gpu=False):

    args = argparse.Namespace(
        model_id=None,
        pretrained_model=pretrained_model,
        conf=conf_file,
        subseg_json=subseg_json,
        embs_out=embs_out,
        batchsize=1,
        use_gpu=use_gpu,
        gpu=gpus,
        rank=0,
        world_size=0
    )
    extract_main(args)

def extract_visual_embeddings(conf_file, videos_dir, onnx_dir, workers=64):
    args = argparse.Namespace(
        conf=conf_file,
        videos=videos_dir,
        onnx_dir=onnx_dir,
        debug_dir="", #TODO
        workers=workers
    )
    visual_main(args)

def cluster_and_postprocess(conf_file, root_dir,):

    args = argparse.Namespace(
        conf=conf_file,
        root=root_dir,
        wav_list_name="clean_wav.list",
    )
    cluster_main(args)

def generate_data( node_dir,pretrained_models_dir,separate_model_path,intro,outro,root_dir = "/path/to/your/dataset/raw_zh",check_audio=False):
    conf_file=os.path.join(node_dir,"speaker_diarization/conf/diar_video.yaml")
    audio_conf_file=os.path.join(node_dir,"speaker_diarization/conf/diar.yaml")

    output_dir=os.path.join( Path(root_dir).parent, "clean/zh")
    model_dict={
        "model":os.path.join(pretrained_models_dir,"speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
        "vad_model":os.path.join(pretrained_models_dir,"speech_fsmn_vad_zh-cn-16k-common-pytorch"),
        "punc_model":os.path.join(pretrained_models_dir,"punc_ct-transformer_zh-cn-common-vocab272727-pytorch"),
        "spk_model":os.path.join(pretrained_models_dir,"speech_campplus_sv_zh-cn_16k-common"),
    }

    # 1.将视频格式、名称标准化；裁剪长视频的片头片尾；提取裁剪后视频的音频。（默认是从起止各裁剪 10 秒。）
    print("[INFO] 1:正在处理视频格式、名称标准化、裁剪长视频的片头片尾、提取裁剪后视频的音频。")
    normalize_and_trim(root_dir,intro,outro)
    
    # 2.Speech Separation. 音频进行人声乐声分离。
    print("[INFO] 2:正在对音频进行人声乐声分离。")
    separation_root(root_dir,os.path.join(node_dir,"speech_separation/configs/config_vocals_mel_band_roformer.yaml"),separate_model_path)
    
    # 3. VideoClipper. 对于长视频，使用 VideoClipper 获取句子级别的字幕文件，并根据时间戳将长视频剪辑成片段。现在它支持中英双语。以下是中文示例。英文建议采用 gpu 加速处理。
    print("[INFO]3:VideoClipper,对于长视频，使用 VideoClipper 获取句子级别的字幕文件，并根据时间戳将长视频剪辑成片段。")
    videoclipper(1,root_dir,output_dir,model_dict,device="cuda")
    videoclipper(2,root_dir,output_dir,model_dict,device="cuda")

    if check_audio:
    # 4.视频时长限制及清理检查。（若不使用--execute参数，则仅打印已预删除的文件。检查后，若需确认删除，请添加--execute参数。）
        print("[INFO]4:检查视频时长及清理文件...")
        clean_video(output_dir)
        clean_srt(output_dir)
        
    # 5 Speaker Diarization. 多模态主动说话人识别，得到 RTTM 文件；识别说话人的面部帧，提取帧级的说话人面部和唇部原始数据，从面部帧中识别说话帧，提取说话帧的面部特征。
    print("[INFO]5:生成数据列表...")
    generate_data_lists(output_dir)
    output_dir = os.path.abspath(output_dir)
    wav_list_files = list(Path(output_dir).rglob("wav.list"))
    print(f"Found {len(wav_list_files)} wav.list files in {output_dir}")
    for index, wav_list in enumerate(wav_list_files):
        wav_list = str(wav_list.absolute())

        dir_path = os.path.dirname(wav_list)
        json_dir = Path(dir_path) / "json"
        embs_dir = Path(dir_path) / "embs_wav"

        print(f"\nProcessing directory: {dir_path}")
        print(f"wav.list file: {wav_list}")
        if not os.path.exists(wav_list):
            print(f"Warning: wav.list not found at {wav_list}, skipping...")
            continue

        print(f"[INFO]5.1,批次{index+1} 重叠检测...")   
        overlap_detection(
            wav_list=wav_list,
            output_dir=dir_path,
            local_dir=os.path.join(pretrained_models_dir,"segmentation-3.0"),
            nj=1,
            overlap_threshold=0.8
        )
        
        print(f"[INFO]5.2,批次{index+1}：生成清理后的列表...")
        filter_clean_lists(
            wav_list=wav_list,
            video_list=os.path.join(dir_path, "video.list"),
            overlap_list=os.path.join(dir_path, "overlap.list"),
            clean_wav_list=os.path.join(dir_path, "clean_wav.list"),
            clean_video_list=os.path.join(dir_path, "clean_video.list")
        )
        
        print(f"[INFO]5.3,批次{index+1}：语音活动检测...")
        json_dir.mkdir(exist_ok=True)
        voice_activity_detection(
            wavs_list=os.path.join(dir_path, "clean_wav.list"),
            out_file=os.path.join(json_dir, "vad.json"),
            local_model=os.path.join(pretrained_models_dir,"speech_fsmn_vad"),
        )

        print(f"[INFO]5.4,批次{index+1}：准备子段信息...")
        prepare_subsegments(
            vad_file=os.path.join(json_dir, "vad.json"),
            out_file=os.path.join(json_dir, "subseg.json")
        )
        
        print(f"[INFO]5.5,批次{index+1}：提取说话人嵌入...")
        embs_dir.mkdir(exist_ok=True)
        extract_speech_embeddings(
            pretrained_model=os.path.join(pretrained_models_dir, "speech_campplus"),
            conf_file=audio_conf_file,
            subseg_json=os.path.join(json_dir, "subseg.json"),
            embs_out=embs_dir,
            gpus=0,
            use_gpu=False
        )


    print("[INFO] 6 视觉说话人嵌入提取...")
    extract_visual_embeddings(
        conf_file=conf_file,
        videos_dir=output_dir,
        onnx_dir=pretrained_models_dir,
        workers=os.cpu_count(),
    )

    print("[INFO] 7 执行聚类分析...")
    cluster_and_postprocess(
        conf_file=conf_file,
        root_dir=output_dir,
    )
    
    return output_dir

def simple_main_infer(video, work_dir, config, pretrained,device="cpu"):
    args = argparse.Namespace(
        video=video,
        work_dir=work_dir,
        hf_token=os.path.join(pretrained,"segmentation-3.0"),
        config=config,
        pretrained=pretrained, #TODO
        batch_size=32,
        device=device,
        jointcluster=True,
        debug_dir=""
    )
    simple_main(args)



import os
import json
import torch
import typing
import time
from moviepy.video.io.VideoFileClip import VideoFileClip, AudioFileClip
from moviepy.audio.AudioClip import CompositeAudioClip
#from modelscope import snapshot_download
from .utils import get_video_duration, generate_jsonl_data, validate_timestamps
import torch
from .funcineforge.auto.auto_frontend import AutoFrontend
from .speaker_diarization.speaker_diarization_sample.run import GlobalModels
from .funcineforge.datasets import FunCineForgeDS
# snapshot_download(
#     repo_id="FunAudioLLM/Fun-CineForge",
#     revision='v1.0.0',
#     local_dir='pretrained_models',
#     ignore_patterns=[
#         "*.md", 
#         ".git*", 
#         "funcineforge_zh_en/llm/config.yaml"
#     ],
#     repo_type="model",
# )


# ==================== 配置区域 ====================
# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# # DEVICE = "cpu"
# SERVER_PORT = 7860  # 如果 7860 被占用，改为 7861
# TEMP_DIR = "temp_workdir"
# CONFIG_FRONTEND = "decode_conf/diar.yaml"
# CONFIG_MODEL = "decode_conf/decode.yaml"
# PRETRAIN = "pretrained_models"
# MAX_SEGMENTS = 3
# DEFAULT_VIDEO_PATH="data/sample.mp4"
# DEFAULT_AUDIO_PATH="data/ref.wav"
# DEFAULT_TEXT = "我军无粮，利在急战。今乘魏兵新败，不敢出兵，出其不意，乘机退去，方可平安无事。"
# DEFAULT_CLUE = "一位中年男性以沉稳但略带担忧的语调，分析我军无粮急战的困境与敌军心败状态。他随即提出一种撤退方案，整体流露出对战局的担忧和谋求生路。"
# 全局模型实例（延迟加载）
model_pool: typing.Optional[GlobalModels] = None
engine = None

def get_segments_data(segments):
    segment_inputs = []
    for seg in segments:
        segment_inputs.extend([
            seg["text"],
            seg["clue"],
            seg["start"],
            seg["end"],
            seg["age"],
            seg["gender"],
            seg["audio"],
            seg["enable"]
        ])
    return segment_inputs

def pre_data_simple(video_path_list, conf_file, pretrained_models_dir, work_dir_list, device,config_path,segments,):
    
    video_duration_list=[]
    for video_path,work_dir in zip(video_path_list,work_dir_list):
        video_duration = get_video_duration(video_path)
        if video_duration <= 0:
            return None, "❌ 无法获取视频时长，请检查视频文件"
        video_duration_list.append(video_duration)
        simple_main_infer(video_path, work_dir, conf_file, pretrained_models_dir,device=device)
    segment_inputs=get_segments_data(segments)
    print(segment_inputs)
    segments_data = []
    for i,video_duration in enumerate(video_duration_list):
        base_idx = i * 8
        enable = segment_inputs[base_idx + 7]  # enable_check
        if not enable:
            continue
    
        text = segment_inputs[base_idx + 0]
        if not text or not text.strip():
            continue
        
        clue = segment_inputs[base_idx + 1]
        start = segment_inputs[base_idx + 2]
        end = segment_inputs[base_idx + 3]
        age = segment_inputs[base_idx + 4]
        gender = segment_inputs[base_idx + 5]
        ref_audio = segment_inputs[base_idx + 6]
        
        errors = validate_timestamps(start, end, video_duration)
        if errors:
            return None, f"❌ 片段 {i+1} 时间戳错误：\n" + "\n".join(errors)
        
        data = {
            "text": str(text).strip(),
            "clue": str(clue) if clue else "",
            "start": float(start) if start else 0.0,
            "end": float(end) if end else 0.0,
            "age": str(age) if age else "不确定",
            "gender": str(gender) if gender else "不确定",
            "ref_audio": str(ref_audio) if ref_audio else ""
        }
        
        segments_data.append(data)
    frontend=init_frontend_models(config_path,pretrained_models_dir,device)
    jsonl_path_list=[]
    jsonl_items_list=[]
    full_report_list=[]
    for video_path,work_dir in zip(video_path_list,work_dir_list):
        jsonl_path, jsonl_items=generate_jsonl_data(frontend, video_path, segments_data, work_dir, video_duration)
        jsonl_path_list.append(jsonl_path)
        jsonl_items_list.append(jsonl_items)
        full_report="got error"
        if jsonl_items:
            report_lines = []
            report_lines.append(f"✅ 任务完成！共生成 **{len(jsonl_items)}** 个片段数据。\n")
            report_lines.append("📄 **详细 JSONL 数据预览：**")
            report_lines.append("=" * 40)
            for idx, item in enumerate(jsonl_items):
                item_json = json.dumps(item, ensure_ascii=False, indent=2)
                report_lines.append(f"\n--- 🎬 片段 #{idx + 1} ---")
                report_lines.append(item_json)
                report_lines.append("-" * 40)
            full_report = "\n".join(report_lines)
        full_report_list.append(full_report)
    return jsonl_path_list,jsonl_items_list,full_report_list

def init_engine(ckpt_path,config_path,output_dir,device):
    engine = AutoFrontend(ckpt_path, config_path, output_dir, device)
    return engine

def init_frontend_models(config_path,pretrained_models_dir,device):
    global model_pool
    model_pool = GlobalModels(
        hf_token = None,
        config_path = config_path,
        pretrained_dir= pretrained_models_dir,
        device = device,
        pool_sizes = {"face": 1, "asd": 1, "fr": 1},
        batch_size = 1,
        preload = True
    )
    return model_pool


def funcineforge_infer(eng, conds, cfg):
    jsonl_path_list=conds[0]
    jsonl_items_list=conds[1]
    video_path_list=conds[2]
    work_dir_list=conds[3]
    
    try:
        audio_lists=[]
        for jsonl_path,jsonl_items,video_file,TEMP_DIR in zip(jsonl_path_list,jsonl_items_list,video_path_list,work_dir_list):
            print("🚀 FunCineForge 模型推理中...")
            # with open(jsonl_path, 'r', encoding='utf-8') as f:
            #      index_ds = [json.loads(line.strip()) for line in f if line.strip()]
            eng.kwargs["output_dir"]=TEMP_DIR
            dataset_conf = eng.kwargs.get("dataset_conf")
            dataset = FunCineForgeDS(jsonl_path, **dataset_conf)
            audio_list=eng.inference(dataset,input_len=len(dataset),**cfg)
            audio_lists.append(audio_list)
            print("✅ FunCineForge 模型推理完成！",audio_lists)
            print("🎵 正在将配音语音粘贴回静音视频...")

            output_wav_dir = os.path.join(TEMP_DIR, "wav")
            final_video_path = os.path.join(TEMP_DIR, "dubbed_video.mp4")
            
            if not os.path.exists(output_wav_dir):
                print(f"⚠️ 创建音频输出目录：{output_wav_dir}")
                return None
            
            wav_files = sorted([f for f in os.listdir(output_wav_dir) if f.endswith('.wav')])
            print(f"🚀 正在处理 {len(wav_files)} 个音频文件...wav_files={wav_files}") #wav_files=['a_134949.wav', 'a_135352.wav', 'a_140102.wav']
            if not wav_files:
                print(f"⚠️ 未生成任何音频文件：{output_wav_dir}")
                return None
            
            time_mapping = {}
            for i, item in enumerate(jsonl_items):
                matched_file = None
                for wf in wav_files:
                    if wf.startswith(item['utt']):
                        matched_file = wf
                        print(f"✅ 匹配到音频文件：{wf}")
                        break
                if matched_file:
                    start_time = float(item['start'])
                    time_mapping[matched_file] = start_time
                
                
            original_clip = VideoFileClip(video_file)
            video_duration = original_clip.duration
            video_only = original_clip.without_audio()
            audio_clips = []
            print("🚀 正在处理音频文件...",time_mapping)
            for wav_file, start_time in time_mapping.items():
                wav_path = os.path.join(output_wav_dir, wav_file)
                audio_clip = AudioFileClip(wav_path)
                audio_clip = audio_clip.with_start(start_time)
                audio_clips.append(audio_clip)
                
            final_audio = CompositeAudioClip(audio_clips)
            if final_audio.duration < video_duration:
                final_audio = final_audio.with_duration(video_duration)
            final_clip = video_only.with_audio(final_audio)
            final_clip.write_videofile(
                final_video_path,
                codec='libx264',
                audio_codec='aac',
                fps=original_clip.fps,
                logger=None
            )
            original_clip.close()
            video_only.close()
            for ac in audio_clips:
                ac.close()
            if 'final_audio' in locals():
                final_audio.close()
            final_clip.close()
            
            print("✅ 配音完成")
        return audio_lists
    except Exception as e:
        import traceback
        traceback.print_exc()
        if "index out of range" in str(e):
            return None, f"⚠️ 模型推理失败。错误：{str(e)}，建议补齐输入的线索描述和说话人属性"
        else: 
            return None, f"⚠️ 模型推理失败。错误：{str(e)}"

def process_dubbing(video_file, *segment_inputs,):
    """主推理流程"""
        
    if not video_file:
        return None, "❌ 请上传视频文件"
    
    video_duration = get_video_duration(video_file)
    if video_duration <= 0:
        return None, "❌ 无法获取视频时长，请检查视频文件"
    
    import shutil
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
        except Exception as e:
            return None, f"❌ 清空临时目录失败：{e}"
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # 解析 segment_inputs
    segments_data = []
    for i in range(MAX_SEGMENTS):
        base_idx = i * 8
        enable = segment_inputs[base_idx + 7]  # enable_check
        if not enable:
            continue
    
        text = segment_inputs[base_idx + 0]
        if not text or not text.strip():
            continue
        
        clue = segment_inputs[base_idx + 1]
        start = segment_inputs[base_idx + 2]
        end = segment_inputs[base_idx + 3]
        age = segment_inputs[base_idx + 4]
        gender = segment_inputs[base_idx + 5]
        ref_audio = segment_inputs[base_idx + 6]
        
        errors = validate_timestamps(start, end, video_duration)
        if errors:
            return None, f"❌ 片段 {i+1} 时间戳错误：\n" + "\n".join(errors)
        
        data = {
            "text": str(text).strip(),
            "clue": str(clue) if clue else "",
            "start": float(start) if start else 0.0,
            "end": float(end) if end else 0.0,
            "age": str(age) if age else "不确定",
            "gender": str(gender) if gender else "不确定",
            "ref_audio": str(ref_audio) if ref_audio else ""
        }
        
        segments_data.append(data)
    
    if not segments_data:
        return None, "❌ 有效片段数据为空，请启用并填写至少一个片段"
    
    try:
        print(0.1, desc="📋 预处理视频，生成 JSONL 数据...")
        frontend = init_frontend_models()
        jsonl_path, jsonl_items = generate_jsonl_data(frontend, video_file, segments_data, TEMP_DIR, video_duration)
        if jsonl_items:
            report_lines = []
            report_lines.append(f"✅ 任务完成！共生成 **{len(jsonl_items)}** 个片段数据。\n")
            report_lines.append("📄 **详细 JSONL 数据预览：**")
            report_lines.append("=" * 40)
            for idx, item in enumerate(jsonl_items):
                item_json = json.dumps(item, ensure_ascii=False, indent=2)
                report_lines.append(f"\n--- 🎬 片段 #{idx + 1} ---")
                report_lines.append(item_json)
                report_lines.append("-" * 40)
            full_report = "\n".join(report_lines)
        
        print(0.3, desc="🔄 FunCineForge 模型加载中...")

        eng = init_engine()
        if eng and jsonl_items:
            try:
                print(0.5, desc="🚀 FunCineForge 模型推理中...")
                eng.inference(jsonl_path)
                
                print(0.8, desc="🎵 正在将配音语音粘贴回静音视频...")

                output_wav_dir = os.path.join(TEMP_DIR, "wav")
                final_video_path = os.path.join(TEMP_DIR, "dubbed_video.mp4")
                
                if not os.path.exists(output_wav_dir):
                    return None, f"⚠️ 未找到音频输出目录：{output_wav_dir}"
                
                wav_files = sorted([f for f in os.listdir(output_wav_dir) if f.endswith('.wav')])
                if not wav_files:
                    return None, f"⚠️ 未生成任何音频文件：{output_wav_dir}"
                
                time_mapping = {}
                for i, item in enumerate(jsonl_items):
                    matched_file = None
                    for wf in wav_files:
                        if wf.startswith(item['utt']):
                            matched_file = wf
                            break
                    if matched_file:
                        start_time = float(item['start'])
                        time_mapping[matched_file] = start_time
                    
                    
                original_clip = VideoFileClip(video_file)
                video_duration = original_clip.duration
                video_only = original_clip.without_audio()
                audio_clips = []
                for wav_file, start_time in time_mapping.items():
                    wav_path = os.path.join(output_wav_dir, wav_file)
                    audio_clip = AudioFileClip(wav_path)
                    audio_clip = audio_clip.with_start(start_time)
                    audio_clips.append(audio_clip)
                    
                final_audio = CompositeAudioClip(audio_clips)
                if final_audio.duration < video_duration:
                    final_audio = final_audio.with_duration(video_duration)
                final_clip = video_only.with_audio(final_audio)
                final_clip.write_videofile(
                    final_video_path,
                    codec='libx264',
                    audio_codec='aac',
                    fps=original_clip.fps,
                    logger=None
                )
                original_clip.close()
                video_only.close()
                for ac in audio_clips:
                    ac.close()
                if 'final_audio' in locals():
                    final_audio.close()
                final_clip.close()
                
                print(1.0, desc="✅ 配音完成")
                return final_video_path, full_report
            except Exception as e:
                import traceback
                traceback.print_exc()
                if "index out of range" in str(e):
                    return None, f"⚠️ 模型推理失败。错误：{str(e)}，建议补齐输入的线索描述和说话人属性"
                else: 
                    return None, f"⚠️ 模型推理失败。错误：{str(e)}"
        else:
            time.sleep(1)
            print(1.0, desc="模拟完成")
            return video_file, full_report

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ 发生错误：{str(e)}"
    finally:
        # 生产环境开启清理
        # shutil.rmtree(work_dir)
        pass