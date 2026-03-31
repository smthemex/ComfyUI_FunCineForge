import os
import shutil
import json
from pydub import AudioSegment
from moviepy.video.io.VideoFileClip import VideoFileClip
from .speaker_diarization.speaker_diarization_sample.local.vision_processer import VisionProcesser

# ==================== 工具函数 ====================

def get_video_duration(video_path):

    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        clip.close()
        return duration,"done"
    except Exception as e:
        return 0.0,str(e)
    
def extract_audio_from_video(video_path: str, wav_path: str, sample_rate: int = 16000):
    """Extract mono 16kHz WAV from video."""
    #print(f"[INFO] Extracting audio from {video_path} to {wav_path}")
    audio = AudioSegment.from_file(video_path)
    audio = audio.set_frame_rate(sample_rate).set_channels(1)
    audio.export(wav_path, format="wav")


def extract_visual_embeddings(frontend, vad_list, video_path, wav_path, pkl_path):
    try:
        vp = VisionProcesser(
            video_file_path = video_path,
            audio_file_path = wav_path,
            audio_vad = vad_list,
            out_feat_path = pkl_path,
            visual_models = frontend,
            conf = frontend.conf,
            out_video_path=None
        )
        vp.run()
    except Exception as e:
        print(f"[ERROR] Failed to process {video_path}: {e}")
        raise
    finally:
        if 'vp' in locals():
            vp.close()
    return


def detect_video_type(video_path):
    """【占位函数】检测视频类型"""
    return "独白"

def clip_video_segment(video_path, start_time, end_time, output_dir, clip_name):
    """裁切视频片段"""
    try:
        video_clip = os.path.join(output_dir, f"{clip_name}.mp4")
        audio_clip = os.path.join(output_dir, f"{clip_name}.wav")
        clip = VideoFileClip(video_path).subclipped(start_time, end_time)
        clip.write_videofile(video_clip, codec="libx264", audio_codec='aac', logger=None)
        clip.audio.write_audiofile(audio_clip, codec="pcm_s16le", logger=None)
        clip.close()
        return video_clip, audio_clip
    except Exception as e:
        return None

def generate_jsonl_data(frontend, video_path, segments_data, work_dir, video_duration,refer_audio,text,clue,video_type,fps):
    """生成 JSONL 格式数据"""
    #video_type = detect_video_type(video_path)
    
    #jsonl_items = []
    
    #for idx, (seg,video_type,spk) in enumerate(zip(segments_data, video_type_list, spk_list)):
    utt_name = f"clip_{0}"
    end_list = []
    start_list = []
    content = []
    for seg in segments_data:
        end=seg['end']
        end_list.append(end)
        start=seg['start']
        start_list.append(start)
        content.append(seg)

    max_end=max(end_list)
    min_start=min(start_list)

    #start, end = max(0.0, float(min_start) - 0.1), min(float(max_end) + 0.1, video_duration)
    duration = max_end - min_start

    video_clip, audio_clip = clip_video_segment(
        video_path, min_start, max_end, 
        work_dir, utt_name
    )

    
    pkl_path = os.path.join(work_dir, f"{utt_name}.pkl")
    
    extract_visual_embeddings(
        frontend,
        vad_list = [[0.0, duration]],
        video_path = video_clip, 
        wav_path = audio_clip, 
        pkl_path = pkl_path
    )
    
    ref_audio_path = audio_clip
    if  os.path.exists(refer_audio):
        src = refer_audio
        dst = os.path.join(work_dir, f"{utt_name}_ref.wav")
        shutil.copy(src, dst)
        ref_audio_path = dst
    
    item = {
        "messages": [
            {"role": "text", "content": text},
            {"role": "vocal", "content": ref_audio_path},
            {"role": "video", "content": video_clip},
            {"role": "face", "content": pkl_path},
            {"role": "dialogue", "content": content},
            {"role": "clue", "content": clue}
        ],
        "utt": utt_name,
        "type": video_type,
        "speech_length": int(duration * fps),
        "start": start,
        "end": end
    }
    #sonl_items.append(item)
    
    jsonl_path = os.path.join(work_dir, "input_data.jsonl")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        #for item in jsonl_items:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return jsonl_path, item

def validate_timestamps(start, end, video_duration):
    """验证时间戳合法性"""
    errors = []
    if start < 0:
        errors.append(f"起始时间 ({start}s) 不能小于 0")
    if end > video_duration:
        errors.append(f"终止时间 ({end}s) 不能大于视频总时长 ({video_duration}s)")
    duration = end - start
    if duration <= 0:
        errors.append(f"起始时间 ({start}s) 必须小于终止时间 ({end}s)")
    if duration >= 0 and duration <= 2:
        errors.append(f"配音时长 ({duration}s) 太短，必须大于 2s")
    if duration >= 30:
        errors.append(f"配音时长 ({duration}s) 太长，请小于 30s")
    return errors