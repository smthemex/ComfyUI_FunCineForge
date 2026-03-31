import torch
import torch.nn as nn
import logging
import numpy as np
import os
import torchaudio
import time
import shutil
from ..register import tables
from ..auto.auto_model import AutoModel
from ..utils.set_all_random_seed import set_all_random_seed
from moviepy.video.io.VideoFileClip import VideoFileClip, AudioFileClip
from datetime import datetime

@tables.register("model_classes", "FunCineForgeInferModel")
class FunCineForgeInferModel(nn.Module):
    def __init__(
        self,
        lm_model: AutoModel,
        fm_model: AutoModel,
        voc_model: AutoModel,
        **kwargs
    ):
        super().__init__()
        self.tokenizer = lm_model.kwargs["tokenizer"]
        self.frontend = fm_model.kwargs["frontend"]
        self.lm_model = lm_model.model
        self.fm_model = fm_model.model
        self.voc_model = voc_model.model
        mel_extractor = self.fm_model.mel_extractor
        #print(f"[mel_extractor]: {mel_extractor}")
        if mel_extractor:
            self.mel_frame_rate = mel_extractor.sampling_rate // mel_extractor.hop_length
            self.sample_rate = mel_extractor.sampling_rate
            #print(f"[self.mel_frame_rate1]: {self.mel_frame_rate}, [self.sample_rate]: {self.sample_rate}") #[self.mel_frame_rate1]: 50, [self.sample_rate]: 24000
        else:
            self.mel_frame_rate = self.fm_model.sample_rate // 480
            self.sample_rate = self.fm_model.sample_rate
            #print(f"[self.mel_frame_rate2]: {self.mel_frame_rate}, [self.sample_rate]: {self.sample_rate}")


    @torch.no_grad()
    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        **kwargs,
    ):
        #uttid = key[0]
        #logging.info(f"generating {uttid}") #clip_0
        # text -> codec in [1, T]
        kwargs["tokenizer"] = self.tokenizer
        set_all_random_seed(kwargs.get("random_seed", 0))
        lm_time = time.time()
        codec, hit_eos, states = self.lm_model.inference(data_in, data_lengths, key, **kwargs)
        # logging.info(f"[llm time]: {((time.time()-lm_time)*1000):.2f} ms, [hit_eos]: {hit_eos}, [gen len]: {codec.shape[1]}, [speech tokens]: {codec[0].cpu().tolist()}")
        
        wav, batch_data_time = None, 1.0
        #print(f"[self.sample_rate]: {self.sample_rate}")
        if codec.shape[1] > 0:
            fm_time = time.time()
            data_in[0]["codec"] = codec
            set_all_random_seed(kwargs.get("random_seed", 0))
            feat = self.fm_model.inference(data_in, data_lengths, key, **kwargs)
            # feat -> wav
            set_all_random_seed(kwargs.get("random_seed", 0))
            wav = self.voc_model.inference([feat[0]], data_lengths, key, **kwargs)
            #print(f"[wav shape]: {wav.shape}") #[wav shape]: torch.Size([1, 51840])
            # output save
            output_dir = kwargs.get("output_dir", None)
            if output_dir is not None:
                #audio_file_prefix = datetime.now().strftime("%y%m%d%H%M%S")[-6:] 
                feat_out_dir = os.path.join(output_dir, "feat")
                os.makedirs(feat_out_dir, exist_ok=True)
                np.save(os.path.join(feat_out_dir, f"{key[0]}.npy"), feat[0].cpu().numpy())

                wav_out_dir = os.path.join(output_dir, "wav")
                os.makedirs(wav_out_dir, exist_ok=True)
                output_wav_path = os.path.join(wav_out_dir, f"{key[0]}.wav")
                torchaudio.save(
                    output_wav_path, wav.cpu(),
                    sample_rate=self.sample_rate, encoding='PCM_S', bits_per_sample=16
                )
                
                silent_video_path = data_in[0]["video"]
                if isinstance(silent_video_path, str):
                    if os.path.exists(silent_video_path):
                        video_out_dir = os.path.join(output_dir, "mp4")
                        video_gt_dir = os.path.join(output_dir, "gt")
                        os.makedirs(video_out_dir, exist_ok=True)
                        os.makedirs(video_gt_dir, exist_ok=True)
                        output_video_path = os.path.join(video_out_dir, f"{key[0]}.mp4")
                        copy_video_path = os.path.join(video_gt_dir, f"{key[0]}.mp4")
                        shutil.copy2(silent_video_path, copy_video_path)
                        self.merge_video_audio(
                            silent_video_path=silent_video_path,
                            wav_path=output_wav_path,
                            output_path=output_video_path,
                        )
                    else:
                        pass
                else: #pli list
                    pass
            #logging.info(f"fm_voc time: {((time.time()-fm_time)*1000):.2f} ms") # (inference_model:104) INFO: fm_voc time: 13958.33 ms

            batch_data_time = wav.shape[1] / self.voc_model.sample_rate
            #print(f"[batch_data_time]: {batch_data_time:.2f}, [ self.voc_model.sample_rate]: {self.voc_model.sample_rate} s") #[batch_data_time]: 8.16, [ self.voc_model.sample_rate]: 24000 s

        return [[wav]], {"batch_data_time": batch_data_time}
    
    def merge_video_audio(self, silent_video_path, wav_path, output_path):
        
        video_clip = VideoFileClip(silent_video_path)
        video_duration = video_clip.duration
        audio_clip = AudioFileClip(wav_path)
        audio_duration = audio_clip.duration
        
        if audio_duration >= video_duration:
            audio_clip = audio_clip.subclipped(0, video_duration)
        
        video_clip = video_clip.with_audio(audio_clip)
        video_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=video_clip.fps,
            logger=None
        )
        video_clip.close()
        audio_clip.close()