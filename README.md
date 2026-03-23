# ComfyUI\_FunCineForge
[Fun-CineForge](https://github.com/FunAudioLLM/FunCineForge): A Unified Dataset Pipeline and Model for Zero-Shot Movie Dubbing in Diverse Cinematic Scenes

Update
-----

*  多个视频可以串联seg节点使用，先试单个视频吧
  
1.Installation
-----

In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_FunCineForge
```
2.requirements  
----

* 注意安装pyannote.audio 库引发numpy 版本提升到2.0以上，记得装回低版本
```
pip install -r requirements.txt
```
3.checkpoints 
-----

* ALL FunCineForge  [ FunCineForge ](https://huggingface.co/FunAudioLLM/Fun-CineForge/tree/main) 
* speech_fsmn_vad_zh-cn-16k-common-pytorch[links](https://www.modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch)
* speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch[links](https://www.modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch)
* speech_campplus_sv_zh-cn_16k-common[links](https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common)
* punc_ct-transformer_zh-cn-common-vocab272727-pytorch[links](https://www.modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch)
* 去我[云盘](https://pan.quark.cn/s/4d76d1185088)拉取全部模型的zip吧,解压到funcineforge目录下（结构如下图）
![](https://github.com/smthemex/ComfyUI_FunCineForge/blob/main/example_workflows/dir.png)
```
├── ComfyUI/models/   # 13G
|     ├── funcineforge/llm   #注意 没有模型库那样的子目录，直接是config和模型文件，其他几个一样
|          ├── config.yaml
|          ├── mp_rank_00_model_states.pt
... 
```

4 Example
-----
![](https://github.com/smthemex/ComfyUI\_FunCineForge/blob/main/example\_workflows/example.png)

5 Citation
-----
```
@misc{liu2026funcineforgeunifieddatasettoolkit,
    title={FunCineForge: A Unified Dataset Toolkit and Model for Zero-Shot Movie Dubbing in Diverse Cinematic Scenes}, 
    author={Jiaxuan Liu and Yang Xiang and Han Zhao and Xiangang Li and Zhenhua Ling},
    year={2026},
    eprint={2601.14777},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
}
```



