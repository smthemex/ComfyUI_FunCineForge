# ComfyUI\_FunCineForge

\[Fun-CineForge](https://github.com/FunAudioLLM/FunCineForge): A Unified Dataset Pipeline and Model for Zero-Shot Movie Dubbing in Diverse Cinematic Scenes



\# Update

\*  多个视频可以串联seg节点使用，先试单个视频吧



1.Installation  

\-----

&#x20; In the ./ComfyUI/custom\_nodes directory, run the following:   

```

git clone https://github.com/smthemex/ComfyUI\_FunCineForge

```

2.requirements  

\----

* 注意安装pyannote.audio 库引发numpy 版本提升到2.0以上，记得装回低版本

```

pip install -r requirements.txt

```

3.checkpoints 

\----

\* ALL FunCineForge  \[ FunCineForge ](https://huggingface.co/FunAudioLLM/Fun-CineForge/tree/main) 

\* speech\_fsmn\_vad\_zh-cn-16k-common-pytorch\[links](https://www.modelscope.cn/models/iic/speech\_fsmn\_vad\_zh-cn-16k-common-pytorch)

\* speech\_seaco\_paraformer\_large\_asr\_nat-zh-cn-16k-common-vocab8404-pytorch\[links](https://www.modelscope.cn/models/iic/speech\_seaco\_paraformer\_large\_asr\_nat-zh-cn-16k-common-vocab8404-pytorch)

\* speech\_campplus\_sv\_zh-cn\_16k-common\[links](https://www.modelscope.cn/models/iic/speech\_campplus\_sv\_zh-cn\_16k-common)

\* punc\_ct-transformer\_zh-cn-common-vocab272727-pytorch\[links](https://www.modelscope.cn/models/iic/punc\_ct-transformer\_zh-cn-common-vocab272727-pytorch)



\# 去我云盘拉取全部模型的zip吧，人麻了

!\[](https://github.com/smthemex/ComfyUI\_FunCineForge/blob/main/example\_workflows/dir.png)

```

├── ComfyUI/models/   # 13G

|     ├── funcineforge/llm   #注意 没有模型库那样的子目录，直接是config和模型文件，其他几个一样

|          ├── config.yaml

|          ├── mp\_rank\_00\_model\_states.pt

... 



```





\# 4 Example

!\[](https://github.com/smthemex/ComfyUI\_FunCineForge/blob/main/example\_workflows/example.png)





\# 5 Citation

\------

@misc{liu2026funcineforgeunifieddatasettoolkit,

&#x20;   title={FunCineForge: A Unified Dataset Toolkit and Model for Zero-Shot Movie Dubbing in Diverse Cinematic Scenes}, 

&#x20;   author={Jiaxuan Liu and Yang Xiang and Han Zhao and Xiangang Li and Zhenhua Ling},

&#x20;   year={2026},

&#x20;   eprint={2601.14777},

&#x20;   archivePrefix={arXiv},

&#x20;   primaryClass={cs.CV},

}```



