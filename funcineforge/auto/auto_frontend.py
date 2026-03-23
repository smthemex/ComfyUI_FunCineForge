import os
import torch
import logging
from omegaconf import OmegaConf
from ..utils.hinter import get_logger
from ..models.utils import dtype_map
from ..datasets import FunCineForgeDS
from pathlib import Path
class AutoFrontend:
    def __init__(
        self,
        ckpt_path: str,
        config_path: str,
        output_dir: str,
        device: str = "cuda:0"
    ):
        self.logger = get_logger(log_level=logging.INFO, local_rank=1, world_size=1)
        self.device = device
        self.output_dir = output_dir
        self.lm_model = None
        self.fm_model = None
        self.voc_model = None
        self.model = None
        self.index_ds_class = None
        
        self.dataset_conf = None
        self.kwargs = OmegaConf.load(config_path)
        
        if device.startswith("cuda"):
            try:
                device_id = int(device.split(":")[-1])
                torch.cuda.set_device(device_id)
            except (ValueError, IndexError):
                self.logger.warning(f"Invalid cuda device string {device}, defaulting to 0")
                torch.cuda.set_device(0)
        else:
            self.logger.info(f"Running on CPU")
        def extract_path_components(ckpt_path):
            if ckpt_path is None:
                return None, None, None
            path = Path(ckpt_path)
            ckpt_id = path.name
            model_name = path.parent.name
            exp_dir = path.parent.parent
            return str(exp_dir), model_name, ckpt_id
        
        lm_ckpt_path = os.path.join(ckpt_path, "llm/mp_rank_00_model_states.pt")
        fm_ckpt_path = os.path.join(ckpt_path, "flow/mp_rank_00_model_states.pt")
        voc_ckpt_path = os.path.join(ckpt_path, "vocoder/avg_5_removewn.pt")
        
        lm_exp_dir, lm_model_name, lm_ckpt_id = extract_path_components(lm_ckpt_path)
        self.logger.info(f"init LM model form {lm_ckpt_path}")
        
        from .auto_model import AutoModel
        self.lm_model = (AutoModel(
            model=os.path.join(lm_exp_dir, lm_model_name),
            init_param=lm_ckpt_path,
            output_dir=None,
            device=device,
        ))
        self.lm_model.model.to(dtype_map[self.kwargs.get("llm_dtype", "fp32")])
        
        fm_exp_dir, fm_model_name, fm_ckpt_id, _ = fm_ckpt_path.rsplit("/", 3)
        self.logger.info(f"build FM model form {fm_ckpt_path}")
        self.fm_model = AutoModel(
            model=os.path.join(fm_exp_dir, fm_model_name),
            init_param=fm_ckpt_path,
            output_dir=None,
            device=device,
        )
        self.fm_model.model.to(dtype_map[self.kwargs.get("fm_dtype", "fp32")])
        
        voc_exp_dir, voc_model_name, voc_ckpt_id, _ = voc_ckpt_path.rsplit("/", 3)
        self.logger.info(f"build VOC model form {voc_ckpt_path}")
        self.voc_model = AutoModel(
            model=os.path.join(voc_exp_dir, voc_model_name),
            init_param=voc_ckpt_path,
            output_dir=None,
            device=device,
        )
        self.voc_model.model.to(dtype_map[self.kwargs.get("voc_dtype", "fp32")])
        
        self.logger.info(f"build inference model {self.kwargs.get('model')}")
        self.kwargs["output_dir"] = output_dir
        self.kwargs["tokenizer"] = None
        self.model = AutoModel(
            **self.kwargs,
            lm_model=self.lm_model,
            fm_model=self.fm_model,
            voc_model=self.voc_model,
        )
        self.dataset_conf = self.kwargs.get("dataset_conf")
        
    def inference(self, jsonl_path: str):
        if not self.model:
            raise RuntimeError("Model class not initialized.")
            
        dataset = FunCineForgeDS(jsonl_path, **self.dataset_conf)
        self.logger.info(f"Starting inference on {len(dataset)} items...")
        
        self.model.inference(input=dataset, input_len=len(dataset))
        self.logger.info("Inference finished.")