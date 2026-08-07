import sys
import os
import json
import hashlib
import re

from tqdm import tqdm

sys.path.insert(0, "eval_tools")



class GenModel:
    def __init__(self, model_name, save_mode="video") -> None:
        self.save_mode = save_mode
        if model_name == "vc2":
            from eval_models.VC2.vc2_predict import VideoCrafter
            self.predictor = VideoCrafter("vc2")
        elif model_name == "vc09":
            from eval_models.VC09.vc09_predict import VideoCrafter09
            self.predictor = VideoCrafter09()
        elif model_name == "modelscope":
            from eval_models.modelscope.modelscope_predict import ModelScope
            self.predictor = ModelScope()
        elif model_name == "latte1":
            from eval_models.latte.latte_1_predict import Latte1
            self.predictor = Latte1()
        elif model_name == "cogvideox-2b":
            from eval_models.cogvideox.cogvideox_predict import CogVideoX2B
            self.predictor = CogVideoX2B()
        elif model_name == "cogvideox-5b":
            from eval_models.cogvideox.cogvideox_predict import CogVideoX5B
            self.predictor = CogVideoX5B()
        elif model_name == "show1":
            from eval_models.show1.show1_predict import Show1
            self.predictor = Show1()
            
        elif model_name == "SDXL-1":
            from eval_models.SD.sd_predict import SDXL
            self.predictor = SDXL()
        elif model_name == "SD-21":
            from eval_models.SD.sd_predict import SD21
            self.predictor = SD21()
        elif model_name == "SD-14":
            from eval_models.SD.sd_predict import SD14
            self.predictor = SD14()
        elif model_name == "SD-3":
            from eval_models.SD.sd_predict import SD3
            self.predictor = SD3() 
        else:
            raise ValueError(f"This {model_name} has not been implemented yet")
    
    
    def predict(self, prompt, save_path):
        os.makedirs(save_path, exist_ok=True)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:10]
        clean_prompt = re.sub(r"[^A-Za-z0-9._-]+", "_", prompt.strip())
        clean_prompt = clean_prompt.strip("._-")[:160] or "sample"
        name = f"{clean_prompt}_{digest}"
        if self.save_mode == "video":
            save_name = os.path.join(save_path, f"{name}.mp4")
        elif self.save_mode == "img":
            save_name = os.path.join(save_path, f"{name}.png")
        else:
            raise NotImplementedError(f"Wrong mode -- {self.save_mode}")
        
        self.predictor.predict(prompt, save_name)
        return prompt, save_name





class ToolBox:
    def __init__(self) -> None:
        pass
    

    def call(self, tool_name, video_pairs):
        method = getattr(self, tool_name, None)
        
        
        if callable(method):
            return method(video_pairs)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{tool_name}'")
    
    def color_binding(self, image_pairs):
        sys.path.insert(0, "eval_tools/t2i_comp/BLIPvqa_eval")
        from eval_tools.t2i_comp.BLIPvqa_eval.BLIP_vqa_eval_agent import calculate_attribute_binding
        results = calculate_attribute_binding(image_pairs)
        return results
    
    def shape_binding(self, image_pairs):
        sys.path.insert(0, "eval_tools/t2i_comp/BLIPvqa_eval")
        from eval_tools.t2i_comp.BLIPvqa_eval.BLIP_vqa_eval_agent import calculate_attribute_binding
        results = calculate_attribute_binding(image_pairs)
        return results

    def texture_binding(self, image_pairs):
        sys.path.insert(0, "eval_tools/t2i_comp/BLIPvqa_eval")
        from eval_tools.t2i_comp.BLIPvqa_eval.BLIP_vqa_eval_agent import calculate_attribute_binding
        results = calculate_attribute_binding(image_pairs)
        return results
    

    def non_spatial(self, image_pairs):
        sys.path.insert(0, "eval_tools/t2i_comp/CLIPScore_eval")
        from eval_tools.t2i_comp.CLIPScore_eval.CLIP_similarity_eval_agent import calculate_clip_score
        results = calculate_clip_score(image_pairs)
        return results
    
    
    def overall_consistency(self, video_pairs):
        from eval_tools.vbench.overall_consistency import compute_overall_consistency
        results = compute_overall_consistency(video_pairs)
        return results
    
    
    def aesthetic_quality(self, video_pairs):
        from eval_tools.vbench.aesthetic_quality import compute_aesthetic_quality
        results = compute_aesthetic_quality(video_pairs)
        return results

    def appearance_style(self, video_pairs):
        from eval_tools.vbench.appearance_style import compute_appearance_style
        results = compute_appearance_style(video_pairs)
        return results
    
    
    def background_consistency(self, video_pairs):
        from eval_tools.vbench.background_consistency import compute_background_consistency
        results = compute_background_consistency(video_pairs)
        return results

    def color(self, video_pairs):
        from eval_tools.vbench.color import compute_color
        results = compute_color(video_pairs)
        return results
    
    def dynamic_degree(self, video_pairs):
        from eval_tools.vbench.dynamic_degree import compute_dynamic_degree
        results = compute_dynamic_degree(video_pairs)
        return results

    def human_action(self, video_pairs):
        from eval_tools.vbench.human_action import compute_human_action
        results = compute_human_action(video_pairs)
        return results

    def imaging_quality(self, video_pairs):
        from eval_tools.vbench.imaging_quality import compute_imaging_quality
        results = compute_imaging_quality(video_pairs)
        return results

    def motion_smoothness(self, video_pairs):
        from eval_tools.vbench.motion_smoothness import compute_motion_smoothness
        results = compute_motion_smoothness(video_pairs)
        return results

    def multiple_objects(self, video_pairs):
        from eval_tools.vbench.multiple_objects import compute_multiple_objects
        results = compute_multiple_objects(video_pairs)
        return results

    def object_class(self, video_pairs):
        from eval_tools.vbench.object_class import compute_object_class
        results = compute_object_class(video_pairs)
        return results
    
    def scene(self, video_pairs):
        from eval_tools.vbench.scene import compute_scene
        results = compute_scene(video_pairs)
        return results
    
    def spatial_relationship(self, video_pairs):
        from eval_tools.vbench.spatial_relationship import compute_spatial_relationship
        results = compute_spatial_relationship(video_pairs)
        return results

    def subject_consistency(self, video_pairs):
        from eval_tools.vbench.subject_consistency import compute_subject_consistency
        results = compute_subject_consistency(video_pairs)
        return results

    def temporal_style(self, video_pairs):
        from eval_tools.vbench.temporal_style import compute_temporal_style
        results = compute_temporal_style(video_pairs)
        return results



        
class ToolCalling:
    def __init__(self, sample_model, save_mode):
        self.gen = GenModel(sample_model, save_mode)
        self.eval_tools = ToolBox()
        self.vlm_gpt = None


    def sample(self, prompts, save_path):
        info_list = []
        for prompt in tqdm(prompts):
            prompt, content = self.gen.predict(prompt, save_path)
            info_list.append({
                "prompt":prompt,
                "content_path":content
            })
        return info_list


    def eval(self, tool_name, video_pairs):
        results = self.eval_tools.call(tool_name, video_pairs)
        return results
    
    
    def vlm_eval(self, content_path, question):
        if self.vlm_gpt is None:
            from eval_tools.vlm.gpt import GPT

            self.vlm_gpt = GPT()
        response = self.vlm_gpt.predict(content_path, question)
        return response




def save_json(content, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(content, json_file, indent=4)
        
