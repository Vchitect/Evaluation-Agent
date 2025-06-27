[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2412.09645)
[![Project Page](https://img.shields.io/badge/Evaluation-Website-green?logo=googlechrome&logoColor=green)](https://vchitect.github.io/Evaluation-Agent-project/)
![Badge](https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2FVchitect%2FEvaluation-Agent&label=Visitors&icon=people&color=%233d8bfd)



This repository contains the implementation of the following work:
> **Evaluation Agent: Efficient and Promptable Evaluation Framework for Visual Generative Models**<br>
> [Fan Zhang](https://github.com/zhangfan-p)<sup>∗</sup>, [Shulin Tian](https://shulin16.github.io/)<sup>∗</sup>, [Ziqi Huang](https://ziqihuangg.github.io/)<sup>∗</sup>, [Yu Qiao](http://mmlab.siat.ac.cn/yuqiao/index.html)<sup>+</sup>, [Ziwei Liu](https://liuziwei7.github.io/)<sup>+</sup><br>
> The 63rd Annual Meeting of the Association for Computational Linguistics (**ACL 2025**), Oral




<a name="overview"></a>
## :mega: Overview

Recent advancements in visual generative models have enabled high-quality image and video generation, opening diverse applications. However, evaluating these models often demands sampling hundreds or thousands of images or videos, making the process computationally expensive, especially for diffusion-based models with inherently slow sampling. Moreover, existing evaluation methods rely on rigid pipelines that overlook specific user needs and provide numerical results without clear explanations. In contrast, humans can quickly form impressions of a model's capabilities by observing only a few samples. To mimic this, we propose the Evaluation Agent framework, which employs human-like strategies for efficient, dynamic, multi-round evaluations using only a few samples per round, while offering detailed, user-tailored analyses. It offers four key advantages: 1) efficiency, 2) promptable evaluation tailored to diverse user needs, 3) explainability beyond single numerical scores, and 4) scalability across various models and tools. Experiments show that Evaluation Agent reduces evaluation time to 10% of traditional methods while delivering comparable results. The Evaluation Agent framework is fully open-sourced to advance research in visual generative models and their efficient evaluation.

![Framework](./assets/fig_framework.jpg)


**Overview of Evaluation Agent Framework.** This framework leverages LLM-powered agents for efficient and flexible visual model assessments. As shown, it consists of two stages: (a) the Proposal Stage, where user queries are decomposed into sub-aspects, and prompts are generated, and (b) the Execution Stage, where visual content is generated and evaluated using an Evaluation Toolkit. The two stages interact iteratively to dynamically assess models based on user queries.

<a name="installation"></a>
## :hammer: Installation

1. Clone the repository.

```bash
git clone https://github.com/Vchitect/Evaluation-Agent.git
cd Evaluation-Agent
```

2. Install the environment.
```bash
conda create -n eval_agent python=3.10
conda activate eval_agent
pip install -r requirements.txt
```



<a name="usage"></a>
## Usage

First, you need to configure the `open_api_key`. You can do it as follows:
```
export OPENAI_API_KEY="your_api_key_here"
```

### Evaluation of Open-ended Questions on T2I Models


```
python open_ended_eval.py --user_query $USER_QUERY --model $MODEL
```
- `$USER_QUERY` can be any question regarding the model’s capabilities, such as ‘How well does the model generate trees in anime style?’
- `$MODEL` refers to the image generation model you want to evaluate. Currently, we support four models: [SD-14](https://huggingface.co/CompVis/stable-diffusion-v1-4), [SD-21](https://huggingface.co/stabilityai/stable-diffusion-2-1), [SDXL-1](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), and [SD-3](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers). You can integrate new models in the following path: `./eval_agent/eval_models/`


### Evaluation Based on the VBench Tools on T2V Models

#### Preparation

1. Configure the VBench Environment

- You need to configure the VBench environment on top of the existing environment. For details, refer to [VBench](https://github.com/Vchitect/VBench).

2. Prepare the Model to be Evaluated

- Download the weights of the target model for evaluation and place them in `./eval_agent/eval_models/{model_name}/checkpoints/`. 

- Currently, we support four models: [latte](https://github.com/Vchitect/Latte/tree/main), [modelscope](https://modelscope.cn/models/iic/text-to-video-synthesis/summary), [videocrafter-0.9](https://github.com/AILab-CVC/VideoCrafter/tree/30048d49873cbcd21077a001e6a3232e0909d254), and [videocrafter-2](https://github.com/AILab-CVC/VideoCrafter). These models may also have specific environment requirements. For details, please refer to the respective model links.

#### Command

```
python eval_agent_for_vbench.py --user_query $USER_QUERY --model $MODEL
```
- `$USER_QUERY` need to be related to the 15 dimensions of VBench. These dimensions are: `subject_consistency`, `background_consistency`, `motion_smoothness`, `dynamic_degree`, `aesthetic_quality`, `imaging_quality`, `object_class`, `multiple_objects`, `human_action`, `color`, `spatial_relationship`, `scene`, `temporal_style`, `appearance_style`, and `overall_consistency`.
- `$MODEL` refers to the video generation model you want to evaluate.



### Evaluation Based on the T2I-CompBench Tools on T2I Models

#### Preparation

1. Configure the T2I-CompBench Environment

- You need to configure the T2I-CompBench environment on top of the existing environment. For details, refer to [T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench/tree/6ea770ada4eea55fa7b09caa2d2fb63fe4d6bf8f).

2. Prepare the Model to be Evaluated

#### Command

```
python eval_agent_for_t2i_compbench.py --user_query $USER_QUERY --model $MODEL
```
- `$USER_QUERY` need to be related to the 4 dimensions of T2I-CompBench. These dimensions are: `color_binding`, `shape_binding`, `texture_binding`, `non-spatial relationship`.
- `$MODEL` refers to the image generation model you want to evaluate.





## Open-Ended User Query Dataset
We propose the **Open-Ended User Query Dataset**, developed through a user study. As part of this process, we gathered questions from various sources, focusing on aspects users consider most important when evaluating new models. After cleaning, filtering, and expanding the initial set, we compiled a refined dataset of 100 open-ended user queries.

Check out the details of the [open-ended user query dataset](https://github.com/Vchitect/Evaluation-Agent/tree/main/dataset) 

![statistic](./assets/open_dataset_stats.png)
The three graphs give an overview of the distributions and types of our curated open queries set. Left: the distribution of question types, which are categorized as `General` or `Specific`. Middle: the distribution of the ability types, which are categorized as `Prompt Following`, `Visual Quality`, `Creativity`, `Knowledge` and `Others`. Right: the distribution of the content categories, which are categorized as `History and Culture`, `Film and Entertainment`, `Science and Education`, `Fashion`, `Medical`, `Game Design`, `Architecture and Interior Design`, `Law`.


## Citation

If you find our repo useful for your research, please consider citing our paper:

```bibtex
@article{zhang2024evaluationagent,
    title = {Evaluation Agent: Efficient and Promptable Evaluation Framework for Visual Generative Models},
    author = {Zhang, Fan and Tian, Shulin and Huang, Ziqi and Qiao, Yu and Liu, Ziwei},
    journal={arXiv preprint arXiv:2412.09645},
    year = {2024}
}
```

## Related Links

Our related projects: [VBench](https://github.com/Vchitect/VBench)

```bibtex
@InProceedings{huang2023vbench,
    title={{VBench}: Comprehensive Benchmark Suite for Video Generative Models},
    author={Huang, Ziqi and He, Yinan and Yu, Jiashuo and Zhang, Fan and Si, Chenyang and Jiang, Yuming and Zhang, Yuanhan and Wu, Tianxing and Jin, Qingyang and Chanpaisit, Nattapol and Wang, Yaohui and Chen, Xinyuan and Wang, Limin and Lin, Dahua and Qiao, Yu and Liu, Ziwei},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2024}
}

@article{huang2024vbench++,
    title={{VBench++}: Comprehensive and Versatile Benchmark Suite for Video Generative Models},
    author={Huang, Ziqi and Zhang, Fan and Xu, Xiaojie and He, Yinan and Yu, Jiashuo and Dong, Ziyue and Ma, Qianli and Chanpaisit, Nattapol and Si, Chenyang and Jiang, Yuming and Wang, Yaohui and Chen, Xinyuan and Chen, Ying-Cong and Wang, Limin and Lin, Dahua and Qiao, Yu and Liu, Ziwei},
    journal={arXiv preprint arXiv:2411.13503},
    year={2024}
}

@article{zheng2025vbench2,
    title={{VBench-2.0}: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness},
    author={Zheng, Dian and Huang, Ziqi and Liu, Hongbo and Zou, Kai and He, Yinan and Zhang, Fan and Zhang, Yuanhan and He, Jingwen and Zheng, Wei-Shi and Qiao, Yu and Liu, Ziwei},
    journal={arXiv preprint arXiv:2503.21755},
    year={2025}
}
```
