<div align="center">

# **LiveStar: Live Streaming Assistant for Real-World Online Video Understanding**

[\[🤖 HF Model (Anonymous)\]](https://huggingface.co/Anonymous4LiveStar/LiveStar_8B) [\[🤗 HF Dataset (Anonymous)\]](https://huggingface.co/datasets/Anonymous4LiveStar/OmniStar-RNG) [\[🎬 Base Model\]](https://huggingface.co/Anonymous4LiveStar/LiveStar_InternVideo_8B)

</div>

This is the **anonymous repository** for the paper ***LiveStar: Live-Streaming Assistant for Real-World Online Video Understanding***, providing code, data, and pipeline to support the LiveStar model and the OmniStar dataset introduced in the work. 🚀🚀🚀


## News & Updates 🚀  
- `2025-05-27`:  
  🔥 **Anonymous Release of LiveStar**: We've launched the [LiveStar-8B model](https://huggingface.co/Anonymous4LiveStar/LiveStar_8B) on Hugging Face for immediate online inference!  
  - **Current Features**:  
    ✔️ Full model weights accessible  
    ✔️ Basic inference pipeline integration  
  - **Coming Soon**:  
    📅 **OmniStar Dataset**: Full release pending completion of the peer-review process  
    ⚙️ **Extended Tools**: Enhanced training scripts and evaluation protocols  

## **Overview**

**Illustration of online video understanding.** (a) Taking the RNG task as an example, online video understanding requires Video-LLMs to handle continuous streams and output at appropriate times; (b) Existing methods overly rely on learning the EOS token, leading to poor inference performance; (c)-(e) LiveStar establishes an effective response-silence training and inference framework by SCAM and SVeD without compromising basic video understanding capabilities.

![overview](/assets/images/overview.png)

### **Abstract**

Despite significant progress in Video Large Language Models (Video-LLMs) for offline video understanding, existing online Video-LLMs typically struggle to simultaneously process continuous frame-by-frame inputs and determine optimal response timing, often compromising real-time responsiveness and narrative coherence. To address these limitations, we introduce LiveStar, a pioneering live streaming assistant that achieves always-on proactive responses through adaptive streaming decoding. Specifically, LiveStar incorporates: (1) a training strategy enabling incremental video-language alignment for variable-length video streams, preserving temporal consistency across dynamically evolving frame sequences; (2) a response-silence decoding framework that determines optimal proactive response timing via a single forward pass verification; (3) memory-aware acceleration via peak-end memory compression for online inference on 10+ minute videos, combined with streaming key-value cache to achieve 1.53× faster inference. We also construct an OmniStar dataset, a comprehensive dataset for training and benchmarking that encompasses 15 diverse real-world scenarios and 5 evaluation tasks for online video understanding. Extensive experiments across three benchmarks demonstrate LiveStar's state-of-the-art performance, achieving an average 19.5% improvement in semantic correctness with 18.1% reduced timing difference compared to existing online Video-LLMs, while improving FPS by 12.0% across all five OmniStar tasks. Our model and dataset can be accessed at https://anonymous.4open.science/r/LiveStar-5272.

## **Getting Started**

This guide provides step-by-step instructions to set up the LiveStar framework, including environment configuration, model acquisition, and dataset preparation. Current implementations focus on inference capabilities with partial resource availability.

### **Installation**

1. Clone the repository (Click `Download file`)
2. Install Python dependencies (Ensure you have Python version >= 3.9 installed). For GPU support, CUDA 12.2 or compatible drivers are required.
```bash
conda create -n LiveStar -y python=3.9.21
conda activate LiveStar
conda install -y -c pytorch pytorch=2.5.1 torchvision=0.10.1
pip install transformers=4.37.2 opencv-python=4.11.0.84 imageio=2.37.0 decord=0.6.0 gradio=4.44.1
pip install flash-attn --no-build-isolation
```
Alternative: Install via requirements.txt (recommended):
```bash
pip install -r requirements.txt
```

### **Model Acquisition**

1. Download Fine-Tuned LiveStar Model (Recommended):

(1) Download the LiveStar-8B model from Hugging Face:

```Bash
git clone https://huggingface.co/Anonymous4LiveStar/LiveStar_8B
```

(2) Move model weights to the inference directory:

```Bash
mv LiveStar_8B/*.safetensors inference/
```

2. SFT Training from Scratch (Advanced):

(1) Download the base pre-trained model:

```bash
git clone https://huggingface.co/Anonymous4LiveStar/LiveStar_InternVideo_8B
```

(2) Prepare weights for fine-tuning:
```bash
mv LiveStar_InternVideo_8B/*.safetensors inference/
```

### **Data Preparation**

(1) Download the OmniStar dataset from Hugging Face (We will open source it after the review process is completed):



```bash
git clone https://huggingface.co/datasets/XXX/OmniStar
```

*For review purposes only*: You may examine sample annotations from the OmniStar-RNG subset:  

```bash
git clone https://huggingface.co/datasets/Anonymous4LiveStar/OmniStar-RNG
```

(2) Navigate to the dataset directory:



```bash
cd OmniStar
```


(3) Concatenate the split files:

Use the cat command to concatenate all the split files into a single file. The split files are named from allVideos.part_aa to allVideos.part_ch, you can use the following command:

```Bash
cat allVideos_tar_sep/allVideos.part_* > allVideo.tar.gz
```

(4) Verify the integrity of the file (optional):

Use the md5sum command to compute the checksum of the concatenated file and compare it with the provided checksum 43d6777701f8bfbfcc7854304245cc2c:

```Bash
md5sum allVideo.tar.gz
```

The output should look like this:

```Bash
43d6777701f8bfbfcc7854304245cc2c  allVideo.tar.gz
```

If the checksum matches 43d6777701f8bfbfcc7854304245cc2c, the file is intact and correct.

(5) Extract the concatenated file:

Use the tar command to extract the contents of allVideo.tar.gz:

```Bash
tar -xzvf allVideo.tar.gz
```

After completing these steps, you should see the extracted video files in the current directory.

## **Inference**

## **Training**

## **OmniStar**

This section provides instructions for reproducing the annotation and evaluation of OmniStar.

![framework](/assets/images/framework.png)

### **1. Data Filtering**

Run the following commands to obtain filtered videos. 

Firstly, you should install [Open-Sora](https://github.com/hpcaitech/Open-Sora/tree/main/tools), and have a raw video dataset prepared. A meta file of the dataset information is needed for data processing. To create a meta file from a folder, run:

```Bash
python -m Data_Filtering/Open-Sora-main/tools.datasets.convert video /path_to_your_video_folder --output /path_to_save_your_meta.csv
```

Then, run the following commands to get aesthetic scores and optical flow scores of your videos. Make sure the meta file has column 'path'.

```Bash
torchrun --nproc_per_node 8 -m Data_Filtering/Open-Sora-main/tools.scoring.aesthetic.inference /path_to_save_your_meta_with_aesthetic_scores.csv --bs 1024 --num_workers 16
torchrun --standalone --nproc_per_node 8 Data_Filtering/Open-Sora-main/tools/scoring/optical_flow/inference.py /path_to_save_your_meta_with_optical_flow_scores.csv
```

With these information of videos above, you can filtering is conducted to retain only those videos containing 5 to 15 scenes,Then you can retain videos with an aesthetic score of 4 or above and with optical flow scores within the range of 0.5 to 100

### **2. Scene Detection and Video Splitting**

First you should have a meta file with column 'path' for the videos. Then run the following command:

```Bash
python Data_Filtering/Open-Sora-main/tools.scene_cut.scene_detect.py ---output /path_to_meta.csv
```

The output is {prefix}_timestamp.csv with column timestamp. Each cell in column timestamp is a list of tuples, with each tuple indicating the start and end timestamp of a scene (e.g., [('00:00:01.234', '00:00:02.345'), ('00:00:03.456', '00:00:04.567')]).

### **3. Video Frame Extracting**

Video frame extraction can be directly run the following code. Run the following command:

```Bash
python extract_video_frame/extract_video_frame_1s.py --data_dir allVideo --output_dir allVideo_frame
```




### **Evaluation**

Before running the script, please ensure you modify the following paths to match your local directory structure. This is crucial for the script to locate the necessary files and directories correctly. Below are the paths that need to be updated:

- answers_path: This should point to the directory where your answer files are stored.
- gt_dir: This should point to the directory containing your ground truth data.
- con_dir: This should point to the directory where your ConQA data is located.
- eval_SingleQA_result: This should be the path where you want to save the evaluation results for SingleQA.
- eval_chainQA_result: This should be the path where you want to save the evaluation results for ChainQA.
- eval_ConQA_result: This should be the path where you want to save the evaluation results for ConQA.

Example

```Bash
answers_path = '/path/to/your/answers_path'
gt_dir = "/path/to/your/gt_dir"
con_dir = "/path/to/your/conQA_data_dir"
eval_SingleQA_result = '/path/to/your/save_file_for_SingleQA'
eval_chainQA_result = '/path/to/your/save_file_for_ChainQA'
eval_ConQA_result = '/path/to/your/save_file_for_ConQA'
```

#### **8.1.QA Evaluation**

Run the following command:

```Bash
python evaluation/eval_SingleQA_gpt4_gpt4o.py
```

#### **8.2.Dialogue Evaluation**

Run the following command:

```Bash
python evaluation/eval_chainQA_gpt4_gpt4o.py
```

#### **8.3.Streaming Evaluation**

Run the following command:

```Bash
python evaluation/eval_ConQA_gpt4_gpt4o.py
```

## **Evaluation of StreamingChat**

![model](/assets/images/model_framework.png)

1. ### **Data Preparation**

Data download reference：

https://internvl.readthedocs.io/en/latest/get_started/eval_data_preparation.html

1. ### **MMBench**

Run the following command:

```Bash
GPUS=8 bash evaluate.sh work_dirs/internvl_chat_v2_0/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_yzy_v6_merge mmbench-test-en --dynamic
```

Then, submit the results to the[ evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission). The expected test results are: 

overall: 80.66

1. ### **CCBench**

Run the following command:

```Bash
GPUS=8 bash evaluate.sh work_dirs/internvl_chat_v2_0/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_yzy_v6_merge ccbench-dev --dynamic
```

Then, submit the results to the[ evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission). The expected test results are: 

overall:74.71

1. ### **Tiny LVLM**

Run the following command:

```Bash
GPUS=8 bash evaluate.sh work_dirs/internvl_chat_v2_0/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_yzy_v6_merge tiny_lvlm --dynamic
```

The expected test results are:

```Bash
Visual_Perception: 0.4825
ObjecCHallucination: 0.9033333333333333
Visual_Commonsense: 0.636
Visual_Knowledge_Acquisition: 0.6842857142857143
Visual_Reasoning: 0.6654545454545454
Overall: 3.371573593073593
```

1. ### **MM-Vet**

Run the following command:

```Bash
GPUS=8 bash evaluate.sh work_dirs/internvl_chat_v2_0/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_yzy_v6_merge mmvet --dynamic
```

Then, submit the results to the[ evaluation server](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator). The expected test results are:

runs:36.9

1. ### **MMMU**

Run the following command:

```Bash
GPUS=8 bash evaluate.sh work_dirs/internvl_chat_v2_0/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_yzy_v6_merge mmmu-val --dynamic
```

The expected test results are:

```Bash
{'Overall-Art and Design': {'num': 120, 'acc': 0.592}, 'Art': {'num': 30, 'acc': 0.733}, 'Art_Theory': {'num': 30, 'acc': 0.6},  'Overall': {'num': 900, 'acc': 0.49}}
```

1. ### **MMBench Video**

```Bash
git clone https://github.com/open-compass/VLMEvalKit.git && cd VLMEvalKit && pip install -e .
```

You can place the required keys in $VLMEvalKit/.env or directly set them as the environment variable. If you choose to create a .env file, its content will look like:

```Bash
# The .env file, place it under $VLMEvalKit
# API Keys of Proprietary VLMs
# QwenVL APIs
DASHSCOPE_API_KEY=
# Gemini w. Google Cloud Backends
GOOGLE_API_KEY=
# OpenAI API
OPENAI_API_KEY=
OPENAI_API_BASE=
# StepAI API
STEPAI_API_KEY=
# REKA API
REKA_API_KEY=
# GLMV API
GLMV_API_KEY=
# CongRong API
CW_API_BASE=
CW_API_KEY=
# SenseChat-V API
SENSECHAT_AK=
SENSECHAT_SK=
# Hunyuan-Vision API
HUNYUAN_SECRET_KEY=
HUNYUAN_SECRET_ID=
# You can also set a proxy for calling api models during the evaluation stage
EVAL_PROXY=
```

Fill the blanks with your API keys (if necessary). Those API keys will be automatically loaded when doing the inference and evaluation.

Run the following command:

```Bash
torchrun --nproc-per-node=8 run.py --data MMBench-Video --model InternVL2-8B --verbose --nframe 8
```

The expected test results are:

```Bash
 "coarse_all": {
        "CP": "1.53",
        "FP-S": "1.41",
        "FP-C": "1.16",
        "HL": "0.21",
        "LR": "1.06",
        "AR": "1.55",
        "RR": "1.59",
        "CSR": "1.37",
        "TR": "1.31",
        "Perception": "1.35",
        "Reasoning": "1.39",
        "Overall": "1.37"
    },
    "coarse_valid": {
        "CP": "1.53",
        "FP-S": "1.41",
        "FP-C": "1.16",
        "HL": "0.21",
        "LR": "1.06",
        "AR": "1.55",
        "RR": "1.59",
        "CSR": "1.37",
        "TR": "1.31",
        "Perception": "1.35",
        "Reasoning": "1.39",
        "Overall": "1.37"
    }
}
```
