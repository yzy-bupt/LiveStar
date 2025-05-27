import json
import os
import shutil
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# model setting
model_path = 'LiveStar/inference'

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda().to(torch.bfloat16)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices


def get_num_frames_by_duration(duration):
        # local_num_frames = 4      
        local_num_frames = 1  
        num_segments = int(duration // local_num_frames)
        if num_segments == 0:
            num_frames = local_num_frames
        else:
            num_frames = local_num_frames * num_segments
        
        num_frames = min(512, num_frames)
        # num_frames = max(128, num_frames)
        num_frames = max(1, num_frames)
        
        return num_frames

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32, get_frame_by_duration = False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = f'./examples/frames/{video_name}'
    if os.path.exists(save_dir):
        save_flag = False
    else:
        save_flag = True
        os.makedirs(save_dir, exist_ok=True)
        destination_path = f'./examples/videos/{os.path.basename(video_path)}'
        os.makedirs(destination_path, exist_ok=True)
        shutil.copy(video_path, destination_path)
        print(f"Video copied to {destination_path}")
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    if get_frame_by_duration:
        duration = max_frame / fps
        num_segments = get_num_frames_by_duration(duration)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)

    for i in range(len(frame_indices)):
        img = Image.fromarray(vr[frame_indices[i]].asnumpy()).convert("RGB")
        if save_flag:
            save_path = os.path.join(save_dir, f'frame_{i}.png')
            img.save(save_path)
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def load_images(image_path, root_dir, input_size=448, max_num=6):
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    for path in image_path:
        image = Image.open(root_dir+path).convert("RGB")
        image = dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(tile) for tile in image]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def frame_to_timestamp(frame_idx):
    total_seconds = int(frame_idx)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02}:{seconds:02}"
    # hours, minutes = divmod(minutes, 60)
    # return f"{hours:02}:{minutes:02}:{seconds:02}"


# evaluation setting
max_num_frames = 512
generation_config = dict(
    temperature=0.0,
    max_new_tokens=1024,
    top_p=0.1,
    num_beams=1,
    repetition_penalty = 1.05,
)
decode_factor = 1.04


def online_inference_evaluate():
    eval_dir = "eval_result/"
    eval_files = [
        "OmniStar/eval_datasets/RNG.jsonl",
        "OmniStar/eval_datasets/FDQ.jsonl", 
    ]
    for input_file in eval_files:
        with torch.no_grad():
            print("Start evaluating...")
            with open(input_file, "r", encoding="utf-8") as infile:
                total_lines = sum(1 for _ in infile)
                infile.seek(0)
                
                for line in tqdm(infile, total=total_lines, desc="Processing"):
                    data = json.loads(line.strip())
                    image_path = data["image"]
                    conversations = data["conversations"]
                    first_path = image_path[0]
                    video_id = os.path.basename(os.path.dirname(first_path))
                    
                    pixel_values, num_patches_list = load_images(image_path, max_num=1)
                    pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
                    
                    batch_frame = 1
                    result = []
                    for i in range(0, len(num_patches_list), batch_frame):
                        video_frame = "".join([f"Frame-{i+j+1}: <image>\n" for j in range(batch_frame)])
                        if i == 0:
                            question = conversations[0]['value']
                            output_last, chat_history, past_key_values = model.chat(
                                tokenizer, 
                                pixel_values[:i+batch_frame, ...], 
                                question, 
                                generation_config, 
                                num_patches_list=num_patches_list[:i+batch_frame], 
                                history=None, 
                                return_history=True,
                            )                            
                            output_perplexity, _ = model.chat(
                                tokenizer, 
                                pixel_values[:i+batch_frame, ...], 
                                video_frame, 
                                generation_config, 
                                num_patches_list=num_patches_list[:i+batch_frame], 
                                history=chat_history, 
                                return_history=False, 
                                check_answer=output_last,
                                self_check=True
                            )
                            decode_threshold = output_perplexity
                            result.append((frame_to_timestamp(i), output_last))
                            
                        else:
                            filtered_pixel_values = pixel_values[:i + batch_frame, ...]
                            filtered_num_patches_list = num_patches_list[:i + batch_frame]
                            question = video_frame
                            output_perplexity, _ = model.chat(
                                tokenizer, 
                                filtered_pixel_values, 
                                question, 
                                generation_config, 
                                num_patches_list=filtered_num_patches_list, 
                                history=chat_history, 
                                return_history=False, 
                                check_answer=output_last
                            )
                            
                            if output_perplexity > decode_threshold * decode_factor:
                                output_last, chat_history, past_key_values = model.chat(
                                    tokenizer, 
                                    filtered_pixel_values, 
                                    question, 
                                    generation_config, 
                                    num_patches_list=filtered_num_patches_list, 
                                    history=chat_history, 
                                    return_history=True,
                                )
                                decode_threshold, _ = model.chat(
                                    tokenizer, 
                                    filtered_pixel_values, 
                                    question, 
                                    generation_config, 
                                    num_patches_list=filtered_num_patches_list, 
                                    history=chat_history, 
                                    return_history=False, 
                                    check_answer=output_last,
                                    self_check=True
                                )
                                result.append((frame_to_timestamp(i), output_last))
                            else:
                                chat_history[-1] = (chat_history[-1][0] + video_frame, chat_history[-1][1])

                    json_line = {
                        "video_id": video_id,
                        "result": result
                    }
                    with open(eval_dir+f"online_eval_result.jsonl", "a", encoding="utf-8") as fout:
                        fout.write(json.dumps(json_line, ensure_ascii=False) + "\n")
                        
    print("result saved to:", eval_dir+f"online_eval_result.jsonl")


def offline_inference_evaluate():
    eval_dir = "eval_result/"
    eval_files = [
        "OmniStar/eval_datasets/RNG.jsonl",
        "OmniStar/eval_datasets/FDQ.jsonl", 
    ]
    for input_file in eval_files:
        with torch.no_grad():
            print("Start offline evaluating...")
            with open(input_file, "r", encoding="utf-8") as infile:
                total_lines = sum(1 for _ in infile)
                infile.seek(0)
                
                for line in tqdm(infile, total=total_lines, desc="Processing"):
                    data = json.loads(line.strip())
                    image_path = data["image"]
                    conversations = data["conversations"]
                    first_path = image_path[0]
                    video_id = os.path.basename(os.path.dirname(first_path))
                    
                    pixel_values, num_patches_list = load_images(image_path, max_num=1)
                    pixel_values = pixel_values.to(torch.bfloat16).to(model.device)

                    result = []
                    chat_history=None
                    count = 0
                    for human, gpt in zip(conversations[::2], conversations[1::2]):
                        question = human["value"]
                        count += question.count("<image>")
                        output_last, chat_history, past_key_values = model.chat(
                            tokenizer, 
                            pixel_values[:count, ...], 
                            question, 
                            generation_config, 
                            num_patches_list=num_patches_list[:count], 
                            history=chat_history, 
                            return_history=True,
                        )
                        result.append((gpt["value"], output_last))
                        
                    json_line = {
                        "video_id": video_id,
                        "gt_result_pair": result
                    }
                    with open(eval_dir+"offline_eval_result.jsonl", "a", encoding="utf-8") as fout:
                        fout.write(json.dumps(json_line, ensure_ascii=False) + "\n")
                        
    print("result saved to:", eval_dir+"offline_eval_result.jsonl")


def compute_ppl_and_acc():
    eval_dir = "eval_result/"
    eval_files = [
        "OmniStar/eval_datasets/RNG.jsonl",
        "OmniStar/eval_datasets/FDQ.jsonl", 
    ]
    for input_file in eval_files:
        with torch.no_grad():
            print("Start evaluating...")
            lm_ppl_list, lm_correctness_list = [], []
                    
            with open(input_file, "r", encoding="utf-8") as infile:
                total_lines = sum(1 for _ in infile)
                infile.seek(0)
                
                for line in tqdm(infile, total=total_lines, desc="Processing"):
                    data = json.loads(line.strip())
                    image_path = data["image"]
                    pixel_values, num_patches_list = load_images(image_path, max_num=1)
                    pixel_values = pixel_values.to(torch.bfloat16).to(model.device)

                    conversations = data["conversations"]
                    question = ""
                    
                    lm_ppl, lm_correctness = model.chat(
                        tokenizer,
                        pixel_values,
                        question,
                        generation_config,
                        num_patches_list=num_patches_list,
                        history=None,
                        return_history=False,
                        eval=True,
                        conversations=conversations,
                    )
                    lm_ppl_list.append(lm_ppl)
                    lm_correctness_list.append(lm_correctness)

            lm_ppl = torch.stack(lm_ppl_list).mean() if lm_ppl_list else -1
            lm_correctness = torch.stack(lm_correctness_list).mean() if lm_correctness_list else -1
            
            print("lm_ppl", lm_ppl)
            print("lm_correctness", lm_correctness)
            
            with open(eval_dir + "ppl_and_acc_result.txt", "a", encoding="utf-8") as fout:
                fout.write(f"{input_file}\n")
                fout.write(f"lm_ppl: {lm_ppl}\n")
                fout.write(f"lm_correctness: {lm_correctness}\n")

    print("Saved to: ", eval_dir + "ppl_and_acc_result.txt")
    
    
if __name__ == "__main__":
    online_inference_evaluate()  # get online inference result
    offline_inference_evaluate()  # get offline inference result
    compute_ppl_and_acc()  # compute PPL and TokAcc
    
    
    