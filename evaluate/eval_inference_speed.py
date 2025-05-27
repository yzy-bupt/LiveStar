import os
import shutil
import time
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from itertools import groupby
from math import ceil

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
    # print("frame_indices", frame_indices)
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


def delete_past_key_values(past_key_values, del_list, start_idx, token_block_size=29):
    # 每一项 del_list 是 (轮次/组别, 组内第几个片段)
    delete_ranges = []
    del_count_per_group = {}  # 记录每组要删几段
    
    for group, local_idx in del_list:
        start = start_idx[group - 1]
        delete_start = start + 2 + local_idx * token_block_size
        delete_end = delete_start + token_block_size
        delete_ranges.append((delete_start, delete_end))

        del_count_per_group[group] = del_count_per_group.get(group, 0) + 1
        
    # 对 past_key_values 的每一层执行删除（沿 seq_len 维度）
    seq_len = past_key_values[0][0].shape[2]
    keep_mask = torch.ones(seq_len, dtype=torch.bool)
    for start, end in delete_ranges:
        keep_mask[start:end] = False
        
    new_past_key_values = []
    for key, value in past_key_values:
        new_key = key[:, :, keep_mask, :]
        new_value = value[:, :, keep_mask, :]
        new_past_key_values.append((new_key, new_value))

    # update start_idx
    new_start_idx = start_idx[:]
    total_groups = len(start_idx)
    for group, count in del_count_per_group.items():
        shift = count * token_block_size
        for i in range(group, total_groups):  # 注意 group 是从 1 开始的
            new_start_idx[i] -= shift        

    return new_past_key_values, new_start_idx


def peak_end_and_streaming_kvcache(ppl_list, chat_history, past_key_values, kvcache_len):
    # Step 1: Sort by group and PPL
    sorted_ppl = sorted(ppl_list, key=lambda x: (x[0], x[1]))
    group_to_top_indices = {}
    for group_key, items in groupby(sorted_ppl, key=lambda x: x[0]):
        item_list = list(items)
        num_to_select = max(ceil(len(item_list) / 2), 1)
        top_indices = [index for _, _, index in item_list[:num_to_select]]
        group_to_top_indices[group_key] = top_indices
    
    # Step 2: Calculate del_idx
    keep_idx = sorted(sum(group_to_top_indices.values(), []))
    del_idx = [i for i in range(1, len(ppl_list) + 1) if i not in keep_idx]
    
    # Step 3: Build mapping from global index -> (group, local_idx)
    index_to_group_and_local = {}
    group_to_local_counter = {}
    for group, _, global_index in sorted(ppl_list, key=lambda x: x[2]):  # sort by global index
        if group not in group_to_local_counter:
            group_to_local_counter[group] = 0
        local_idx = group_to_local_counter[group]
        index_to_group_and_local[global_index] = (group, local_idx)
        group_to_local_counter[group] += 1
        
    # Step 4: Update past_key_values and kvcache_len 
    del_list = [index_to_group_and_local[i] for i in del_idx if i in index_to_group_and_local]
    past_key_values, kvcache_len = delete_past_key_values(past_key_values, del_list, kvcache_len)
    
    # Step 5: Update chat_history
    updated_chat_history = []
    for idx, (prompt, response) in enumerate(chat_history):
        group_id = idx + 1
        if group_id not in group_to_top_indices:
            updated_chat_history.append((prompt, response))
            continue
        keep_indices = set(group_to_top_indices[group_id])
        lines = prompt.splitlines()
        new_lines = [
            line for line in lines 
            if not line.startswith('Frame-') or any(f'Frame-{i}:' in line for i in keep_indices)
        ]
        new_prompt = '\n'.join(new_lines) + '\n'
        updated_chat_history.append((new_prompt, response))
        
    return updated_chat_history, keep_idx, past_key_values, kvcache_len


def truncate_past_key_values(past_key_values, useful_kvcache_len):
    new_past_key_values = []
    for key, value in past_key_values:
        truncated_key = key[:, :, :useful_kvcache_len, :]
        truncated_value = value[:, :, :useful_kvcache_len, :]
        new_past_key_values.append((truncated_key, truncated_value))
    return new_past_key_values


# evaluation setting
max_num_frames = 512
generation_config = dict(
    temperature=0.0,
    max_new_tokens=1024,
    top_p=0.1,
    num_beams=1,
    repetition_penalty = 1.05,
)

video_path = "your_video.mp4"
num_segments=128
decode_factor = 1.04
use_kvcache = True
use_peak_end = True
peak_end_window_len = 24


with torch.no_grad():
    pixel_values, num_patches_list = load_video(video_path, num_segments=num_segments, max_num=1, get_frame_by_duration=True)
    pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
    batch_frame = 1
    
    mask = [True] * len(num_patches_list)
    ppl_list = []
    kvcache_len = []
    
    start_time = time.time()
    for i in tqdm(range(0, len(num_patches_list), batch_frame)):
        video_frame = "".join([f"Frame-{i+j+1}: <image>\n" for j in range(batch_frame)])
        if i == 0:
            caption_prompt = "You are an expert in real-time streaming video description. I will provide video frames sequentially, and you need to comprehend each frame's content in real-time while dynamically generating concise descriptions. Use transitional phrases to maintain textual coherence and avoid repeating already described content.\n"
            question = caption_prompt + video_frame
            output_last, chat_history, past_key_values = model.chat(
                tokenizer, 
                pixel_values[:i+batch_frame, ...], 
                question, 
                generation_config, 
                num_patches_list=num_patches_list[:i+batch_frame], 
                history=None, 
                return_history=True,
                use_kvcache=use_kvcache
            )
            if use_kvcache:
                kvcache_len.append(past_key_values[0][0].shape[2])
            
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
            ppl_list.append((len(chat_history), output_perplexity, i+1))
            decode_threshold = output_perplexity

        else:                
            
            if len(ppl_list) > peak_end_window_len and use_peak_end and use_kvcache:
                chat_history, keep_list, past_key_values, kvcache_len = peak_end_and_streaming_kvcache(ppl_list, chat_history, past_key_values, kvcache_len)
                mask = [i+1 in keep_list for i in range(len(mask))]
                ppl_list = [item for item in ppl_list if item[2] in keep_list]            
            
            # Filter out frames where mask is False
            mask[i:i+batch_frame] = [True] * batch_frame
            filtered_pixel_values = pixel_values[:i+batch_frame, ...][mask[:i+batch_frame]]
            filtered_num_patches_list = [num_patches_list[j] for j in range(i+batch_frame) if mask[j]]
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
                    past_key_values=past_key_values,
                    use_kvcache=use_kvcache
                )
                if use_kvcache:
                    kvcache_len.append(past_key_values[0][0].shape[2])
                    past_key_values = truncate_past_key_values(past_key_values, kvcache_len[len(chat_history)-2])
            
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
                ppl_list.append((len(chat_history), output_perplexity, i+1))
            else:
                ppl_list.append((len(chat_history), output_perplexity, i+1))
                chat_history[-1] = (chat_history[-1][0] + video_frame, chat_history[-1][1])