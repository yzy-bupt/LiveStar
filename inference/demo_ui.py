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
import gradio as gr

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

def load_video(video_path, input_size=448, max_num=1, num_segments=32, get_frame_by_duration=False, sample_fps=1):
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
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.copy(video_path, destination_path)
        print(f"Video copied to {destination_path}")

    transform = build_transform(input_size=input_size)
    pixel_values_list, num_patches_list = [], []

    if get_frame_by_duration:
        duration = max_frame / fps
        # num_segments = get_num_frames_by_duration(duration)  # original sampling method
        num_segments = int(duration * sample_fps)

    interval = max(1, int(fps / sample_fps))
    frame_indices = list(range(0, max_frame + 1, interval))
    frame_indices = frame_indices[:num_segments]

    for i, idx in enumerate(frame_indices):
        img = Image.fromarray(vr[idx].asnumpy()).convert("RGB")

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


# inference setting
max_num_frames = 512
generation_config = dict(
    temperature=0.0,
    max_new_tokens=1024,
    top_p=0.1,
    num_beams=1,
    repetition_penalty = 1.05
)
num_segments=128


def process_video_ui(video_path, num_segments, decode_factor, caption_prompt_input):
    with torch.no_grad():
        default_prompt = (
            "You are an expert in real-time streaming video description. I will provide video frames sequentially, "
            "and you need to comprehend each frame's content in real-time while dynamically generating concise descriptions. "
            "Use transitional phrases to maintain textual coherence and avoid repeating already described content.\n"
        )

        caption_prompt = caption_prompt_input.strip() if caption_prompt_input.strip() else default_prompt
        pixel_values, num_patches_list = load_video(video_path, num_segments=num_segments, max_num=1, get_frame_by_duration=True)
        pixel_values = pixel_values.to(torch.bfloat16).to(model.device)

        batch_frame = 1
        output_texts = []
        question = ""
        output_last = None
        decode_threshold = None

        for i in range(0, len(num_patches_list), batch_frame):
            video_frame = "".join([f"Frame-{i+j+1}: <image>\n" for j in range(batch_frame)])
            if i == 0:
                question = caption_prompt + video_frame

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
                output_texts.append(f"🟢 Frame-{i+1}: {output_last} (Decode Threshold: {decode_threshold:.3f})")

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

                    output_texts.append(f"🟢 Frame-{i+1}: {output_last} (Perplexity: {output_perplexity:.3f}, New Threshold: {decode_threshold:.3f})")
                else:
                    chat_history[-1] = (chat_history[-1][0] + video_frame, chat_history[-1][1])
                    output_texts.append(f"⚪ Frame-{i+1}: Slient.")

            yield gr.update(value="\n".join(output_texts))


def clear_all():
    return None, "", ""


css="""
#video-box video {
    height: 300px !important;
    object-fit: contain !important;
}

#title {
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin-top: 10px;
    margin-bottom: 10px;
}
"""

# Gradio UI
with gr.Blocks(css=css) as demo:
    
    gr.Markdown("# 🎥 Streaming Video Chat", elem_id="title")
    gr.Markdown("Upload the video and the model will process it frame by frame and generate real-time subtitles.")

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Video...", elem_id="video-box", width="100%")
            slider_segments = gr.Slider(16, 256, value=128, step=8, label="num_segments")
            slider_factor = gr.Slider(1, 1.3, value=1.05, step=0.001, label="Tunable Scaling Factor α")

            with gr.Row():
                clear_btn = gr.Button("Clear Output")
                submit_btn = gr.Button("Start Processing")

        with gr.Column(scale=1):
            system_prompt_input = gr.Textbox(
                label="System Prompt or Question (Optional)",
                placeholder="You are an expert in real-time streaming video description. I will provide video frames sequentially, and you need to comprehend each frame's content in real-time while dynamically generating concise descriptions. Use transitional phrases to maintain textual coherence and avoid repeating already described content.\n",
                lines=3
            )
            output_box = gr.Textbox(label="Model Output:", lines=18)

    clear_btn.click(fn=clear_all, outputs=[video_input, output_box, system_prompt_input])
    submit_btn.click(
        fn=process_video_ui,
        inputs=[video_input, slider_segments, slider_factor, system_prompt_input],
        outputs=output_box
    )
    
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=9999)
