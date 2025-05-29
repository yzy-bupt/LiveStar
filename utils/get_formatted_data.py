import re
import os
import json
import glob
from PIL import Image
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

DEFAULT_SYSTEM_PROMPT = ""
VIDEO_FRAME_BASE_PATH_PREFIX = "xxx/StreamingCaption/data/video_frames" # Example, adjust if needed


def convert_time_to_seconds(time_string: str) -> int:
    """
    Converts a time string in HH:MM:SS format to total seconds.

    Args:
        time_string (str): The time string to convert (e.g., "01:23:45").

    Returns:
        int: The total number of seconds.
    """
    hours, minutes, seconds = map(int, time_string.split(':'))
    return hours * 3600 + minutes * 60 + seconds


def group_and_merge_video_conversations(input_jsonl_path: str, output_jsonl_path: str):
    """
    Reads single-turn conversation entries from a JSONL file, groups them by video ID,
    and merges them into multi-turn conversation entries.

    The video ID is extracted from the directory name of the first image path in an entry.
    Entries within each video group are sorted by their original 'id' before merging.

    Args:
        input_jsonl_path (str): Path to the input JSONL file.
                                Each line is a JSON object representing a single-turn entry.
                                Expected keys: 'image' (list of paths), 'id', 'height_list',
                                'width_list', 'conversations'.
        output_jsonl_path (str): Path to save the merged multi-turn JSONL file.
                                 Each line is a JSON object with merged data.
    """
    video_data_groups = defaultdict(list)
    with open(input_jsonl_path, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            entry = json.loads(line)
            if not entry.get('image'): # Skip if no images
                print(f"Warning: Entry with id {entry.get('id')} has no 'image' field. Skipping.")
                continue
            # Extract video_id from the parent directory of the first image path
            first_image_path = Path(entry['image'][0])
            video_id = first_image_path.parent.name
            video_data_groups[video_id].append(entry)

    merged_entries_data = []
    for group_idx, (video_id, entries_in_group) in enumerate(video_data_groups.items()):
        # Sort entries by original 'id' to maintain chronological order
        sorted_entries = sorted(entries_in_group, key=lambda x: x['id'])

        all_images = []
        all_heights = []
        all_widths = []
        full_conversation = []

        for entry in sorted_entries: # Removed 'i' and enumerate as it was only for system_prompt
            all_images.extend(entry.get('image', []))
            all_heights.extend(entry.get('height_list', []))
            all_widths.extend(entry.get('width_list', []))

            current_conversations = entry.get('conversations', [])
            full_conversation.extend(current_conversations) # Directly extend, no system prompt logic

        merged_entry = {
            "id": group_idx,
            "image": all_images,
            "height_list": all_heights,
            "width_list": all_widths,
            "conversations": full_conversation
        }
        merged_entries_data.append(merged_entry)

    with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
        for entry in merged_entries_data:
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Multi-turn conversation processing completed. Data saved to: {output_jsonl_path}")


def _parse_raw_video_text_entry(
    raw_entry_lines: list[str],
    all_json_entries: list[dict],
    video_frames_base_dir: str,
    entry_start_id: int
):
    """
    Parses a single raw text block (for one video) and generates JSONL-like dictionary entries.
    This is a helper function for `convert_text_to_jsonl`.

    The raw text block contains a video path followed by lines with timestamps and captions.
    It extracts frames based on timestamps and creates structured entries.

    Args:
        raw_entry_lines (list[str]): Lines of text for a single video entry.
                                     First line is video path, rest is content.
        all_json_entries (list[dict]): List to which new JSON entries will be appended.
        video_frames_base_dir (str): Base directory where video frames are stored.
        entry_start_id (int): The starting ID for the entries generated from this text block.
    """
    if not raw_entry_lines:
        return

    video_file_path_str = raw_entry_lines[0]
    video_name = Path(video_file_path_str).stem # e.g., "my_video" from "/path/to/my_video.mp4"
    raw_content = ' '.join(raw_entry_lines[1:])

    # Regex to find "Timestamp: HH:MM:SS Caption: Some text"
    # Uses positive lookahead for robust parsing of captions that might span lines.
    timestamp_caption_pairs = re.findall(
        r'Timestamp:\s*(\d{2}:\d{2}:\d{2})\s*Caption:(.*?)(?=\s*Timestamp:\s*|$)',
        raw_content,
        re.DOTALL | re.IGNORECASE # Added IGNORECASE for flexibility
    )

    if not timestamp_caption_pairs:
        print(f"Warning: No timestamp-caption pairs found for {video_name}. Skipping.")
        return

    # Timestamp adjustment
    adjusted_timestamp_caption_pairs = []
    if timestamp_caption_pairs[0][0] == "00:00:00":
        adjusted_timestamp_caption_pairs.append(timestamp_caption_pairs[0]) # Keep first as is
        if len(timestamp_caption_pairs) > 1:
            # Force second timestamp to 00:00:01
            adjusted_timestamp_caption_pairs.append(('00:00:01', timestamp_caption_pairs[1][1]))
            # For the rest, increment from the *original* previous timestamp
            for i in range(2, len(timestamp_caption_pairs)):
                try:
                    prev_original_time_obj = datetime.strptime(timestamp_caption_pairs[i-1][0], '%H:%M:%S')
                    new_time_obj = prev_original_time_obj + timedelta(seconds=1)
                    new_time_str = new_time_obj.strftime('%H:%M:%S')
                    adjusted_timestamp_caption_pairs.append((new_time_str, timestamp_caption_pairs[i][1]))
                except ValueError:
                    print(f"Warning: Could not parse time {timestamp_caption_pairs[i-1][0]} for {video_name}. Skipping segment.")
                    continue
    else:
        # Force first timestamp to 00:00:00
        adjusted_timestamp_caption_pairs.append(('00:00:00', timestamp_caption_pairs[0][1]))
        # For the rest, increment from the *original* previous timestamp
        for i in range(1, len(timestamp_caption_pairs)):
            try:
                prev_original_time_obj = datetime.strptime(timestamp_caption_pairs[i-1][0], '%H:%M:%S')
                new_time_obj = prev_original_time_obj + timedelta(seconds=1)
                new_time_str = new_time_obj.strftime('%H:%M:%S')
                adjusted_timestamp_caption_pairs.append((new_time_str, timestamp_caption_pairs[i][1]))
            except ValueError:
                print(f"Warning: Could not parse time {timestamp_caption_pairs[i-1][0]} for {video_name}. Skipping segment.")
                continue

    video_specific_frame_dir = Path(video_frames_base_dir) / video_name
    if not video_specific_frame_dir.exists():
        print(f"Warning: Frame directory not found: {video_specific_frame_dir}. Skipping video {video_name}.")
        return

    frame_files = sorted(glob.glob(str(video_specific_frame_dir / f"{video_name}_*.jpg")))
    
    # Extract frame numbers and ensure they are digits
    valid_frame_numbers = []
    for f_path_str in frame_files:
        f_path = Path(f_path_str)
        try:
            # Assumes format like video_name_XXXX.jpg
            frame_num_str = f_path.stem.split('_')[-1]
            if frame_num_str.isdigit():
                valid_frame_numbers.append(int(frame_num_str))
        except (IndexError, ValueError):
            print(f"Warning: Could not parse frame number from {f_path.name}. Skipping file.")
            
    if not valid_frame_numbers:
        print(f"Warning: No valid frame numbers found in {video_specific_frame_dir}. Skipping video {video_name}.")
        return

    extracted_frame_step = 1 # Default if only one frame
    if len(valid_frame_numbers) > 1:
        extracted_frame_step = valid_frame_numbers[1] - valid_frame_numbers[0]
        if extracted_frame_step <= 0: # Should not happen with sorted list
            print(f"Warning: Non-positive frame step {extracted_frame_step} for {video_name}. Defaulting to 1.")
            extracted_frame_step = 1


    img_width, img_height = -1, -1
    try:
        # Use the first valid frame file found
        first_frame_filename = f"{video_name}_{valid_frame_numbers[0]:04d}.jpg"
        with Image.open(video_specific_frame_dir / first_frame_filename) as img:
            img_width, img_height = img.size
    except FileNotFoundError:
        print(f"Warning: First frame image not found for {video_name}. Cannot get dimensions.")
        return
    except Exception as e:
        print(f"Warning: Could not open image to get dimensions for {video_name}: {e}")
        return

    MAX_FRAME_NUMBER_PLACEHOLDER = 1_000_000 # A large number for the last segment's end_frame

    for i, (time_str, description) in enumerate(adjusted_timestamp_caption_pairs):
        description = description.strip()
        try:
            current_segment_start_seconds = convert_time_to_seconds(time_str)
        except ValueError:
            print(f"Warning: Invalid time string '{time_str}' for {video_name}. Skipping segment.")
            continue

        segment_start_frame_value = current_segment_start_seconds * extracted_frame_step

        segment_end_frame_value: int
        if i < len(adjusted_timestamp_caption_pairs) - 1:
            next_time_str, _ = adjusted_timestamp_caption_pairs[i+1]
            try:
                next_segment_start_seconds = convert_time_to_seconds(next_time_str)
                # End frame is one "step" before the next segment starts
                segment_end_frame_value = (next_segment_start_seconds * extracted_frame_step) - extracted_frame_step
            except ValueError:
                print(f"Warning: Invalid next time string '{next_time_str}' for {video_name}. Using placeholder for end frame.")
                segment_end_frame_value = MAX_FRAME_NUMBER_PLACEHOLDER # Fallback
        else:
            # For the last timestamp, extend to a very large frame number
            segment_end_frame_value = MAX_FRAME_NUMBER_PLACEHOLDER

        current_segment_frames = [
            f_num for f_num in valid_frame_numbers
            if segment_start_frame_value <= f_num <= segment_end_frame_value # and f_num % extracted_frame_step == 0
        ]

        if not current_segment_frames:
            # print(f"Debug: No frames for {video_name} segment {i} ({time_str}). Start: {segment_start_frame_value}, End: {segment_end_frame_value}, Step: {extracted_frame_step}, ValidFrames: {valid_frame_numbers[:5]}... ")
            continue

        num_frames_in_segment = len(current_segment_frames)
        all_json_entries.append({
            "id": entry_start_id + i,
            "image": [
                # Ensure consistent path separators
                str(Path(VIDEO_FRAME_BASE_PATH_PREFIX) / video_name / f"{video_name}_{f_num:04d}.jpg").replace("\\","/")
                for f_num in current_segment_frames
            ],
            "height_list": [img_height] * num_frames_in_segment,
            "width_list": [img_width] * num_frames_in_segment,
            "conversations": [
                {"from": "human", "value": "<image>\n" * num_frames_in_segment},
                {"from": "gpt", "value": description}
            ]
        })


def convert_text_to_jsonl(input_text_file: str, output_jsonl_file: str, video_frames_base_dir: str):
    """
    Processes a custom formatted text file and converts its content into a JSONL file.
    Each entry in the text file starts with a video path (e.g., "/RNG/.../video.mp4")
    followed by lines containing timestamps and captions.

    Annotation example:
    /RNG_videos/9Uly5zVvl3Q.mp4
    Timestamp: 00:00:00 Caption: The screen shows a pink dividing line with black above and below.
    Timestamp: 00:00:06 Caption: Then, the upper and lower frames are the same. In a forest at night, a brown bear is walking on a country road with a lantern and basket. The camera zooms out to show a table covered with a tablecloth. The table is filled with jams and teacups, and a little girl in a purple dress is standing next to it.
    Timestamp: 00:00:12 Caption: Then, the little girl is standing next to a golden teapot, pouring water into a teacup, steam rising from the hot water. As the brown bear walks up to the table, the little girl pushes it towards him. The brown bear looks down at it, and the frame cuts to the jams on the table: carrot jam, cucumber jam, cherry jam, and tomato jam.
    Timestamp: 00:00:19 Caption: The little girl then pours a jar of red jam into a large bowl filled with pine nuts. The picture switches back to the frowning brown bear, head lowered. The background turns white and shows the words "COMING ON MAY 17!". 

    Args:
        input_text_file (str): Path to the input text file.
        output_jsonl_file (str): Path to save the output JSONL file.
        video_frames_base_dir (str): Base directory where video frames are stored.    
    """
    TEXT_ENTRY_MARKER = "/RNG_videos" # Assuming this marks the start of a new video entry
    
    all_json_entries = []
    current_raw_entry_lines = []

    with open(input_text_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            stripped_line = line.strip()
            if not stripped_line: # Skip empty lines
                continue

            if stripped_line.startswith(TEXT_ENTRY_MARKER):
                if current_raw_entry_lines: # Process previous entry
                    # The start_id for _parse_raw_video_text_entry should be the current length of all_json_entries
                    _parse_raw_video_text_entry(current_raw_entry_lines, all_json_entries, video_frames_base_dir, len(all_json_entries))
                current_raw_entry_lines = [stripped_line] # Start new entry
            elif current_raw_entry_lines: # Append to current entry
                current_raw_entry_lines.append(stripped_line)

        # Process the last entry in the file
        if current_raw_entry_lines:
            _parse_raw_video_text_entry(current_raw_entry_lines, all_json_entries, video_frames_base_dir, len(all_json_entries))

    with open(output_jsonl_file, 'w', encoding='utf-8') as f_out:
        for entry in all_json_entries:
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Text to JSONL processing completed. Data saved to: {output_jsonl_file}")


def add_prompts_and_frame_tags(input_jsonl_path: str, output_jsonl_path: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    """
    Processes a JSONL file, adding a system prompt to the first human turn and
    numbered frame tags (Frame-1: <image>, Frame-2: <image>) to human messages.

    The frame counter is continuous across all human turns within a single JSONL entry.

    Args:
        input_jsonl_path (str): Path to the input JSONL file.
        output_jsonl_path (str): Path to save the modified JSONL file.
        system_prompt (str, optional): A system prompt to add. Defaults to DEFAULT_SYSTEM_PROMPT.
    """
    with open(input_jsonl_path, 'r', encoding='utf-8') as infile, \
         open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            
            frame_tag_counter = 1 # Reset for each JSONL entry (each video's full conversation)
            first_human_turn_processed = False

            if 'conversations' in data:
                for conversation_turn in data['conversations']:
                    if conversation_turn.get('from') == 'human':
                        original_value = conversation_turn.get('value', "")
                        
                        # Add system prompt to the very first human turn of this entry
                        if not first_human_turn_processed and system_prompt:
                            original_value = system_prompt + original_value
                            first_human_turn_processed = True
                        
                        # Add frame tags
                        parts = original_value.split('<image>')
                        new_value_parts = []
                        for i, part in enumerate(parts):
                            new_value_parts.append(part)
                            if i < len(parts) - 1: # If not the last part, an <image> tag followed
                                new_value_parts.append(f"Frame-{frame_tag_counter}: <image>")
                                frame_tag_counter += 1
                        conversation_turn['value'] = ''.join(new_value_parts)
            
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
    print(f"Prompt and frame tag addition completed. Data saved to: {output_jsonl_path}")


def get_interleaved_frame_caption(input_jsonl_path: str, output_jsonl_path: str):
    """
    Processes a JSONL file to transform conversations. If a human turn contains
    multiple "<image>\n" tags, it splits this into multiple (human, gpt) pairs,
    where each new human turn has one "<image>\n" and the gpt response is duplicated.

    Example:
    Human: <image>\n<image>\n
    GPT: Caption for both.
    
    Becomes:
    Human: <image>\n
    GPT: Caption for both.
    Human: <image>\n
    GPT: Caption for both.

    Args:
        input_jsonl_path (str): Path to the input JSONL file.
        output_jsonl_path (str): Path to save the modified JSONL file.
    """
    with open(input_jsonl_path, 'r', encoding='utf-8') as infile, \
         open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
        
        for line_idx, line in enumerate(infile):
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON on line {line_idx+1}. Skipping.")
                continue
            
            original_conversations = data.get("conversations", [])
            if not original_conversations or len(original_conversations) % 2 != 0:
                print(f"Warning: Entry {data.get('id', 'N/A')} has no conversations or odd number of turns. Writing as is.")
                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            new_interleaved_conversations = []
            
            # Iterate through (human, gpt) pairs
            for i in range(0, len(original_conversations), 2):
                human_turn = original_conversations[i]
                gpt_turn = original_conversations[i+1] # Assumes gpt_turn always follows human_turn

                if human_turn.get("from") != "human" or gpt_turn.get("from") != "gpt":
                    print(f"Warning: Unexpected turn structure for entry {data.get('id', 'N/A')} at pair index {i//2}. Appending as is.")
                    new_interleaved_conversations.append(human_turn)
                    new_interleaved_conversations.append(gpt_turn)
                    continue

                human_value = human_turn.get("value", "")
                image_tag_placeholder = "<image>\n" # The specific tag to count and replicate
                num_image_tags = human_value.count(image_tag_placeholder)

                if num_image_tags == 0: # No image tags, keep as is
                    new_interleaved_conversations.append(human_turn)
                    new_interleaved_conversations.append(gpt_turn)
                    continue
                
                # Handle prefix text for the very first image of the first human turn in the entry
                prefix_text = ""
                if i == 0: # This is the first (human, gpt) pair of the entry
                    if image_tag_placeholder in human_value:
                        prefix_text = human_value.split(image_tag_placeholder, 1)[0]
                    else: # Should not happen if num_image_tags > 0
                        prefix_text = human_value

                for j in range(num_image_tags):
                    new_human_value = ""
                    if j == 0 and i == 0: # First image of the first pair
                        new_human_value = prefix_text + image_tag_placeholder
                    else:
                        new_human_value = image_tag_placeholder
                    
                    new_interleaved_conversations.append({"from": "human", "value": new_human_value})
                    new_interleaved_conversations.append(gpt_turn.copy()) # Duplicate GPT response
            
            data["conversations"] = new_interleaved_conversations
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            
    print(f"Interleaving frames with captions completed. Data saved to: {output_jsonl_path}")


def run_full_processing_pipeline():
    """
    Coordinates the full data processing pipeline by calling various
    processing functions in sequence.
    """
    
    # Path for initial text processing
    raw_text_input_file = '/your_path/raw_text_input_file.txt'
    jsonl_after_text_processing = '/your_path/single_turn.jsonl'
    video_frames_dir = '/your_path/video_frames'
    
    # Path for multi-turn conversion
    jsonl_after_multi_turn = "/your_path/multi_turn.jsonl"
    
    # Path for interleaving
    interleaved_output_jsonl = "/your_path/interleaved_frame_caption.jsonl"

    # Path for adding prompts
    final_output_jsonl = "/your_path/formatted_data.jsonl"

    print("Starting Step 1: Converting text file to JSONL...")
    convert_text_to_jsonl(raw_text_input_file, jsonl_after_text_processing, video_frames_dir)
    print(f"Step 1 finished. Output: {jsonl_after_text_processing}\n")
    
    print("Starting Step 2: Converting to multi-turn format...")
    group_and_merge_video_conversations(jsonl_after_text_processing, jsonl_after_multi_turn)
    print(f"Step 2 finished. Output: {jsonl_after_multi_turn}\n")

    print("Starting Step 3: Interleaving frames with captions...")
    get_interleaved_frame_caption(jsonl_after_multi_turn, interleaved_output_jsonl)
    print(f"Step 3 finished. Output: {interleaved_output_jsonl}\n")
    
    print("Starting Step 4: Adding prompts and frame tags...")
    add_prompts_and_frame_tags(interleaved_output_jsonl, final_output_jsonl) # Uses DEFAULT_SYSTEM_PROMPT
    print(f"Step 4 finished. Final output: {final_output_jsonl}\n")

    print("Full processing pipeline completed.")


if __name__ == "__main__":
    run_full_processing_pipeline()