import re, os, json
from datetime import datetime
from math_verify import parse, verify


from PIL import Image, ImageDraw

def draw_boxes(image_path, boxes, output_path):
    # 打开图像
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)

        # 遍历每个box并绘制
        for box in boxes:
            # box格式为 [左上x, 左上y, 右下x, 右下y]
            draw.rectangle(box, outline="red", width=2)

        # 保存带有box的新图像
        img.save(output_path)
        print(f"Annotated image saved at: {output_path}")

def extract_bbox(response):
    start_tag = "<answer>"
    end_tag = "</answer>"
    input_str = response
    # Check if the start tag is in the string
    if start_tag in input_str:
        # Extract the content between the start tag and end tag
        start_idx = input_str.find(start_tag) + len(start_tag)
        end_idx = input_str.find(end_tag)
        
        # If end_tag is not found (i.e., the string is truncated), assume it should be at the end
        if end_idx == -1:
            end_idx = len(input_str)
    
        content_str = input_str[start_idx:end_idx]
    
        # Check if it ends with a closing bracket, if not, fix it
        if not content_str.endswith("]"):
            # If the string is truncated, remove the incomplete part
            content_str = content_str.rsplit("},", 1)[0] + "}]"
    
        # Replace single quotes with double quotes for valid JSON
        content_str_corrected = content_str.replace("'", '"')
    
        # Convert the corrected string to a list of dictionaries (JSON format)
        try:
            bbox_list = json.loads(content_str_corrected)
        except json.JSONDecodeError as e:
            bbox_list = None
    else:
        bbox_list = None
    return bbox_list

def calculate_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    xi1 = max(x1, x1_2)
    yi1 = max(y1, y1_2)
    xi2 = min(x2, x2_2)
    yi2 = min(y2, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection_area = (xi2 - xi1) * (yi2 - yi1)
    
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = area1 + area2 - intersection_area
    
    iou = intersection_area / union_area
    return iou

def sort_and_calculate_iou(list1, list2, iou_threshold=0.5):
    list2_sorted = sorted(list2, key=lambda x: x['Confidence'], reverse=True)
    
    iou_results = []
    
    matched_list1_indices = set()

    for bbox2 in list2_sorted:
        max_iou = 0
        matched_bbox1 = -1
        best_iou = 0
        for i, bbox1 in enumerate(list1):
            if i not in matched_list1_indices:
                iou = calculate_iou(bbox1['Position'], bbox2['Position'])
                if iou > best_iou:
                    best_iou = iou
                    matched_bbox1 = i

        if best_iou > iou_threshold:
            iou_results.append((best_iou, bbox2['Confidence']))
            matched_list1_indices.add(matched_bbox1)
        else:
            iou_results.append((0, bbox2['Confidence']))
    
    ### [(0.7192676547515258, 1.0), (0, 0.7)]
    return iou_results

def remove_duplicates(bbox_list):
    seen = set()
    unique_bboxes = []
    
    for bbox in bbox_list:
        # Convert the position tuple to a tuple for set hashing
        position_tuple = tuple(bbox['Position'])
        
        if position_tuple not in seen:
            seen.add(position_tuple)
            unique_bboxes.append(bbox)
    
    return unique_bboxes

# V1
def compute_reward_iou(iou_results):
    iou_reward = 0.0
    confidence_reward = 0.0
    for i in range(len(iou_results)):
        temp_iou = iou_results[i][0]
        temp_confidence = iou_results[i][1]

        temp_iou_reward = temp_iou
        if temp_iou == 0:
            temp_confidence_reward = (1-temp_iou)*(1-temp_confidence)
        else:
            temp_confidence_reward = temp_confidence

        iou_reward += temp_iou_reward
        confidence_reward += temp_confidence_reward
        
    iou_reward = iou_reward/len(iou_results)
    confidence_reward = confidence_reward/len(iou_results)
    return iou_reward

# V2
def compute_reward_iou_v2(iou_results, len_gt):
    iou_reward = 0.0
    confidence_reward = 0.0
    for i in range(len(iou_results)):
        temp_iou = iou_results[i][0]
        temp_confidence = iou_results[i][1]

        temp_iou_reward = temp_iou
        if temp_iou == 0:
            temp_confidence_reward = (1-temp_iou)*(1-temp_confidence)
        else:
            temp_confidence_reward = temp_confidence

        iou_reward += temp_iou_reward
        confidence_reward += temp_confidence_reward
        
    if len_gt>=len(iou_results):
        iou_reward = iou_reward/len_gt
    else:
        iou_reward = iou_reward/len(iou_results)
    return iou_reward

def compute_reward_confidence(iou_results):
    iou_reward = 0.0
    confidence_reward = 0.0
    for i in range(len(iou_results)):
        temp_iou = iou_results[i][0]
        temp_confidence = iou_results[i][1]

        temp_iou_reward = temp_iou
        if temp_iou == 0:
            temp_confidence_reward = (1-temp_iou)*(1-temp_confidence)
        else:
            temp_confidence_reward = temp_confidence

        iou_reward += temp_iou_reward
        confidence_reward += temp_confidence_reward
        
    iou_reward = iou_reward/len(iou_results)
    confidence_reward = confidence_reward/len(iou_results)
    return confidence_reward

def accuracy_reward_iou(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        student_answer_bbox = []
        ground_truth_bbox = []
        iou_results = []
        show_flage = 0

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                show_flage = 1
                # Extract answer from solution if it has think/answer tags
                ground_truth = sol.strip()
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                student_answer = '<answer>'+student_answer+'</answer>'

                # fix format error
                student_answer = student_answer.replace("[[",'[')  
                student_answer = student_answer.replace("]]",']')  
                student_answer = student_answer.replace("\n",'')  
                # [{'Position': [254, 303, 291, 365], 'Confidence': 0.9}, {'Position': [100, 100, 200, 200], 'Confidence': 0.8}]
                ground_truth_bbox = extract_bbox(ground_truth)
                student_answer_bbox = extract_bbox(student_answer)
                # pdb.set_trace()
                if student_answer_bbox==None or type(student_answer_bbox[0])!=dict:
                    reward = 0.0
                else:
                    student_answer_bbox = remove_duplicates(student_answer_bbox)   # remove duplicates
                    iou_results = sort_and_calculate_iou(ground_truth_bbox, student_answer_bbox)
                    ### new iou reward
                    reward = compute_reward_iou_v2(iou_results, len(ground_truth_bbox))
                    if reward>1:
                        reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        # import pdb; pdb.set_trace()
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward of IoU: {reward} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"sol: {sol}\n")
                if show_flage==1:
                    f.write(f"student_answer_bbox: {student_answer_bbox}\n")
                    f.write(f"ground_truth_bbox: {ground_truth_bbox}\n")
                    if student_answer_bbox!=None:
                        f.write(f"iou_results: {iou_results}\n")
        show_flage = 0 
    return rewards

def accuracy_reward_confidence(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    img_list= kwargs['image']

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    cnt= 0
    for content, sol, img in zip(contents, solution, img_list):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        student_answer_bbox = []
        ground_truth_bbox = []
        iou_results = []
        show_flage = 0

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                show_flage = 1
                # Extract answer from solution if it has think/answer tags
                ground_truth = sol.strip()
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                student_answer = '<answer>'+student_answer+'</answer>'

                # fix format error
                student_answer = student_answer.replace("[[",'[')
                student_answer = student_answer.replace("]]",']')
                student_answer = student_answer.replace("\n",'')
                # [{'Position': [254, 303, 291, 365], 'Confidence': 0.9}, {'Position': [100, 100, 200, 200], 'Confidence': 0.8}]
                ground_truth_bbox = extract_bbox(ground_truth)
                student_answer_bbox = extract_bbox(student_answer)
                # pdb.set_trace()
                if student_answer_bbox==None or type(student_answer_bbox[0])!=dict:  # wrong bbox
                    reward = 0.0
                else:
                    student_answer_bbox = remove_duplicates(student_answer_bbox)   # remove duplicates
                    iou_results = sort_and_calculate_iou(ground_truth_bbox, student_answer_bbox)
                    reward = compute_reward_confidence(iou_results)
                    if reward>1:
                        reward = 1.0
                    if reward<0:
                        reward = 0.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        # import pdb; pdb.set_trace()
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))

            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward of Confidence: {reward} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"sol: {sol}\n")
                if show_flage==1:
                    
                    f.write(f"student_answer_bbox: {student_answer_bbox}\n")
                    f.write(f"ground_truth_bbox: {ground_truth_bbox}\n")
                    if student_answer_bbox!=None:
                        f.write(f"iou_results: {iou_results}\n")

                        # 画图存图
                        if 0:
                            image_path= img
                            f.write(f"image_path: {image_path}\n")
                            outres= f'{os.path.dirname(log_path)}/train_rollout_save/{cnt}-{os.path.basename(image_path)}'
                            cnt+=1
                            os.makedirs(os.path.dirname(outres), exist_ok=True)
                            try:
                                draw_boxes(image_path, [box['Position'] for box in student_answer_bbox], outres)
                            except:
                                pass
        show_flage = 0 
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    # pattern = r"<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    # matches = [re.match(pattern, content) for content in completion_contents]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]
