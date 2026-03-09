import pandas as pd
import ollama
import json
from tqdm import tqdm
from media_helper import parse_media_urls, fetch_and_encode_image, extract_frames_from_video

tqdm.pandas()

def extract_visual_context(row):
    image_b64_list = []
    
    img_urls = parse_media_urls(row.get('ad_image_urls', ''))
    if img_urls:
        b64 = fetch_and_encode_image(img_urls[0]) # Just take the first image
        if b64:
            image_b64_list.append(b64)
    elif not image_b64_list:
        vid_urls = parse_media_urls(row.get('ad_video_urls', ''))
        if vid_urls:
            frames = extract_frames_from_video(vid_urls[0], num_frames=1)
            if frames:
                image_b64_list.append(frames[0])
                
    if not image_b64_list:
        return ""
        
    try:
        response = ollama.chat(
            model='qwen3.5:9b',
            messages=[{
                'role': 'user', 
                'content': 'อธิบายสิ่งที่คุณเห็นในภาพนี้หรือรูปภาพโฆษณานี้อย่างละเอียดเป็นภาษาไทย', 
                'images': image_b64_list
            }]
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error extracting visual context for ID {row.get('id', 'Unknown')}: {e}")
        return ""

def stage1_fast_screening(row, visual_desc):
    """
    Stage 1: Binary classification using iapp/chinda-qwen3-4b.
    Enforced JSON output.
    """
    visual_text = f"\n    รายละเอียดรูปภาพ/วิดีโอ (จาก AI): {visual_desc}" if visual_desc else ""
    
    prompt = f"""วิเคราะห์ข้อมูลโฆษณานี้:
    ชื่อผู้ลง (Ad Name): {row.get('ad_name', 'N/A')}
    คีย์เวิร์ดที่ใช้ (Keyword): {row.get('keyword', 'N/A')}
    URL ของเพจ (Page URL): {row.get('page_url', 'N/A')}
    ระยะเวลาเปิดโฆษณา (Active Time Hours): {row.get('active_time_hr', 'N/A')}
    วันที่ลงโฆษณา (Ad Date): {row.get('ad_date', 'N/A')}
    URL โปรไฟล์ (Profile URL): {row.get('ad_profile_url', 'N/A')}
    แคปชั่น (Caption): {row.get('ad_caption', 'N/A')}
    ลิ้งก์ในโฆษณา (Links): {row.get('ad_links', 'N/A')}{visual_text}
    
    โฆษณานี้มีแนวโน้มที่จะเป็นข้อความหลอกลวง (Scam) หรือไม่?
    ตอบเป็น JSON เท่านั้น โดยมี key ชื่อ "is_scam" และ value เป็น boolean (true หรือ false)
    ตัวอย่าง: {{"is_scam": true}}"""
    
    try:
        response = ollama.chat(
            model='iapp/chinda-qwen3-4b', # Use Chinda for fast screening
            messages=[{'role': 'user', 'content': prompt}],
            format='json' # Force JSON output
        )
        
        result = json.loads(response['message']['content'])
        return result.get('is_scam', False)
        
    except Exception as e:
        print(f"Error in Stage 1 ID {row.get('id', 'Unknown')}: {e}")
        return True # Default to True to force Stage 2 inspection on failure

def stage2_deep_analysis(row, visual_desc):
    """
    Stage 2: Deep analysis using Typhoon-2.
    Enforced JSON output for clean parsing.
    """
    visual_text = f"\n    รายละเอียดรูปภาพ/วิดีโอ (จาก AI): {visual_desc}" if visual_desc else ""
    
    prompt = f"""โฆษณาด้านล่างนี้ถูกตั้งข้อสงสัยว่าเป็น Scam จงวิเคราะห์อย่างละเอียด:
    ชื่อผู้ลง (Ad Name): {row.get('ad_name', 'N/A')}
    คีย์เวิร์ดที่ใช้ (Keyword): {row.get('keyword', 'N/A')}
    URL ของเพจ (Page URL): {row.get('page_url', 'N/A')}
    ระยะเวลาเปิดโฆษณา (Active Time Hours): {row.get('active_time_hr', 'N/A')}
    วันที่ลงโฆษณา (Ad Date): {row.get('ad_date', 'N/A')}
    URL โปรไฟล์ (Profile URL): {row.get('ad_profile_url', 'N/A')}
    แคปชั่น (Caption): {row.get('ad_caption', 'N/A')}
    ลิ้งก์ในโฆษณา (Links): {row.get('ad_links', 'N/A')}{visual_text}
    
    ระบุระดับความเสี่ยง และบอกเหตุผลที่ชัดเจนเป็นภาษาไทย
    ตอบเป็น JSON เท่านั้น โดยใช้รูปแบบนี้:
    {{
        "risk_level": "High" หรือ "Medium" หรือ "Low",
        "reason": "อธิบายเหตุผลสั้นๆ"
    }}"""
    
    try:
        response = ollama.chat(
            model='scb10x/llama3.1-typhoon2-8b-instruct', # Use Typhoon-2 
            messages=[{'role': 'user', 'content': prompt}],
            format='json' # Force JSON output
        )
        
        # Parse the JSON response
        result = json.loads(response['message']['content'])
        
        risk_level = result.get('risk_level', 'Unknown')
        reason = result.get('reason', 'ไม่สามารถดึงเหตุผลได้')
        
        return pd.Series([risk_level, reason])
        
    except json.JSONDecodeError:
        print(f"JSON Parse Error in Stage 2 ID {row.get('id', 'Unknown')}")
        return pd.Series(["Error", "LLM did not return valid JSON"])
    except Exception as e:
        print(f"Error in Stage 2 ID {row.get('id', 'Unknown')}: {e}")
        return pd.Series(["Error", str(e)])

def process_ad_pipeline(row):
    """The main orchestrator."""
    visual_desc = extract_visual_context(row)
    
    is_suspicious = stage1_fast_screening(row, visual_desc)
    
    if not is_suspicious:
        return pd.Series(["Low", "Normal advertisement, no obvious scam markers found."])
    
    return stage2_deep_analysis(row, visual_desc)

# ==========================================
# Execution
# ==========================================

df = pd.read_csv("meta_ad_response_rows.csv")[:1000]
df_batch = df.copy()

print("Starting strict-JSON Ensembled 3-Model Pipeline...")
df_batch[['calculated_risk_level', 'calculated_risk_reason']] = df_batch.progress_apply(process_ad_pipeline, axis=1)

print("\nCleaned Results:")
print(df_batch[['ad_name', 'calculated_risk_level', 'calculated_risk_reason']])

df_batch.to_csv("analyzed_ads_data.csv", index=False)