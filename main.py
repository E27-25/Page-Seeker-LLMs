import pandas as pd
import ollama
import json
from tqdm import tqdm

tqdm.pandas()

def stage1_fast_screening(row):
    """
    Stage 1: Binary classification using Qwen. 
    Enforced JSON output.
    """
    prompt = f"""วิเคราะห์ข้อมูลโฆษณานี้:
    ชื่อผู้ลง: {row['ad_name']}
    แคปชั่น: {row['ad_caption']}
    ลิ้งก์: {row['ad_links']}
    
    โฆษณานี้มีแนวโน้มที่จะเป็นข้อความหลอกลวง (Scam) หรือไม่?
    ตอบเป็น JSON เท่านั้น โดยมี key ชื่อ "is_scam" และ value เป็น boolean (true หรือ false)
    ตัวอย่าง: {{"is_scam": true}}"""
    
    try:
        response = ollama.chat(
            model='iapp/chinda-qwen3-4b', 
            messages=[{'role': 'user', 'content': prompt}],
            format='json' # Force JSON output
        )
        
        # Parse the JSON response
        result = json.loads(response['message']['content'])
        return result.get('is_scam', False)
        
    except Exception as e:
        print(f"Error in Stage 1 ID {row['id']}: {e}")
        return True # Default to True to force Stage 2 inspection on failure

def stage2_deep_analysis(row):
    """
    Stage 2: Deep analysis using Typhoon-2.
    Enforced JSON output for clean parsing.
    """
    prompt = f"""โฆษณาด้านล่างนี้ถูกตั้งข้อสงสัยว่าเป็น Scam จงวิเคราะห์อย่างละเอียด:
    ชื่อผู้ลง: {row['ad_name']}
    แคปชั่น: {row['ad_caption']}
    ลิ้งก์: {row['ad_links']}
    
    ระบุระดับความเสี่ยง และบอกเหตุผลที่ชัดเจนเป็นภาษาไทย
    ตอบเป็น JSON เท่านั้น โดยใช้รูปแบบนี้:
    {{
        "risk_level": "High" หรือ "Medium" หรือ "Low",
        "reason": "อธิบายเหตุผลสั้นๆ"
    }}"""
    
    try:
        response = ollama.chat(
            model='scb10x/llama3.1-typhoon2-8b-instruct', 
            messages=[{'role': 'user', 'content': prompt}],
            format='json' # Force JSON output
        )
        
        # Parse the JSON response
        result = json.loads(response['message']['content'])
        
        risk_level = result.get('risk_level', 'Unknown')
        reason = result.get('reason', 'ไม่สามารถดึงเหตุผลได้')
        
        return pd.Series([risk_level, reason])
        
    except json.JSONDecodeError:
        print(f"JSON Parse Error in Stage 2 ID {row['id']}")
        return pd.Series(["Error", "LLM did not return valid JSON"])
    except Exception as e:
        print(f"Error in Stage 2 ID {row['id']}: {e}")
        return pd.Series(["Error", str(e)])

def process_ad_pipeline(row):
    """The main orchestrator."""
    is_suspicious = stage1_fast_screening(row)
    
    if not is_suspicious:
        return pd.Series(["Low", "Normal advertisement, no obvious scam markers found."])
    
    return stage2_deep_analysis(row)

# ==========================================
# Execution
# ==========================================

df = pd.read_csv("meta_ad_response_rows.csv")[:100]
df_batch = df.copy()

print("Starting strict-JSON 2-Model Pipeline...")
df_batch[['calculated_risk_level', 'calculated_risk_reason']] = df_batch.progress_apply(process_ad_pipeline, axis=1)

print("\nCleaned Results:")
print(df_batch[['ad_name', 'calculated_risk_level', 'calculated_risk_reason']])

df_batch.to_csv("analyzed_ads_data.csv", index=False)