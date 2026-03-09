import pandas as pd
import ollama
import json
import asyncio
import aiohttp
from tqdm.asyncio import tqdm
from media_helper import parse_media_urls, fetch_and_encode_image_async
from verification_helper import build_verification_map
from rag_helper import SafeAdRAG

async def extract_visual_context_async(row, session, async_client):
    image_b64_list = []
    
    img_urls = parse_media_urls(row.get('feed_image_urls', ''))
    if img_urls:
        b64 = await fetch_and_encode_image_async(img_urls[0], session)
        if b64:
            image_b64_list.append(b64)
                
    if not image_b64_list:
        return ""
        
    try:
        response = await async_client.chat(
            model='deepseek-ocr:latest',
            messages=[{
                'role': 'user', 
                'content': "ถอดข้อความทั้งหมดที่เห็นในภาพ (OCR) ออกมา", 
                'images': image_b64_list
            }]
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error extracting visual context for ID {row.get('id', 'Unknown')}: {e}")
        return ""

async def stage1_fast_screening_async(row, async_client, is_verified=False, rag_text=""):
    """
    Stage 1: Text-only fast screening using iapp/chinda-qwen3-4b.
    Early Exit logic.
    """
    verification_text = "โปรไฟล์ได้รับการตรวจสอบยืนยันแล้ว (Verified Page/Account)" if is_verified else "บัญชีผู้ใช้นี้ยังไม่ได้รับการตรวจสอบ (Unverified)"
    
    prompt = f"""วิเคราะห์ข้อมูลโพสต์นี้ว่าน่าจะเป็นข้อความหลอกลวง (Scam) หรือไม่:
    ชื่อผู้โพสต์ (Name): {row.get('feed_name', 'N/A')}
    สถานะเพจ (Page Status): {verification_text}
    URL โปรไฟล์ (Profile URL): {row.get('feed_profile_urls', 'N/A')}
    แคปชั่น (Caption): {row.get('feed_caption', 'N/A')}
    เนื้อหาเต็ม (Raw Text): {row.get('feed_raw_text', 'N/A')}
    ลิ้งก์ในโพสต์ (Links): {row.get('feed_links', 'N/A')}{rag_text}
    
    คำแนะนำพิเศษ: 
    1. หากโพสต์นี้เป็นการขายสินค้าทั่วไปที่มีอยู่จริง (เช่น ปุ๋ย, อุปกรณ์การเกษตร, เสื้อผ้า, ของใช้ทั่วไป, อาหาร) วิดีโอคลิปตลก หรือการขายทั่วไป ให้ถือว่าโพสต์นี้เป็น "ปกติ (Low-Medium Risk)" และไม่เป็น Scam แม้ว่าชื่อเพจอาจจะดูแปลกๆ หรือไม่เกี่ยวข้องกันก็ตาม
    2. กลุ่มการลงทุน (Investment): ให้ถือว่าเป็น Scam ก็ต่อเมื่อมีลักษณะหลอกลวงชัดเจน เช่น โฆษณาให้ผลตอบแทนสูงเกินจริง การันตีรายได้ หรือแอบอ้างบุคคลมีชื่อเสียง แต่หากเป็นข้อมูลจากแพลตฟอร์มการเงินหรือธนาคารที่น่าเชื่อถือ (เช่น KBank, SCB, SET, แพลตฟอร์มเทรดที่ถูกกฎหมาย) ให้ระบุเป็นโพสต์ปกติ
    
    ตอบเป็น JSON เท่านั้น โดยมี key ชื่อ "is_scam" (boolean) 
    บอกว่าโพสต์นี้มีแนวโน้มเป็น Scam หรือไม่
    ตัวอย่าง: {{"is_scam": true}} หรือ {{"is_scam": false}}"""
    
    try:
        response = await async_client.chat(
            model='iapp/chinda-qwen3-4b', 
            messages=[{'role': 'user', 'content': prompt}],
            format='json'
        )
        result = json.loads(response['message']['content'])
        return result.get('is_scam', False)
    except Exception as e:
        # Default to True so it falls back to detailed analysis
        return True

async def stage2_deep_analysis_async(row, visual_desc, async_client, is_verified=False, rag_text=""):
    """
    Stage 2: Deep analysis using scb10x/llama3.1-typhoon2-8b-instruct.
    """
    visual_text = f"\n    รายละเอียดรูปภาพ (จาก AI): {visual_desc}" if visual_desc else ""
    verification_text = "โปรไฟล์ได้รับการตรวจสอบยืนยันแล้ว (Verified Page/Account)" if is_verified else "บัญชีผู้ใช้นี้ยังไม่ได้รับการตรวจสอบ (Unverified)"
    
    prompt = f"""วิเคราะห์โพสต์เพื่อหาความเสี่ยง Scam โดยลดความเข้มงวดลง ใช้เกณฑ์ดังนี้:
    1. ความไม่สอดคล้อง: อนุโลมให้เพจขายของทั่วไปที่อาจจะซื้อเพจเก่ามาทำใหม่ (หากเนื้อหาเป็นการขายสินค้าทั่วไป ให้ถือว่าความเสี่ยงต่ำ-ปานกลาง)
    2. การลงทุน (Investment): หากเป็นโฆษณาสถาบันการเงินที่น่าเชื่อถือ หรือแพลตฟอร์มเทรดที่ถูกกฎหมาย ถือว่าน่าเชื่อถือ แต่ถ้าเป็นการชวนลงทุนที่ได้ผลตอบแทนการันตีสูงเว่อร์ รวยไว หรือแอบอ้างคนดัง ถือว่าเป็น High Risk
    3. การกระตุ้นอารมณ์: ใช้ความกลัว หรือความโลภแบบเกินจริงแจกเงินฟรีๆ ส่วนคำโปรยโฆษณาสินค้าลดแลกแจกแถมปกติ "ไม่นับเป็น Scam"
    4. สินค้าทั่วไป: หากแอดโฆษณาเป็นการขายสินค้าทางกายภาพ (เช่น ปุ๋ย, เครื่องจักร, ของใช้ทั่วไป, อาหาร) โพสต์คลิปทั่วไป ให้วิเคราะห์เป็น Low หรือ Medium Risk เสมอ (เว้นแต่แอบอ้างแบรนด์ดังในราคาถูกเกินกว่าจะเป็นไปได้มาก)
    5. URL ที่น่าสงสัย: ลิงก์ย่อ หรือลิงก์ที่จงใจซ่อนหน้าเว็บจริงเพื่อดูดข้อมูล (Phishing)
    6. สิ่งที่เห็นในภาพ: {visual_text}

    ข้อมูลโพสต์:
    - ชื่อผู้โพสต์ (Name): {row.get('feed_name', 'N/A')}
    - สถานะเพจ (Page Status): {verification_text}
    - แคปชั่น (Caption): {row.get('feed_caption', 'N/A')}
    - เนื้อหาเต็ม (Raw Text): {row.get('feed_raw_text', 'N/A')}
    - URL ลิ้งก์ (Links): {row.get('feed_links', 'N/A')}
    - URL โปรไฟล์ (Profile URL): {row.get('feed_profile_urls', 'N/A')}{rag_text}
    
    ตอบเป็น JSON เท่านั้น โดยใช้รูปแบบนี้:
    {{
        "risk_level": "High" หรือ "Medium" หรือ "Low",
        "reason": "อธิบายเหตุผลสั้นๆ",
        "scam_type": "ประเภทของการหลอกลวง (เช่น Phishing, Investment Scam, Fake Product, ฯลฯ) หรือระบุ 'None' ถ้าปกติ"
    }}"""
    
    try:
        response = await async_client.chat(
            model='scb10x/llama3.1-typhoon2-8b-instruct',
            messages=[{'role': 'user', 'content': prompt}],
            format='json'
        )
        result = json.loads(response['message']['content'])
        
        risk_level = result.get('risk_level', 'Unknown')
        reason = result.get('reason', 'ไม่สามารถดึงเหตุผลได้')
        scam_type = result.get('scam_type', 'Unknown')
        
        return pd.Series([risk_level, reason, scam_type, visual_desc])
        
    except Exception as e:
        return pd.Series(["Error", str(e), "Error", visual_desc])

async def process_feed_pipeline_async(row, session, async_client, rag):
    """The main orchestrated async pipeline for feed items."""
    
    # We can check verified status directly from feed_is_verified column
    is_verified = bool(row.get('feed_is_verified', False))
    
    # HARD EARLY EXIT (Verified Page)
    if is_verified:
        return pd.Series(["Low", "Verified page. Automatically classified as Low Risk.", "None", "No visual extraction triggered (Verified page)"])
        
    # RAG INJECTION for Unverified Pages
    caption = str(row.get('feed_caption', 'N/A')).strip()
    safe_examples = rag.get_similar_safe_ads(caption) if rag else []
    rag_text = ""
    if safe_examples:
        rag_text = "\n\n    [ข้อมูลเปรียบเทียบ: ตัวอย่างแคปชั่นโพสต์/โฆษณาที่ปลอดภัยคล้ายคลึงกัน]\n" + "\n".join([f"    > {ex}" for ex in safe_examples]) + "\n    (หากแคปชั่นนี้มีรูปแบบคล้ายตัวอย่างที่ปลอดภัยด้านบน ให้ประเมินว่าเป็นโพสต์ปกติ)\n"
    
    # EARLY EXIT LOGIC: Stage 1 (Text-only)
    is_suspicious = await stage1_fast_screening_async(row, async_client, is_verified, rag_text)
    
    if not is_suspicious:
        # High confidence safe based on text alone
        return pd.Series(["Low", "Normal post based on fast screening. No obvious scam markers.", "None", "No visual extraction triggered (Low risk text)"])
    
    # If suspicious, extract visual context (if any)
    visual_desc = await extract_visual_context_async(row, session, async_client)
    if not visual_desc:
        visual_desc = "No valid media found or unable to extract."
        
    # Run deep analysis
    return await stage2_deep_analysis_async(row, visual_desc, async_client, is_verified, rag_text)

async def run_batch(df_batch, rag):
    async_client = ollama.AsyncClient()
    # Use a TCPConnector to limit concurrent connections and prevent overwhelming the server/network
    # Limits simultaneous LLM requests and network connections
    connector = aiohttp.TCPConnector(limit_per_host=5, limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [process_feed_pipeline_async(row, session, async_client, rag) for _, row in df_batch.iterrows()]
        
        # We can just use asyncio.gather and wrap it with tqdm for a progress bar
        results = await tqdm.gather(*tasks, desc="Processing Feeds Async")
        return results

def main():
    # Read the feed rows csv
    df = pd.read_csv("meta_feed_response_rows.csv")
    df_batch = df[:1000].copy() # Process up to 1000 rows

    # To build a comprehensive safe verification map, we use the standard verification helper
    print("Building verification map from feed data...")
    verified_map = build_verification_map("meta_feed_response_rows.csv")
    print(f"Loaded {len(verified_map)} verified pages.")

    print("Building RAG database of safe text...")
    df_ads = pd.read_csv("meta_ad_response_rows.csv")
    rag = SafeAdRAG()
    rag.build_index(df_ads, verified_map)

    print(f"Starting Asynchronous Pipeline on {len(df_batch)} feed rows...")
    
    results = asyncio.run(run_batch(df_batch, rag))
    
    # Assign results back to df safely 
    # Notice we map risk level back to feed_risk_level and feed_risk_reason
    df_batch['feed_risk_level'] = [r.iloc[0] if len(r) > 0 else "Error" for r in results]
    df_batch['feed_risk_reason'] = [r.iloc[1] if len(r) > 1 else "Unknown" for r in results]
    df_batch['calculated_scam_type'] = [r.iloc[2] if len(r) > 2 else "Unknown" for r in results]
    df_batch['visual_description_log'] = [r.iloc[3] if len(r) > 3 else "" for r in results]
    
    # SAME-PAGE EXCEPTION RULE (Post-Processing)
    # 1. Identify all feed_name that produced at least one "Low" risk post
    low_risk_pages = df_batch[df_batch['feed_risk_level'] == 'Low']['feed_name'].dropna().unique()
    
    # 2. Find any posts from those same pages that were flagged as High or Medium
    mask = df_batch['feed_name'].isin(low_risk_pages) & (df_batch['feed_risk_level'] != 'Low')
    
    # 3. Apply downgrade
    if mask.any():
        num_corrected = mask.sum()
        print(f"\nApplying Same-Page Exception Rule: Downgrading {num_corrected} posts to 'Low Risk' based on their safe sibling posts.")
        df_batch.loc[mask, 'feed_risk_reason'] += " [Same-Page Exception: Downgraded because another post from this page was deemed Safe]"
        df_batch.loc[mask, 'feed_risk_level'] = 'Low'
        df_batch.loc[mask, 'calculated_scam_type'] = 'None'
    
    print("\nCleaned Results:")
    print(df_batch[['feed_name', 'feed_risk_level', 'calculated_scam_type', 'visual_description_log']])
    
    df_batch.to_csv("analyzed_feeds_data.csv", index=False)

if __name__ == "__main__":
    main()
