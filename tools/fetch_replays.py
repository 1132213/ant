import os
import time
import json
import random
import requests
import zstandard as zstd
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==========================================
# 抓取配置
# ==========================================
# 填入你想抓取的目标列表，格式为 (用户名, 模型名, 版本号)
# 如果不限制模型名或版本号，可以填 None
TARGET_MODELS = [
    ("zyfkid", "zyfkid", 18),
    ("Xishiqing", "nnn", 2),
    ("zzy25","2_rl",3),
    ("abcd1235","神经",1),
    ("waxray","Claude",12)
    # ("用户名", "模型名", 2),
]

# 每个目标组合最多下载多少局
TARGET_COUNT_PER_MODEL = 20

# 对局列表 API（支持分页 limit 和 offset）
LIST_API_URL = "https://api.saiblo.net/api/matches/?limit=20&offset={offset}&username={username}"

# 根据之前的推断，直接下载 JSON 的地址应该是这个 (但现在我们发现 API 直接返回了 url 字段)
# 如果 API 里的 url 不能直接用，我们还是可以用这个模板
REPLAY_URL_TEMPLATE = "https://api.saiblo.net{url}"

# 3. 把你在 F12 Network 面板里抓到的 Request Headers 里的 Cookie 复制过来
# 这是平台鉴权必须的，否则他不会让你下载天梯高手的回放。
AUTHORIZATION_TOKEN = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzc2MzU4MDQyLCJpYXQiOjE3NzU3NTMyNDIsImp0aSI6ImQ2Nzc3M2Y3YjU0YzRiNzViZGRlYjdlOWJlNjMyMTEyIiwidXNlcl9pZCI6NDYzM30.mLzI8KWIrxjr4nHr-BEs6GCiwgl_dtHZEiteeRHr1yo"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
]

def get_session():
    """创建一个带有自动重试机制的 requests.Session"""
    session = requests.Session()
    # 遇到 429 (Too Many Requests), 500, 502, 503, 504 时自动重试，最多 3 次，带指数退避
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# 全局 session 实例
session = get_session()

def get_random_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json",
        "Authorization": AUTHORIZATION_TOKEN,
        "Accept-Encoding": "gzip, deflate, br, zstd",
    }

SAVE_DIR = Path(__file__).resolve().parents[1] / "replays_online"

def fetch_match_list(username: str, offset: int):
    """获取指定用户的对局列表"""
    url = LIST_API_URL.format(username=username, offset=offset)
    print(f"正在请求对局列表: {url}")
    try:
        response = session.get(url, headers=get_random_headers(), timeout=15)
        response.raise_for_status()
        
        # 检查是否被 zstd 压缩了
        content_encoding = response.headers.get("content-encoding", "")
        raw_data = response.content
        
        if "zstd" in content_encoding.lower():
            try:
                dctx = zstd.ZstdDecompressor()
                # 如果 zstd 数据帧头部没有包含原始大小，直接 decompress 会报错。
                # 传入 max_output_size 允许它分配足够大的缓冲区进行解压 (10MB 足够一个回放 JSON 了)
                raw_data = dctx.decompress(raw_data, max_output_size=10 * 1024 * 1024)
            except Exception as e:
                print(f"列表数据 Zstd 解压失败: {e}")
                return []
                
        try:
            json_str = raw_data.decode("utf-8")
            data = json.loads(json_str)
        except json.JSONDecodeError:
            print(f"获取对局列表失败：服务器返回的不是有效的 JSON。")
            print(f"返回内容前 200 个字符: {raw_data[:200]}")
            return []
        
        # 【重要】根据真实返回的 JSON 结构调整这里的解析逻辑
        if "results" in data:
            return data["results"]
        elif "data" in data and "matches" in data["data"]:
            return data["data"]["matches"]
        else:
            print(f"无法从返回数据中解析出 matches 列表。返回数据示例：{str(data)[:200]}")
            return []
            
    except Exception as e:
        print(f"获取对局列表请求异常 ({username}, offset {offset}): {e}")
        return []

def download_replay(match_id: str, save_path: Path, download_url: str):
    """下载单个对局的回放文件（处理 zstd 压缩）"""
    if save_path.exists():
        print(f"回放 {match_id} 已存在，跳过。")
        return True

    # 尝试解析并使用直接链接
    url = REPLAY_URL_TEMPLATE.format(url=download_url) if download_url.startswith("/") else download_url
    
    try:
        response = session.get(url, headers=get_random_headers(), timeout=20, stream=True)
        response.raise_for_status()
        
        # 检查响应头里有没有刚才你发给我的 content-encoding: zstd
        content_encoding = response.headers.get("content-encoding", "")
        
        raw_data = response.content
        
        # 核心：解压 zstd 数据
        if "zstd" in content_encoding.lower():
            try:
                dctx = zstd.ZstdDecompressor()
                # 同样，对于回放文件，如果缺失头部大小，我们预留一个较大的缓冲区（例如 20MB，以防神仙局太长）
                raw_data = dctx.decompress(raw_data, max_output_size=20 * 1024 * 1024)
            except Exception as e:
                print(f"Zstd 解压失败 (Match {match_id}): {e}")
                return False
                
        # 确保能解析成正常的 JSON，排除 HTML 报错页面的可能
        try:
            json_str = raw_data.decode("utf-8")
            json_data = json.loads(json_str)
        except json.JSONDecodeError:
            print(f"下载的数据不是有效的 JSON (Match {match_id})，可能是报错信息，跳过。")
            return False

        # 保存到本地
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False)
            
        print(f"成功下载回放: {match_id} -> {save_path.name}")
        return True
        
    except Exception as e:
        print(f"下载回放失败 (Match {match_id}): {e}")
        return False

def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    total_downloaded = 0
    
    # 全局去重集合，防止 A 和 B 对战的同一局被重复下载
    downloaded_match_ids = set()

    print(f"准备开始抓取，目标列表: {TARGET_MODELS}")
    
    # 提前提取出所有目标用户的用户名，用于快速判断对手是否在名单内
    target_usernames = {t[0] for t in TARGET_MODELS if len(t) == 3}

    for target in TARGET_MODELS:
        if len(target) == 3:
            target_username, expected_model_name, expected_version = target
        else:
            print(f"配置错误，跳过 {target}，格式必须为 (用户名, 模型名, 版本)")
            continue

        downloaded_count = 0
        offset = 0
        desc = f"{target_username} (模型: {expected_model_name}, 版本: {expected_version})"
        print(f"\n--- 开始抓取 {desc} 的对局 ---")
        
        while downloaded_count < TARGET_COUNT_PER_MODEL:
            matches = fetch_match_list(target_username, offset)
            if not matches:
                print(f"{desc} 没有更多对局了。")
                break

            for match in matches:
                if downloaded_count >= TARGET_COUNT_PER_MODEL:
                    break
                    
                match_id = str(match.get("id"))
                
                # 0. 全局去重：如果已经在内存白名单中（说明刚才下载过，或者之前检测到本地已有）
                if match_id in downloaded_match_ids:
                    print(f"对局 {match_id} 已在本次运行或本地缓存中，秒速跳过。")
                    downloaded_count += 1
                    continue
                
                # 1. 检查本地硬盘是否已经有这个文件
                existing_files = list(SAVE_DIR.glob(f"match_{match_id}_*.json"))
                if existing_files:
                    print(f"对局 {match_id} 已存在于本地硬盘 ({existing_files[0].name})，加入内存缓存。")
                    downloaded_match_ids.add(match_id)
                    downloaded_count += 1
                    continue

                # 2. 检查全局评测状态
                if match.get("state") != "评测成功":
                    print(f"对局 {match_id} 整体状态异常 ({match.get('state')})，跳过。")
                    continue
                
                # 3. 深入 info 数组检查具体玩家情况
                target_version = "未知"
                target_model_name = "无"
                all_players_ok = True
                found_target = False
                both_players_in_target = True
                player_names = []  # 新增：提取两名玩家的名字，用于统一文件命名
                
                for info in match.get("info", []):
                    # 强制校验：任何一方的 end_state 不是 "OK"，这局作废
                    if info.get("end_state") != "OK":
                        all_players_ok = False
                        print(f"对局 {match_id} 有玩家状态异常 ({info.get('end_state')})，跳过。")
                        break
                        
                    user_info = info.get("user", {})
                    current_username = user_info.get("username")
                    
                    if current_username:
                        player_names.append(current_username)
                    
                    # 检查当前这名玩家是否在我们的全局目标名单中
                    if current_username not in target_usernames:
                        both_players_in_target = False
                        # 我们可以选择提前 break，但为了拿到完整的 print 信息，这里继续循环
                        
                    if current_username == target_username:
                        code_info = info.get("code", {})
                        target_version = code_info.get("version", "未知")
                        target_model_name = code_info.get("entity", "无") # 使用 entity 作为模型名
                        found_target = True

                if not all_players_ok or not found_target:
                    continue
                    
                # 【新增校验】如果双方有一方不在大名单里，直接跳过
                if not both_players_in_target:
                    print(f"对局 {match_id} 对手不在目标名单中，跳过。")
                    continue
                
                # 4. 校验我们指定的“模型名”和“版本”
                if expected_model_name is not None and target_model_name != expected_model_name:
                    print(f"对局 {match_id} 模型名不匹配 (实际: {target_model_name} != 期望: {expected_model_name})，跳过。")
                    continue
                    
                if expected_version is not None and target_version != expected_version:
                    print(f"对局 {match_id} 版本不匹配 (实际: {target_version} != 期望: {expected_version})，跳过。")
                    continue
                
                print(f"命中对局 {match_id} | 玩家: {' vs '.join(player_names)}")
                
                download_url = match.get("url")
                if not download_url:
                    print(f"无法找到对局 {match_id} 的下载链接，跳过。")
                    continue

                # 统一文件名：为了去重和方便查找，我们把对战双方都写在文件名里，并把 ID 放前面
                # 排序玩家名字，保证 A vs B 和 B vs A 生成的文件名完全一致！
                sorted_players = "_vs_".join(sorted(player_names))
                save_path = SAVE_DIR / f"match_{match_id}_{sorted_players}.json"
                
                success = download_replay(match_id, save_path, download_url)
                
                if success:
                    # 加入全局去重集合
                    downloaded_match_ids.add(match_id)
                    downloaded_count += 1
                    total_downloaded += 1
                    print(f"进度: {downloaded_count}/{TARGET_COUNT_PER_MODEL}")
                    # 引入随机休眠抖动，防止被封 IP
                    time.sleep(random.uniform(1.5, 3.5))
                    
            offset += 20
            time.sleep(random.uniform(2.0, 4.0))

    print(f"\n全部抓取完成！共下载 {total_downloaded} 局回放，保存在 {SAVE_DIR} 目录下。")

if __name__ == "__main__":
    if "Bearer eyJhb" not in AUTHORIZATION_TOKEN:
        print("【错误】请先在脚本中填入你的 Saiblo Authorization Token 和真实的 API 地址！")
    else:
        main()
