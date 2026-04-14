#!/usr/bin/env python3
import sys
import json
import math
import argparse
from pathlib import Path

try:
    import pygame
except ImportError:
    print("【错误】未安装 pygame 库！")
    print("请运行以下命令安装：")
    print("pip install pygame")
    sys.exit(1)

# 动态导入项目的常量定义
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from SDK.utils.constants import MAP_PROPERTY, Terrain, PLAYER_BASES, MAP_SIZE
    from SDK.backend.engine import GameState, Tower, Ant
    from SDK.utils.constants import TowerType, AntStatus, AntKind, AntBehavior
    from SDK.utils.features import FeatureExtractor
except ImportError:
    print("无法导入 SDK 相关模块，请确保脚本在项目根目录或 tools 目录下运行。")
    sys.exit(1)

# 渲染常量
HEX_SIZE = 18
START_X = 150
START_Y = 100

# 颜色定义
C_BG = (30, 30, 30)
C_TEXT = (255, 255, 255)
C_PATH = (220, 220, 220)
C_BARRIER = (80, 80, 80)
C_P0_HIGH = (255, 180, 180)
C_P1_HIGH = (180, 180, 255)
C_P0 = (220, 50, 50)
C_P1 = (50, 50, 220)
C_P0_BASE = (255, 0, 0)
C_P1_BASE = (0, 0, 255)

TOWER_NAMES = {
    0: "Bsc", 1: "Hvy", 2: "Qck", 3: "Mrt", 4: "Prd",
    11: "Hvy+", 12: "Ice", 13: "Bwt",
    21: "Qck+", 22: "Dbl", 23: "Snp",
    31: "Mrt+", 32: "Pls", 33: "Msl",
    41: "Prd+", 42: "Sge", 43: "Mdc"
}

def get_pixel_pos(x, y):
    # odd-r pointy topped
    px = START_X + HEX_SIZE * math.sqrt(3) * (x + 0.5 * (y % 2))
    py = START_Y + HEX_SIZE * 1.5 * y
    return px, py

def draw_hex(surface, color, x, y, size=HEX_SIZE, width=0):
    px, py = get_pixel_pos(x, y)
    points = []
    for i in range(6):
        angle_deg = 60 * i - 30
        angle_rad = math.pi / 180 * angle_deg
        points.append((px + size * math.cos(angle_rad), py + size * math.sin(angle_rad)))
    pygame.draw.polygon(surface, color, points, width)
    return px, py

def parse_replay(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    frames = []
    current_towers = {}

    for turn in data:
        state = turn.get("round_state", {})
        if not state:
            continue

        # 增量更新防御塔
        for t in state.get("towers", []):
            if isinstance(t, dict):
                tid = t.get("id")
                ttype = t.get("type")
                if ttype == -1:
                    if tid in current_towers:
                        del current_towers[tid]
                else:
                    if tid not in current_towers:
                        current_towers[tid] = {}
                    current_towers[tid].update(t)
            elif isinstance(t, list) and len(t) >= 6:
                tid, player, x, y, ttype, hp = t[:6]
                if ttype == -1:
                    if tid in current_towers:
                        del current_towers[tid]
                else:
                    current_towers[tid] = {
                        "id": tid, "player": player, "type": ttype, "hp": hp,
                        "pos": {"x": x, "y": y}
                    }

        # 全量更新蚂蚁
        ants = state.get("ants", [])
        parsed_ants = []
        for a in ants:
            if isinstance(a, dict):
                parsed_ants.append(a)
            elif isinstance(a, list) and len(a) >= 9:
                parsed_ants.append({
                    "id": a[0], "player": a[1], "pos": {"x": a[2], "y": a[3]},
                    "hp": a[4], "level": a[5], "age": a[6], "status": a[7],
                    "behavior": a[8], "kind": a[9] if len(a) > 9 else 0
                })

        # 基地血量
        camps = state.get("camps", state.get("camps_hp", [50, 50]))
        if isinstance(camps, list) and len(camps) > 0 and isinstance(camps[0], dict):
            camps = [c.get("hp", 50) for c in camps]

        frames.append({
            "round": state.get("round_index", len(frames)),
            "coins": state.get("coins", [0, 0]),
            "camps_hp": camps,
            "anthp_lv": state.get("anthpLv", [0, 0]),
            "speed_lv": state.get("speedLv", [0, 0]),
            "weapon_cd": state.get("weaponCooldowns", [[0]*4, [0]*4]),
            "towers": {k: v.copy() for k, v in current_towers.items()},
            "ants": parsed_ants,
            "active_effects": state.get("activeEffects", [])
        })
        
    print("正在计算每回合估值 (FeatureExtractor)...")
    fe = FeatureExtractor()
    for frame in frames:
        gs = GameState.initial()
        gs.round_index = frame["round"]
        gs.coins = list(frame["coins"])
        gs.bases[0].hp = frame["camps_hp"][0]
        gs.bases[1].hp = frame["camps_hp"][1]
        gs.bases[0].ant_level = frame["anthp_lv"][0]
        gs.bases[1].ant_level = frame["anthp_lv"][1]
        gs.bases[0].generation_level = frame["speed_lv"][0]
        gs.bases[1].generation_level = frame["speed_lv"][1]
        
        for tid, t in frame["towers"].items():
            gs.towers.append(Tower(
                tower_id=int(tid), player=t["player"], x=t["pos"]["x"], y=t["pos"]["y"],
                tower_type=TowerType(t["type"]), cooldown_clock=0.0, hp=t.get("hp", -1)
            ))
            
        for a in frame["ants"]:
            hp = a.get("hp", 0)
            status = a.get("status", 0)
            if hp <= 0 or status in (2, 3):
                continue
            gs.ants.append(Ant(
                ant_id=a["id"], player=a["player"], x=a["pos"]["x"], y=a["pos"]["y"],
                hp=hp, level=a.get("level", 0), kind=AntKind(a.get("kind", 0)),
                age=a.get("age", 0), status=AntStatus(status)
            ))
            
        frame["eval_p0"] = fe.evaluate(gs, 0)
        frame["eval_p1"] = fe.evaluate(gs, 1)
        
    return frames

def render_frame(screen, font, ui_font, frame, max_frames, all_evals=None, max_abs_eval=1.0):
    screen.fill(C_BG)
    
    # 1. 绘制地图
    for x in range(MAP_SIZE):
        for y in range(MAP_SIZE):
            terrain = MAP_PROPERTY[x][y]
            if terrain == Terrain.VOID:
                continue
            color = C_PATH
            if terrain == Terrain.BARRIER: color = C_BARRIER
            elif terrain == Terrain.PLAYER0_HIGHLAND: color = C_P0_HIGH
            elif terrain == Terrain.PLAYER1_HIGHLAND: color = C_P1_HIGH
            
            if (x, y) == PLAYER_BASES[0]: color = C_P0_BASE
            if (x, y) == PLAYER_BASES[1]: color = C_P1_BASE
            
            draw_hex(screen, color, x, y)
            draw_hex(screen, (80, 80, 80), x, y, width=1) # 六边形边框
            
    # 2. 绘制超级武器区域
    for effect in frame.get("active_effects", []):
        if isinstance(effect, dict):
            # 可能是 {"x": 15, "y": 11, "type": 1} 或者嵌套 {"pos": {"x":15, "y":11}, "type": 1}
            ex = effect.get("x", effect.get("pos", {}).get("x", -1))
            ey = effect.get("y", effect.get("pos", {}).get("y", -1))
            etype = effect.get("type", 0)
        elif isinstance(effect, (list, tuple)) and len(effect) >= 4:
            etype, _, ex, ey = effect[:4]
        else:
            continue
            
        if ex != -1 and ey != -1:
            # 闪电(黄) / EMP(蓝) / 护盾(绿) / 回避(紫)
            ecolors = {1: (255, 255, 0), 2: (0, 191, 255), 3: (0, 255, 0), 4: (255, 0, 255)}
            draw_hex(screen, ecolors.get(etype, (255, 255, 255)), ex, ey, size=HEX_SIZE*3, width=3)

    # 3. 绘制防御塔
    for tid, t in frame["towers"].items():
        x, y = t["pos"]["x"], t["pos"]["y"]
        color = C_P0 if t["player"] == 0 else C_P1
        px, py = draw_hex(screen, color, x, y, size=HEX_SIZE-2)
        
        # 塔类型文字
        name = TOWER_NAMES.get(t["type"], "?")
        text = font.render(name, True, C_TEXT)
        screen.blit(text, text.get_rect(center=(px, py-4)))
        
        # 血量
        hp_text = font.render(str(t.get("hp", "")), True, (200, 255, 200))
        screen.blit(hp_text, hp_text.get_rect(center=(px, py+6)))

    # 4. 绘制蚂蚁
    for ant in frame["ants"]:
        hp = ant.get("hp", 0)
        status = ant.get("status", 0)
        if hp <= 0 or status in (2, 3): # 死亡或老死
            continue
        
        pos = ant.get("pos", {})
        x, y = pos.get("x", -1), pos.get("y", -1)
        if x == -1: continue
        
        px, py = get_pixel_pos(x, y)
        
        # 稍微偏移一点防止和塔完全重合
        offset = -6 if ant.get("player") == 0 else 6
        px += offset
        
        color = C_P0 if ant.get("player") == 0 else C_P1
        kind = ant.get("kind", 0)
        
        # 兵种：普通蚂蚁(圆)，战斗蚂蚁(方块)
        if kind == 1:
            pygame.draw.rect(screen, color, (px-5, py-5, 10, 10))
        else:
            pygame.draw.circle(screen, color, (int(px), int(py)), 5)
            
        # 状态边框：免控(金)、蛊惑(紫)、冻结(青)
        behavior = ant.get("behavior", 0)
        if behavior == 4:
            pygame.draw.circle(screen, (255, 215, 0), (int(px), int(py)), 7, 2)
        elif behavior == 3:
            pygame.draw.circle(screen, (255, 0, 255), (int(px), int(py)), 7, 2)
        if status == 4:
            pygame.draw.circle(screen, (0, 255, 255), (int(px), int(py)), 8, 2)
            
    # 5. 绘制顶部 UI 状态栏
    p0_cd = frame.get("weapon_cd", [[0]*4, [0]*4])[0]
    p1_cd = frame.get("weapon_cd", [[0]*4, [0]*4])[1]
    
    ui_texts = [
        f"Round: {frame['round']} / {max_frames-1}",
        f"P0(Red)  Coins:{frame['coins'][0]:<4} HP:{frame['camps_hp'][0]:<3} ALv:{frame.get('anthp_lv', [0,0])[0]} SLv:{frame.get('speed_lv', [0,0])[0]} CD:{p0_cd}",
        f"P1(Blue) Coins:{frame['coins'][1]:<4} HP:{frame['camps_hp'][1]:<3} ALv:{frame.get('anthp_lv', [0,0])[1]} SLv:{frame.get('speed_lv', [0,0])[1]} CD:{p1_cd}",
        "Controls: [Space] Play/Pause | [Left/Right] Step | [Up/Down] Speed | Click progress bar to seek"
    ]
    for i, text in enumerate(ui_texts):
        surf = ui_font.render(text, True, C_TEXT)
        screen.blit(surf, (20, 20 + i * 25))

    # 6. 绘制进度条
    bar_x, bar_y, bar_w, bar_h = 20, 660, 860, 20
    pygame.draw.rect(screen, (80, 80, 80), (bar_x, bar_y, bar_w, bar_h)) # 背景
    
    if max_frames > 1:
        progress = frame['round'] / (max_frames - 1)
        pygame.draw.rect(screen, (150, 150, 150), (bar_x, bar_y, int(bar_w * progress), bar_h)) # 已播放进度
        
        # 滑块
        slider_x = bar_x + int(bar_w * progress)
        pygame.draw.rect(screen, (220, 220, 220), (slider_x - 5, bar_y - 5, 10, bar_h + 10))

    # 7. 绘制形势估计曲线
    if all_evals:
        graph_x, graph_y, graph_w, graph_h = 650, 20, 230, 100
        # 黑色半透明背景框
        bg_surf = pygame.Surface((graph_w, graph_h))
        bg_surf.set_alpha(200)
        bg_surf.fill((50, 50, 50))
        screen.blit(bg_surf, (graph_x, graph_y))
        pygame.draw.rect(screen, (150, 150, 150), (graph_x, graph_y, graph_w, graph_h), 1)
        
        # 绘制 0 线
        zero_y = graph_y + graph_h // 2
        pygame.draw.line(screen, (100, 100, 100), (graph_x, zero_y), (graph_x + graph_w, zero_y), 1)
        
        points = []
        for i, val in enumerate(all_evals):
            px = graph_x + (i / max(1, len(all_evals) - 1)) * graph_w
            # 裁剪 val 以防突破边框
            val_clipped = max(min(val, max_abs_eval), -max_abs_eval)
            py = zero_y - (val_clipped / max_abs_eval) * (graph_h / 2 - 5)
            points.append((px, py))
            
        if len(points) >= 2:
            pygame.draw.lines(screen, (255, 255, 0), False, points, 2)
            
        # 当前回合游标
        curr_x = graph_x + (frame['round'] / max(1, max_frames - 1)) * graph_w
        pygame.draw.line(screen, (255, 255, 255), (curr_x, graph_y), (curr_x, graph_y + graph_h), 2)
        
        # 文字
        eval_text = font.render(f"P0 Eval: {frame.get('eval_p0', 0):.1f}", True, C_TEXT)
        screen.blit(eval_text, (graph_x + 5, graph_y + 5))

def get_frame_from_mouse(mouse_x, bar_x, bar_w, max_frames):
    if max_frames <= 1: return 0
    rel_x = max(0, min(mouse_x - bar_x, bar_w))
    progress = rel_x / bar_w
    return int(progress * (max_frames - 1))

def main():
    parser = argparse.ArgumentParser(description="Ant-Game Local Replay Player")
    parser.add_argument("replay", type=str, nargs='?', default="local_match_output/replay.json", help="Path to replay.json")
    args = parser.parse_args()
    
    if not Path(args.replay).exists():
        print(f"【错误】找不到回放文件: {args.replay}")
        sys.exit(1)
        
    print("正在解析回放文件...")
    frames = parse_replay(args.replay)
    if not frames:
        print("【错误】回放文件解析失败或没有有效的帧！")
        sys.exit(1)
        
    pygame.init()
    screen = pygame.display.set_mode((900, 700))
    pygame.display.set_caption(f"Ant-Game Local Player - {args.replay}")
    
    # 字体
    font = pygame.font.SysFont("consolas,arial", 12, bold=True)
    ui_font = pygame.font.SysFont("consolas,arial", 20, bold=True)
    
    clock = pygame.time.Clock()
    running = True
    playing = False
    dragging = False
    current_frame = 0
    fps = 10
    
    print("解析完成！已启动图形界面。")
    
    all_evals = [f.get("eval_p0", 0) for f in frames]
    non_terminal = [abs(e) for e in all_evals if abs(e) < 5000]
    max_abs_eval = max(non_terminal) if non_terminal else 1.0
    max_abs_eval = max(max_abs_eval * 1.1, 10.0) # padding

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    playing = not playing
                elif event.key == pygame.K_RIGHT:
                    current_frame = min(current_frame + 1, len(frames) - 1)
                elif event.key == pygame.K_LEFT:
                    current_frame = max(current_frame - 1, 0)
                elif event.key == pygame.K_UP:
                    fps = min(fps + 5, 60)
                elif event.key == pygame.K_DOWN:
                    fps = max(fps - 5, 1)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # 左键
                    mx, my = event.pos
                    # 进度条区域: x: 20-880, y: 650-690 (稍微扩大点击范围)
                    if 20 <= mx <= 880 and 650 <= my <= 690:
                        dragging = True
                        current_frame = get_frame_from_mouse(mx, 20, 860, len(frames))
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    mx, my = event.pos
                    current_frame = get_frame_from_mouse(mx, 20, 860, len(frames))
                    
        if playing and not dragging:
            current_frame = min(current_frame + 1, len(frames) - 1)
            if current_frame == len(frames) - 1:
                playing = False
                
        render_frame(screen, font, ui_font, frames[current_frame], len(frames), all_evals, max_abs_eval)
        pygame.display.flip()
        clock.tick(fps)
        
    pygame.quit()

if __name__ == "__main__":
    main()