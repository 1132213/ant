import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import setproctitle
except ImportError:
    setproctitle = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SDK.alphazero import PolicyValueNet, build_policy_value_net, PolicyValueNetConfig
from SDK.training.alphazero import SelfPlayBatch
from SDK.utils.features import FeatureExtractor
from SDK.utils.actions import ActionCatalog
from SDK.backend.engine import GameState
from SDK.backend.model import Operation
from SDK.backend.state import PythonBackendState
from SDK.utils.constants import OperationType, TowerType, SuperWeaponType

def _parse_operation_from_replay(action_data) -> Operation | None:
    """将回放中的 action 转换为内部 Operation 对象"""
    # 如果 action_data 是一个空字典或者空列表，说明没有操作
    if not action_data:
        return None
        
    # 赛博平台返回的动作通常是放在一个 list 里
    action_dict = action_data[0] if isinstance(action_data, list) and len(action_data) > 0 else action_data
    
    if not isinstance(action_dict, dict):
        return None

    # 这里的 type 是一个数字 (对应 OperationType 的 IntEnum)
    action_type = action_dict.get("type", 0)
    
    try:
        op_type = OperationType(action_type)
    except ValueError:
        return None
    
    if op_type == OperationType.BUILD_TOWER:
        pos = action_dict.get("pos", {})
        return Operation(op_type, pos.get("x", 0), pos.get("y", 0))
    elif op_type == OperationType.UPGRADE_TOWER:
        # 在原版常量里 UPGRADE_TOWER 需要 tower_id 和 tower_type
        # 赛博平台的 JSON 里：id 存的是 tower_id，args 存的是目标 tower_type
        return Operation(op_type, action_dict.get("id", 0), action_dict.get("args", 0))
    elif op_type == OperationType.DOWNGRADE_TOWER:
        return Operation(op_type, action_dict.get("id", 0))
    elif op_type in (OperationType.UPGRADE_GENERATED_ANT, OperationType.UPGRADE_GENERATION_SPEED):
        return Operation(op_type)
    elif op_type in (OperationType.USE_LIGHTNING_STORM, OperationType.USE_EMP_BLASTER, 
                     OperationType.USE_DEFLECTOR, OperationType.USE_EMERGENCY_EVASION):
        pos = action_dict.get("pos", {})
        return Operation(op_type, pos.get("x", 0), pos.get("y", 0))
        
    return None

def _match_action_to_bundle_index(catalog: ActionCatalog, state, player: int, target_op: Operation | None) -> int:
    """在当前状态的所有合法动作包中，找到与录像动作最匹配的一个"""
    bundles = catalog.build(state, player)
    
    # 如果录像里没有动作，返回 hold (通常是索引 0)
    if target_op is None:
        return 0
        
    # 尝试匹配具体的动作
    for i, bundle in enumerate(bundles):
        for op in bundle.operations:
            if op.op_type == target_op.op_type and op.arg0 == target_op.arg0 and op.arg1 == target_op.arg1:
                return i
                
    # 如果在合法动作里没找到（可能因为状态不一致或剪枝了），回退到 hold
    return 0

def process_replay_file(filepath: Path, max_actions: int, target_user: str | None = None) -> list[tuple]:
    """
    解析单个回放文件，提取 (obs, mask, policy, value) 元组列表
    在多进程环境中，必须在进程内部重新实例化 FeatureExtractor 和 ActionCatalog。
    """
    try:
        # 子进程内部实例化组件
        feature_extractor = FeatureExtractor(max_actions=max_actions)
        catalog = ActionCatalog(max_actions=max_actions, feature_extractor=feature_extractor)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            turns = json.load(f)
    except Exception as e:
        print(f"读取文件失败 {filepath}: {e}")
        return []

    if not turns:
        return []
        
    # 如果装了 setproctitle，我们在处理每个文件时把进程名改一下，方便 top 命令监控
    if setproctitle:
        setproctitle.setproctitle("python")

    # 判断胜负，用于计算 Value Target (Z)
    last_turn = turns[-1]
    last_state = last_turn.get("round_state", {})
    camps = last_state.get("camps", last_state.get("camps_hp", [50, 50]))
    
    hp_p0 = camps[0].get("hp", 0) if isinstance(camps[0], dict) else camps[0]
    hp_p1 = camps[1].get("hp", 0) if isinstance(camps[1], dict) else camps[1]
    
    if hp_p0 <= 0 and hp_p1 > 0:
        winner = 1
    elif hp_p1 <= 0 and hp_p0 > 0:
        winner = 0
    else:
        print(f"跳过平局/无法判断胜负的对局: {filepath.name}")
        return []

    samples = []
    
    # 获取需要学习的玩家视角
    # 文件名格式: match_id_playerA_vs_playerB.json 或者带有特定目标用户的旧格式
    # 由于文件名比较复杂，我们可以直接学习胜利方的操作（这是最好的）
    # 如果指定了 target_user，我们就只学习这个高手的操作
    learning_players = []
    if target_user is None:
        # 如果不指定高手，默认只学习胜利者（只学成功经验）
        learning_players = [winner]
    else:
        # 这个需要在实际工程中根据 JSON 里的玩家映射去匹配 player=0 或 1
        # 在赛博的 JSON 里，通常在 `info` 或者第一回合里能看出来，
        # 为了通用，如果指定了高手，我们假设他赢了才学，输了也学（但是反向 label）
        # 这里简化：只学习胜利方的。如果高手输了，我们这局其实可以扔掉。
        learning_players = [winner]

    # 初始化引擎
    # 这里我们通过解析第一回合获得 seed
    first_state = turns[0].get("round_state", {})
    seed = first_state.get("seed", 0)
    
    gs = GameState.initial(seed=seed)
    backend_state = PythonBackendState(gs)

    # 遍历回合
    for turn_idx, turn in enumerate(turns):
        op0_raw = turn.get("op0", {})
        op1_raw = turn.get("op1", {})
            
        op_p0 = _parse_operation_from_replay(op0_raw)
        op_p1 = _parse_operation_from_replay(op1_raw)
            
        # 收集要学习的视角的样本（在结算前收集特征）
        for player in learning_players:
            target_op = op_p0 if player == 0 else op_p1
            
            # 【核心优化】：过滤掉大量的“发呆”回合 (None/WAIT)
            # 如果没有做任何操作，就没必要让神经网络强行去学“在这里应该发呆”
            # 因为发呆大概率是被迫的（没钱），或者正在跑路。
            # 只提取真正做出了建筑、升级、放技能等主动决策的回合
            if target_op is None:
                continue

            # 1. 提取合法动作掩码
            bundles = catalog.build(backend_state, player)
            mask = catalog.action_mask(bundles).astype(np.float32)

            # 2. 提取当前状态的特征张量
            obs_dict = feature_extractor.encode_observation(backend_state, player, mask)
            obs = feature_extractor.flatten_observation(obs_dict)
            
            # 3. 匹配真实动作
            action_index = _match_action_to_bundle_index(catalog, backend_state, player, target_op)
            
            # 4. 生成 One-hot 策略
            policy = np.zeros(catalog.max_actions, dtype=np.float32)
            policy[action_index] = 1.0
            
            # 5. 确定价值 Z (因为我们目前只收集 winner，所以这里肯定是 +1.0)
            value = 1.0 if winner == player else -1.0
            
            samples.append((obs, mask, policy, value))
            
        # 引擎结算当前回合，推进到下一状态
        backend_state.resolve_turn(
            operations0=(op_p0,) if op_p0 else (),
            operations1=(op_p1,) if op_p1 else ()
        )
        backend_state.advance_round()
            
    return samples

def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero from Online Replays")
    parser.add_argument("--replays-dir", type=str, default="replays_online", help="目录包含抓取的 json 回放文件")
    parser.add_argument("--epochs", type=int, default=5, help="在回放数据集上训练多少轮")
    parser.add_argument("--lr", type=float, default=1e-3, help="监督学习的学习率")
    parser.add_argument("--max-actions", type=int, default=96, help="动作空间大小")
    parser.add_argument("--save-path", type=str, default="checkpoints/ai_alphazero_sl_pretrained.npz", help="保存预训练权重的路径")
    parser.add_argument("--resume", type=str, default=None, help="基于某个现有的 checkpoint 继续训练")
    args = parser.parse_args()

    # 在主进程开始时，设置进程名
    if setproctitle:
        setproctitle.setproctitle("python")

    replay_dir = Path(args.replays_dir)
    if not replay_dir.exists():
        print(f"【错误】找不到回放目录: {replay_dir}")
        sys.exit(1)
        
    print(f"扫描回放目录: {replay_dir}")
    json_files = list(replay_dir.glob("*.json"))
    print(f"找到 {len(json_files)} 个回放文件。")
    
    if len(json_files) == 0:
        return
        
    print("\n【注意】为了将 JSON 转换为张量，我们需要在后台重新跑一遍对局来收集精确的特征。")
    print("这部分代码需要调用 SDK.backend.engine.GameState...")
    
    # 多进程处理录像文件
    all_samples = []
    
    # 动态确定可以使用的进程数，通常等于逻辑 CPU 核心数
    max_workers = os.cpu_count() or 4
    print(f"\n🚀 启动多进程解析，工作线程数: {max_workers}")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 将所有的解析任务提交到线程池
        future_to_file = {
            executor.submit(process_replay_file, f, args.max_actions, None): f 
            for f in json_files
        }
        
        # 收集结果
        completed_count = 0
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            completed_count += 1
            try:
                samples = future.result()
                if samples:
                    all_samples.extend(samples)
                    print(f"[{completed_count}/{len(json_files)}] 解析完成: {file_path.name} -> 提取了 {len(samples)} 条有效操作样本")
                else:
                    print(f"[{completed_count}/{len(json_files)}] 跳过无样本对局: {file_path.name}")
            except Exception as exc:
                print(f"[{completed_count}/{len(json_files)}] 解析报错 {file_path.name}: {exc}")
            
    if not all_samples:
        print("【错误】未能从回放中提取出任何有效样本！")
        return
    
    # 构建模型，此时再实例化 FeatureExtractor
    feature_extractor = FeatureExtractor(max_actions=args.max_actions)
    
    if args.resume and Path(args.resume).exists():
        print(f"加载现有模型: {args.resume}")
        model = PolicyValueNet.from_checkpoint(args.resume)
    else:
        print("初始化全新的 PolicyValueNet...")
        model = build_policy_value_net(
            feature_extractor=feature_extractor,
            action_dim=args.max_actions,
            config=PolicyValueNetConfig(hidden_dim=128, hidden_dim2=64, seed=42)
        )
        
    print(f"\n预训练准备就绪！共加载了 {len(all_samples)} 条样本 (State-Action Pairs)。")
    
    # 拼装为 SelfPlayBatch 格式
    batch = SelfPlayBatch(
        observations=np.asarray([s[0] for s in all_samples], dtype=np.float32),
        masks=np.asarray([s[1] for s in all_samples], dtype=np.float32),
        policies=np.asarray([s[2] for s in all_samples], dtype=np.float32),
        values=np.asarray([s[3] for s in all_samples], dtype=np.float32),
    )
    
    print(f"开始进行监督学习预训练，Epochs: {args.epochs}, LR: {args.lr} ...")
    
    # 因为我们的数据可能非常大，我们需要打乱顺序并多次更新
    for epoch in range(args.epochs):
        # model.update 内部实际上执行了 Pytorch 的 forward 和 backward
        metrics = model.update(
            observations=batch.observations,
            masks=batch.masks,
            policy_targets=batch.policies,
            value_targets=batch.values,
            learning_rate=args.lr,
            value_weight=1.0,  # 在专家样本上，我们同样重视价值
            l2_weight=1e-5
        )
        print(f"Epoch {epoch+1}/{args.epochs} | Policy Loss: {metrics.get('policy_loss', 0):.4f} | Value Loss: {metrics.get('value_loss', 0):.4f}")
        
    # 保存权重
    model.save(args.save_path)
    print(f"训练完成！模型已保存至: {args.save_path}")

if __name__ == "__main__":
    main()
