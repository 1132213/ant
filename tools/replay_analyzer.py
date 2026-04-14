import json
import argparse
from collections import defaultdict
from pathlib import Path

# Operation Types (from SDK.utils.constants)
BUILD_TOWER = 11
UPGRADE_TOWER = 12
UPGRADE_GENERATION_SPEED = 31
UPGRADE_GENERATED_ANT = 32

def parse_replay(replay_path: Path) -> dict:
    try:
        with open(replay_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {replay_path.name}: {e}")
        return None
        
    if not data:
        print(f"Replay {replay_path.name} is empty.")
        return None
        
    # Get winner
    winner = data[-1].get("round_state", {}).get("winner", None)
    if winner is None:
        print(f"Match {replay_path.name} ended in a draw or crashed.")
        return None
        
    # Statistics for this match
    build_counts = defaultdict(int)
    first_20_rounds_builds = []
    base_upgrades = []
    
    # Iterate through rounds
    for round_idx, turn in enumerate(data):
        # We care about the winner's operations
        op_key = f"op{winner}"
        if op_key not in turn:
            continue
            
        operations = turn[op_key]
        coins = turn.get("round_state", {}).get("coins", [0, 0])[winner]
        
        for op in operations:
            op_type = op.get("type")
            
            # Record builds
            if op_type == BUILD_TOWER:
                pos = op.get("pos", {})
                x, y = pos.get("x", -1), pos.get("y", -1)
                if x != -1 and y != -1:
                    # Mirror coordinates to Player 0's perspective if winner is Player 1
                    # Player 0 base is at (2, EDGE-1), Player 1 base is at (MAP_SIZE-3, EDGE-1)
                    # For simplicity in stats, we normalize all coordinates relative to the player's base
                    # But for absolute coordinates, if winner == 1, we should flip X. MAP_SIZE is usually 19 (for EDGE=10)
                    normalized_x = x
                    if winner == 1:
                        normalized_x = 18 - x # assuming MAP_SIZE = 19
                    
                    build_counts[(normalized_x, y)] += 1
                    if round_idx < 20:
                        first_20_rounds_builds.append((normalized_x, y, round_idx))
                        
            # Record base upgrades
            if op_type in (UPGRADE_GENERATION_SPEED, UPGRADE_GENERATED_ANT):
                up_name = "Generation Speed" if op_type == UPGRADE_GENERATION_SPEED else "Ant Level"
                base_upgrades.append({
                    "round": round_idx,
                    "type": up_name,
                    "coins_before_op": coins
                })
                
    return {
        "file": replay_path.name,
        "winner": winner,
        "total_rounds": len(data),
        "builds": build_counts,
        "early_builds": first_20_rounds_builds,
        "upgrades": base_upgrades
    }

def analyze_batch(directory_path: str):
    p = Path(directory_path)
    if not p.is_dir():
        # If it's a file, just analyze the single file
        if p.exists() and p.is_file():
            files = [p]
        else:
            print(f"Path not found: {directory_path}")
            return
    else:
        files = list(p.glob("*.json"))
        
    print(f"Found {len(files)} replay files. Analyzing...")
    
    global_build_heatmap = defaultdict(int)
    global_early_builds = defaultdict(int)
    global_upgrades = []
    
    successful_matches = 0
    
    for f in files:
        result = parse_replay(f)
        if result:
            successful_matches += 1
            for pos, count in result["builds"].items():
                global_build_heatmap[pos] += count
            for x, y, _ in result["early_builds"]:
                global_early_builds[(x, y)] += 1
            global_upgrades.extend(result["upgrades"])
            
    print(f"\n--- Batch Analysis Results ({successful_matches} Matches) ---")
    
    print("\nTop 10 Early Game Build Locations (First 20 Rounds, normalized to Player 0's side):")
    sorted_early = sorted(global_early_builds.items(), key=lambda item: item[1], reverse=True)
    for (x, y), count in sorted_early[:10]:
        print(f"  - ({x}, {y}) : Built {count} times")
        
    print("\nTop 15 Build Locations Overall (normalized to Player 0's side):")
    sorted_builds = sorted(global_build_heatmap.items(), key=lambda item: item[1], reverse=True)
    for (x, y), count in sorted_builds[:15]:
        print(f"  - ({x}, {y}) : Built {count} times")
        
    print("\nBase Upgrades Timing (Top 15 earliest):")
    if not global_upgrades:
        print("  None")
    else:
        sorted_upgrades = sorted(global_upgrades, key=lambda x: x["round"])
        for up in sorted_upgrades[:15]:
            print(f"  - Round {up['round']:03d}: Upgraded {up['type']} (Coins remaining ~{up['coins_before_op']})")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to replay.json or directory of replays")
    args = parser.parse_args()
    
    analyze_batch(args.path)
