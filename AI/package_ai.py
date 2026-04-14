import os
import sys
import zipfile
import shutil
import tempfile
from pathlib import Path

def clean_tree(root_dir: Path):
    for root, dirs, files in os.walk(root_dir):
        if '__pycache__' in dirs:
            shutil.rmtree(os.path.join(root, '__pycache__'))
            dirs.remove('__pycache__')
        for file in files:
            if file.endswith('.pyc') or file == '.DS_Store' or file.endswith('.so') or file.endswith('.dylib') or file.endswith('.pyd'):
                os.remove(os.path.join(root, file))

def copy_file_mapping(output_dir: Path, mapping: str):
    source_str, target_str = mapping.rsplit(':', 1)
    source = Path(source_str)
    target = output_dir / target_str
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

def copy_tree_mapping(output_dir: Path, mapping: str):
    source_str, target_str = mapping.rsplit(':', 1)
    source = Path(source_str)
    target = output_dir / target_str
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    clean_tree(target)

def assemble_layout(repo_root: Path, output_dir: Path, file_mappings: list, tree_mappings: list):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Base files
    copy_file_mapping(output_dir, f"{repo_root}/AI/main.py:main.py")
    copy_file_mapping(output_dir, f"{repo_root}/AI/common.py:common.py")
    copy_file_mapping(output_dir, f"{repo_root}/AI/protocol.py:protocol.py")
    copy_tree_mapping(output_dir, f"{repo_root}/SDK:SDK")
    copy_tree_mapping(output_dir, f"{repo_root}/tools:tools")

    for mapping in file_mappings:
        copy_file_mapping(output_dir, mapping)
        
    for mapping in tree_mappings:
        copy_tree_mapping(output_dir, mapping)

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("usage: python package_ai.py <random|mcts|greedy|example|custom|mcts_custom|alphazero> [output_path_or_dir]", file=sys.stderr)
        sys.exit(1)

    target = sys.argv[1]
    output_arg = sys.argv[2] if len(sys.argv) == 3 else None

    script_dir = Path(__file__).parent.resolve()
    repo_root = script_dir.parent

    archive_name = ""
    file_mappings = []
    tree_mappings = []

    if target == "random":
        archive_name = "ai_rand.zip"
        file_mappings.append(f"{repo_root}/AI/ai_random.py:ai.py")
    elif target == "mcts":
        archive_name = "ai_mcts.zip"
        file_mappings.append(f"{repo_root}/AI/ai_mcts.py:ai.py")
    elif target == "greedy":
        archive_name = "ai_greedy.zip"
        file_mappings.append(f"{repo_root}/AI/ai_greedy.py:ai.py")
        tree_mappings.append(f"{repo_root}/AI/ai_greedy:ai_greedy")
    elif target == "example":
        archive_name = "ai_example.zip"
        file_mappings.append(f"{repo_root}/AI/ai_example.py:ai.py")
    elif target == "custom":
        archive_name = "ai_custom.zip"
        file_mappings.append(f"{repo_root}/AI/ai_custom.py:ai.py")
        file_mappings.append(f"{repo_root}/AI/custom_utils.py:custom_utils.py")
    elif target == "mcts_custom":
        archive_name = "ai_mcts_custom.zip"
        file_mappings.append(f"{repo_root}/AI/ai_mcts_custom.py:ai.py")
        file_mappings.append(f"{repo_root}/AI/custom_utils.py:custom_utils.py")
    elif target == "alphazero":
        archive_name = "ai_alphazero.zip"
        file_mappings.append(f"{repo_root}/AI/ai_alphazero.py:ai.py")
        file_mappings.append(f"{repo_root}/AI/custom_utils.py:custom_utils.py")
        # 复制预训练好的模型
        if (repo_root / "checkpoints" / "ai_alphazero_latest.npz").exists():
            file_mappings.append(f"{repo_root}/checkpoints/ai_alphazero_latest.npz:ai_alphazero_latest.npz")
    else:
        print(f"Unknown target: {target}", file=sys.stderr)
        sys.exit(1)

    if output_arg and not output_arg.endswith('.zip'):
        output_dir = Path(output_arg)
        if output_dir.exists() and not output_dir.is_dir():
            print(f"output path exists and is not a directory: {output_dir}", file=sys.stderr)
            sys.exit(1)
        if output_dir.exists() and any(output_dir.iterdir()):
            print(f"output directory must be empty: {output_dir}", file=sys.stderr)
            sys.exit(1)
        
        assemble_layout(repo_root, output_dir, file_mappings, tree_mappings)
        print(output_dir)
        sys.exit(0)

    output_zip = Path(output_arg) if output_arg else script_dir / archive_name
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    
    with tempfile.TemporaryDirectory(prefix=f"agent-tradition-{target}-") as staging_dir:
        staging_path = Path(staging_dir)
        assemble_layout(repo_root, staging_path, file_mappings, tree_mappings)
        
        if output_zip.exists():
            output_zip.unlink()
            
        # Create zip file
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(staging_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, staging_path)
                    zipf.write(file_path, arcname)
                    
    print(output_zip.resolve())

if __name__ == "__main__":
    main()