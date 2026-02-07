import os
import yaml

"""
从文件系统重新扫描所有 .pkl 文件，生成新的 twist2_dataset_gmr.yaml

用法：
python legged_gym/motion_data_configs/gen_twist2_dataset_from_scratch.py
"""

ROOT_PATH = "/home/huanghao/source/datasets/gmr_retarget_x"
OUTPUT_DIR = "/home/huanghao/source/code/TWIST2/legged_gym/motion_data_configs"


def generate_output_filename(selected_folders, special_weights, default_weight, total_count):
    """
    根据选择的文件夹和权重动态生成输出文件名
    格式: folder1_weight1_folder2_weight2_..._总数.yaml
    """
    parts = []
    for folder in selected_folders:
        weight = special_weights.get(folder, default_weight)
        # 使用简短的文件夹名（取前几个字符或缩写）
        short_name = folder.replace("_g1_GMR_30fps", "").replace("_gmr_120fps", "")
        parts.append(f"{short_name}_w{weight}")
    
    # 添加总数
    parts.append(f"total{total_count}")
    
    filename = "_".join(parts) + ".yaml"
    
    # 如果文件名过长（超过200字符），使用哈希缩短
    if len(filename) > 200:
        import hashlib
        # 使用配置内容的哈希，保持确定性
        config_str = "_".join(parts)
        hash_obj = hashlib.md5(config_str.encode())
        short_hash = hash_obj.hexdigest()[:8]
        filename = f"dataset_mix_{short_hash}_total{total_count}.yaml"
        
    return os.path.join(OUTPUT_DIR, filename)

# ============================================================
# 手动配置要扫描的文件夹列表
# 注释掉不需要的文件夹即可排除
# ============================================================
SELECTED_FOLDERS = [
    # "AAAaaaaaaaatest",                    # 测试文件夹
    "AMASS_numpy123",                        # AMASS 数据集
    "CORE4D_Real_numpy123",                  # CORE4D Real 数据集
    "EgoBody_g1_GMR_30fps_numpy123",         # EgoBody 数据集
    "HuMMan_numpy123",                       # HuMMan 数据集
    # "MotionMillion_g1_GMR_30fps_numpy123",   # MotionMillion 数据集
    # "MotionMillion_g1_GMR_30fps_numpy123_mirror",
    "embody3d_numpy123",                     # Embody3D 数据集
    "OMOMO_numpy123",                        # OMOMO 数据集
    "inter_x_gmr_120fps_numpy123",                    # Inter-X 数据集 (120fps)
    "interhuman_numpy123",                   # InterHuman 数据集
    "lafan1_numpy123",                       # LaFAN1 数据集
    "pico_numpy123",                         # TWIST2 Pico 清洗后数据
    "twist1_to_twist2_numpy123",             # TWIST1 转换数据
    "v1_v2_v3_g1_numpy123",                  # V1/V2/V3 G1 数据
    # "ViMoGen_228K_20fps_for_gmr_numpy123"
]

# 配置特殊目录及其对应的权重
# 格式: {"目录名": 权重, ...}
SPECIAL_WEIGHTS = {
    "v1_v2_v3_g1_numpy123": 20,
    "pico_numpy123": 30,
    "ViMoGen_228K_20fps_for_gmr_numpy123": 0.2,
    "MotionMillion_g1_GMR_30fps_numpy123": 0.05,
    "MotionMillion_g1_GMR_30fps_numpy123_mirror": 0.05,
}
DEFAULT_WEIGHT = 1


def scan_pkl_files(root_path, selected_folders=None):
    """
    扫描指定文件夹下所有 .pkl 文件，返回相对于 root_path 的路径列表
    
    Args:
        root_path: 根目录路径
        selected_folders: 要扫描的子文件夹列表，如果为 None 则扫描所有
    """
    pkl_files = []
    
    if selected_folders:
        # 只扫描指定的文件夹
        for folder in selected_folders:
            folder_path = os.path.join(root_path, folder)
            if not os.path.isdir(folder_path):
                print(f"警告：文件夹不存在，已跳过: {folder}")
                continue
            for dirpath, dirnames, filenames in os.walk(folder_path):
                for filename in filenames:
                    if filename.endswith('.pkl'):
                        abs_path = os.path.join(dirpath, filename)
                        rel_path = os.path.relpath(abs_path, root_path)
                        pkl_files.append(rel_path)
    else:
        # 扫描所有文件夹（原有逻辑）
        for dirpath, dirnames, filenames in os.walk(root_path):
            for filename in filenames:
                if filename.endswith('.pkl'):
                    abs_path = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(abs_path, root_path)
                    pkl_files.append(rel_path)
    
    return pkl_files


def generate_yaml_data(root_path, pkl_files):
    """
    根据文件列表生成 yaml 数据结构
    """
    motions = []
    
    for rel_path in sorted(pkl_files):
        # 判断权重：根据 SPECIAL_WEIGHTS 配置
        weight = DEFAULT_WEIGHT
        path_segments = rel_path.split(os.sep)
        
        for special_dir, sp_weight in SPECIAL_WEIGHTS.items():
            if special_dir in path_segments:
                weight = sp_weight
                break
        
        motions.append({
            'file': rel_path,
            'weight': weight,
            'description': 'general movement'
        })
    
    data = {
        'root_path': root_path,
        'motions': motions
    }
    
    return data


def main():
    print(f"正在扫描目录: {ROOT_PATH}")
    print(f"已选择的文件夹: {SELECTED_FOLDERS}")
    print()
    
    if not os.path.isdir(ROOT_PATH):
        print(f"错误：找不到目录 {ROOT_PATH}")
        return
    
    # 扫描指定文件夹中的 pkl 文件
    pkl_files = scan_pkl_files(ROOT_PATH, SELECTED_FOLDERS)
    print(f"找到 {len(pkl_files)} 个 .pkl 文件")
    
    # 按文件夹统计数量
    folder_stats = {folder: 0 for folder in SELECTED_FOLDERS}
    
    for f in pkl_files:
        # 获取一级文件夹名称
        top_folder = f.split(os.sep)[0]
        if top_folder in folder_stats:
            folder_stats[top_folder] += 1
    
    # 计算表格宽度
    max_folder_len = max(len(folder) for folder in SELECTED_FOLDERS) if SELECTED_FOLDERS else 10
    max_folder_len = max(max_folder_len, len("文件夹"))
    
    # 打印表格
    print()
    print("=" * (max_folder_len + 40))
    print(f"{'文件夹':<{max_folder_len}} | {'数量':>8} | {'权重':>6} | {'占比':>8}")
    print("-" * (max_folder_len + 40))
    
    total_count = 0
    total_weighted = 0
    
    for folder in SELECTED_FOLDERS:
        count = folder_stats.get(folder, 0)
        weight = SPECIAL_WEIGHTS.get(folder, DEFAULT_WEIGHT)
        total_count += count
        total_weighted += count * weight
    
    for folder in SELECTED_FOLDERS:
        count = folder_stats.get(folder, 0)
        weight = SPECIAL_WEIGHTS.get(folder, DEFAULT_WEIGHT)
        percent = (count / total_count * 100) if total_count > 0 else 0
        print(f"{folder:<{max_folder_len}} | {count:>8} | {weight:>6} | {percent:>7.2f}%")
    
    print("-" * (max_folder_len + 40))
    print(f"{'合计':<{max_folder_len}} | {total_count:>8} | {'-':>6} | {'100.00%':>8}")
    print("=" * (max_folder_len + 40))
    
    # 生成输出文件名
    output_yaml = generate_output_filename(SELECTED_FOLDERS, SPECIAL_WEIGHTS, DEFAULT_WEIGHT, total_count)
    
    # 生成 yaml 数据
    data = generate_yaml_data(ROOT_PATH, pkl_files)
    
    # 写入文件
    with open(output_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False
        )
    
    print(f"\n已生成配置文件: {output_yaml}")
    print(f"root_path: {ROOT_PATH}")
    print(f"总计 {len(pkl_files)} 个动作文件")


if __name__ == "__main__":
    main()

