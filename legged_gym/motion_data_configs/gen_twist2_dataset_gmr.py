import os
import yaml

"""
从文件系统重新扫描所有 .pkl 文件，生成新的 twist2_dataset_gmr.yaml

用法：
python legged_gym/motion_data_configs/gen_twist2_dataset_from_scratch.py
"""

ROOT_PATH = "/home/huanghao/source/datasets/gmr_retarget_x"
OUTPUT_YAML = "/home/huanghao/source/code/TWIST2/legged_gym/motion_data_configs/test.yaml"
# 配置特殊目录及其对应的权重
# 格式: {"目录名": 权重, ...}
SPECIAL_WEIGHTS = {
    "v1_v2_v3_g1": 10,
    "twist2_pico_clean": 10,
    "twist2_pico_no_clean": 10,
}
DEFAULT_WEIGHT = 1


def scan_pkl_files(root_path):
    """
    递归扫描 root_path 下所有 .pkl 文件，返回相对于 root_path 的路径列表
    """
    pkl_files = []
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
    
    if not os.path.isdir(ROOT_PATH):
        print(f"错误：找不到目录 {ROOT_PATH}")
        return
    
    # 扫描所有 pkl 文件
    pkl_files = scan_pkl_files(ROOT_PATH)
    print(f"找到 {len(pkl_files)} 个 .pkl 文件")
    
    # 统计权重分布
    stats = {k: 0 for k in SPECIAL_WEIGHTS.keys()}
    stats['default'] = 0
    
    for f in pkl_files:
        path_segments = f.split(os.sep)
        matched = False
        for special_dir in SPECIAL_WEIGHTS.keys():
            if special_dir in path_segments:
                stats[special_dir] += 1
                matched = True
                break
        if not matched:
            stats['default'] += 1

    for special_dir, count in stats.items():
        if special_dir == 'default':
            pass # 稍后打印
        else:
            weight = SPECIAL_WEIGHTS[special_dir]
            print(f"  - {special_dir} 目录下: {count} 个 (权重={weight})")
            
    print(f"  - 其他目录: {stats['default']} 个 (权重={DEFAULT_WEIGHT})")
    
    # 生成 yaml 数据
    data = generate_yaml_data(ROOT_PATH, pkl_files)
    
    # 写入文件
    with open(OUTPUT_YAML, 'w', encoding='utf-8') as f:
        yaml.dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False
        )
    
    print(f"\n已生成配置文件: {OUTPUT_YAML}")
    print(f"root_path: {ROOT_PATH}")
    print(f"总计 {len(pkl_files)} 个动作文件")


if __name__ == "__main__":
    main()

