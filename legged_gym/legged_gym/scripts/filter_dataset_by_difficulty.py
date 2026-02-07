#!/usr/bin/env python3
"""
TWIST2 数据集筛选脚本

根据动作难度评估结果筛选数据集，生成清理后的YAML配置文件。

使用方法：
    # 筛选出难度分数低于阈值的动作
    python filter_dataset_by_difficulty.py \
        --difficulty_csv difficulty_scores.csv \
        --original_config motion_data_configs/original_dataset.yaml \
        --output_config motion_data_configs/filtered_dataset.yaml \
        --max_difficulty 1.5

    # 只保留完成率高于阈值的动作
    python filter_dataset_by_difficulty.py \
        --difficulty_csv difficulty_scores.csv \
        --original_config motion_data_configs/original_dataset.yaml \
        --output_config motion_data_configs/filtered_dataset.yaml \
        --min_completion 0.8

    # 排除特定终止原因的动作
    python filter_dataset_by_difficulty.py \
        --difficulty_csv difficulty_scores.csv \
        --original_config motion_data_configs/original_dataset.yaml \
        --output_config motion_data_configs/filtered_dataset.yaml \
        --exclude_reasons contact roll_pitch
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import yaml
import numpy as np
from rich import print
from rich.console import Console
from rich.table import Table


def load_difficulty_results(csv_path):
    """加载难度评估结果"""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['difficulty_score'] = float(row['difficulty_score'])
            row['completion_rate'] = float(row['completion_rate'])
            results.append(row)
    return results


def load_original_config(config_path):
    """加载原始数据集配置"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def filter_motions(difficulty_results, original_config, args):
    """根据条件筛选动作"""

    # 构建motion文件到结果的映射
    motion_to_result = {}
    for r in difficulty_results:
        motion_file = r['motion_file']
        if motion_file not in motion_to_result:
            motion_to_result[motion_file] = []
        motion_to_result[motion_file].append(r)

    # 筛选条件
    def passes_filter(result):
        # 难度分数筛选
        if args.max_difficulty is not None:
            if result['difficulty_score'] > args.max_difficulty:
                return False

        # 完成率筛选
        if args.min_completion is not None:
            if result['completion_rate'] < args.min_completion:
                return False

        # 终止原因筛选
        if args.exclude_reasons:
            if result['termination_reason'] in args.exclude_reasons:
                return False

        # 只保留指定终止原因
        if args.only_reasons:
            if result['termination_reason'] not in args.only_reasons:
                return False

        return True

    # 筛选动作
    filtered_motions = []
    excluded_motions = []
    excluded_reasons_count = defaultdict(int)

    for motion in original_config.get('motions', []):
        motion_file = motion.get('file', '')
        weight = motion.get('weight', 1.0)

        # 获取该文件的所有评估结果
        results = motion_to_result.get(motion_file, [])

        if not results:
            # 没有评估结果，根据策略决定
            if args.skip_unevaluated:
                excluded_motions.append(motion)
                excluded_reasons_count['no_evaluation'] += 1
                continue
            else:
                filtered_motions.append(motion)
                continue

        # 检查该文件的所有评估结果是否通过筛选
        all_pass = all(passes_filter(r) for r in results)

        if all_pass:
            filtered_motions.append(motion)
        else:
            excluded_motions.append(motion)
            # 记录排除原因
            for r in results:
                if not passes_filter(r):
                    if args.max_difficulty and r['difficulty_score'] > args.max_difficulty:
                        excluded_reasons_count['high_difficulty'] += 1
                    elif args.min_completion and r['completion_rate'] < args.min_completion:
                        excluded_reasons_count['low_completion'] += 1
                    elif args.exclude_reasons and r['termination_reason'] in args.exclude_reasons:
                        excluded_reasons_count[f"excluded_{r['termination_reason']}"] += 1
                    elif args.only_reasons and r['termination_reason'] not in args.only_reasons:
                        excluded_reasons_count[f"not_{r['termination_reason']}"] += 1

    return filtered_motions, excluded_motions, excluded_reasons_count


def save_filtered_config(filtered_motions, original_config, output_path):
    """保存筛选后的配置"""
    output_config = {
        'root_path': original_config.get('root_path', ''),
        'motions': filtered_motions
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(output_config, f, default_flow_style=False, sort_keys=False)

    print(f"Filtered config saved to: {output_path}")


def print_summary(original_config, filtered_motions, excluded_motions, excluded_reasons_count, args):
    """打印筛选摘要"""
    console = Console()

    total_original = len(original_config.get('motions', []))
    total_filtered = len(filtered_motions)
    total_excluded = len(excluded_motions)

    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    console.print(f"[bold cyan]                    筛选摘要                              [/bold cyan]")
    console.print(f"[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("类别", style="cyan")
    table.add_column("数量", justify="right")
    table.add_column("占比", justify="right")

    table.add_row("原始动作数", str(total_original), "100%")
    table.add_row("保留动作数", f"[green]{total_filtered}[/green]", f"{total_filtered/total_original*100:.1f}%")
    table.add_row("排除动作数", f"[red]{total_excluded}[/red]", f"{total_excluded/total_original*100:.1f}%")

    console.print(table)

    if excluded_reasons_count:
        console.print(f"\n[bold]排除原因统计:[/bold]")
        for reason, count in sorted(excluded_reasons_count.items(), key=lambda x: -x[1]):
            console.print(f"  {reason}: {count}")

    console.print(f"\n[bold]筛选条件:[/bold]")
    if args.max_difficulty is not None:
        console.print(f"  最大难度分数: {args.max_difficulty}")
    if args.min_completion is not None:
        console.print(f"  最小完成率: {args.min_completion}")
    if args.exclude_reasons:
        console.print(f"  排除终止原因: {', '.join(args.exclude_reasons)}")
    if args.only_reasons:
        console.print(f"  只保留终止原因: {', '.join(args.only_reasons)}")
    if args.skip_unevaluated:
        console.print(f"  跳过未评估的动作: True")


def main():
    parser = argparse.ArgumentParser(description="TWIST2 数据集筛选脚本")
    parser.add_argument('--difficulty_csv', type=str, required=True,
                        help='难度评估结果CSV文件路径')
    parser.add_argument('--original_config', type=str, required=True,
                        help='原始数据集YAML配置文件路径')
    parser.add_argument('--output_config', type=str, required=True,
                        help='输出的筛选后数据集YAML配置文件路径')
    parser.add_argument('--max_difficulty', type=float, default=None,
                        help='保留难度分数小于此值的动作')
    parser.add_argument('--min_completion', type=float, default=None,
                        help='保留完成率大于此值的动作')
    parser.add_argument('--exclude_reasons', type=str, nargs='*', default=[],
                        choices=['completed', 'contact', 'height_diff', 'roll_pitch',
                                'pose_tracking', 'root_tracking', 'timeout', 'unknown'],
                        help='排除具有这些终止原因的动作')
    parser.add_argument('--only_reasons', type=str, nargs='*', default=[],
                        choices=['completed', 'contact', 'height_diff', 'roll_pitch',
                                'pose_tracking', 'root_tracking', 'timeout', 'unknown'],
                        help='只保留具有这些终止原因的动作')
    parser.add_argument('--skip_unevaluated', action='store_true',
                        help='跳过没有评估结果的动作')
    parser.add_argument('--top_k', type=int, default=None,
                        help='只保留难度最低的K个动作')
    parser.add_argument('--bottom_k', type=int, default=None,
                        help='只保留难度最高的K个动作（用于分析最难动作）')

    args = parser.parse_args()

    console = Console()

    # 验证输入
    if not os.path.exists(args.difficulty_csv):
        console.print(f"[red]Error: CSV file not found: {args.difficulty_csv}[/red]")
        return

    if not os.path.exists(args.original_config):
        console.print(f"[red]Error: Config file not found: {args.original_config}[/red]")
        return

    # 加载数据
    console.print(f"[cyan]加载难度评估结果...[/cyan]")
    difficulty_results = load_difficulty_results(args.difficulty_csv)
    console.print(f"[green]加载了 {len(difficulty_results)} 条评估结果[/green]")

    console.print(f"[cyan]加载原始配置...[/cyan]")
    original_config = load_original_config(args.original_config)
    console.print(f"[green]加载了 {len(original_config.get('motions', []))} 个动作配置[/green]")

    # 特殊处理：top_k / bottom_k
    if args.top_k is not None or args.bottom_k is not None:
        # 按难度排序
        sorted_results = sorted(difficulty_results, key=lambda x: x['difficulty_score'])

        if args.top_k is not None:
            # 选择难度最低的K个
            selected_results = sorted_results[:args.top_k]
            selected_files = set(r['motion_file'] for r in selected_results)
        else:
            # 选择难度最高的K个
            selected_results = sorted_results[-args.bottom_k:]
            selected_files = set(r['motion_file'] for r in selected_results)

        # 构建筛选后的motion列表
        filtered_motions = []
        excluded_motions = []
        for motion in original_config.get('motions', []):
            if motion.get('file', '') in selected_files:
                filtered_motions.append(motion)
            else:
                excluded_motions.append(motion)

        excluded_reasons_count = {'top_k_or_bottom_k_filter': len(excluded_motions)}

    else:
        # 常规筛选
        filtered_motions, excluded_motions, excluded_reasons_count = filter_motions(
            difficulty_results, original_config, args
        )

    # 保存结果
    save_filtered_config(filtered_motions, original_config, args.output_config)

    # 打印摘要
    print_summary(original_config, filtered_motions, excluded_motions,
                  excluded_reasons_count, args)


if __name__ == "__main__":
    main()
