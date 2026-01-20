#!/bin/bash
# 批量处理 twist2_pico 数据集
# 四个人：huanghao(180cm), huanghao2(180cm), xuanyu(170cm), zhaobing(170cm)
# 处理 raw 和 raw_clean 目录，统一输出到 raw_retarget 和 raw_clean_retarget

set -e  # 遇到错误立即退出

# ========================================
# 环境配置
# ========================================
# GMR 环境配置（用于第一步：动作重定向）
GMR_ENV="/home/huanghao/miniconda3/envs/gmr"
GMR_PYTHON="${GMR_ENV}/bin/python"

# 检查 GMR 环境是否存在
if [ ! -f "${GMR_PYTHON}" ]; then
    echo "错误: GMR 环境不存在: ${GMR_ENV}"
    exit 1
fi

# 设置 GMR 环境变量
export PYTHONPATH="${GMR_ENV}/lib/python3.8/site-packages:${PYTHONPATH}"
export LD_LIBRARY_PATH="${GMR_ENV}/lib:${LD_LIBRARY_PATH}"

# 数据集根目录
DATASET_ROOT="/home/huanghao/source/datasets/twist2_pico"

# 统一输出目录
OUTPUT_RAW="${DATASET_ROOT}/raw_retarget"
OUTPUT_RAW_CLEAN="${DATASET_ROOT}/raw_clean_retarget"

# 创建输出目录
mkdir -p "${OUTPUT_RAW}"
mkdir -p "${OUTPUT_RAW_CLEAN}"

# 脚本路径
SCRIPT_PATH="/home/huanghao/source/code/TWIST2/deploy_real/batch_retarget_raw.py"

echo "========================================="
echo "批量处理 TWIST2 Pico 数据集"
echo "========================================="
echo "GMR 环境: ${GMR_ENV}"
echo ""

# 处理函数
process_person() {
    local person=$1
    local height=$2
    
    echo "========================================="
    echo "处理: ${person} (身高: ${height}m)"
    echo "========================================="

    # 为每个人创建独立的输出子目录
    local output_raw_person="${OUTPUT_RAW}/${person}"
    local output_raw_clean_person="${OUTPUT_RAW_CLEAN}/${person}"
    mkdir -p "${output_raw_person}"
    mkdir -p "${output_raw_clean_person}"

    # 处理 raw 目录
    local input_raw="${DATASET_ROOT}/${person}/raw"
    if [ -d "${input_raw}" ]; then
        echo ""
        echo "[1/2] 处理 raw 目录..."
        echo "  输出到: ${output_raw_person}"
        "${GMR_PYTHON}" "${SCRIPT_PATH}" \
            --input_dir "${input_raw}" \
            --output_dir "${output_raw_person}" \
            --actual_human_height ${height} \
            --optimize_latency
        echo "✓ raw 处理完成"
    else
        echo "跳过: raw 目录不存在"
    fi
    
    # 处理 raw_clean 目录
    local input_raw_clean="${DATASET_ROOT}/${person}/raw_clean"
    if [ -d "${input_raw_clean}" ]; then
        echo ""
        echo "[2/2] 处理 raw_clean 目录..."
        echo "  输出到: ${output_raw_clean_person}"
        "${GMR_PYTHON}" "${SCRIPT_PATH}" \
            --input_dir "${input_raw_clean}" \
            --output_dir "${output_raw_clean_person}" \
            --actual_human_height ${height} \
            --optimize_latency
        echo "✓ raw_clean 处理完成"
    else
        echo "跳过: raw_clean 目录不存在"
    fi
    
    echo ""
}

# 处理每个人的数据
process_person "huanghao" 1.80
process_person "huanghao2" 1.80
process_person "xuanyu" 1.70
process_person "zhaobing" 1.70

echo "========================================="
echo "全部处理完成！"
echo "========================================="
echo "输出目录："
echo "  - raw_retarget: ${OUTPUT_RAW}"
echo "  - raw_clean_retarget: ${OUTPUT_RAW_CLEAN}"
echo ""
echo "文件统计（按人员）："
for person in huanghao huanghao2 xuanyu zhaobing; do
    raw_count=$(find "${OUTPUT_RAW}/${person}" -name "*.pkl" 2>/dev/null | wc -l)
    clean_count=$(find "${OUTPUT_RAW_CLEAN}/${person}" -name "*.pkl" 2>/dev/null | wc -l)
    echo "  ${person}:"
    echo "    raw_retarget: ${raw_count} 个文件"
    echo "    raw_clean_retarget: ${clean_count} 个文件"
done
echo ""
total_raw=$(find "${OUTPUT_RAW}" -name "*.pkl" 2>/dev/null | wc -l)
total_clean=$(find "${OUTPUT_RAW_CLEAN}" -name "*.pkl" 2>/dev/null | wc -l)
echo "总计："
echo "  raw_retarget: ${total_raw} 个文件"
echo "  raw_clean_retarget: ${total_clean} 个文件"
echo ""

# ========================================
# 第二步：转换为 numpy 1.23 版本
# ========================================
# 上面的步骤使用 gmr 环境运行生成 numpy 2.x 版本的 pkl 文件
# 下面使用 np123 环境将其转化为 numpy 1.23 版本

CONVERT_SCRIPT="/home/huanghao/source/tool/convert_pkl_numpy2_to_numpy123.py"
OUTPUT_RAW_NP123="${OUTPUT_RAW}_numpy123"
OUTPUT_RAW_CLEAN_NP123="${OUTPUT_RAW_CLEAN}_numpy123"

echo "========================================="
echo "开始转换为 numpy 1.23 版本"
echo "========================================="
echo ""

# 检查转换脚本是否存在
if [ ! -f "${CONVERT_SCRIPT}" ]; then
    echo "警告: 转换脚本不存在: ${CONVERT_SCRIPT}"
    echo "跳过 numpy 版本转换步骤"
    exit 0
fi

# 配置 np123 环境路径
NP123_ENV="/home/huanghao/miniconda3/envs/np123"
NP123_PYTHON="${NP123_ENV}/bin/python"

# 检查 np123 环境是否存在
if [ ! -f "${NP123_PYTHON}" ]; then
    echo "警告: np123 环境不存在: ${NP123_ENV}"
    echo "跳过 numpy 版本转换步骤"
    exit 0
fi

# 设置 np123 环境变量
export PYTHONPATH="${NP123_ENV}/lib/python3.10/site-packages:${PYTHONPATH}"
export LD_LIBRARY_PATH="${NP123_ENV}/lib:${LD_LIBRARY_PATH}"

echo "使用环境: ${NP123_ENV}"
echo ""

# 转换 raw_retarget
if [ -d "${OUTPUT_RAW}" ]; then
    echo "[1/2] 转换 raw_retarget..."
    "${NP123_PYTHON}" "${CONVERT_SCRIPT}" \
        --root "${OUTPUT_RAW}" \
        --out-root "${OUTPUT_RAW_NP123}" \
        --no-sort \
        --skip-existing \
        --workers 64 \
        --max-in-flight 64 \
        --batch-size 1
    echo "✓ raw_retarget 转换完成"
    echo ""
fi

# 转换 raw_clean_retarget
if [ -d "${OUTPUT_RAW_CLEAN}" ]; then
    echo "[2/2] 转换 raw_clean_retarget..."
    "${NP123_PYTHON}" "${CONVERT_SCRIPT}" \
        --root "${OUTPUT_RAW_CLEAN}" \
        --out-root "${OUTPUT_RAW_CLEAN_NP123}" \
        --no-sort \
        --skip-existing \
        --workers 64 \
        --max-in-flight 64 \
        --batch-size 1
    echo "✓ raw_clean_retarget 转换完成"
    echo ""
fi

echo "========================================="
echo "numpy 版本转换完成！"
echo "========================================="
echo "输出目录："
echo "  - raw_retarget_numpy123: ${OUTPUT_RAW_NP123}"
echo "  - raw_clean_retarget_numpy123: ${OUTPUT_RAW_CLEAN_NP123}"
echo ""
echo "文件统计（按人员）："
for person in huanghao huanghao2 xuanyu zhaobing; do
    raw_count=$(find "${OUTPUT_RAW_NP123}/${person}" -name "*.pkl" 2>/dev/null | wc -l)
    clean_count=$(find "${OUTPUT_RAW_CLEAN_NP123}/${person}" -name "*.pkl" 2>/dev/null | wc -l)
    echo "  ${person}:"
    echo "    raw_retarget_numpy123: ${raw_count} 个文件"
    echo "    raw_clean_retarget_numpy123: ${clean_count} 个文件"
done
echo ""
total_raw=$(find "${OUTPUT_RAW_NP123}" -name "*.pkl" 2>/dev/null | wc -l)
total_clean=$(find "${OUTPUT_RAW_CLEAN_NP123}" -name "*.pkl" 2>/dev/null | wc -l)
echo "总计："
echo "  raw_retarget_numpy123: ${total_raw} 个文件"
echo "  raw_clean_retarget_numpy123: ${total_clean} 个文件"
echo ""