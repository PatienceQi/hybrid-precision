#!/usr/bin/env python3
"""
从HotpotQA训练集中抽取medium级别的样本
创建时间: 2024-01-14
"""

import json
import os
from datetime import datetime
from tqdm import tqdm

def analyze_hotpot_file(input_file):
    """分析HotpotQA文件，统计各级别样本数量"""
    print(f"正在分析文件: {input_file}")

    level_counts = {"easy": 0, "medium": 0, "hard": 0}
    total_samples = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_samples = len(data)

    for item in data:
        level = item.get('level', 'unknown')
        if level in level_counts:
            level_counts[level] += 1
        else:
            print(f"发现未知级别: {level}")

    print(f"\n文件统计信息:")
    print(f"总样本数: {total_samples}")
    print(f"Easy级别: {level_counts['easy']}")
    print(f"Medium级别: {level_counts['medium']}")
    print(f"Hard级别: {level_counts['hard']}")

    return data, level_counts

def quality_check(item, sample_id):
    """质量检查函数"""
    # 检查必要字段
    required_fields = ['question', 'answer', 'supporting_facts', 'context', 'level']
    for field in required_fields:
        if field not in item:
            print(f"样本{sample_id}缺少字段: {field}")
            return False

    # 检查内容质量
    question = item.get('question', '')
    answer = item.get('answer', '')
    supporting_facts = item.get('supporting_facts', [])
    context = item.get('context', [])

    # 基本长度检查
    if not (10 <= len(question.split()) <= 200):
        print(f"样本{sample_id}问题长度异常: {len(question.split())}词")
        return False

    if not (2 <= len(answer.split()) <= 100):
        print(f"样本{sample_id}答案长度异常: {len(answer.split())}词")
        return False

    # supporting_facts检查
    if not supporting_facts or len(supporting_facts) == 0:
        print(f"样本{sample_id}supporting_facts为空")
        return False

    # context检查
    if not context or len(context) == 0:
        print(f"样本{sample_id}context为空")
        return False

    return True

def extract_medium_samples(data, target_count=1000, start_index=0):
    """抽取medium级别样本"""
    medium_samples = []
    processed_count = 0

    print(f"\n开始抽取medium级别样本...")
    print(f"目标数量: {target_count}")
    print(f"起始位置: {start_index}")

    # 从指定位置开始处理
    for i in tqdm(range(start_index, len(data)), desc="处理样本"):
        item = data[i]

        # 检查是否为medium级别
        if item.get('level') == 'medium':
            processed_count += 1

            # 质量检查
            if quality_check(item, i):
                # 添加原始索引信息
                item['_original_index'] = i
                medium_samples.append(item)

                # 达到目标数量时停止
                if len(medium_samples) >= target_count:
                    break

    print(f"\n抽取完成:")
    print(f"处理的medium样本: {processed_count}")
    print(f"通过质量检查的样本: {len(medium_samples)}")

    return medium_samples

def save_samples(samples, output_file):
    """保存抽取的样本"""
    output_data = {
        "metadata": {
            "total_samples": len(samples),
            "extraction_date": datetime.now().isoformat(),
            "level": "medium",
            "description": "从HotpotQA训练集中抽取的medium级别样本"
        },
        "samples": samples
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n样本已保存到: {output_file}")

def create_batch_files(samples, batch_size=200):
    """创建批次文件"""
    total_samples = len(samples)
    num_batches = (total_samples + batch_size - 1) // batch_size

    print(f"\n创建批次文件:")
    print(f"总样本数: {total_samples}")
    print(f"批次大小: {batch_size}")
    print(f"批次数量: {num_batches}")

    batch_files = []

    for batch_id in range(num_batches):
        start_idx = batch_id * batch_size
        end_idx = min(start_idx + batch_size, total_samples)
        batch_samples = samples[start_idx:end_idx]

        batch_file = f"/Users/qipatience/Desktop/混合检索指标集成在 RAGAS 中的评估/dataset/hotpot_medium_batch_{batch_id+1}.json"

        batch_data = {
            "metadata": {
                "batch_id": batch_id + 1,
                "batch_size": len(batch_samples),
                "sample_range": f"{start_idx+1}-{end_idx}",
                "total_batches": num_batches,
                "level": "medium",
                "creation_date": datetime.now().isoformat()
            },
            "samples": batch_samples
        }

        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, indent=2, ensure_ascii=False)

        batch_files.append(batch_file)
        print(f"批次 {batch_id+1}: {len(batch_samples)} 个样本 -> {batch_file}")

    return batch_files

def main():
    """主函数"""
    # 输入文件
    input_file = "/Users/qipatience/Desktop/混合检索指标集成在 RAGAS 中的评估/dataset/hotpot_train_v1.1.json"

    # 输出文件
    output_file = "/Users/qipatience/Desktop/混合检索指标集成在 RAGAS 中的评估/dataset/hotpot_medium_1000.json"

    try:
        # 步骤1: 分析文件
        print("="*60)
        print("步骤1: 分析HotpotQA训练文件")
        print("="*60)

        data, level_counts = analyze_hotpot_file(input_file)

        # 检查medium样本数量
        if level_counts['medium'] < 1000:
            print(f"\n警告: medium级别样本只有 {level_counts['medium']} 个，少于目标数量1000")
            target_count = level_counts['medium']
        else:
            target_count = 1000

        # 步骤2: 抽取medium样本
        print("\n" + "="*60)
        print("步骤2: 抽取medium级别样本")
        print("="*60)

        medium_samples = extract_medium_samples(data, target_count)

        # 步骤3: 保存完整结果
        print("\n" + "="*60)
        print("步骤3: 保存抽取结果")
        print("="*60)

        if medium_samples:
            save_samples(medium_samples, output_file)

            # 步骤4: 创建批次文件
            print("\n" + "="*60)
            print("步骤4: 创建批次文件")
            print("="*60)

            batch_files = create_batch_files(medium_samples)

            print(f"\n✅ 抽取完成！")
            print(f"总计抽取了 {len(medium_samples)} 个medium级别样本")
            print(f"创建了 {len(batch_files)} 个批次文件")
            print(f"完整数据集: {output_file}")

            # 显示批次文件列表
            print(f"\n批次文件:")
            for i, batch_file in enumerate(batch_files, 1):
                print(f"  批次{i}: {batch_file}")

        else:
            print("\n❌ 未能抽取到符合条件的medium级别样本")

    except FileNotFoundError:
        print(f"\n❌ 文件未找到: {input_file}")
    except json.JSONDecodeError:
        print(f"\n❌ JSON文件格式错误: {input_file}")
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")

if __name__ == "__main__":
    main()