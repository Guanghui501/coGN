#!/usr/bin/env python
"""
修复 train_with_cross_modal_attention.py 中的 bool 参数问题

这个脚本会在现有的训练脚本基础上修改 bool 类型的参数，
使用 lambda 函数来正确解析 True/False 字符串。

用法:
    python fix_bool_args.py
"""

import re

def str2bool(v):
    """将字符串转换为布尔值"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError(f'Boolean value expected, got: {v}')

# 读取原始文件
with open('train_with_cross_modal_attention.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 替换所有 type=bool 为 type=str2bool
replacements = [
    (r"parser\.add_argument\('--use_cross_modal', type=bool,",
     "parser.add_argument('--use_cross_modal', type=str2bool,"),

    (r"parser\.add_argument\('--use_middle_fusion', type=bool,",
     "parser.add_argument('--use_middle_fusion', type=str2bool,"),

    (r"parser\.add_argument\('--use_fine_grained_attention', type=bool,",
     "parser.add_argument('--use_fine_grained_attention', type=str2bool,"),

    (r"parser\.add_argument\('--fine_grained_use_projection', type=bool,",
     "parser.add_argument('--fine_grained_use_projection', type=str2bool,"),

    (r"parser\.add_argument\('--use_contrastive', type=bool,",
     "parser.add_argument('--use_contrastive', type=str2bool,"),

    (r"parser\.add_argument\('--use_preprocessed', type=bool,",
     "parser.add_argument('--use_preprocessed', type=str2bool,"),
]

# 应用替换
for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content)

# 在 get_parser 函数开头添加 str2bool 函数定义
if 'def str2bool(v):' not in content:
    # 找到 def get_parser(): 的位置
    parser_def_pos = content.find('def get_parser():')
    if parser_def_pos != -1:
        # 在这之前插入 str2bool 函数
        str2bool_func = '''
def str2bool(v):
    """将字符串转换为布尔值（用于 argparse）"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError(f'Boolean value expected, got: {v}')


'''
        content = content[:parser_def_pos] + str2bool_func + content[parser_def_pos:]

# 写入新文件
output_file = 'train_with_cross_modal_attention_fixed.py'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"✅ 修复完成！")
print(f"新文件: {output_file}")
print(f"\n修改内容:")
print(f"  - 添加了 str2bool 函数")
print(f"  - 将所有 type=bool 改为 type=str2bool")
print(f"\n现在可以正确使用:")
print(f"  --use_fine_grained_attention True")
print(f"  --use_fine_grained_attention False")
print(f"  --use_fine_grained_attention 1")
print(f"  --use_fine_grained_attention 0")
