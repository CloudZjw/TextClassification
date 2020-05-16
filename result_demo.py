#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文本分类预测模块
"""

import argparse


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--m', required=True, help='模型文件, 例如: model.bin')
    parser.add_argument(
        '--c', required=True, help='检验数据, 例如: val.txt')
    parser.add_argument(
        '--o', required=True, help='结果文件, 例如: result.txt')
    args, unparsed = parser.parse_known_args()
    return args


def load_model(model_filepath):
    """加载模型
    Args:
        * model_filepath: 模型文件
    Returns:
         模型对象: onject, <class XXXXX>
    """


def predict(model):
    """预测

    Args:
        * model: 模型对象
    Returns:
         预测结果: string, "dataid|predicted_label|content"
    """


def main():
    args = parse_arg()
    print("loading...")
    model = load_model(args.m)
    print("predicting...")
    predict(model)
    print("DONE")


if __name__ == "__main__":
    main()

