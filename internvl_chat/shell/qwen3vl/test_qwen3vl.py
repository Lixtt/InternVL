#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import argparse
from typing import Dict, Optional, Union, List

import torch
from PIL import Image

# 添加代码目录到路径中
sys.path.append("/fs-computility/ai-shen/mllm_safety-shared/projects/mllm-reasoning/Qwen3-VL/internvl_chat/")
sys.path.append("/fs-computility/ai-shen/mllm_safety-shared/projects/mllm-reasoning/Qwen3-VL/internvl_chat/internvl")

from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Qwen3-VL 模型交互测试脚本")
    
    # 模型相关参数
    parser.add_argument("--checkpoint", type=str, default='internvl_chat/checkpoint-4b/',
                       help="模型检查点路径")
    parser.add_argument("--load-in-8bit", action="store_true", 
                       help="以8位精度加载模型以节省显存")
    parser.add_argument("--load-in-4bit", action="store_true",
                       help="以4位精度加载模型以节省更多显存")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="运行设备 (cuda/cpu)")
    
    # 图像相关参数
    parser.add_argument("--image", type=str, default=None,
                       help="图像文件路径，如果不提供则使用纯文本模式")
    parser.add_argument("--dynamic-image", action="store_true",
                       help="使用动态图像预处理")
    parser.add_argument("--max-patches", type=int, default=12,
                       help="最大图像块数(仅在动态图像处理时使用)")
    parser.add_argument("--image-size", type=int, default=None,
                       help="图像尺寸，默认使用模型配置")
    parser.add_argument("--use-thumbnail", action="store_true",
                       help="是否使用缩略图")
    
    # 生成相关参数
    parser.add_argument("--prompt", type=str, default=None,
                       help="输入提示文本，如果不提供则进入交互模式")
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                       help="生成的最大token数")
    parser.add_argument("--num-beams", type=int, default=1,
                       help="束搜索的束数，1表示贪婪搜索")
    parser.add_argument("--do-sample", action="store_true",
                       help="是否使用采样生成")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="生成的温度参数，较低的值使输出更确定性，较高的值增加随机性")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="nucleus采样的概率阈值")
    parser.add_argument("--top-k", type=int, default=50,
                       help="top-k采样的k值")
    parser.add_argument("--verbose", action="store_true",
                       help="显示详细输出")
    
    return parser.parse_args()


def load_and_prepare_model(args):
    """加载并准备模型和分词器"""
    print(f"正在加载模型：{args.checkpoint}...")
    
    # 准备模型参数
    model_args = argparse.Namespace(
        checkpoint=args.checkpoint,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        auto=False
    )
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_args)
    
    # 将模型移动到指定设备并设置为评估模式
    model = model.to(args.device)
    model = model.eval()
    
    return model, tokenizer


def load_and_preprocess_image(image_path: str, model, args) -> Optional[torch.Tensor]:
    """加载并预处理图像"""
    if image_path is None:
        return None
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在：{image_path}")
    
    # 获取模型配置参数
    image_size = args.image_size or model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = args.use_thumbnail if args.use_thumbnail is not None else model.config.use_thumbnail
    
    # 使用命令行参数覆盖默认配置
    dynamic = args.dynamic_image if args.dynamic_image is not None else model.config.dynamic_image_size
    max_num = args.max_patches if args.max_patches is not None else model.config.max_dynamic_patch
    
    print(f"图像处理参数：尺寸={image_size}, 动态={dynamic}, 使用缩略图={use_thumbnail}, 最大块数={max_num}")
    
    try:
        image = Image.open(image_path).convert('RGB')
        transform = build_transform(is_train=False, input_size=image_size)
        
        if dynamic:
            images = dynamic_preprocess(
                image, 
                image_size=image_size,
                use_thumbnail=use_thumbnail,
                max_num=max_num
            )
        else:
            images = [image]
            
        # 确保所有图像通过transform处理为张量
        pixel_values_list = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values_list)
        
        # 确保张量在正确的设备和数据类型上
        return pixel_values.to(model.device).to(model.dtype)
    
    except Exception as e:
        print(f"图像处理出错：{e}")
        raise


def generate_response(model, tokenizer, question: str, pixel_values: Optional[torch.Tensor] = None, args = None) -> str:
    """生成对话回应"""
    # 确保args存在，否则使用默认值
    if args is None:
        args = argparse.Namespace(
            num_beams=1,
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
            top_p=0.9,
            top_k=50,
            verbose=False
        )
    
    # 构建生成配置
    generation_config = {
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    if args.do_sample:
        generation_config.update({
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        })

    # 调用模型的chat方法
    response = model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,  # 如果是纯文本模式，则为None
        question=question,
        generation_config=generation_config,
        verbose=args.verbose,
    )
    
    return response


def interactive_mode(model, tokenizer, args):
    """交互模式：用户可以多次提问"""
    pixel_values = None
    if args.image:
        try:
            pixel_values = load_and_preprocess_image(args.image, model, args)
            print(f"已加载图片：{args.image}")
        except Exception as e:
            print(f"图片加载失败：{e}")
            if input("是否继续纯文本模式？(y/n): ").lower() != 'y':
                return

    print("\n欢迎使用Qwen3-VL交互模式！输入'exit'或'quit'退出。")
    
    while True:
        try:
            question = input("\n请输入您的问题: ")
            if question.lower() in ['exit', 'quit']:
                break
                
            if not question.strip():
                continue
                
            print("模型思考中...")
            response = generate_response(model, tokenizer, question, pixel_values, args)
            print(f"\n回答: {response}")
            
        except KeyboardInterrupt:
            print("\n已退出交互模式")
            break
        except Exception as e:
            print(f"出错：{e}")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 加载模型和分词器
    model, tokenizer = load_and_prepare_model(args)
    
    # 加载和预处理图像（如果提供了图像路径）
    pixel_values = None
    if args.image:
        try:
            pixel_values = load_and_preprocess_image(args.image, model, args)
        except Exception as e:
            print(f"图像处理失败：{e}")
            if args.prompt:  # 如果有提示文本，询问是否以纯文本模式继续
                if input("是否以纯文本模式继续？(y/n): ").lower() != 'y':
                    return
    
    # 根据提供的参数决定运行模式
    if args.prompt:
        # 单次问答模式
        response = generate_response(model, tokenizer, args.prompt, pixel_values, args)
        print("\n-------------------")
        print(f"问题: {args.prompt}")
        print(f"回答: {response}")
    else:
        # 交互模式
        interactive_mode(model, tokenizer, args)
    

if __name__ == "__main__":
    main()
