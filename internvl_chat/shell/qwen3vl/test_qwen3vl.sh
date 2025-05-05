
python /fs-computility/ai-shen/mllm_safety-shared/projects/mllm-reasoning/Qwen3-VL/internvl_chat/qwen3vl/test_qwen3vl.py \
    --checkpoint /fs-computility/ai-shen/mllm_safety-shared/projects/mllm-reasoning/Qwen3-VL/internvl_chat/checkpoints/checkpoint-1.7b-nothink-0505/checkpoint-100 \
    --max-new-tokens 512 \
    --do-sample \
    --verbose \
    --prompt "你是谁？" \
    # --image /fs-computility/ai-shen/mllm_safety-shared/projects/mllm-reasoning/Qwen3-VL/internvl_chat/examples/image1.jpg \
    



    # # 模型相关参数
    # parser.add_argument("--checkpoint", type=str, default='internvl_chat/checkpoint-4b/',
    #                    help="模型检查点路径")
    # parser.add_argument("--load-in-8bit", action="store_true", 
    #                    help="以8位精度加载模型以节省显存")
    # parser.add_argument("--load-in-4bit", action="store_true",
    #                    help="以4位精度加载模型以节省更多显存")
    # parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
    #                    help="运行设备 (cuda/cpu)")
    
    # # 图像相关参数
    # parser.add_argument("--image", type=str, default=None,
    #                    help="图像文件路径，如果不提供则使用纯文本模式")
    # parser.add_argument("--dynamic-image", action="store_true",
    #                    help="使用动态图像预处理")
    # parser.add_argument("--max-patches", type=int, default=12,
    #                    help="最大图像块数(仅在动态图像处理时使用)")
    # parser.add_argument("--image-size", type=int, default=None,
    #                    help="图像尺寸，默认使用模型配置")
    # parser.add_argument("--use-thumbnail", action="store_true",
    #                    help="是否使用缩略图")
    
    # # 生成相关参数
    # parser.add_argument("--prompt", type=str, default=None,
    #                    help="输入提示文本，如果不提供则进入交互模式")
    # parser.add_argument("--max-new-tokens", type=int, default=512,
    #                    help="生成的最大token数")
    # parser.add_argument("--num-beams", type=int, default=1,
    #                    help="束搜索的束数，1表示贪婪搜索")
    # parser.add_argument("--do-sample", action="store_true",
    #                    help="是否使用采样生成")
    # parser.add_argument("--temperature", type=float, default=1.0,
    #                    help="生成的温度参数，较低的值使输出更确定性，较高的值增加随机性")
    # parser.add_argument("--top-p", type=float, default=0.9,
    #                    help="nucleus采样的概率阈值")
    # parser.add_argument("--top-k", type=int, default=50,
    #                    help="top-k采样的k值")
    # parser.add_argument("--verbose", action="store_true",
    #                    help="显示详细输出")