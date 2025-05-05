import sys
sys.path.append("/fs-computility/ai-shen/mllm_safety-shared/projects/mllm-reasoning/Qwen3-VL/internvl_chat/")
sys.path.append("/fs-computility/ai-shen/mllm_safety-shared/projects/mllm-reasoning/Qwen3-VL/internvl_chat/internvl")

from internvl.train.internvl_chat_finetune import main
if __name__ == "__main__":
    main()
