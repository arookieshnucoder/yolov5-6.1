import torch

# 检查CUDA是否可用
cuda_available = torch.cuda.is_available()
if cuda_available:
    print("CUDA 可用！")
else:
    print("CUDA 不可用。")

if cuda_available:
    # 获取当前可用的GPU数量
    gpu_count = torch.cuda.device_count()
    print(f"当前有 {gpu_count} 个GPU 可用：")

    # 遍历每个GPU并打印其名称和显存大小
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # 显存大小，单位GB
        print(f"GPU {i}: {gpu_name}, 显存大小: {gpu_memory:.2f} GB")

