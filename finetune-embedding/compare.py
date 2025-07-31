import torch

# 加载两个模型的 state_dict
base_sd = torch.load("similarity_model/pytorch_model.bin", map_location="cpu")  #使用 torch.load 加载到 CPU:contentReference[oaicite:0]{index=0}
ft_sd   = torch.load("finetuned_model/pytorch_model.bin", map_location="cpu")

changed_layers = []

for k in base_sd:
    if k in ft_sd:
        # 先统一转为 float32，再比较张量是否相近
        a = base_sd[k].float()
        b = ft_sd[k].float()
        if not torch.allclose(a, b, atol=1e-6, rtol=1e-5):  #torch.allclose 用于逐元素比较，超出容差即返回 False:contentReference[oaicite:1]{index=1}
            changed_layers.append(k)  #使用列表的 append 方法收集变化层:contentReference[oaicite:2]{index=2}

# 打印结果
print(f"共有 {len(changed_layers)} 层参数发生了变化：")
for name in changed_layers:
    print(f" - {name}")
