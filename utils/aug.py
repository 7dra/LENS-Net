import os

path = "datasets/test_dataset/ORI-4199/GT"

if not os.path.exists(path):
    print(f"路径 '{path}' 不存在！")
else:
    # 获取所有jpg文件
    jpg_files = [f for f in os.listdir(path) if f.lower().endswith('.png')]
    
    # 检查文件数量是否为2199
    if len(jpg_files) != 2199:
        print(f"警告：目录下共有 {len(jpg_files)} 个jpg文件，但期望2199个！")
    
    # 按当前顺序重命名
    for idx, filename in enumerate(jpg_files, start=1):
        old_path = os.path.join(path, filename)
        print(idx)
        idx = str(idx).zfill(4)
        new_path = os.path.join(path, f"{idx}.png")
        
        # 避免覆盖已有文件
        if os.path.exists(new_path):
            print(f"跳过：{new_path} 已存在！")
        else:
            os.rename(old_path, new_path)
            print(f"重命名：{filename} -> {idx}.png")