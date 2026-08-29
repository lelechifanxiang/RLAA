"""
使用Python生成Mermaid框图的PNG图片
"""
import subprocess
import sys

def generate_mermaid_png(mmd_file: str, output_file: str):
    """使用mermaid.ink API生成PNG"""
    import urllib.request
    import urllib.parse
    import base64

    # 读取mermaid文件
    with open(mmd_file, 'r', encoding='utf-8') as f:
        mermaid_code = f.read()

    # 使用mermaid.ink在线服务
    # 编码mermaid代码
    encoded = base64.b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
    url = f"https://mermaid.ink/img/{encoded}"

    print(f"生成 {output_file}...")
    print(f"URL: {url}")

    # 下载图片
    try:
        urllib.request.urlretrieve(url, output_file)
        print(f"✓ 成功生成: {output_file}")
        return True
    except Exception as e:
        print(f"✗ 生成失败: {e}")
        return False

if __name__ == "__main__":
    # 生成两个图片
    print("=" * 60)
    print("生成RLAA系统架构图")
    print("=" * 60)

    success1 = generate_mermaid_png(
        "optical_modeling.mmd",
        "optical_modeling.png"
    )

    print()

    success2 = generate_mermaid_png(
        "rl_training.mmd",
        "rl_training.png"
    )

    print()
    print("=" * 60)
    if success1 and success2:
        print("✓ 所有图片生成成功！")
        print()
        print("生成的图片:")
        print("  1. optical_modeling.png  - 光学建模体系")
        print("  2. rl_training.png       - RL训练体系")
    else:
        print("✗ 部分图片生成失败")
    print("=" * 60)
