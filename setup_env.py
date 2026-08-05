"""
环境安装小工具
1. 让用户选择安装 GPU 版（CUDA）还是 CPU 版 PyTorch
2. 逐项检查已装依赖，已满足要求的跳过，只装缺失/不符合的

用法：
    python setup_env.py
"""
import subprocess
import sys
from importlib import metadata

# pip 包名 -> 最低版本（None 表示不检查版本，装了就行）
DEPENDENCIES = {
    "Flask": "2.3.0",
    "Pillow": "10.0.0",
    "numpy": "1.24.0",
    "opencv-python": "4.8.0",
    "tqdm": "4.65.0",
    "PyYAML": "6.0",
    "lpips": "0.1.4",
    "scipy": "1.10.0",
    "scikit-learn": "1.3.0",
    "optuna": "3.3.0",
    "matplotlib": "3.7.0",
    "albumentations": "1.3.0",
}

# 有版本上限的依赖：pip 包名 -> (最低版本, 最高版本-不含)
DEPENDENCY_CAPS = {
    "numpy": ("1.24.0", "2.0.0"),
}

TORCH_PACKAGES = ["torch", "torchvision", "torchaudio"]

PYTORCH_INDEX = {
    "1": ("CUDA 12.1", "https://download.pytorch.org/whl/cu121"),
    "2": ("CUDA 11.8", "https://download.pytorch.org/whl/cu118"),
    "3": ("CPU", "https://download.pytorch.org/whl/cpu"),
}


def parse_ver(v):
    parts = []
    for x in v.split("."):
        num = "".join(c for c in x if c.isdigit())
        parts.append(int(num) if num else 0)
    return tuple(parts)


def installed_version(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def run_pip(args):
    print(f"\n>>> pip {' '.join(args)}\n")
    return subprocess.call([sys.executable, "-m", "pip", *args]) == 0


def choose_pytorch():
    print("=" * 50)
    print("请选择 PyTorch 安装版本：")
    for key, (name, _) in PYTORCH_INDEX.items():
        print(f"  [{key}] {name}")
    print("=" * 50)
    while True:
        choice = input("输入序号（默认 1）: ").strip() or "1"
        if choice in PYTORCH_INDEX:
            return choice, *PYTORCH_INDEX[choice]
        print("输入无效，请重新输入")


def check_torch(choice, index_url):
    """检查 torch 是否已装且类型（CPU/GPU）与选择一致"""
    ver = installed_version("torch")
    want_gpu = choice in ("1", "2")
    if ver is None:
        print("[torch] 未安装，需要安装")
        return False
    is_cuda_build = "+cu" in ver
    if want_gpu and is_cuda_build:
        print(f"[torch] 已安装 GPU 版 ({ver})，跳过")
        return True
    if not want_gpu and not is_cuda_build:
        print(f"[torch] 已安装 CPU 版 ({ver})，跳过")
        return True
    cur = "GPU" if is_cuda_build else "CPU"
    want = "GPU" if want_gpu else "CPU"
    print(f"[torch] 已安装 {cur} 版 ({ver})，与选择的 {want} 版不一致，需要重装")
    return False


def install_torch(index_url):
    return run_pip(["install", *TORCH_PACKAGES, "--index-url", index_url])


def check_dependencies():
    missing = []
    for pkg, min_ver in DEPENDENCIES.items():
        ver = installed_version(pkg)
        if ver is None:
            print(f"[{pkg}] 未安装")
            missing.append(pkg)
            continue
        cap = DEPENDENCY_CAPS.get(pkg)
        if cap and parse_ver(ver) >= parse_ver(cap[1]):
            print(f"[{pkg}] 版本过高 ({ver} >= {cap[1]})，需要降级")
            missing.append(f"{pkg}>={cap[0]},<{cap[1]}")
        elif min_ver and parse_ver(ver) < parse_ver(min_ver):
            print(f"[{pkg}] 版本过低 ({ver} < {min_ver})，需要升级")
            missing.append(f"{pkg}>={min_ver}")
        else:
            print(f"[{pkg}] 已安装 ({ver})，跳过")
    return missing


def main():
    print("Focus-StyleGAN 环境安装工具\n")

    # 1. PyTorch
    choice, name, index_url = choose_pytorch()
    if not check_torch(choice, index_url):
        print(f"\n准备安装 PyTorch（{name}）...")
        if not install_torch(index_url):
            print("\n[失败] PyTorch 安装失败，请检查网络后重试")
            sys.exit(1)
    else:
        pass

    # 2. 其余依赖
    print("\n检查其余依赖...")
    missing = check_dependencies()
    if missing:
        print(f"\n需要安装/升级 {len(missing)} 项：{', '.join(missing)}")
        if not run_pip(["install", *missing]):
            print("\n[失败] 部分依赖安装失败，请检查上方报错")
            sys.exit(1)
    else:
        print("\n所有依赖均已满足，无需安装")

    print("\n环境检查完成！")


if __name__ == "__main__":
    main()
