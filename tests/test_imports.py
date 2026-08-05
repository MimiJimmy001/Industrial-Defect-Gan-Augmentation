#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试所有模块导入
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """测试所有关键模块导入"""
    modules_to_test = [
        ("src.data.loader", ["MVTecADDataset", "create_dataloader"]),
        ("src.models.focus_stylegan", ["FocusStyleGAN"]),
        ("src.training.trainer", ["FocusStyleGANTrainer", "OptunaOptimizer"]),
        ("src.evaluation.evaluator", ["Evaluator"]),
        ("src.augmentation.augmentor", ["Augmentor"]),
        ("src.utils.config", ["Config"]),
        ("src.utils.logger", ["setup_logger"]),
    ]

    print("=" * 60)
    print("测试模块导入")
    print("=" * 60)

    all_passed = True

    for module_path, imports in modules_to_test:
        try:
            module = __import__(module_path, fromlist=imports)
            for import_name in imports:
                if hasattr(module, import_name):
                    print(f"[OK] {module_path}.{import_name} 导入成功")
                else:
                    print(f"[FAIL] {module_path}.{import_name} 未找到")
                    all_passed = False
        except Exception as e:
            print(f"[FAIL] {module_path} 导入失败: {e}")
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("所有模块导入成功!")
    else:
        print("部分模块导入失败!")

    return all_passed

def test_config_loading():
    """测试配置文件加载"""
    print("\n" + "=" * 60)
    print("测试配置文件加载")
    print("=" * 60)

    try:
        from model.utils.config import Config
        config = Config(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "integrated_config.yaml"))
        print(f"[OK] 配置文件加载成功")
        print(f"  图像尺寸: {config.data.image_size}")
        print(f"  批次大小: {config.data.batch_size}")
        print(f"  训练轮数: {config.training.epochs}")
        return True
    except Exception as e:
        print(f"[FAIL] 配置文件加载失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n" + "=" * 60)
    print("测试模型创建")
    print("=" * 60)

    try:
        import yaml
        from model.models.focus_stylegan import FocusStyleGAN

        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "integrated_config.yaml"), "r", encoding='utf-8') as f:
            config = yaml.safe_load(f)

        model = FocusStyleGAN(config['model'])
        print(f"[OK] FocusStyleGAN模型创建成功")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"[FAIL] 模型创建失败: {e}")
        return False

if __name__ == "__main__":
    import_test = test_imports()
    config_test = test_config_loading()
    model_test = test_model_creation()

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"模块导入: {'通过' if import_test else '失败'}")
    print(f"配置加载: {'通过' if config_test else '失败'}")
    print(f"模型创建: {'通过' if model_test else '失败'}")

    if all([import_test, config_test, model_test]):
        print("\n[SUCCESS] 所有测试通过!")
    else:
        print("\n[FAILURE] 部分测试失败，请检查代码")