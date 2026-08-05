#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于Focus-StyleGAN的图像增广系统Web API
提供RESTful API接口进行图像增广
"""

import os
import sys
import json
import base64
import logging
from pathlib import Path
from io import BytesIO
from datetime import datetime

# 添加项目根目录到路径（web/ 的上一级），以便导入 core / model 模块
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import numpy as np

# 导入集成增广器
try:
    from core.integrated_augmentor import IntegratedAugmentor
    AUGMENTOR_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入集成增广器: {e}")
    AUGMENTOR_AVAILABLE = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB限制
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'api_outputs'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}

# 创建目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局增广器实例
augmentor = None


def init_augmentor():
    """初始化增广器"""
    global augmentor
    if augmentor is None and AUGMENTOR_AVAILABLE:
        try:
            logger.info("初始化集成增广器...")
            augmentor = IntegratedAugmentor(
                config_path=str(_project_root / "core" / "integrated_config.yaml"),
                checkpoint_path=str(_project_root / "checkpoints" / "final_model.pth"),
                device=None
            )
            logger.info("集成增广器初始化完成")
        except Exception as e:
            logger.error(f"初始化增广器失败: {e}")
            augmentor = None
    return augmentor


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def image_to_base64(image_path):
    """将图像转换为base64字符串"""
    with open(image_path, "rb") as f:
        img_data = f.read()
    return base64.b64encode(img_data).decode('utf-8')


@app.route('/')
def index():
    """API主页"""
    return jsonify({
        "api": "Focus-StyleGAN图像增广系统API",
        "version": "1.0",
        "endpoints": {
            "GET /": "API信息",
            "POST /api/augment": "GAN增广单张图像",
            "POST /api/augment/real": "真实缺陷迁移增广（推荐）",
            "POST /api/augment/retrieval": "相似度检索增广",
            "POST /api/augment/stacking": "缺陷堆叠增广",
            "POST /api/augment/batch": "批量增广图像",
            "GET /api/model/info": "获取模型信息",
            "GET /api/health": "健康检查",
            "GET /api/categories": "获取可用缺陷类别",
            "GET /api/examples": "获取示例图像"
        },
        "status": "running" if AUGMENTOR_AVAILABLE else "augmentor_unavailable"
    })


@app.route('/api/health')
def health_check():
    """健康检查端点"""
    augmentor_status = "available" if init_augmentor() is not None else "unavailable"
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "augmentor": augmentor_status,
        "device": str(augmentor.device) if augmentor else None
    })


@app.route('/api/model/info')
def get_model_info():
    """获取模型信息"""
    augmentor_instance = init_augmentor()
    if augmentor_instance is None:
        return jsonify({"error": "增广器不可用"}), 503

    try:
        info = augmentor_instance.get_model_info()
        return jsonify({
            "success": True,
            "model_info": info
        })
    except Exception as e:
        logger.error(f"获取模型信息失败: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/augment', methods=['POST'])
def augment_single():
    """增广单张图像"""
    augmentor_instance = init_augmentor()
    if augmentor_instance is None:
        return jsonify({"error": "增广器不可用"}), 503

    try:
        # 检查是否有文件上传
        if 'image' not in request.files:
            return jsonify({"error": "没有上传文件"}), 400

        file = request.files['image']

        # 检查文件名
        if file.filename == '':
            return jsonify({"error": "没有选择文件"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "不支持的文件类型"}), 400

        # 获取参数
        n_variations = int(request.form.get('variations', 5))
        defect_intensity = float(request.form.get('intensity', 1.0))
        save_to_disk = request.form.get('save', 'true').lower() == 'true'

        # 保存上传的文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = Path(file.filename).stem
        upload_filename = f"{original_filename}_{timestamp}.png"
        upload_path = Path(app.config['UPLOAD_FOLDER']) / upload_filename

        # 读取并保存图像
        img = Image.open(file.stream)
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        img.save(upload_path, 'PNG')

        logger.info(f"上传图像: {upload_path}")

        # 创建输出目录
        output_dir = Path(app.config['OUTPUT_FOLDER']) / f"augment_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 增广图像
        if defect_intensity != 1.0:
            # 使用自定义缺陷强度
            generated_paths = []
            for i in range(n_variations):
                output_path = output_dir / f"augmented_{i}.png"
                augmentor_instance.generate_with_custom_intensity(
                    image_path=str(upload_path),
                    output_path=str(output_path),
                    defect_intensity=defect_intensity
                )
                generated_paths.append(output_path)
        else:
            # 使用默认增广
            generated_paths = augmentor_instance.augment_single_image(
                image_path=str(upload_path),
                output_dir=str(output_dir),
                n_variations=n_variations
            )

        # 准备响应
        result = {
            "success": True,
            "original_image": f"/api/files/{upload_path.name}",
            "augmented_count": len(generated_paths),
            "augmented_images": [],
            "output_dir": str(output_dir) if save_to_disk else None
        }

        # 将生成的图像转换为base64
        for i, img_path in enumerate(generated_paths):
            if save_to_disk:
                # 保存到磁盘，返回URL
                result["augmented_images"].append({
                    "url": f"/api/files/{img_path.name}",
                    "path": str(img_path)
                })
            else:
                # 转换为base64
                img_base64 = image_to_base64(img_path)
                result["augmented_images"].append({
                    "data": f"data:image/png;base64,{img_base64}",
                    "index": i
                })

        logger.info(f"增广完成: 生成 {len(generated_paths)} 张图像")
        return jsonify(result)

    except Exception as e:
        logger.error(f"增广失败: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/augment/batch', methods=['POST'])
def augment_batch():
    """批量增广图像"""
    augmentor_instance = init_augmentor()
    if augmentor_instance is None:
        return jsonify({"error": "增广器不可用"}), 503

    try:
        # 检查是否有文件上传
        if 'images' not in request.files:
            return jsonify({"error": "没有上传文件"}), 400

        files = request.files.getlist('images')
        if not files:
            return jsonify({"error": "没有选择文件"}), 400

        # 获取参数
        n_variations = int(request.form.get('variations', 3))
        save_to_disk = request.form.get('save', 'true').lower() == 'true'

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(app.config['OUTPUT_FOLDER']) / f"batch_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        uploaded_paths = []

        # 处理每个文件
        for i, file in enumerate(files):
            if file.filename == '' or not allowed_file(file.filename):
                continue

            try:
                # 保存上传的文件
                original_filename = Path(file.filename).stem
                upload_filename = f"{original_filename}_{i}_{timestamp}.png"
                upload_path = Path(app.config['UPLOAD_FOLDER']) / upload_filename

                img = Image.open(file.stream)
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                img.save(upload_path, 'PNG')
                uploaded_paths.append(upload_path)

                # 为每张图像创建子目录
                img_output_dir = output_dir / original_filename
                img_output_dir.mkdir(exist_ok=True)

                # 增广图像
                generated_paths = augmentor_instance.augment_single_image(
                    image_path=str(upload_path),
                    output_dir=str(img_output_dir),
                    n_variations=n_variations
                )

                results.append({
                    "original": file.filename,
                    "uploaded": str(upload_path),
                    "augmented_count": len(generated_paths),
                    "output_dir": str(img_output_dir) if save_to_disk else None,
                    "augmented_images": [
                        f"/api/files/{p.name}" for p in generated_paths
                    ] if save_to_disk else []
                })

                logger.info(f"处理完成 [{i+1}/{len(files)}]: {file.filename}")

            except Exception as e:
                logger.error(f"处理文件失败 {file.filename}: {e}")
                results.append({
                    "original": file.filename,
                    "error": str(e),
                    "success": False
                })

        return jsonify({
            "success": True,
            "timestamp": timestamp,
            "total_processed": len(results),
            "output_dir": str(output_dir),
            "results": results
        })

    except Exception as e:
        logger.error(f"批量增广失败: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/augment/real', methods=['POST'])
def augment_real():
    """使用真实缺陷图像进行增广（推荐）"""
    try:
        from core.integrated_augmentor import REAL_DEFECT_AVAILABLE
        if not REAL_DEFECT_AVAILABLE:
            return jsonify({"error": "真实缺陷模块不可用"}), 503

        from core.real_defect_blender import RealDefectBlender, get_categories

        if 'image' not in request.files:
            return jsonify({"error": "没有上传文件"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "没有选择文件"}), 400

        category = request.form.get('category', None)
        n_variations = int(request.form.get('variations', 5))
        n_defects = request.form.get('defects', None)
        if n_defects is not None:
            n_defects = int(n_defects)
        intensity = float(request.form.get('intensity', 1.0))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = Path(file.filename).stem
        upload_filename = f"{original_filename}_{timestamp}.png"
        upload_path = Path(app.config['UPLOAD_FOLDER']) / upload_filename

        img = Image.open(file.stream)
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        img.save(upload_path, 'PNG')

        output_dir = Path(app.config['OUTPUT_FOLDER']) / f"real_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        blender = RealDefectBlender(category=category)
        generated_paths = blender.generate_batch(
            good_image_path=str(upload_path),
            output_dir=str(output_dir),
            n_variations=n_variations,
            n_defects=n_defects,
            intensity=intensity,
        )

        result = {
            "success": True,
            "category": blender.category,
            "original_image": f"/api/files/{upload_path.name}",
            "augmented_count": len(generated_paths),
            "augmented_images": [
                {"url": f"/api/files/{p.name}", "path": str(p)}
                for p in generated_paths
            ],
            "output_dir": str(output_dir),
        }

        logger.info(f"真实缺陷增广: {len(generated_paths)} 张, 类别={blender.category}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"真实缺陷增广失败: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/augment/retrieval', methods=['POST'])
def augment_retrieval():
    """相似度检索增广：从数据集中查找最相似的同类图像"""
    try:
        from core.retrieval_augmentor import RetrievalAugmentor

        if 'image' not in request.files:
            return jsonify({"error": "没有上传文件"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "没有选择文件"}), 400

        category = request.form.get('category', None)
        if not category:
            return jsonify({"error": "需要指定category参数（产品类别）"}), 400

        n_variations = int(request.form.get('variations', 5))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = Path(file.filename).stem
        upload_filename = f"{original_filename}_{timestamp}.png"
        upload_path = Path(app.config['UPLOAD_FOLDER']) / upload_filename

        img = Image.open(file.stream)
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        img.save(upload_path, 'PNG')

        output_dir = Path(app.config['OUTPUT_FOLDER']) / f"retrieval_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        ret_aug = RetrievalAugmentor()
        ret_aug.build_index()
        generated_paths = ret_aug.augment_by_retrieval(
            str(upload_path), str(output_dir), category, n_variations=n_variations,
        )

        result = {
            "success": True,
            "category": category,
            "original_image": f"/api/files/{upload_path.name}",
            "augmented_count": len(generated_paths),
            "augmented_images": [
                {"url": f"/api/files/{p.name}", "path": str(p)}
                for p in generated_paths
            ],
            "output_dir": str(output_dir),
        }

        logger.info(f"检索增广: {len(generated_paths)} 张, 类别={category}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"检索增广失败: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/augment/stacking', methods=['POST'])
def augment_stacking():
    """缺陷堆叠增广：同类型不同属性缺陷堆叠融合"""
    try:
        from core.retrieval_augmentor import DefectStackingAugmentor

        if 'image' not in request.files:
            return jsonify({"error": "没有上传文件"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "没有选择文件"}), 400

        category = request.form.get('category', None)
        if not category:
            return jsonify({"error": "需要指定category参数（产品类别）"}), 400

        n_variations = int(request.form.get('variations', 5))
        n_defects = request.form.get('defects', None)
        if n_defects is not None:
            n_defects = int(n_defects)
        intensity = float(request.form.get('intensity', 1.0))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = Path(file.filename).stem
        upload_filename = f"{original_filename}_{timestamp}.png"
        upload_path = Path(app.config['UPLOAD_FOLDER']) / upload_filename

        img = Image.open(file.stream)
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        img.save(upload_path, 'PNG')

        output_dir = Path(app.config['OUTPUT_FOLDER']) / f"stacking_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        stack_aug = DefectStackingAugmentor(category=category)
        generated_paths = stack_aug.generate_batch(
            str(upload_path), str(output_dir),
            n_variations=n_variations,
            n_defects=n_defects,
            intensity=intensity,
        )

        result = {
            "success": True,
            "category": category,
            "original_image": f"/api/files/{upload_path.name}",
            "augmented_count": len(generated_paths),
            "augmented_images": [
                {"url": f"/api/files/{p.name}", "path": str(p)}
                for p in generated_paths
            ],
            "output_dir": str(output_dir),
        }

        logger.info(f"堆叠增广: {len(generated_paths)} 张, 类别={category}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"堆叠增广失败: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/augment/dataset', methods=['POST'])
def augment_dataset():
    """增广数据集目录"""
    augmentor_instance = init_augmentor()
    if augmentor_instance is None:
        return jsonify({"error": "增广器不可用"}), 503

    try:
        data = request.get_json()
        if not data or 'dataset_path' not in data:
            return jsonify({"error": "需要dataset_path参数"}), 400

        dataset_path = data['dataset_path']
        n_samples = data.get('samples', 100)
        output_dir = data.get('output_dir', None)

        if not Path(dataset_path).exists():
            return jsonify({"error": f"数据集目录不存在: {dataset_path}"}), 400

        # 增广数据集
        stats = augmentor_instance.augment_dataset(
            input_dir=dataset_path,
            output_dir=output_dir,
            n_samples=n_samples
        )

        return jsonify({
            "success": True,
            "dataset_path": dataset_path,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"数据集增广失败: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/files/<filename>')
def get_file(filename):
    """获取文件"""
    # 先在uploads目录查找
    upload_path = Path(app.config['UPLOAD_FOLDER']) / filename
    if upload_path.exists():
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    # 在outputs目录查找
    output_path = Path(app.config['OUTPUT_FOLDER'])
    for root, dirs, files in os.walk(output_path):
        if filename in files:
            return send_from_directory(root, filename)

    return jsonify({"error": "文件不存在"}), 404


@app.route('/api/examples')
def get_examples():
    """获取示例图像列表"""
    examples_dir = Path("example_images")
    examples = []

    if examples_dir.exists():
        for img_path in examples_dir.glob("*.jpg"):
            examples.append({
                "name": img_path.name,
                "path": str(img_path),
                "url": f"/api/files/{img_path.name}"
            })
        for img_path in examples_dir.glob("*.png"):
            examples.append({
                "name": img_path.name,
                "path": str(img_path),
                "url": f"/api/files/{img_path.name}"
            })

    return jsonify({
        "success": True,
        "examples": examples,
        "count": len(examples)
    })


@app.route('/api/categories')
def get_categories():
    """获取可用的产品类别和缺陷类型"""
    try:
        from core.real_defect_blender import get_categories as get_cats, get_defect_types as get_dts
        cats = get_cats()
        result = {}
        for c in cats:
            result[c] = get_dts(c)
        return jsonify({
            "success": True,
            "categories": result,
            "count": len(cats)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # 初始化增广器
    if AUGMENTOR_AVAILABLE:
        init_augmentor()

    print("=" * 60)
    print("基于Focus-StyleGAN的图像增广系统Web API")
    print("=" * 60)
    print(f"增广器状态: {'可用' if augmentor is not None else '不可用'}")
    print("API端点:")
    print("  GET  /                    - API信息")
    print("  GET  /api/health          - 健康检查")
    print("  GET  /api/model/info      - 模型信息")
    print("  POST /api/augment         - 增广单张图像")
    print("  POST /api/augment/batch   - 批量增广")
    print("  POST /api/augment/dataset - 增广数据集目录")
    print("  GET  /api/examples        - 示例图像")
    print("  GET  /api/files/<filename> - 获取文件")
    print()
    print("启动服务...")
    print("访问地址: http://127.0.0.1:5000")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)