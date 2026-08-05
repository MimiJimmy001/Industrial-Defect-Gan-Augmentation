import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from model.models.focus_stylegan import FocusStyleGAN
from model.utils.config import Config
config = Config(str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'core' / 'integrated_config.yaml'))
model = FocusStyleGAN(config=config.model)
print('模型创建成功')
import torch
batch_size = 2
real_images = torch.randn(batch_size, 3, 256, 256)
z_defect = torch.randn(batch_size, config.model.generator.latent_dim)
z_background = torch.randn(batch_size, config.model.generator.latent_dim)
print('开始前向传播...')
try:
    outputs = model(real_images, z_defect, z_background)
    print('前向传播成功')
    print('fused_images形状:', outputs['fused_images'].shape)
except Exception as e:
    print('错误:', e)
    import traceback
    traceback.print_exc()