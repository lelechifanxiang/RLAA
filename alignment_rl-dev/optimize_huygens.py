"""优化Huygens积分的显存使用。

问题分析：
当前实现在 _huygens_integral 中创建了巨大的中间张量：
- phase_tilt: [num_fields, num_waves, num_batches, num_rays, grid_h, grid_w]
- 对于 1视场×3波长×1024光线×128×128网格，需要1.1GB显存
- 实际峰值显存达到4.6GB（包括所有中间张量）

优化方案：分块计算光线维度，降低显存峰值
"""
import torch
import time

def _huygens_integral_chunked(
    image_points,
    ray_directions,
    opl,
    chief_points,
    image_x,
    image_y,
    image_z,
    wavelength_mm,
    valid_points,
    pupil_weights,
    chunk_size: int = 256,
    compute_ideal_psf: bool = True,
):
    """分块计算Huygens积分，降低显存峰值。

    Args:
        chunk_size: 每次处理的光线数量，降低可减少显存使用
    """
    num_rays = image_points.shape[3]

    # 准备工作：计算公共活塞相位
    ray_to_chief = chief_points[:, :, :, None, :] - image_points
    path_at_chief = opl + torch.sum(ray_directions * ray_to_chief, dim=-1)
    integration_weight = torch.where(valid_points, pupil_weights.reshape(1, 1, 1, -1), 0.0)
    weight_sum = integration_weight.sum(dim=-1, keepdim=True)
    integration_weight = torch.where(
        weight_sum > 0.0,
        integration_weight / weight_sum.clamp_min(torch.finfo(torch.float64).eps),
        torch.zeros_like(integration_weight),
    )
    valid_path_at_chief = torch.where(valid_points, path_at_chief, torch.zeros_like(path_at_chief))
    piston = torch.sum(integration_weight * valid_path_at_chief, dim=-1, keepdim=True)
    relative_path_at_chief = path_at_chief - piston

    wave_number = 2.0 * torch.pi / wavelength_mm

    # 构建像面网格（需要广播到 [batch, field, wave, grid_h, grid_w, 3]）
    # image_x: [1, field, 1, 1, grid_w] -> [1, field, 1, grid_h, grid_w]
    # image_y: [1, field, 1, grid_h, 1] -> [1, field, 1, grid_h, grid_w]
    # image_z: [1, field, 1, 1, 1] -> [1, field, 1, grid_h, grid_w]
    grid_h = image_y.shape[-2]
    grid_w = image_x.shape[-1]

    image_grid_x = image_x.expand(-1, -1, -1, grid_h, -1)
    image_grid_y = image_y.expand(-1, -1, -1, -1, grid_w)
    image_grid_z = image_z.expand(-1, -1, -1, grid_h, grid_w)

    image_grid = torch.stack([image_grid_x, image_grid_y, image_grid_z], dim=-1)
    grid_offset = image_grid - chief_points[:, :, :, None, None, :]

    # 分块计算：累加复振幅
    amplitude = torch.zeros(
        (image_points.shape[0], image_points.shape[1], image_points.shape[2], grid_h, grid_w),
        dtype=torch.complex128,
        device=image_points.device,
    )

    for start_idx in range(0, num_rays, chunk_size):
        end_idx = min(start_idx + chunk_size, num_rays)

        # 当前块的数据切片
        ray_chunk = slice(start_idx, end_idx)
        chunk_directions = ray_directions[:, :, :, ray_chunk, :]
        chunk_path = relative_path_at_chief[:, :, :, ray_chunk]
        chunk_weight = integration_weight[:, :, :, ray_chunk]

        # 计算相位倾斜（这是显存瓶颈）
        phase_tilt = torch.sum(
            chunk_directions[:, :, :, :, None, None, :] * grid_offset[:, :, :, None, :, :, :],
            dim=-1,
        )

        # 计算复指数核
        phase = wave_number[:, :, :, None, None, None] * (
            chunk_path[:, :, :, :, None, None] + phase_tilt
        )
        kernel = torch.complex(torch.cos(phase), torch.sin(phase))
        kernel = kernel * chunk_weight[:, :, :, :, None, None]

        # 累加到振幅
        amplitude += kernel.sum(dim=3)

    # 计算PSF
    psf = torch.real(amplitude * torch.conj(amplitude))

    if not compute_ideal_psf:
        return psf, None

    # 理想PSF（衍射极限）
    # TODO: 实现理想PSF计算
    return psf, None


def benchmark_chunked():
    """对比原始实现与分块实现的性能。"""
    from optics_core.huygens_psf import _huygens_integral

    # 模拟输入数据
    num_fields = 1
    num_wavelengths = 3
    num_rays = 1024
    grid_size = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 生成测试数据
    image_points = torch.randn(1, num_fields, num_wavelengths, num_rays, 3, device=device, dtype=torch.float64)
    ray_directions = torch.randn(1, num_fields, num_wavelengths, num_rays, 3, device=device, dtype=torch.float64)
    ray_directions = ray_directions / ray_directions.norm(dim=-1, keepdim=True)
    opl = torch.randn(1, num_fields, num_wavelengths, num_rays, device=device, dtype=torch.float64)
    chief_points = torch.randn(1, num_fields, num_wavelengths, 3, device=device, dtype=torch.float64)
    wavelength_mm = torch.tensor([0.486, 0.588, 0.656], device=device, dtype=torch.float64).reshape(1, 1, 3)
    valid_points = torch.ones(1, num_fields, num_wavelengths, num_rays, device=device, dtype=torch.bool)
    pupil_weights = torch.ones(num_rays, device=device, dtype=torch.float64)

    # 生成网格（保持与原实现相同的shape）
    image_x = torch.linspace(-10, 10, grid_size, device=device, dtype=torch.float64).reshape(1, num_fields, 1, 1, grid_size)
    image_y = torch.linspace(-10, 10, grid_size, device=device, dtype=torch.float64).reshape(1, num_fields, 1, grid_size, 1)
    image_z = torch.zeros(1, num_fields, 1, 1, 1, device=device, dtype=torch.float64)

    print(f'Device: {device}')
    print(f'Config: {num_fields} fields, {num_wavelengths} waves, {num_rays} rays, {grid_size}×{grid_size} grid')
    print()

    # 测试分块实现
    for chunk_size in [128, 256, 512, 1024]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        start = time.time()
        psf, _ = _huygens_integral_chunked(
            image_points, ray_directions, opl, chief_points,
            image_x, image_y, image_z, wavelength_mm,
            valid_points, pupil_weights,
            chunk_size=chunk_size,
            compute_ideal_psf=False,
        )
        torch.cuda.synchronize()
        elapsed = time.time() - start

        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f'Chunked (chunk_size={chunk_size}): {elapsed*1000:.1f}ms, {peak_mem:.1f}MB peak')


if __name__ == '__main__':
    benchmark_chunked()
