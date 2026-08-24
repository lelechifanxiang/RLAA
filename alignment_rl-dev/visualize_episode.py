"""单次对准过程可视化 CLI 入口。"""
from __future__ import annotations

import argparse
import os

from visualization.lens_2d_visualizer import render_lens_2d_frame
from visualization.lens_multi_visualizer import render_lens_multi_frame
from visualization.visualize_utils import (
    generate_frames,
    infer_lens_dof_names,
    infer_task,
    is_single_lens_2d_case,
    load_manual_results,
    load_results_file,
    make_gif,
    normalize_episode_data,
    print_available_methods,
    safe_name,
)


def select_renderer(results_data: dict, method: str, episode_idx: int):
    task, config, _ep, states, _metric_values, _mtf_obs_log, _threshold = normalize_episode_data(
        results_data,
        method,
        episode_idx,
    )

    dof_names = infer_lens_dof_names(config, len(states[0]))
    if is_single_lens_2d_case(config, len(dof_names)):
        return render_lens_2d_frame
    return render_lens_multi_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="镜头对准过程逐步可视化（帧序列 / GIF / 手动 rollout）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--results", default=None, metavar="PKL_PATH", help="evaluate.py 保存的 .pkl 结果文件路径；不提供时进入手动 rollout 模式")
    parser.add_argument("--model_path", default=None, help="手动 rollout 模式下指定 RL 模型路径")
    parser.add_argument("--algo", default="sac", choices=["sac", "td3", "ppo"], help="手动 rollout 模式下的算法类型")
    parser.add_argument("--method", default=None, help="算法名称，如 'RL (SAC)' 或 'Hill Climbing'；pkl 模式下不指定时列出所有可用方法")
    parser.add_argument("--episode", type=int, default=0, help="要可视化的 episode 索引（默认 0）")
    parser.add_argument("--seed", type=int, default=42, help="手动 rollout 模式的起始随机种子（目标 episode 使用 seed + episode）")
    parser.add_argument("--output", default=None, metavar="DIR", help="帧图片输出目录（默认自动生成）")
    parser.add_argument("--stride", type=int, default=1, help="帧间隔步数（默认 1 = 每步一帧；对长 episode 可设为 5~10）")
    parser.add_argument("--dpi", type=int, default=120, help="图片分辨率 DPI（默认 120）")
    parser.add_argument("--make_gif", action="store_true", help="生成帧后自动合成 GIF（需要 imageio：pip install imageio）")
    parser.add_argument("--fps", type=int, default=8, help="GIF 帧率，单位 fps（默认 8）")
    args = parser.parse_args()

    if args.results is None and args.model_path is None:
        parser.error("未提供 --results 时，必须提供 --model_path 进入手动 rollout 模式")

    if args.results is not None:
        results_data = load_results_file(args.results)
        task = infer_task(results_data)
        if args.method is None:
            print_available_methods(results_data)
            return
        method_name = args.method
        display_episode_idx = args.episode
        actual_episode_idx = args.episode
    else:
        results_data, method_name, display_episode_idx = load_manual_results(
            task="lens",
            model_path=args.model_path,
            algo=args.algo,
            episode_idx=args.episode,
            seed=args.seed,
        )
        task = "lens"
        actual_episode_idx = 0
        print(f"手动 rollout 完成：task=lens, algo={args.algo}, episode={args.episode}, seed={args.seed}")

    renderer = select_renderer(results_data, method_name, actual_episode_idx)
    output_dir = args.output or os.path.join("results", f"frames_{safe_name(method_name)}_lens_ep{display_episode_idx}")
    frame_paths = generate_frames(
        results_data,
        method=method_name,
        episode_idx=actual_episode_idx,
        output_dir=output_dir,
        renderer=renderer,
        stride=args.stride,
        dpi=args.dpi,
        display_episode_idx=display_episode_idx,
    )

    if args.make_gif and frame_paths:
        gif_path = output_dir + ".gif"
        make_gif(frame_paths, gif_path, fps=args.fps)

    print(f"\n完成！{len(frame_paths)} 帧已保存至 {output_dir}/")


if __name__ == "__main__":
    main()
