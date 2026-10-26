# 第一版 RGB 条件 → 物体未来视频

代码已接通数据、模型、训练和推理入口，供检查。正式训练尚未启动；下面的 GPU 训练命令尚未执行。

## 先看哪些文件

| 文件 | 检查重点 |
| --- | --- |
| [训练配置](../configs/obj_video_baseline_v1.yaml) | Stage-2 初始化、采样、前缀 loss 权重、10k steps |
| [dataset.py](../src/vpp2_obj/dataset.py) | 两路视频、共享去重索引、历史补齐、文本缓存 |
| [model.py](../src/vpp2_obj/model.py) | 25 帧 VAE 编码、原总 loss、分段日志、推理截取 |
| [train.py](../src/vpp2_obj/train.py) / [启动脚本](../scripts/train_obj_video.sh) | 训练前检查、冻结组件、复用上游 VideoTrainer |
| [infer.py](../src/vpp2_obj/infer.py) | 仅读取 RGB 条件与文本，输出 16 帧未来视频 |
| [检查脚本](../scripts/check_obj_video_baseline.py) | 独立解码核对、采样预览、可选冻结 VAE 重建 |

适配代码位于项目自己的 `src/vpp2_obj/`，复用 `third_party/vpp2/` 的模型与训练器。另对上游训练器的终端/W&B 吞吐日志补上梯度累积倍数；优化器和总 loss 公式不变。

## 梯度累积调用修复（2026-10-19）

早期短训练直接调用 DeepSpeed engine 委托的 `training_loss`，绕过了 engine 的 `forward`。本机 DeepSpeed 0.18.5 在 forward 输出上注册梯度缩放与反向准备 hook，因此旧路径在累积次数大于 1 时未正确除以累积次数。

现在 [distributed.py](../src/vpp2_obj/distributed.py) 在 `accelerator.prepare` 和恢复状态完成后，只为本次 engine 实例绑定 `training_loss → engine(...)`；模型新增 `forward → 原 training_loss`。这样运行完整 DeepSpeed 前向/反向生命周期，仍由 DeepSpeed 做累积平均，不再额外手工除 loss，前缀权重保持 1.0。

旧 `obj_video_batch4_acc8_smoke_v1` 完成两次更新，证明该配置当时没有 OOM，但不作为正确梯度累积的验收结果。修复后的完整 GPU 路径需要重测；旧短测不续训到正式训练。

累积 K 次时，吞吐应按 `optimizer steps/s × 每卡 batch × 卡数 × K` 计算。旧日志漏乘 K；原 profiler 的 optimizer-step 耗时已累计整个更新窗口，可以按正确全局 batch 重算。

## 数据与边界

保持现有划分：训练 2155 集，测试 240 集，105 集待复核 prompt 继续排除。每行使用当前 prompt 的文本缓存，保留真实 bool padding mask；不重新编码、不拼接额外提示词。

令 `s` 为去重后的当前帧序号，`N` 为去重后的集长：

```text
RGB 条件 9 帧：I0 + [I(s-210), I(s-180), ..., I(s-30), Is]
obj 未来 16 帧：[O(s+5), O(s+10), ..., O(s+80)]
训练 video：  [9 帧 RGB | 16 帧 obj]，形状 [3, 25, 224, 448]
condition：   [9 帧 RGB]，形状 [3, 9, 224, 448]
```

历史负序号用第 0 帧补齐，同时记录 `image_is_pad`；episode 首帧和当前帧为有效位置。训练仅采样 `0 <= s <= N-81`，保证 16 帧未来真实存在，不重复末帧充当未来监督。当前 2395 集全部至少有一个完整未来窗口。

先在去重序列上取位置，再通过同一 NPZ 的 `source_frame_indices` 映射到原 RGB/obj MP4。相同时间戳只保留第一帧，保留其他短时间间隔；没有转换到均匀 25/30 Hz。`210` 个保留帧间隔不保证精确 7 秒，实际时间可查看输出的时间戳。

训练按任务均衡、任务内按合法窗口抽样；随机性由 seed、epoch、样本序号决定。测试默认每集取固定中间窗口，推理 CLI 也支持指定 `s`。训练采样数设为 703872，使每个 epoch 恰好有 846 个完整 global batch；这是有放回采样次数，不是新增 episode。原始合法窗口数为训练 703130、测试 76576。

## 条件、损失与推理

从 `vpp2-video-stage2-17f.pth` 初始化，只训练 Video DiT。VAE、CLIP 冻结并保持 eval；UMT5 使用已有冻结缓存，不加载文本编码器。

完整 `[RGB9 | obj16]` 一次 VAE 编码为 7 个 latent 时间位置；RGB9 单独编码为 3 个条件位置。当前观测已在 RGB9 内，不再添加独立 observation 分支。CLIP 使用 RGB episode 首帧。

保留 `condition_latent_loss_weight=1.0` 和上游有效位置归一化、padding 规则、timestep 权重。训练优化原 `loss_video`，另记录 `loss_rgb_prefix`、`loss_obj_future`。无 padding 时，按相同 timestep 权重计算的总 loss 为 `(3 × RGB均值 + 4 × obj均值) / 7`；不会将两个分段均值直接相加作为新目标。

推理继续沿用上游对全部 7 个 latent 的联合去噪，不新增前缀 clamp。完整解码 25 帧，只导出 `[9:25]` 共 16 帧。推理接口只接收 RGB 条件、缓存文本和 mask；CLI 不读取 obj 真值。`--fps` 仅决定输出播放速度，不改变采样索引或仿真时间。

## 已完成的验证

- [33 项 CPU 测试全部通过](../runs/obj_video_baseline_v1_review/cpu_tests.log)，覆盖采样与补齐、两路视频对齐、缓存校验、任务均衡与多 worker 确定性、训练配置、推理不读真值，以及使用小型替身模型对比上游总 loss、梯度和采样行为。
- [真实数据检查](../runs/obj_video_baseline_v1_review/verification.json)：`adjust_bottle/episode_0000016` 的首、中、末 3 个合法窗口，独立解码的 RGB/obj 像素差均为 0。[中间窗口预览](../runs/obj_video_baseline_v1_review/window_001/contact_sheet.png)。
- [真实冻结 VAE 检查](../runs/obj_video_baseline_v1_vae_review/window_000/vae_check.json)：该集 `s=98`，完整 latent 为 `[1,16,7,28,56]`，条件 latent 为 `[1,16,3,28,56]`，前 3 个 latent 最大差为 0。[重建预览](../runs/obj_video_baseline_v1_vae_review/window_000/vae_reconstruction.png) 是真值的 VAE 重建，不是模型预测效果。

[汇总报告](../runs/obj_video_baseline_v1_review/code_review_summary.json)记录检查结果与代码哈希。

尚未验证完整 14B DiT 的 GPU 前向/反向、多卡训练、断点恢复或实际预测质量。CPU 替身测试和单窗口 VAE 验证不能代替这些检查。默认 micro-batch 的显存和吞吐仍需用短训练测量。

## 复查命令

在项目目录运行：

```bash
source env_train.sh
python -m unittest discover -s tests -p 'test_obj_video_*.py' -v
bash scripts/train_obj_video.sh --dry-run
bash scripts/train_obj_video.sh --config configs/obj_video_smoke_v1.yaml --dry-run
```

`--dry-run` 只检查清单与配置，不创建训练输出或加载模型。独立检查脚本会创建预览目录，已有目录拒绝覆盖，请使用新路径：

```bash
python scripts/check_obj_video_baseline.py \
  --config configs/obj_video_baseline_v1.yaml --split test --max-episodes 1 \
  --output runs/obj_video_review_new
```

推理入口可先做 dry-run；以下 Stage-2 checkpoint 仅用于验证入口参数，尚未针对物体视频微调：

```bash
python -m vpp2_obj.infer \
  --config configs/obj_video_baseline_v1.yaml \
  --checkpoint weights/checkpoints_video/vpp2-video-stage2-17f.pth \
  --task adjust_bottle --episode episode_0000016 --anchor 98 \
  --output runs/obj_video_infer_new --dry-run
```

实际预测时指定训练得到的 DiT checkpoint，移除 `--dry-run`，选择空闲 GPU。输出为 `future.mp4` 与记录条件索引、prompt、mask、checkpoint 来源的 `trace.json`。

## 后续训练命令（尚未执行）

启动脚本默认 GPU 4–7，4 个进程。正式配置为每卡 batch 1、梯度累积 208，global batch 832；学习率 `5e-6`、warmup 200、cosine、10000 optimizer steps。`max_steps` 控制训练长度，训练器会跨 epoch 继续训练。

先执行短训练配置，检查真实梯度、显存和吞吐；它只运行 2 个 optimizer steps，global batch 4，不保存大 checkpoint：

```bash
bash scripts/train_obj_video.sh \
  --config configs/obj_video_smoke_v1.yaml --allow-batch-change
```

完成检查后，正式配置的入口为：

```bash
bash scripts/train_obj_video.sh
```

需要重新运行时使用新的 `output_dir=...`，已有目录不会静默覆盖。恢复入口复用上游：`resume=...` 用于完整训练状态，`resume_ckpt=...` 用于权重初始化，两者互斥。真实恢复尚未经过 GPU 验证。
