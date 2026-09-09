# 实时 Human Mesh 流水线优化记录

本文记录 `mesh_live_o3d.py` 中 HMR2 与 HybrIK 两条实时人体 mesh
流水线的延迟优化、RGB-D 配准、可见面定义、离开 FOV 后的姿态保持，以及
临时双模型对比工具。记录对应当前代码状态，后续修改时应保证两个模型的
公共优化同步生效。

## 1. 当前入口和模型

- 正式统一入口：`mesh_live_o3d.py`
  - `--engine hmr2`：HMR2.0 / 4D-Humans，ViT-H，默认模型。
  - `--engine hybrik`：HybrIK，HRNet-W48。
- 公共实时流水线：`hybrik_live_o3d.py`
- 双模型叠加对比：`compare_pose_overlay_tmp.py`
  - HMR2 为红色。
  - HybrIK 为蓝色。
  - 同时显示原始 RGB-D 点云。
- HMR2 TensorRT 运行时：`hmr2_trt_runtime.py`
- HMR2 TensorRT 导出工具：`export_hmr2_trt.py`

正式入口通过替换 `hybrik_live_o3d.py` 中的模型加载和推理 hook 来切换
模型。YOLO、深度定位、配准、平滑、离屏保持、Open3D 渲染和 profiling
均复用同一套代码，因此公共逻辑的改动会同时作用于 HMR2 和 HybrIK。

运行示例：

```bash
DISPLAY=:0 ~/anaconda3/envs/human-pose/bin/python mesh_live_o3d.py --engine hmr2
DISPLAY=:0 ~/anaconda3/envs/human-pose/bin/python mesh_live_o3d.py --engine hybrik
DISPLAY=:0 ~/anaconda3/envs/human-pose/bin/python compare_pose_overlay_tmp.py
```

## 2. 当前端到端处理链路

一帧 mesh 的关键路径如下：

1. ROS2 接收 RGB、深度和相机内参，QoS 使用 `KEEP_LAST/depth=1`。
2. 解码 RGB 和毫米深度，深度转换为米。
3. YOLO segmentation 找到最大人体，得到 bbox 和 person mask。
4. HMR2 或 HybrIK 预测 root-relative SMPL mesh。
5. 使用模型预测的 pelvis 像素和人体深度，将 pelvis 放置到相机坐标系。
6. 从当前相机位置计算 FOV 内的 SMPL 前表面。
7. 反投影 person mask 内的深度，生成配准目标人体点云。
8. 执行深度方向 refine。
9. 执行有界的上半身 4DOF（XYZ + yaw）refine。
10. 对 mesh 和 joints 做时间 EMA。
11. 对离开 FOV 或发生不合理折叠的身体部位执行 torso-relative hold。
12. 重新计算最终可见面，构造 Open3D mesh、骨架和关节几何。
13. GUI 主线程只替换最新几何并记录 topic-to-render 延迟。

原始点云走独立快速线程，不再阻塞 mesh 推理。

## 3. Profiling 方法

`hybrik_live_o3d.py` 对以下阶段使用 `time.perf_counter()` 计时：

- `topic_wait`
- `decode_copy`
- `yolo`
- `detection_post`
- `model_pre`
- `model_gpu`
- `model_post`
- `metric_root`
- `cloud_refine`
- `upper_4dof_refine`
- `temporal_head`
- `visible_surface`
- `geometry_build`
- `worker_total`
- `render_submit`
- `topic_to_render`

关闭 Open3D 窗口时，程序丢弃前 5 个冷启动样本，输出各阶段的中位数和
P90。GPU 模型计时前后使用 CUDA synchronize，避免只测到异步 kernel
提交时间。

最近一次 HMR2 正式流水线实测（约 274 个 warm frames）：

| 阶段 | 中位数 | P90 |
|---|---:|---:|
| topic wait | 10.67 ms | 29.67 ms |
| decode/copy | 0.84 ms | 1.49 ms |
| YOLO | 25.91 ms | 30.38 ms |
| detection post | 1.27 ms | 1.65 ms |
| model preprocess | 1.83 ms | 2.91 ms |
| model GPU | 14.79 ms | 17.84 ms |
| model postprocess | 0.41 ms | 0.91 ms |
| metric root | 0.16 ms | 1.69 ms |
| cloud refine | 2.99 ms | 4.38 ms |
| upper-body 4DOF | 0.11 ms | 3.71 ms |
| temporal/part hold | 3.28 ms | 4.97 ms |
| visible surface | 11.29 ms | 14.44 ms |
| geometry build | 1.23 ms | 2.35 ms |
| worker total | 66.96 ms | 73.73 ms |
| render submit | 0.84 ms | 1.01 ms |
| topic to render | 85.75 ms | 103.85 ms |

另一次加入更严格可见性实验后的 HMR2 测量约为 93 ms 中位
topic-to-render。该实验的 mask/depth 交集后来因可见区域过小而撤销，
当前实现只保留 HPR 与 FOV 条件。

早期用户日志中的 HybrIK 单帧时间为
`95, 135, 146, 153, 120, 100, 131, 93 ms`，简单平均约 `121.6 ms`。
这组数字来自不同瞬时负载，不应替代完整的 median/P90 profiling。

## 4. 第一轮：调度、点云和渲染优化

### 4.1 最新帧优先

ROS QoS 设置为 `KEEP_LAST`、队列深度 1。推理线程每次读取当前最新的
消息引用，不维护待处理帧队列，避免推理速度低于相机帧率时延迟持续累积。

GUI 侧设置 `mesh_update_pending`。如果主线程尚未处理上一次回调，推理线程
不会继续堆积 mesh 更新任务；回调执行时读取最新 `mesh_geo`，自动丢弃已经
过时的中间显示结果。

### 4.2 点云与 mesh 解耦

点云和 mesh 使用两个生产线程：

- 点云线程只做深度反投影和显示几何构造。
- mesh 线程执行 YOLO、模型、配准和 mesh 构造。
- GUI 主线程仅上传并替换几何。

当前参数：

```python
CLOUD_HZ = 15
DISPLAY_STRIDE = 6
```

提高点云频率或减小 stride 会增加 CPU、GPU 上传和 GIL 竞争，mesh 延迟会
上升。`15 Hz / stride 6` 是当前的交互性与 mesh 延迟折中。

`person_cloud()` 同样使用 stride 6，避免配准目标点数过大。Open3D
`Vector3dVector` 输入显式转成连续 `float64`，避免逐元素转换的高开销。

### 4.3 几何构造移出 GUI 线程

`compute_vertex_normals`、骨架和关节球体构造均在 worker 中完成。GUI 回调
只执行 remove/add geometry，使 topic-to-render 更稳定。

## 5. 模型推理优化

### 5.1 通用 PyTorch 设置

- 推理使用 `torch.inference_mode()`。
- 固定输入尺寸启用 `torch.backends.cudnn.benchmark = True`。
- HybrIK HRNet 使用 `channels_last`，让卷积优先使用合适的 Tensor Core /
  cuDNN kernel。
- 关闭 HybrIK flip test：

```python
FLIP_TEST = False
```

flip test 需要第二次前向传播，不适合低延迟 live 模式。

### 5.2 YOLO TensorRT

当前优先使用：

```text
/home/oliver/Documents/semantic_perception/yolo26m-seg-custom_20260817_rtx4070.engine
```

engine 不存在时自动回退：

```text
/home/oliver/Documents/semantic_perception/yolo26m-seg-custom_20260817.pt
```

实测 YOLO 仍约占 26 ms，是当前关键路径中的主要耗时之一。

### 5.3 HMR2 TensorRT

HMR2 的 ViT backbone 与 SMPL transformer head 导出为固定
`1x3x256x256` FP16 TensorRT engine：

```text
engines/hmr2_vit_head_256_fp16.engine
```

运行时采用 TensorRT 10 tensor API，并让输入输出直接绑定 PyTorch CUDA
tensor 地址。TensorRT 只运行神经网络回归器；SMPL forward 保持 FP32，
避免姿态旋转和蒙皮精度问题。

载入 TRT engine 后释放 PyTorch 的 `backbone` 和 `smpl_head`，降低显存
占用。engine 不可用时自动回退 PyTorch FP16 eager。

TRT 与原模型的输出做过数值对比，误差较小；最近实测 HMR2 `model_gpu`
中位数约 15 ms。此前尝试过 `torch.compile`，在当前模型/环境中没有稳定
收益且存在失败，最终移除。

## 6. 深度定位和配准

### 6.1 metric grounding

单目模型输出的 mesh 先保持 root-relative，再使用 pelvis 的真实图像位置
和人体深度确定相机坐标系中的根节点。

HMR2 必须使用其扩展 44-joint 定义中的 pelvis index 39。不能直接用 bbox
中心：bbox 中心曾造成约 50 像素、约 20 cm 的横向偏移，而后续仅 z 方向
refine 无法纠正这种误差。

深度测量是人体前表面，pelvis 位于躯干内部，因此使用：

```python
ROOT_OFFSET = 0.10
```

### 6.2 相机可见 mesh 表面

当前 `visible_vertices()` 的定义为：

```text
可见源顶点 = HPR 前表面 ∩ 投影位于相机 FOV
```

- HPR（hidden point removal）从相机原点剔除 mesh 背面和自遮挡表面。
- 相机内参投影剔除图像范围外的顶点。
- 上一帧的 HPR 顶点索引可以复用，但每帧必须用当前姿态重新执行 FOV 投影。

绿色表示上述 mesh 源表面，深蓝色表示 mesh 自遮挡、背面或 FOV 外部分。

重要：不要把源 mesh 可见面再与 YOLO mask 或逐像素深度做交集。曾尝试：

```text
HPR ∩ FOV ∩ person mask ∩ valid depth ∩ depth occlusion
```

结果绿色面积显著小于实际可见表面，原因包括：

- segmentation 在头发、手和轮廓处会内缩；
- mesh 在配准前本来就与深度存在误差；
- 深度孔洞和边缘混合像素会错误剔除顶点。

正确的职责划分是：

- HPR + FOV 定义配准的 mesh 源表面；
- person mask + depth 定义配准的目标人体点云。

### 6.3 z-only refine

第一阶段 refine 只优化全局 z：

- 横向位置由 pelvis 像素和相机内参可靠确定。
- 全 6DOF ICP 曾出现约 92° 翻转和约 2 m 平移的错误低残差解。
- z-only 避免 mesh 在光滑躯干表面横向滑动。
- 最多 2 次迭代。
- 最大对应距离 0.35 m。
- 总 z 位移限制为 0.4 m。
- 有效对应少于 100 个时不更新。
- 加权中位数代替均值，降低离群点影响。

### 6.4 上半身 4DOF refine

第二阶段优化全局 XYZ 与绕竖直轴 yaw，但仅用相机可见的上半身顶点：

```text
spine, neck, collars, shoulders, elbows, wrists, hands
```

关键约束：

- 最多采样 700 个源顶点。
- 首帧 yaw 候选为 `[-8°, -4°, 0°, 4°, 8°]`。
- 后续帧以上一帧为 warm start，只测 `previous ± 2°` 和 previous。
- XYZ shift 限制为：
  - x/y：±8 cm
  - z：±6 cm
- 对应距离阈值 20 cm。
- 使用 80% 截断后的加权距离评分。
- 评分中加入相对上一帧 shift 和 yaw 的小幅连续性惩罚。

所以并非每帧从零开始搜索；正常情况下从上一帧的 yaw 和 shift 开始。
没有检测到人时会清空 warm-start 状态。

HMR2 和 HybrIK 都启用了相同的 4DOF 算法。区别仅在顶点可靠性权重：

- HybrIK 使用模型输出的逐关节 `pred_sigma`。
- HMR2 没有同类 uncertainty，使用固定 SMPL-24 先验。

HMR2 固定权重特意提高肩、肘、腕的可靠性，并降低头部和下肢的影响。
`sigma` 越小，配准权重越高；肩、肘、腕设为 0.004，脊柱约
0.006–0.007，默认值 0.020。头部还乘以 `HEAD_W = 0.10`，因为 RGB-D
点云包含头发而裸 SMPL 头模更小，直接强配准容易拉偏身体。

## 7. 时间平滑与离开 FOV 的防折叠

### 7.1 EMA

mesh 和 joints 使用：

```python
MeshEMA(a=0.6)
```

时间滤波位于配准后、离屏部位保持前。它抑制逐帧抖动，但不会单独解决模型
对不可见身体部位产生的不合理姿态。

### 7.2 问题原因

当头、手臂或腿离开 crop/FOV 后，单帧模型没有图像证据，姿态由训练先验
决定。模型可能把不可见关节折回身体，而不是保持中立姿势。EMA 只能平滑
折叠过程，无法判断这个姿态不合理。

### 7.3 torso-relative part hold

`HeadHold` 目前实际处理以下部位：

- head
- left/right arm
- left/right leg

算法：

1. 使用 pelvis、neck 和左右肩构造当前 torso frame。
2. 当一个部位可靠可见时，将该部位 mesh 顶点缓存到 torso-local 坐标。
3. 根据 person mask 接近画面边界、部位中心投影出画面等条件判定截断。
4. 头部额外检查：
   - 头部中心沿 torso-up 是否明显高于 neck；
   - 当前局部头部中心与 neutral rest head 的偏差是否过大。
5. 截断或折叠时，将最后一次可靠的 torso-local 部位重新嵌入当前躯干。
6. 如果没有可靠历史，使用 SMPL T-pose 中的 neutral local part。
7. 替换 mesh 后用 `J_REGRESSOR @ vertices` 重算 24 个关节，使骨架和表面
   一致。

最初只替换 `VERT_JOINT == 15` 的头盖顶点。虽然日志显示
`head-hold=True`，蓝色 HybrIK 头部仍会折叠，因为被 neck joint 主导的
头颈连接环没有被替换。当前改为：

```python
PART_MASKS["head"] = lbs[:, 15] > 0.05
```

即使用 SMPL LBS 权重大于 0.05 的连续头颈过渡区域，避免“头盖已保持、
颈部仍向下拉”的接缝问题。

HMR2 和 HybrIK 必须各自保存 `VERT_JOINT`、`PART_MASKS`、
`PART_LOCAL_REST` 和 `J_REGRESSOR`。叠加脚本顺序运行两个模型时会切换到
对应模型的 SMPL 表，两个 `HeadHold` 和两个 4DOF state 也彼此独立。

## 8. 双模型对比工具

`compare_pose_overlay_tmp.py` 用于在完全相同的 RGB-D 帧、检测结果和点云
上比较：

- 红色：HMR2
- 蓝色：HybrIK
- 原始 RGB-D 点云

两个模型分别执行：

- 自己的模型推理；
- 相同 metric grounding；
- 相机 HPR + FOV 可见源面；
- z-only refine；
- 独立 warm-start 的 4DOF refine；
- 独立 torso-relative part hold。

脚本每 20 帧输出红、蓝两套 head hold 是否触发，用来区分：

- 没有检测到离屏/折叠；
- 已触发保持但替换顶点范围不够；
- 保持正确但模型整体配准仍偏移。

## 9. 已知权衡和后续优化方向

1. YOLO 当前约 26 ms，是最大单项之一。可尝试更小 segmentation 模型、
   降低 detector 输入尺寸，或在人稳定时降低检测频率并复用 tracking
   box/mask。
2. HPR 当前约 11–14 ms。可以评估 GPU rasterization/depth buffer，或在
   相邻帧基于拓扑与法向增量更新，但必须保持 HPR + FOV 定义一致。
3. Open3D 点云上传会与推理争用资源。调高 `CLOUD_HZ` 或降低
   `DISPLAY_STRIDE` 前必须重新测 topic-to-render。
4. 4DOF 是有界刚性校正，不能修正单个肩/肘/腕的关节角错误。进一步提升
   局部拟合需要优化 pose parameters，并加入强时间先验和关节角限制，
   否则容易重新引入折叠。
5. 当前 part hold 是刚性 torso-relative 保持。快速转身或部位刚好在边界
   附近时，可增加进入/退出滞回和短期速度外推，降低状态抖动。
6. TensorRT engine 与 GPU、TensorRT 版本和固定输入 shape 相关。迁移机器
   后应重新导出并重新做数值与延迟验证。

## 10. 修改后的验证清单

每次影响公共流水线的优化都应同时验证两个模型：

1. 对三个脚本执行 `python -m py_compile`。
2. 分别运行 `mesh_live_o3d.py --engine hmr2` 和 `--engine hybrik`。
3. 关闭窗口并保存 median/P90 profiling。
4. 运行 `compare_pose_overlay_tmp.py`，检查红、蓝 mesh 与原始点云。
5. 测试人物正面、侧身、头出画面、左右臂出画面和腿出画面。
6. 确认可见绿色区域只受 HPR + FOV 控制，FOV 外部分不得为绿色。
7. 确认注册源使用同一组绿色顶点，目标只使用分割人体点云。
8. 确认离屏部位不折回躯干，重新进入 FOV 后能恢复模型输出并更新缓存。
9. 检查无人体时 EMA、4DOF warm start 和可见顶点缓存均被清空。
10. 对比优化前后的 `topic_to_render`，不能只看模型 forward 时间。
