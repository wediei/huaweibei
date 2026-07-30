# 021-e2e-cgpf Spec

## 背景与问题

项目最终目标是在 Round1 官方榜单上严格超过提交时的实时第 32 名；2026-07-30 的用户提供快照为 `0.5911`。当前已复现基线 O4.1 的线上成绩声明为 `0.552317`。

此前多条方案把 O4.1 冻结后增加低维运输、功率修正、复残差、Gaussian 场或 diffusion 附件，大多得到零收益或极小收益。`020-ac-cgmrf` 虽通过工程、安全和泄漏验证，但真实单折训练的增益、real-over-shuffle 与 support novelty 均为零，残差分支塌缩到 O4.1 恒等解。

用户已明确调整架构原则：O4.1 只保留为评分基准和独立回退版本，不再要求新方案冻结或依附 O4.1。下一方案允许联合训练、替换现有组件或从头构建主干，以争取绝对值级的大幅提升。

多 reference 审查结论：

- WRF-GS 可复用可训练 Gaussian 场、目标位置条件网络和 densify/prune，但原实现只输出实值空间功率谱。
- WRF-GS+ 可复用目标条件静/动态场思想，但实际代码未启用中心变形且相位被取模。
- XFreq-GS 可复用 Gaussian 复系数相干求和，但最终输出仍取模，没有完整 MIMO、Delay 和极化。
- diffusion reference 只处理实值二维功率图，数据规模和条件形式不适合直接迁移。

本任务综合上述有效机制，建立独立于 O4.1 的完整复信道生成主干。

## 目标

实现和验证 `E2E-CGPF`：End-to-End Complex Gaussian Path Field，端到端复高斯路径场。

1. 用官方 PLY 初始化一个可由信道监督联合优化的 Gaussian 场。
2. 根据 Target 位置和场几何，为每个激活 Gaussian 生成多个软路径模态。
3. 每个路径模态显式预测连续 Delay、AoA/AoD、复增益、低秩极化耦合、存在性和可靠度。
4. 使用连续 Delay phase-ramp、阵列 steering 和复数相干求和，直接生成完整 `(256,4,192)` MIMO-OFDM 复信道。
5. 从头联合训练 Gaussian 场、目标条件路径网络和复信道 renderer；O4.1 不进入模型前向图。
6. 先证明表示容量，再验证严格空间泛化和地图因果性；只有达到大增益门槛才生成官方测试结果。

## 非目标

- 不冻结、微调或添加 O4.1 残差分支。
- 不复用 `020-ac-cgmrf` 的零 trust gate 结构或候选分支代码。
- 不直接移植 WRF-GS、WRF-GS+ 或 XFreq-GS 的图像 rasterizer、RGB alpha blending、取模输出头或自定义 CUDA tracer。
- 不使用固定 Gaussian 单次反射公式，不重复 O17。
- 不把每个 Gaussian 强制绑定为唯一单跳反射点。
- 不使用传统硬射线追踪、硬可见性裁剪或材质手工规则。
- 不训练无物理结构的全尺寸 ChannelDecoder。
- 不在原始完整信道张量上训练 diffusion。
- 不使用测试信道、target-visible Oracle 分数或验证位置专属参数作为部署输入。
- 不自动上传榜单；是否提交由用户决定。
- 不重建 SSH 环境，不重复安装现有依赖，不引入必须编译的新 CUDA 扩展。

## 当前方案与约束

- GitHub：`https://github.com/wediei/huaweibei`
- 起始提交：`2ba9796d4eefe7e4c08e5eb4366cba274a18a226`
- 工作任务不得包含已 `DISCARD` 的 `task/020-ac-cgmrf` 实现。
- O4.1 仅用于离线评分对比和失败时回退，不得作为 E2E-CGPF 输入、teacher forcing 目标或冻结前缀。
- 实际训练样本数为 `P_Train=2000`；不得使用 Setup 中冲突的 `2500`。
- 官方输出为 `(500,256,4,192)`、`complex64`、全部有限。
- 用户已经在 SSH 服务器上创建好环境。
- 用户负责连接 SSH 并实际运行训练、验证、测试和推理。
- 执行者负责实现、提供命令、分析用户回传的 tqdm/日志/报告，并在任务范围内修正。
- 运行脚本使用 `python`，不得使用 `conda run`。
- 不得自行创建环境或重复安装 PyTorch/依赖；缺依赖时优先使用现有 PyTorch/NumPy 改写，无法继续时停止报告。

## 建议设计

### 1. 可训练 Gaussian 场

官方 PLY 只用于初始化场几何和提供先验，不把点云点直接等同于固定 RF 反射点。

每个 Gaussian 的共享场状态至少包括：

- 可学习但受边界限制的位置偏移
- 各向异性尺度和方向
- opacity/path activation
- 小容量材料/传播 latent code
- 几何法向和初始化置信度

场参数由复信道监督端到端更新。支持受控 densify/prune：

- densify 依据持续高梯度、高残差和稳定路径贡献；
- prune 依据长期低 activation、低贡献或冗余；
- 设置全局数量上限，防止场规模失控；
- 每次结构更新都写入场统计和可复现记录。

不得使用大规模逐位置或逐频点独立参数表。

### 2. Target 条件路径网络

输入：

- Target 三维位置和 BS 固定位置
- Gaussian center、scale、orientation、normal 和 material code
- BS→Gaussian、Gaussian→Target 的距离、方向及相对几何
- Target 相对场的多尺度位置编码
- 可选的邻域/场上下文，不包含 O4.1 信道

对每个激活 Gaussian 生成多个软路径模态。每个模态输出：

- continuous delay
- BS 阵列侧 AoD/beam 参数
- UE 阵列侧 AoA/beam 参数
- complex path gain
- `2×2` 低秩复极化耦合
- existence、reliability 和 path width
- 路径类型 latent，不强制赋予直达、反射或绕射硬标签

同一 Gaussian 可以贡献多个 Delay/AoA/AoD 模态，从而避免固定单次反射假设。

### 3. 软路径选择

采用可微软门控选择有限数量的重要 Gaussian/path 模态：

- 不使用硬可见性裁剪；
- 不使用 RGB 前后 alpha 遮挡；
- 不把贡献限制为最近或单个散射点；
- 支持直达、遮挡、绕射和多次散射的潜在成分；
- 每个 Target 的激活路径数量受上限控制，防止完整场与全网格笛卡尔积。

需要记录 active path 数、Delay 分布、角度分布、路径能量和场覆盖率。

### 4. 赛题原生复信道 renderer

renderer 不复用参考项目的图像输出头。每个路径模态使用：

```text
H_path[n_rx,n_tx,k]
  = complex_gain
  × polarization_coupling
  × a_rx(AoA)
  × a_tx(AoD)^H
  × exp(-j 2π ν_k delay)
```

其中 `ν_k` 使用与现有 DFT/Beam-Delay 定义一致的归一化子载波坐标，不依赖任务书未提供的物理载频间隔。

所有路径在复数域相干求和：

```text
H_pred = Σ_path H_path
```

要求：

- 显式保持跨子载波相位连续性；
- 显式支持 256 基站天线、4 用户天线及双极化结构；
- 不独立预测每个子载波任意相位；
- 使用纯 PyTorch/现有依赖实现，不新增编译扩展；
- 支持分块和稀疏计算，避免 Path×完整信道的大型稠密中间量。

### 5. 联合训练

Gaussian 场、Target 条件路径网络和 renderer 从头联合训练。O4.1 只在评估阶段提供对比指标。

联合损失至少包含：

- normalized complex channel loss
- PAS cosine loss
- PDP cosine loss
- 与官方指标一致的 differentiable score loss
- path sparsity/activation 正则
- Gaussian geometry/material 平滑和边界正则
- 极化低秩与能量约束
- real-map、zero-map、shuffle-map 因果 margin

不得用参数拟合损失替代官方信道指标。

### 6. Coarse-to-fine 训练课程

#### 阶段 A：表示容量

- 在固定的小训练子集上训练完整模型；
- 不提供位置专属 embedding；
- 目标是证明 renderer 和路径表示能主动拟合复信道；
- 该结果只用于表示容量判断，不作为泛化成绩。

#### 阶段 B：几何、功率和 Delay

- 优先学习路径 activation、Delay、角度和功率；
- 使用 PAS/PDP 和复信道幅度结构；
- Gaussian 位置只允许小范围更新，不 densify。

#### 阶段 C：复相位和极化

- 解锁完整 complex gain、Delay phase-ramp 和极化耦合；
- 加入复信道、NMSE 和相位连续性监督；
- 检查是否出现幅度正确但相位崩坏。

#### 阶段 D：场结构联合优化

- 达到稳定表示后才启用受控 densify/prune；
- 联合优化官方指标和地图因果 margin；
- 每次场结构变化后验证确定性、有限值和参数规模。

#### 阶段 E：严格空间泛化

- 使用可复现的空间分块；
- 验证 Target 不允许拥有位置专属参数；
- real/zero/shuffle 使用相同模型容量和训练预算；
- 所有模型从相同初始化与数据划分比较。

#### 阶段 F：全量训练和条件式推理

- 只有通过完整空间门槛才使用全部 2000 训练位置；
- 只有 full checkpoint 才能生成官方测试 NPY；
- 推理结果必须经过 shape、dtype、finite 和 hash 验证。

### 7. 与 O17 的边界

E2E-CGPF 必须满足：

- Gaussian 场可由信道监督移动、增删和改变支撑；
- 每个 Gaussian 可产生多个软路径模态；
- 路径参数由 Target 条件网络生成；
- 不使用固定单跳几何公式决定唯一反射路径；
- 不使用硬遮挡替代 RF 多径叠加；
- phase 由连续 Delay phase-ramp 和低维复系数生成。

若实现退化为固定 Gaussian、单次反射、硬可见性或固定材质系数，应立即停止并返回架构师。

## 组件、接口或数据流

核心接口语义：

```text
TrainableGaussianField(
    official_scene,
    field_state,
) -> gaussian_state

TargetConditionedPathNetwork(
    target_position,
    bs_position,
    gaussian_state,
) -> path_modes

ComplexMIMOOFDMRenderer(
    path_modes,
    array_spec,
    subcarrier_grid,
) -> complex_channel
```

总数据流：

```text
Official PLY
    ↓ initialization prior
Trainable Gaussian Field
    ↓ target-conditioned shared state
Multi-modal Path Network
    ↓ delay / AoA / AoD / complex gain / polarization
Complex MIMO-OFDM Renderer
    ↓ coherent sum
H_pred [B,256,4,192] complex
    ↓
Complex + PAS + PDP + NMSE + causal training
```

## Reference 依据

- WRF-GS：可训练 Gaussian 场、Target 条件 MappingNetwork、densify/prune。
- WRF-GS+：Target 条件静/动态场状态；其现有取模和未启用变形不直接复用。
- XFreq-GS：Gaussian 复幅相相干求和；其 PNG 功率头和自定义 tracer 不直接复用。
- diffusion：仅保留未来对低维路径 latent 做条件去噪的启发，本任务不采用。
- O17 失败结论：固定 Gaussian 单次反射和硬解析映射无效，E2E-CGPF 必须保持可训练场和多模态软路径。
- O4.1：只作为评分基准、服务器对照和独立回退。

## 交付物

- `specs/021-e2e-cgpf.md`，执行者只读并原样提交。
- 可训练 Gaussian 场和受控 densify/prune。
- Target 条件多模态路径网络。
- 赛题原生复数 MIMO-OFDM renderer。
- 表示容量、阶段训练、空间验证和因果对照入口。
- 场规模、路径统计、复相位、极化和有限值诊断。
- 单元/集成测试：
  - shape、dtype、finite
  - 复数相干叠加
  - Delay phase-ramp
  - 阵列 steering 和极化维度
  - 场参数梯度
  - densify/prune 可复现性
  - 空间划分和位置专属参数泄漏
  - real/zero/shuffle 等容量
- tqdm、`metrics.jsonl`、配置快照、`best.pt`、`last.pt`、SHA-256 边车和阶段报告。
- 表示容量报告、单折报告、完整空间交叉验证报告。
- 达到门槛时：全量训练报告、推理验证、官方格式 NPY 和哈希。
- `tasks/021-e2e-cgpf/执行报告.md`
- 大型 checkpoint、缓存、日志和 NPY 保留在 Git 外部，仅在报告记录路径与哈希。

## 验收条件

### 工程验收

1. 输出为 `[B,256,4,192]` complex tensor，dtype、finite 和设备传播正确。
2. renderer 的同相增强、反相抵消、Delay phase-ramp、阵列响应和极化维度测试通过。
3. Gaussian center/scale/material、路径参数和 renderer 均有有效有限梯度。
4. 不依赖 O4.1 前向结果，不包含 task020 trust gate。
5. 不使用位置专属 embedding、测试标签或验证目标泄漏。
6. densify/prune 有上限、确定性记录和可回退快照。
7. 训练过程使用 tqdm，并生成完整日志、checkpoint hash 和报告。

### 表示容量门槛

1. 在固定小训练子集上，本地官方总分达到至少 `0.80`。
2. complex loss、PAS、PDP 和 NMSE 均显示有效拟合，不得只优化单一功率指标。
3. 至少存在多个稳定激活路径模态，不能退化为全零、单路径或纯平均信道。
4. 若充分训练后不能通过容量门槛，立即停止，不进入空间泛化。

该阶段允许在训练子集上过拟合，只证明表示容量，不得作为部署成绩。

### 单折空间门槛

1. 完整 coarse-to-fine 课程结束后，单折总分至少超过 O4.1 `+0.01`。
2. `real-over-zero` 与 `real-over-shuffle` 均至少 `+0.005`。
3. PAS、PDP 和 NMSE 不得出现结构性崩坏。
4. 路径数量、Delay/角度分布和场贡献必须随 Target 变化，不能只记忆全局均值。

### 完整空间门槛

1. 所有空间折相对 O4.1 均为正增益。
2. 空间折平均总分提升至少 `+0.03`。
3. 最佳稳定配置目标提升至少 `+0.05`。
4. `real-over-zero` 和 `real-over-shuffle` 均至少 `+0.01`。
5. PAS、PDP 不低于 O4.1，NMSE 不恶化。

只有通过完整空间门槛，才允许全量训练、生成官方测试 NPY，并建议用户使用一次线上提交机会。该门槛表示“值得提交”，不保证超过实时第 32 名。

## 风险与取舍

- **2000 位置欠约束**：使用共享场、低维材料码、有限路径模态、阶段训练和严格空间验证。
- **复相位难学习**：用显式 Delay phase-ramp 和低维 complex gain，不独立生成每个子载波相位。
- **场规模失控**：设置 Gaussian 和 active path 上限，densify/prune 延后且受报告约束。
- **退化为 O17**：禁止固定单跳、硬可见性和唯一反射映射。
- **幅度拟合但 NMSE 崩坏**：分阶段解锁相位，并同时观察复 loss、PAS、PDP、NMSE。
- **地图无因果性**：real/zero/shuffle 等容量对照是硬门槛。
- **从头主干初期分数低**：先通过表示容量，再判断空间泛化，不以早期低于 O4.1 直接停止。
- **显存和速度风险**：软选择有限 active path，分块 renderer，不构造全场×全信道稠密张量。
- **现有环境依赖**：优先纯 PyTorch/NumPy；缺依赖时停止，不擅自重建环境。

## 执行拆分

1. **Renderer 与表示容量骨架**：产出复路径数据结构、MIMO-OFDM renderer 和基础测试；验收为工程条件通过。
2. **可训练场与路径网络**：产出官方点云初始化、共享场和多模态路径；验收为有效梯度和受控规模。
3. **小子集容量实验**：由用户在 SSH 运行；验收为达到或触发表示容量门槛。
4. **阶段 B/C 单折训练**：产出功率/Delay、复相位/极化报告；验收为无结构性崩坏。
5. **场结构联合训练**：条件式启用 densify/prune；验收为场规模受控且指标改善。
6. **单折空间评估**：产出 real/zero/shuffle 和 O4.1 对比；验收为通过单折门槛。
7. **完整空间交叉验证**：产出各折及汇总报告；验收为通过完整空间门槛。
8. **全量训练与条件式推理**：仅在门槛通过后执行；验收为官方 NPY 和全部哈希有效。
9. **Git 与任务报告**：提交代码、测试、配置、Spec 和唯一执行报告；不提交大型运行资产。

## 回退或退出方式

- O4.1 作为独立 checkpoint、代码和保留 NPY 持续存在；E2E-CGPF 失败不影响 O4.1。
- 任一阶段触发对应门槛失败，停止后续昂贵阶段，不生成测试提交。
- 若实现需要固定单次反射、硬射线追踪、位置专属参数或大完整解码器，停止并返回架构师。
- 用户运行 SSH 命令发生错误时，执行者在任务范围内修复；需要新依赖、环境重建或新研究方向时停止。
- 未经架构师 `ACCEPT`，不得把 E2E-CGPF 标记为当前方案或替换 O4.1 基线。
