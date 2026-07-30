# 020-ac-cgmrf Spec

## 背景与问题

项目目标是在 Round1 官方榜单上严格超过提交时的实时第 32 名；2026-07-30 的用户提供快照为 `0.5911`。当前 O4.1 线上成绩声明为 `0.552317`，`019-o41-provenance-closure` 已在 SSH 服务器上精确复现保留 NPY，用户负责补齐平台回执并明确接受该基线。

O4.1 的核心限制是：它只对 K=4 Anchor 的已有 Beam-Delay 信道预测 H/V/Delay shift、复增益、存在性和可靠度等低维运输参数。最终增量来自 `moved_anchor - anchor`，因此擅长搬运已有能量，但难以生成目标位置特有、在邻近 Anchor 中缺失的新多径支撑。当前距离晋级线较大，继续小幅 gate 或参数调优缺少足够上限。

R02 对 XFreq-GS 的源码研究确认：其 CUDA tracer 内部会把 Gaussian 幅度和相位组成复数贡献并相干求和，但公开模型随后立即取模，只使用 PNG 实值功率谱监督，也没有显式建模赛题的跨子载波时延相位、MIMO 阵列和极化结构。因此本任务不移植 XFreq-GS 渲染器，只复用“位置条件化 Gaussian 状态、复系数和相干聚合”的机制。

## 目标

实现并验证 `AC-CGMRF`：Anchor-Conditioned Complex Gaussian Multipath Residual Field，锚点条件复高斯多径残差场。

1. 冻结并保留 O4.1 作为稳定基础信道。
2. 根据官方地图 Gaussian、Target/Anchor 几何和 Anchor 信道摘要，生成少量连续 Beam-Delay 复多径原子。
3. 允许新分支在 Anchor 原信道之外创建新的角度—时延—复相位支撑。
4. 通过零初始化可信门控安全融合；关闭分支时精确回退 O4.1。
5. 使用严格空间分块、real/zero/shuffle 因果对照和大增益门槛，判断方案是否值得生成官方测试提交。

## 非目标

- 不直接移植 XFreq-GS 的 CUDA tracer、PNG 监督或取模输出头。
- 不训练新的完整信道大解码器，不直接回归约 20 万个复数值。
- 不恢复 O5-O17、旧 diffusion 或 target-visible Oracle 路线。
- 不修改官方数据和地图，不改变官方输出契约。
- 不替换 O4.1 coarse、Power Refiner、Anchor 选择和已有缓存链。
- 不使用传统射线追踪。
- 不建立大规模每 Gaussian 独立自由参数表。
- 不在本任务中自动上传榜单；是否提交由用户根据本地结果决定。
- 不重建 SSH 环境，不重复安装依赖，不引入必须编译的新 CUDA 扩展。

## 当前方案与约束

- GitHub：`https://github.com/wediei/huaweibei`
- 起始提交：`2ba9796d4eefe7e4c08e5eb4366cba274a18a226`
- 当前基线：O4.1
- 基线保留 NPY SHA-256：`f474dcc61ee5418954a28437bdb09aefdafcfa266559133e130db04c404bfe8c`
- 基线输出：`(500,256,4,192)`、`complex64`、全部有限
- 实际训练样本数：`P_Train=2000`；不得使用 Setup 中冲突的 `2500`
- SSH 环境已经由用户创建完成；执行者不得创建环境或重复安装依赖。
- 用户负责连接 SSH 并实际运行模型训练、验证、测试和推理。
- 执行者负责提供可直接运行的命令、分析用户回传的 tqdm/日志/报告，并在本任务范围内修正实现或命令。
- 运行脚本使用 `python`，不得使用 `conda run`。
- 若现有环境缺少新实现所需依赖，优先改为使用现有 PyTorch/NumPy 能力；确实无法继续时停止并报告，不自行安装。

## 建议设计

### 1. 冻结 O4.1 基线

O4.1 的 coarse、Power Refiner、Gaussian token cache、Anchor→Target path cache、K=4 Anchor 选择和运输 checkpoint 全部冻结。AC-CGMRF 只读取其：

- 最终 O4.1 Beam-Delay 信道 `H_base`
- K=4 Anchor Beam-Delay 信道
- Anchor 位置、Target 位置和距离
- BS→Target、BS→Anchor、Anchor→Target Gaussian path token
- 已有 Beam-Delay 网格与变换规范

新分支不得反向修改 O4.1 权重。

### 2. 地图和 Anchor 条件编码

使用共享编码器形成每个候选 Gaussian token 的条件状态：

- Gaussian center、normal、tangent/normal scale、opacity
- BS、Target、Anchor 相对位置及距离
- Target path、Anchor path、path difference、Anchor→Target corridor
- K=4 Anchor 在相关 Beam-Delay 邻域的复数摘要
- O4.1 在相关邻域的幅度、相位和可靠度摘要

不得依赖大规模逐 Gaussian 独立 embedding。第一版只允许小容量共享网络；若共享网络无法拟合，是否引入低秩场状态必须另行向架构师申请。

### 3. 连续复高斯多径原子

每个 Target 最多生成可配置的少量原子，初始候选范围为 `16–32`。每个原子包含：

- 连续水平 beam 中心 `mu_h`
- 连续垂直 beam 中心 `mu_v`
- 连续 delay 中心 `mu_d`
- 三个受限宽度 `sigma_h/sigma_v/sigma_d`
- 复增益的幅度和相对相位
- 基站/用户双极化的低秩复耦合
- existence、reliability 和 atom gate

中心、宽度和相位使用有界参数化；不得输出无界 shift 或任意全尺寸复张量。

### 4. Anchor 相位参考

原子复相位优先相对以下信号预测：

1. K=4 Anchor 在相邻 Beam-Delay 区域的距离加权复数参考；
2. O4.1 在相邻区域的复相位参考；
3. 只有两者均缺少可靠能量时，才允许使用受限的地图条件相位残差。

delay 通过连续 DFT phase-ramp 显式产生跨子载波相位变化，禁止由独立子载波 MLP 随意生成相位。

### 5. 可微复数相干 splat

在现有 Beam-Delay 网格上使用 PyTorch 实现可微的局部连续 splat：

- Gaussian envelope 决定 H/V/Delay 邻域权重；
- 复增益和极化耦合决定每个格点的复数贡献；
- 所有原子在复数域相干求和，得到 `Delta_H_map`；
- 仅计算每个原子附近的有限支撑，不构造巨大稠密中间张量；
- 不依赖 XFreq-GS 自定义 CUDA 扩展。

该分支必须能够在 Anchor 原始支撑之外产生非零 Beam-Delay 能量，并输出 support-novelty 诊断。

### 6. 安全融合

最终输出：

```text
H_pred = H_base + trust_gate * Delta_H_map
```

- trust gate 的最后一层零初始化。
- 初始状态和显式 `zero` 模式必须精确回退 `H_base`。
- 残差能量相对 `H_base` 受限，并按训练进度逐步放开。
- 支持 `real`、`zero`、`shuffle` 三种地图模式。
- shuffle 只破坏地图—位置绑定，不改变 Anchor、标签和批次统计。

### 7. 训练损失

联合损失至少包含：

- 归一化复信道误差
- PAS cosine loss
- PDP cosine loss
- 与官方加权指标一致的可微 score loss
- residual energy/trust 正则
- atom sparsity/existence 正则
- real-over-zero 和 real-over-shuffle 因果 margin
- 极化耦合和跨子载波相位连续性约束

不得只用参数标签、功率图或 target-visible 残差进行部署式训练。

### 8. 训练课程

1. **恒等与接口阶段**：验证零 gate、shape、dtype、梯度和 O4.1 精确回退。
2. **单折先导阶段**：在一个固定空间折上先学习原子位置、宽度、功率和 existence；相对相位与极化耦合受限。
3. **复相位阶段**：解锁相对相位和低秩极化耦合，加入完整复信道和因果损失。
4. **空间交叉验证**：所有验证 Target 从可用 Anchor 集中排除，防止位置泄漏。
5. **全量训练与推理**：只有达到大增益门槛后才用全部 2000 个训练样本训练并生成测试 NPY。

所有训练使用 batch/epoch 级 tqdm，并保存 `metrics.jsonl`、`best.pt`、`last.pt`、SHA-256 边车、配置快照、验证报告和 `train_report.json`。

## 组件、接口或数据流

核心模型接口应保持模块化，语义等价于：

```text
ACCGMRF.forward(
    base_beam_delay,          # [B,M,N,S] complex
    anchor_beam_delay,        # [B,K,M,N,S] complex
    anchor_positions,         # [B,K,3]
    target_positions,         # [B,3]
    target_path_tokens,
    anchor_path_tokens,
    anchor_target_tokens,
    anchor_mask,
    mode,                     # real / zero / shuffle
) -> {
    prediction,
    residual,
    atoms,
    trust,
    support_novelty,
    diagnostics
}
```

数据流：

```text
O4.1 frozen inference ────────────────────────→ H_base
                                                       ┐
Map/path tokens + positions + K=4 Anchor complex state │
                        ↓                              │
              condition encoder                       │
                        ↓                              │
          16–32 complex Gaussian atoms                 │
                        ↓                              ├→ H_pred
       continuous complex Beam-Delay splat             │
                        ↓                              │
          energy-limited trusted residual ─────────────┘
```

## Reference 依据

- O4.1 `gaussian_anchor_transport`：证明当前输出只来自 Anchor 运输增量和 coarse 融合。
- XFreq-GS `gaussian_renderer`：证明参考实现生成 Gaussian 幅度/相位后调用复 tracer，但最终取模。
- XFreq-GS `cuda_tracer/forward.cu`：证明 Gaussian 复系数可进行相干聚合。
- R02 研究结论：只复用复数相干聚合机制，重写为赛题原生 Beam-Delay、MIMO 和复信道监督。

## 交付物

- `specs/020-ac-cgmrf.md`，执行者只读并原样提交。
- AC-CGMRF 模型、训练、评估和推理入口。
- 空间分块与 Anchor 排除逻辑。
- real/zero/shuffle 控制和 support-novelty 诊断。
- 恒等回退、shape/dtype、复数 splat、梯度、确定性和泄漏防护测试。
- 单折先导报告和完整空间交叉验证报告。
- 若达到门槛：全量训练报告、checkpoint hash、推理验证报告和官方格式 NPY。
- `tasks/020-ac-cgmrf/执行报告.md`
- 大型 checkpoint、缓存、日志和 NPY 保留在 Git 外部，仅记录路径与哈希。

## 验收条件

### 工程验收

1. 显式关闭 AC-CGMRF 时，输出与 O4.1 在数值容差内一致；目标为 `max_abs_error <= 1e-6`。
2. 模型输出 shape、dtype 和 finite 符合官方契约。
3. 连续复数 splat 可反向传播，并通过确定性和复数相干叠加测试。
4. shuffle/zero 模式不泄漏真实地图绑定。
5. 空间验证 Target 不得出现在自身 Anchor 候选中。
6. 无新编译扩展，无大尺寸直接复信道解码头。
7. tqdm、metrics、配置、checkpoint 边车和报告齐全。

### 研究淘汰门槛

1. 单折先导相对 O4.1 总分提升 `< +0.01`，停止。
2. 单折 `real-over-shuffle < +0.005`，停止。
3. 残差主要落在 O4.1 已有支撑内、support-novelty 接近零，说明没有解决核心瓶颈，停止。
4. PAS、PDP 或 NMSE 出现明显结构性崩坏，停止。

### 完整训练门槛

1. 所有空间验证折均为正增益。
2. 空间折平均官方总分提升至少 `+0.03`。
3. 最佳稳定配置相对 O4.1 提升至少 `+0.04`。
4. `real-over-zero` 和 `real-over-shuffle` 均至少 `+0.01`。
5. PAS、PDP 不低于 O4.1，NMSE 不恶化。

只有同时达到完整训练门槛，才允许生成官方测试 NPY，并建议用户使用一次线上提交机会。该门槛表示“值得提交”，不等同于保证超过实时第 32 名。

## 风险与取舍

- **绝对复相位难泛化**：使用 Anchor/O4.1 相位参考和显式 delay phase-ramp，限制自由相位。
- **2000 样本欠约束**：使用共享小模型、少量原子、严格空间折和稀疏/能量正则，不使用大逐点参数表。
- **新支撑可能变成噪声**：设置 trust gate、support-novelty 与因果对照，必须同时改善官方指标。
- **显存和速度风险**：限制原子数量和局部 splat 支撑，不构造全局 Gaussian×全网格张量。
- **本地到线上转化不稳定**：采用高于普通微调幅度的本地门槛，并保留每日提交次数。
- **既有环境依赖风险**：优先纯 PyTorch/NumPy；环境缺依赖时停止，不擅自重建或安装。

## 执行拆分

1. **接口与恒等回退**：产出模型骨架、复数 splat 和零 gate；验收为 O4.1 精确回退及基础测试通过。
2. **空间验证与泄漏防护**：产出固定空间折和 Anchor 排除报告；验收为折分可复现且无目标泄漏。
3. **单折先导训练**：由用户在 SSH 运行，执行者分析日志并修正；验收为通过或触发研究淘汰门槛。
4. **复相位和因果训练**：产出 real/zero/shuffle、support-novelty 和多指标报告；验收为机制增益来自正确地图绑定。
5. **完整空间交叉验证**：产出各折及汇总报告；验收为满足完整训练门槛。
6. **全量训练和条件式推理**：仅在门槛通过后执行；验收为官方 NPY 格式有效、checkpoint 和结果哈希齐全。
7. **任务报告与 Git 交付**：提交源码、测试、配置、Spec 和唯一执行报告；不提交大型运行资产。

## 回退或退出方式

- AC-CGMRF 是 O4.1 旁路残差分支；关闭开关或 gate 即回退 O4.1。
- 任何阶段触发研究淘汰门槛，立即停止后续大规模训练，不生成测试提交。
- 环境、缓存、checkpoint 或数据与 Spec 不一致时，停止并向架构师报告，不自行扩大范围。
- 用户运行 SSH 命令发生错误时，执行者在当前任务范围内修复实现或命令；需要新依赖、环境重建或新研究方向时停止。
- 未经架构师 `ACCEPT`，不得把 AC-CGMRF 替换为项目当前基线。
