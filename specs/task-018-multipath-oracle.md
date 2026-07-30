# TASK-018-multipath-oracle Spec

## 背景与问题

当前 O4.1 对每个 Anchor 的每个 P/N 组只预测一组 H/V/Delay 平移、复增益、存在性和可靠度，
再统一作用于完整 Beam-Delay 块。真实无线信道由多个独立变化的多径分量叠加，单一整体运输
可能构成表示上限。

项目已有 `path_transport_oracle_cli.py`，能够审计单分量 O2 运输的 target-visible 上限。
本任务在它旁边增加多分量 Oracle，以低成本回答“多分量运输是否值得进入可训练方案”。

## 目标

- 将每个 P/N 组的源 Beam-Delay 信号确定性地划分为 `L` 个和为原信号的软分量。
- 使用 target-visible 坐标下降，为每个分量独立拟合整数 H/V/Delay 平移与复增益。
- 对最终多分量候选拟合一个全局可靠度，保持与现有 O2 Oracle 相同的融合语义。
- 比较 `L=1/2/3` 的 PAS、PDP、NMSE 和综合分数，形成是否继续主方案的表示门槛。

## 非目标

- 不训练地图条件网络。
- 不修改 O4.1 正式训练、评价或推理入口。
- 不生成比赛提交。
- 不把 target-visible Oracle 当作可部署成绩。
- 不恢复 diffusion、Radio Field 或 learned Anchor fusion。

## 当前方案与约束

- 输入信道形状遵循 `(B,M,N,S)`，Beam-Delay 形状遵循 `(B,H,V,P,N,D)`。
- `L=1` 必须数值复现现有 O2 `reliability` 阶段。
- 分量划分只依赖源信道，不能查看目标信道。
- 分量拟合可以查看目标信道，因为本任务只测量表示上限。
- 所有输出必须保持 `complex64`、有限值并维持原形状。
- 训练相关工作仍遵守 tqdm、过程日志、mamba 和独立 PyTorch 安装约束；本任务不训练模型。

## 建议设计

### 1. 确定性软分量

对单个复数 `(H,V,D)` 源块：

1. 按功率从高到低稳定选择 `L` 个不同峰值坐标。
2. 计算每个网格点到峰值的周期距离，按各维 `max(1,size/4)` 归一化。
3. 使用温度参数对负平方距离做 softmax。
4. 用 softmax 权重乘源信号，得到 `L` 个分量。

权重在分量轴严格和为 1，因此分量重建必须在浮点误差内等于原信号。`L=1` 的权重恒为 1。

### 2. 多分量 target-visible 拟合

初始时每个移动分量等于源分量。执行固定次数的坐标下降：

1. 暂时移除当前分量，计算其他分量对目标的剩余残差。
2. 复用现有 `_best_beam_delay_shift` 在指定整数范围内寻找最佳三维平移。
3. 复用现有 `_complex_fit` 拟合该分量的复增益。
4. 更新移动分量。

所有分量求和后，复用现有可靠度闭式解，在原信号与多分量候选之间做 `[0,1]` 融合。

### 3. 审计报告

报告必须包含：

- `kind=path_transport_multi_component_target_visible_ceiling`
- `target_visible=true`
- `deployable_prediction=false`
- 每个 component count 的 PAS、PDP、NMSE、score
- 相对 `L=1` 的增益
- 分量温度、坐标下降轮数和平移限制
- 最佳 component count
- `promoted`：最佳 `L>1` 相对 `L=1` 增益是否达到 `0.01`

## 组件、接口或数据流

新增纯算法模块：

```python
build_soft_components(
    source: np.ndarray,
    component_count: int,
    temperature: float,
) -> np.ndarray
```

输入 `(H,V,D)` 复数数组，输出 `(L,H,V,D)` 同 dtype 数组。

```python
fit_multi_component_group(
    source: np.ndarray,
    target: np.ndarray,
    component_count: int,
    max_h_shift: int,
    max_v_shift: int,
    max_delay_shift: int,
    temperature: float,
    sweeps: int,
) -> tuple[np.ndarray, dict[str, object]]
```

输出可靠度融合后的 `(H,V,D)` 候选及分量诊断。

```python
audit_multi_component_ceiling(
    reference_channels: np.ndarray,
    target_channels: np.ndarray,
    layout: AntennaLayout,
    component_counts: tuple[int, ...] = (1, 2, 3),
    ...,
) -> dict[str, object]
```

CLI 读取既有 fold、coarse validation 和目标验证信道，调用审计函数并原子写出 JSON。

## Reference 依据

- 当前 O2 Oracle 已提供三维相关、复增益拟合、可靠度闭式解和 MetricAccumulator 流程。
- XFreq-GS 证明按传播单元独立进行复幅度/相位调制后再复数叠加是可实现的机制。
- 不复用 XFreq-GS 的逐 Gaussian 持久参数，避免在 2000 个位置和大点云上欠约束。
- 不直接复用 WRF-GS+ 的 deformation，因为其当前 renderer 未使用 `d_xyz`，且信号相位被取绝对值丢弃。

## 交付物

- `solution/radio_map/learning/multi_component_transport_oracle.py`
- `solution/radio_map/learning/multi_component_transport_oracle_cli.py`
- 对应 unittest 测试
- 服务器运行命令与 JSON 结果路径
- `tasks/task-018-multipath-oracle/执行报告.md`

## 验收条件

1. `L=1` 与现有 O2 `reliability` 输出和指标在数值容差内一致。
2. 软分量确定、有限、保持 dtype，并在浮点容差内重建输入。
3. 合成双路径案例中 `L=2` 明显优于 `L=1`。
4. 非法 component count、温度、sweeps、形状和 dtype 被拒绝。
5. CLI 使用 tqdm 并原子写 JSON。
6. 本地相关测试及完整 `solution/tests` 回归通过。
7. 服务器正式验证中，最佳 `L>1` 相对 `L=1` 综合分数增益至少 `0.01`，才建议进入可训练主方案。

## 风险与取舍

- 软分量由源功率峰定义，不保证等价于真实物理路径；本任务只验证表示容量。
- 坐标下降可能依赖分量顺序，使用稳定峰值排序和固定 sweeps 保证确定性。
- Oracle 增益高只说明值得训练，不能证明地图条件网络能够预测这些参数。
- 若 `L>1` 没有足够增益，停止主方案并转向“路径集合对齐编码器”备选。

## 执行拆分

1. TDD 实现软分量划分。
2. TDD 实现单组多分量坐标下降，并验证 `L=1` 兼容性。
3. TDD 实现批量指标审计和 JSON 报告。
4. TDD 实现 fold CLI。
5. 完整回归、服务器命令、执行报告和远端任务分支。

## 回退或退出方式

新代码与现有 O4.1 入口解耦。放弃任务时不合并任务分支即可；无需修改、回滚或重建现有
checkpoint/cache。服务器结果低于验收门槛时，将结论记为 `DISCARD`，不进入训练方案。
