# 019-o41-provenance-closure Spec

## 背景与问题

项目需要在 Round1 官方榜单上严格超过提交时的实时第 32 名；2026-07-30 的用户提供快照为 `0.5911`。当前文档声明 O4.1 得分为 `0.552317`，但本地只保留源码和提交产物，缺少服务器 checkpoint、运行报告、产物清单和平台回执。

独立验证已确认保留 NPY/ZIP 的格式、内容和哈希有效，但无法建立 `源码 commit → checkpoint → NPY → ZIP → 线上得分` 的完整来源链。没有这一基线，后续创新无法可靠证明恒等回退、真实增益或线上转化。

## 目标

1. 在用户授权的 SSH 服务器上只读盘点 O4.1 所需资产并安全追回可用证据。
2. 建立机器可读的来源清单，把服务器资产、源码身份、本地提交产物和平台证据连接起来。
3. 仅在全部推理前置资产和运行条件满足时，重新生成官方测试 NPY，并与保留 NPY 做精确 SHA-256 比较。
4. 给架构师返回 `PASS / PARTIAL / FAIL` 结论，决定后续是直接进入 O4.1 创新实验，还是先重建基线。

## 非目标

- 不训练或微调任何模型。
- 不修改模型源码、配置、checkpoint、缓存、数据集或既有结果。
- 不恢复 O5-O17，不研究新方案。
- 不上传榜单，不消耗每日提交次数。
- 不删除、移动、覆盖或重命名服务器和本地既有资产。
- 不把大型数据、checkpoint、缓存或 NPY 提交到 Git。
- 不安装或升级依赖；若现有环境不能运行，只记录环境缺口。

## 当前方案与约束

- GitHub：`https://github.com/wediei/huaweibei`
- 起始提交：`e28785d77c843f224bb90645956636a5927f98a1`
- 当前文档基线：O4.1，声明线上分数 `0.552317`
- 保留 NPY SHA-256：`f474dcc61ee5418954a28437bdb09aefdafcfa266559133e130db04c404bfe8c`
- 保留 ZIP SHA-256：`40b6654eb24ca5e6195788afb61dbe80971983b8c2c2fefe009d60766f78b1da`
- 官方输出契约：`(500,256,4,192)`、`complex64`、全部有限
- `Round1_Setup.json` 的 `P_Train=2500` 与训练数组首维 `2000` 冲突；本任务以数组事实 `P_Train=2000` 为准，并保留异常记录。
- 当前共享工作树存在架构文档修改和用户已有的 `specs/task-018-multipath-oracle.md` 删除状态；执行者不得 stash、reset、checkout 或覆盖这些改动。
- SSH 环境若涉及 Conda 系列操作，应使用 mamba；本任务不授权创建环境。验证命令使用 `python`，不得使用 `conda run`。

## 建议设计

### 1. 只读服务器盘点

定向核对文档列出的 O4.1 依赖：

- `dataset/Round1_Map/`
- `solution/artifacts/day3/geometry_1m.npz`
- `solution/artifacts/day7/cache_seed42_stable/`
- `solution/artifacts/day10/a2_directscore_seed42/best.pt`
- `solution/artifacts/day10/a2_directscore_seed42/best.pt.sha256`
- `solution/artifacts/day13/power_refiner_seed42/best.pt`
- `solution/artifacts/day13/power_refiner_seed42/best.pt.sha256`
- `solution/artifacts/day14/gaussian_scene_1m.npz`
- `solution/artifacts/day14/gaussian_tokens_seed42/`
- `solution/artifacts/day14/o3_supervision_seed42/`
- `solution/artifacts/day18/anchor_target_paths_k4/`
- `solution/artifacts/day18/o4_direct_path_seed42/best.pt`
- `solution/artifacts/day18/o4_direct_path_seed42/best.pt.sha256`

同时定向寻找与 O4.1 对应的训练报告、评估报告、指标日志、推理报告、命令记录、源码 commit 标识、产物 manifest 和平台回执。不得扩大为服务器全盘扫描。

### 2. 来源清单

对发现的关键文件记录：

- 规范化相对路径
- 文件类型、大小和修改时间
- SHA-256
- 所属阶段和用途
- 是否存在已有 hash 边车，以及边车是否匹配
- 与起始 commit、checkpoint、输出 NPY、ZIP 和平台回执的关系

目录以其关键文件的逐文件清单表示，不用单一目录时间戳代替证据。清单不得包含 SSH 凭据、令牌或其他秘密。

### 3. 安全追回

只有在用户确认目标位置且空间足够后，才把缺失于本地的关键 checkpoint、报告、manifest 和回执复制到独立的新位置。复制后重新计算哈希；不得移动或覆盖源文件。

大型缓存和数据默认只记录清单与位置，不复制到 Git 工作区。若需要完整备份但目标位置或容量不明确，停止并请求用户决定。

### 4. 条件式精确复现

仅当以下条件全部成立时执行一次 O4.1 官方测试推理：

- 所有必需资产存在且边车哈希有效；
- 源码身份可确定，或差异已被完整记录；
- 现有 Python/CUDA 环境可直接运行，无需安装依赖；
- 推理输出写入全新路径，不覆盖保留产物；
- 预计磁盘空间足够。

新 NPY 必须先验证 shape、dtype 和 finite，再计算 SHA-256，与保留哈希做字节级比较。不得通过改写 NPY 头、复制保留文件或后处理来制造哈希一致。

### 5. 结论分级

- `PASS`：完整资产和来源链存在，重新推理 NPY 哈希精确匹配，且有可信平台回执把该产物绑定到 `0.552317`。
- `PARTIAL`：资产或复现哈希已闭合，但缺平台回执；或平台回执存在但源码/运行链仍有一处无法闭合。
- `FAIL`：关键资产缺失、边车不匹配、重新推理哈希不一致，或源码身份无法合理恢复。

`PARTIAL` 和 `FAIL` 都不得把 O4.1 标记为“已闭合可复现基线”。

## 组件、接口或数据流

```text
服务器资产只读盘点
        ↓
逐文件 manifest 与 hash 边车核对
        ↓
commit / checkpoint / report 身份关联
        ↓
条件满足时：全新路径执行一次推理
        ↓
shape / dtype / finite / SHA-256
        ↓
保留 NPY、ZIP、平台回执交叉绑定
        ↓
PASS / PARTIAL / FAIL
```

## Reference 依据

- `task/赛题任务书.pdf`：官方输入输出和提交契约。
- `总结.md`：O4.1 文档级资产路径、运行方式和历史成绩声明。
- `项目方案迭代记录.md`：R01 的基线验证结果、证据优先级和停止条件。
- `results/8_0.552317/`：当前保留提交产物。

## 交付物

- `specs/019-o41-provenance-closure.md`：本 Spec，执行者只读并原样提交。
- `tasks/019-o41-provenance-closure/server_asset_manifest.json`
- `tasks/019-o41-provenance-closure/provenance_chain.json`
- 条件式推理发生时：`tasks/019-o41-provenance-closure/reproduction_report.json`
- 缺失或不一致时：`tasks/019-o41-provenance-closure/reconstruction_card.md`
- `tasks/019-o41-provenance-closure/执行报告.md`
- Git 外部资产位置和哈希清单；不把大型资产纳入仓库。

## 验收条件

1. 已对所有已知 O4.1 关键资产逐项给出“存在、缺失或未授权访问”的确定状态。
2. 所有发现的关键文件都有路径、大小、SHA-256、用途和关系记录。
3. 已核对 checkpoint hash 边车、训练/评估报告、源码身份和平台回执；缺失项不得用推断补齐。
4. 若执行复现，输出位于全新路径，格式有效，并给出与保留 NPY 的精确哈希比较。
5. 结论严格按 `PASS / PARTIAL / FAIL` 定义生成，且执行报告中的每个关键结论都能定位到 manifest、报告或外部回执。
6. 未修改、覆盖、移动或删除任何既有服务器及本地资产。
7. 未训练、未修改源码、未上传榜单、未提交大型资产。

## 风险与取舍

- 服务器资产可能已删除，只能形成明确的重建卡，不能完成精确复现。
- 目录体积和哈希耗时可能较大；优先关键文件，遇到容量或时间风险时停止扩张。
- 当前源码可能不是生成保留 NPY 的源码；即使 checkpoint 存在，也可能无法字节级复现。
- 平台回执可能只记录分数而没有产物哈希，此时最多判定 `PARTIAL`。
- GPU、CUDA 或环境不可用时，不得临时安装依赖；保留资产证据并把复现标记为未执行。

## 执行拆分

1. **资产盘点**：产出完整状态表；验收为所有已知路径都有确定状态。
2. **证据清单**：产出 manifest 和 provenance chain；验收为关键文件及关系可机器读取。
3. **安全追回**：在用户确认目标位置和容量后复制必要证据；验收为源、目标哈希一致。
4. **条件式复现**：仅在前置条件满足时生成新 NPY；验收为格式检查和哈希比较完整。
5. **结论报告**：产出唯一执行报告及必要的 reconstruction card；验收为 `PASS / PARTIAL / FAIL` 有证据支持。

## 回退或退出方式

- 本任务原则上只读；新增 manifest、报告和新推理输出均使用唯一新路径。
- 任何缺少权限、凭据、空间、关键资产或运行环境的情况都应停止对应步骤并报告，不得绕过。
- 发现文件冲突、hash 边车不匹配或源码身份漂移时，保留现场，不修复、不覆盖、不删除。
- 若结论为 `FAIL`，下一任务应是独立的 O4.1 可复现基线重建，不在本任务内扩展。
