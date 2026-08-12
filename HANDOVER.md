# AP-RD / GIFStream 代码交接(2026-08-03)

这是 AP-RD(Action-Preserving Rate–Distortion)在 GIFStream 上的完整实现,
目标是投 ICLR。本文档包含:代码结构、方法速览、可复现性验证、关键配置、
待完成的改进路线、运行方法。

---

## 1. 包结构

```
ONBOARDING.md        上手指南:阅读顺序、复现阶梯、任务验收标准 ← 从这里开始
method_explainer.html  方法逐节图解(浏览器直接打开;绿色标注 = 任务落点)
GIFStream_APRD/      当前工作代码树(主体,直接在这里开发)
provenance/          可审计的来源链
  patches/           官方 GIFStream → 本实现 的 9 个补丁(patch1…patch8)
  h007_ap_gifstream_u3_patch_chain_*.json   补丁链 manifest(含 sha256)
  local_drift_after_patch8_20260803.patch   补丁链之后的本地改动(见 §4)
eval_scripts/        路径评估脚本(223 配对注册 / 224 全码点评估 / 234 修复版 /
                     h007_export_ap_same_run_reference_f03.py 参考导出)
minimal_codec/       实例一:极简透明编码器(逐高斯轨迹,PanopticSports,
                     见其 README)—— 写作定位的关键素材,见 §2b 与任务 6
```

上游基线:GIFStream 官方仓库(github.com/ShanghaiTech 系,CVPR'25),
commit `c98486632e7dafd830740b1a1692bd08c48b96e3`。

## 2. 方法速览(是什么)

问题:动态高斯的常规压缩按渲染质量取舍,会系统性地丢掉**运动**——
少数高速运动的元素对 PSNR 贡献极小,先被剪掉或粗量化,而它们携带的
持久路径(同一身份随时间的轨迹)正是资产的可复用价值。

AP-RD 在**不增加预算、不外挂轨迹流**的前提下保住路径,四个机制:

| 机制 | 一句话 | 代码落点 |
|---|---|---|
| 参考排序冻结 | 压缩前用未压缩参考解出路径,按累计位移排序取前 ρ,冻结(outcome-blind:选择规则看不到任何压缩结果) | `gsplat/compression/ap_gifstream.py` |
| 等量交换 | 被编码器删掉的受保护身份换回来:放回 1 个踢 1 个,保留数严格不变;时间活跃度同理,以冻结熵模型的估计字节相等交换(内部精确子集和 DP) | 同上,`build_count_preserving_anchor_allocation` / `build_equal_estimated_byte_allocation` |
| 解码态契约 | 身份是坐标权威(存精确编号、由编号重算坐标);邻居图两端确定性重建、仅传 sha256 摘要;保护范围 = 受保护集 ∪ 一跳邻居(解码器读邻居,只保护自己不够),范围内细量化、范围外粗量化 | `h007_path_contract.py`、`gifstream_end2end_compression.py` |
| 路径对齐损失 | 训练时双分支解码(未压缩停梯度 vs 压缩模拟),按动作量加权罚差 | `examples/simple_trainer_GIFStream.py` ~2288 |

计率原则:码率 = 完整落盘档案大小(掩码、表头、量化器参数、元数据全部
计入),不用熵估计;每个候选走干净解码验收。

## 2b. 定位:一个原理、两个实例(写作主线)

AP-RD 是**分配 + 契约 + 度量**的一层,不是编码器,必然落在宿主上。
论文以 AP-RD 框架为主角,给出两个实例:

- **实例一 · 极简透明编码器**(`minimal_codec/`):逐高斯轨迹 +
  按冻结动作排序分配 keyframe 密度,匹配 key 预算;数据集自带 3D GT,
  机制全透明 → 证明**原理**。
- **实例二 · GIFStream**(`GIFStream_APRD/`):完整学习编码器
  (anchor + 熵模型 + 端到端 RD)→ 证明原理在**真实 SOTA 码流**上成立。

为什么宿主选 GIFStream:AP-RD 对宿主有三条硬要求(稳定唯一身份全集、
确定性 identity-conditioned 解码器、可恢复身份映射),满足契约的编码器里
GIFStream 最强、最新、代码公开。契约筛选(写进论文前逐一核对各方法的
身份机制,不要照抄):

| 编码器类型 | 契约 | 说明 |
|---|---|---|
| GIFStream | 满足 | 持久 anchor + 每身份时间流 → 本文实例 |
| 4DGC 类(motion grid) | 部分 | 身份持久性弱 |
| TC3DGS / RD4DGS(显式轨迹编码) | — | 是对照组,不是宿主 |
| 形变场类(4D-GS 等) | 不满足 | 无逐身份编码状态可重分配 |
| 前馈 I/P(D-FCGS 等) | 不满足 | 身份映射不可恢复 |

第二个学习型宿主(如 TED-4DGS)写 future work,本轮不做。

## 3. 可复现性验证(怎么确认代码没被动过)

补丁链有密码学锚点,任何时候可以自查:

```bash
# 1) 官方基线 + 9 个补丁重放
git clone <GIFStream官方> tree && cd tree && git checkout c9848663
for p in ../provenance/patches/*.patch; do patch -p1 < "$p"; done

# 2) 用代码库自带的函数算规范化树 hash
python3 -c "
from pathlib import Path; import importlib.util
s=importlib.util.spec_from_file_location('p','gsplat/compression/h007_runtime_provenance.py')
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
print(m.normalized_code_tree(Path('.'))['sha256'])"
# 应得 0f94bc919c616f30bdf8b32d1e28f8a80668bae04178aa10a0ee182def40b64f
# (与 provenance/ 里 manifest 的 normalized_code_tree.sha256 一致)
```

## 4. 本地漂移(重要)

`GIFStream_APRD/` = 补丁链结果 + `local_drift_after_patch8_20260803.patch`。
漂移内容是 patch8 之后的运行时修复(kNN 的 float64 确定性、factor 激活
处理、估计字节的边界守卫等,11 个文件、约 500 行)。**这部分尚未在集群上
完整回归验证** —— 接手后第一件事建议:跑一遍测试 + 一次小规模训练回归,
确认漂移无害,然后把它固化为 patch9。

```bash
cd GIFStream_APRD && python -m pytest tests/ -x -q
```

## 5. 关键配置事实(论文里必须写、代码里已固定)

- `ap_path_loss_lambda = 0.01`;`ap_path_loss_every = 50`(每 50 步一次、
  权重 ×50 补偿 → 时间平均等价每步 0.01,梯度是脉冲式的)
- `n_knn`:依赖闭包的邻居数,配置项(消融要用,见 §6)
- 训练必须开 `compression_sim` + `entropy_model_opt`(不开会 raise)
- 码率档位:`FROZEN_RATE_LAMBDAS[cfg.rate]`
- 确定性:所有排序并列由规范身份编号裁定;编解码两端 kNN 图各自重建,
  sha256 摘要不一致 = 解码作废

## 6. 改进路线(ICLR 前要完成的,按优先级)

1. **D_path 评估器(要写新代码)**:方法的目标量
   `D_path = Σ匹配身份的路径误差 + Σ缺失身份 × π`,目前只分项报
   (MTE / missing-action),从未合成报告。定为 `w_i = 1`,
   `π = 冻结参考集动作量的中位数`(只依赖参考,outcome-blind),
   并扫 π ∈ 0.5×–2× 证明结论不随 π 翻转。基于 `eval_scripts/234_*.py` 扩展。
2. **闭包消融(改配置重跑)**:`n_knn: 8 → 0`,即「只保护受保护集自己」
   vs「含一跳邻居」。这是全方法最花码率的设计,目前没有直接消融;
   同时报告 |保护范围| / |保留集| 的实测占比。
   注意 `h007_path_contract.py` 中 `retained_rows.numel() <= count` 守卫,
   n_knn=0 跑前先确认不触发别处异常。
3. **donor 双键(小代码改动 + 重跑)**:目前交换踢出的 donor 只按
   动作量最低排序,而动作量对视觉重要性零信息,可能踢掉静止但重要的
   元素。改为两把键:动作量升序 → backbone 自身冻结重要性估计升序。
   落点 `_lexicographic_rank` 加第二键。保持确定性与 outcome-blind。
   (不要做成可学习的选择器:从下游损失学出来的选择器会破坏
   outcome-blind 论证,那是本方法最强的防线。)
4. **写作注意 A**:时间预算交换的"估计字节恰好相等"由一个可达子集和
   DP 实现(`exact_subset`,带状态数上限)。它有两种失败:凑不出
   (真不可行)与超出状态帽(算力放弃)。写作时分开陈述、分别报告
   触发次数,不要合并成一句 "infeasible"。
5. **写作注意 B**:一跳闭包只对一层状态解码器严格成立;声明适用范围时
   写清"本实例 L=1,L 层解码器推广为 L 跳、规模按 k^L 增长",
   不要写成适用于任何编码器。
6. **写作定位(见 §2b)**:AP-RD 框架为主角,GIFStream 降为实例二;
   `minimal_codec/` 提为实例一;附契约筛选表(写前逐一核对);
   宿主选择理由一句话;第二学习宿主进 future work。

可选探索(有余力再做,先跑最便宜的验证):熵模型目前是 conditional
Gaussian;假设它对运动瞬变系统性次优(因果 context 预测不了运动起始 →
瞬变处条件残差重尾;高斯率项对大偏差施加无界收缩力 → 路径误差随幅度
线性增长)。最便宜的检验:从现有 checkpoint 抖出 (ŷ, μ, σ),算归一化
残差 z=(y−μ)/σ,按动作十分位/瞬变帧切片做 PIT 均匀性与峰度检验。
若 z 处处 ≈ N(0,1) 则假设死,止损;若尾部集中在瞬变处,这是一个
独立的诊断贡献方向。

## 7. 运行

```bash
# 训练(dev 场景 flame_salmon,N3DV 协议)
python examples/simple_trainer_GIFStream.py \
  --compression_sim --entropy_model_opt --rate <档位> \
  --ap_variant <变体> ...   # 变体与全部 flag 见文件头 Config

# 干净解码 + 容器验证
python examples/h007_clean_decode_gifstream.py ...

# 路径评估(配对注册 → 全码点评估)
python eval_scripts/223_*.py ...
python eval_scripts/234_*.py ...
```

数据:N3DV/Neu3D 半分辨率协议(dev 用 flame_salmon_1,确认集五场景:
coffee_martini / cook_spinach / cut_roasted_beef / flame_steak / sear_steak,
300 帧,留出中心视角);PanopticSports 用于轨迹 GT 锚定。
数据预处理沿用 GIFStream 官方 `dataset_process/`。

## 8. 约定

- 别破坏三条不变量:交换的**数量守恒**与**估计字节守恒**、以及
  **所有排序的确定性并列裁定**。测试会抓(`tests/test_ap_gifstream_core.py`)。
- 改任何进码流的东西(掩码、量化器、图规则)时,同步更新解码端验证,
  否则 sha 校验会拒绝解码 —— 这是特性,不是 bug。
- 报码率永远用落盘档案大小,不用熵估计。
