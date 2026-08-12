# 上手指南(ONBOARDING)

按顺序走。原则:**每一步都有"跑通了才算懂"的验收** —— 先复现,再动手。

## 0. 阅读顺序

1. `HANDOVER.md` —— 是什么、结构、关键配置、任务定义(§6 是任务的权威编号)
2. `method_explainer.html` —— 方法逐节图解,浏览器直接打开;绿色标注 = 五项任务在方法里的落点
3. 代码,按这个顺序读:

```
gsplat/compression/ap_gifstream.py            排序冻结 + 两种等量交换(核心,先读)
gsplat/compression/h007_path_contract.py      依赖闭包 + 两档精度掩码
gsplat/compression/gifstream_end2end_compression.py   编解码两端的契约(kNN 摘要、掩码进码流)
examples/simple_trainer_GIFStream.py          损失组装(~2280 起)、冻结时机、路径损失(~2288)
gsplat/compression/h007_sequence_container.py + examples/h007_clean_decode_gifstream.py   容器与干净解码
eval_scripts/223 → 224 → 234                  配对注册 → 全码点路径评估
minimal_codec/(README 先读)                  实例一:极简透明编码器
```

## 1. 起步阶段:只复现,不写新代码

| # | 做什么 | 验收 |
|---|---|---|
| 1 | 装环境,`python -m pytest tests/ -x -q` | 全绿 |
| 2 | 按 HANDOVER §3 重放补丁链、算规范化树 hash | 得到 `0f94bc91…40b64f`,与 manifest 一致 |
| 3 | 对重放树应用 `provenance/local_drift_after_patch8_20260803.patch`,再跑测试 | 与 `GIFStream_APRD/` 一致,测试全绿 |
| 4 | 小规模训练一次(dev 场景)+ 干净解码 | 容器通过 sha 校验;audit dict 里 promoted/demoted 数字合理 |

第 3 步通过后,把 drift 固化为 patch9(和 provenance 里其他补丁同格式),以后所有改动都走"补丁 + manifest"这条链。

## 2. 任务阶梯(编号同 HANDOVER §6,顺序有依赖)

```
任务3 donor 双键(小代码)──┐
                            ├→ 重跑 full 模型 ──→ 任务2 n_knn=0 消融(对新 full 消融)
任务1 D_path 评估器(独立,可并行写)
任务4/5/6 写作注意与定位(起草 method 时用)
```

每项的完成标准:

- **任务 3(donor 双键)**:`_lexicographic_rank` 加第二键(backbone 冻结重要性);
  audit dict 的 `demoted_canonical_ids` 相应变化;为双键并列裁定新增一个测试用例;
  仍是确定性排序、仍 outcome-blind(不许引入任何看得到压缩结果的信号)。
- **任务 2(闭包消融)**:`n_knn: 8 → 0` 跑通(先确认 `h007_path_contract.py`
  的 `retained_rows.numel() <= count` 守卫在 0 时不误触发);消融表加一行;
  报告 |保护范围| / |保留集| 实测占比。
- **任务 1(D_path 评估器)**:基于 `eval_scripts/234_*.py` 扩展;π 的取值在
  评估器里显式打印;π ∈ 0.5×–2× 扫描下结论方向不变;只吃干净解码产物,
  不吃任何训练时张量。
- **任务 4/5**:写 method 时落实(DP 两种失败分开报;闭包声明写清 L=1)。
- **任务 6(写作定位)**:全文以 AP-RD 框架为主线,两个实例并列
  (minimal_codec 证原理 / GIFStream 证真实码流);契约筛选表进正文,
  表中每一行写前自行核对该方法的身份机制;宿主选择理由一句话。

## 3. 三条设计问答(动手前先读,少走弯路)

**Q:保护集为什么不用学习的方法选(比如 Gumbel-softmax)?**
A:选择规则的全部辩护在于它**看不到任何压缩结果**(排序在压缩前冻结)。
从下游损失学出来的选择器按构造就是 outcome-dependent,这条防线就没了。
这是设计原则,不是没想到。

**Q:时间预算为什么要"恰好相等",不用 ≤?**
A:精确等式让"没有多占预算"成为可证的不变量而不是近似声明,且代码里的
子集和 DP 实践中解得出。写作时把两种失败分开报告即可(任务 4)。

**Q:路径损失为什么每 50 步才一次?**
A:双分支解码贵。×50 权重补偿后时间平均等价于每步 0.01,梯度呈脉冲式。
照实写,这是开销折衷,不是缺陷。

## 4. 约定(改代码前必读)

1. **三条不变量不许破坏**:交换的数量守恒、估计字节守恒、所有排序的
   确定性并列裁定(规范身份编号是最终权威)。测试会抓。
2. **改任何进码流的东西**(掩码、量化器参数、图规则)时,同步更新解码端
   验证,否则 sha 校验拒绝解码 —— 这是特性,不是 bug,不要绕过它。
3. **码率只认落盘档案大小**,不认熵估计。
4. 改动积累后及时固化为新补丁 + 更新 manifest,保持链条可重放。
