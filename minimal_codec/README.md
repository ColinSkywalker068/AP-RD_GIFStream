# 实例一:极简透明编码器(minimal codec)

AP-RD 的第一个实例,落在**逐高斯轨迹表示**上(Dynamic3DGS 风格,
PanopticSports 公开数据集,自带 3D GT 轨迹)。它的价值:机制全透明
(没有学习组件的干扰)、有真值轨迹可以直接量 —— 用来证明**原理与宿主无关**,
GIFStream(实例二)则证明原理在真实 SOTA 学习码流上成立。

## 管线(5 步)

```
panoptic_make_temporal_variants.py     分配:按冻结的动作排序给每个高斯分配
                                       keyframe 密度/量化档,在匹配的 key 预算下
                                       与 baseline 变体对照(对应 AP-RD 的
                                       排序冻结 + 预算守恒分配)
panoptic_pack_trajectory_payloads.py   打包:量化 keyframe payload
panoptic_entropy_codec.py              熵编码器(无损,payload → bitstream)
panoptic_pack_entropy_bitstreams.py    批量出码流 + 记账
panoptic_eval_decoded_payload_mte.py   解码 payload → 逐身份 MTE(含 top-10% 路径)
panoptic_eval_entropy_bitstream_mte.py 从真实码流解码再评(端到端闭环)
panoptic_render_eval.py                渲染质量核对(保运动没有伤画质)
panoptic_track_eval.py                 轨迹指标工具(被上面两个评估脚本 import)
```

依赖:numpy、torch、Pillow,以及 `diff_gaussian_rasterization`
(3DGS 官方光栅化器,Dynamic 3D Gaussians 项目用的同一个,pip 从源码装)。

## 原则(与主实现一致)

- 排序在分配前冻结,分配规则看不到任何压缩结果
- 预算匹配:AP 变体与对照变体的 key 预算相当,赢在分配而不是多花
- 评估以**解码产物**为准:MTE 从解出的 payload/码流上算,码率按实际
  bitstream 字节报

脚本均为独立 CLI,路径全部走参数;数据准备见 PanopticSports 公开发布
(Dynamic 3D Gaussians 项目)。
