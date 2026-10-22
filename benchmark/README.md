# U-Net benchmark 实验约定

## 固定条件

- U-Net 结构保持不变；输入 RGB，输出背景/宠物两类。默认 256×256。
- 官方 trainval 按原来的 split seed=0 固定划分：2944 训练、736 验证。test 不用于调参。
- 忽略标签255；主指标为全局前景验证 Dice，不是逐图 Dice 的平均值。
- 第一轮预算固定5个epoch，交叉熵、无增强、float32、num_workers=0。
- 每次从随机初始化开始；同一轮学习率对比使用相同 seed。微调实验不能和从零训练直接混合排名。
- seed 保持数据划分不变，只影响初始化和训练顺序。固定种子不能保证跨硬件逐位一致。

## 分阶段比较

1. 学习率：0.001 / 0.0003 / 0.0001，其他参数不变。
2. batch size：1 / 2 / 4，记录显存、耗时；固定epoch时更新次数不同，BatchNorm行为也会变，结论是完整配置的效果。
3. weight decay：0 / 0.00001 / 0.0001。Adam是耦合L2，AdamW是解耦权重衰减，解释时区分。
4. 优化器：Adam / AdamW，先固定相同学习率和非零衰减；候选优化器可能各有不同的最佳学习率。
5. 后续可扩展损失CE与CE+Dice、增强、调度器、分辨率和AMP，当前没有实现这些实验开关。

plan.csv 是待运行计划，不会自动执行；它不存成绩。results.csv 是实际实验汇总。
不要一开始进行全部参数的笛卡尔积。先单变量初筛，再对前两名用 seed=0、1、2 重复，报告均值和样本标准差。
初筛5轮只能说明短预算表现，不代表充分收敛后的排名；胜出配置再用相同更长预算验证。

## 运行

完整学习率实验（依次训练三个配置，分别预测完整测试集，再绘制柱状图）：

```powershell
python -m benchmark.run_learning_rates
```

代码位于 `benchmark/run_learning_rates.py` 和 `benchmark/evaluate.py`。
每轮实验汇总位于 `benchmark/suites/learning_rates_<时间>/`，包括运行日志、清单、结果 CSV 和 PNG/SVG 柱状图。
权重及逐图预测位于 `benchmark/runs/<实验名>/`。新实验不会覆盖根目录旧权重。
学习率比较只按验证集最佳 Dice 选择；完整测试集成绩作为固定配置的描述性结果，不能继续据此调参。
每个学习率仅执行一个种子，不据此声称统计显著或长期训练的最优学习率。

```powershell
python train_step.py --train --lr 0.001 --epochs 5 --seed 0
python train_step.py --train --lr 0.0003 --epochs 5 --seed 0
python train_step.py --train --lr 0.0001 --epochs 5 --seed 0
python plot_benchmark.py
```

实验串行执行。每次建立独立 runs/<run_id>，包含 config.json、split.json、源代码快照、epochs.csv、result.csv、best.pth。
不会覆盖根目录已有的 unet_pets.pth。预测新实验要明确指定：

```powershell
python predict.py --model benchmark/runs/<run_id>/best.pth --output-dir outputs/<run_id>
```

## 记录与图表

- epochs.csv：逐轮训练/验证 Loss、Dice、学习率、epoch_seconds 和 elapsed_seconds。
- results.csv：每个实验一行，包含配置、状态、最佳轮次、该轮验证Loss/Dice、耗时、PyTorch峰值已分配显存MB、来源。
- 耗时包括训练、验证、保存权重，累计时间还包含日志开销，不包含模型初始化；不是纯GPU算子耗时。
- 显存是 PyTorch max_memory_allocated，不是整张GPU的总占用。
- 每次实验图：训练/验证 Loss 曲线、Dice 曲线。
- 有正式完成结果后：最佳验证Dice柱形图、质量与耗时散点图。绘图不会训练。
- 不同配置混合图只用于查看，推断某变量效果时必须核对其他条件相同。
- Train指标来自训练模式并横跨更新中的参数；Val指标来自固定轮末权重的eval模式，二者差值不能机械解释为过拟合。
- 若进程中断，已完成轮次仍写入CSV，但状态可能保留running；应视为未完成实验，不参与完成实验对比。

## 历史数据

historical_001 的5轮指标来自用户在对话中粘贴的原始日志，配置依据当时脚本记录。原始耗时、显存和权重身份无法追溯，留空。
此记录单独画曲线，不纳入正式benchmark比较。重新训练后的6张测试图评分不能充当完整验证集成绩。

## 下一步完善

- 进度条、异常/OOM状态记录和完整断点恢复。
- 自动执行计划前先审阅预计预算；当前不会批量启动实验。
- 固定验证样本可视化，逐图Dice分布、低分样本分析。
- 主配置筛选完成后再做最终测试评估，减少测试集反馈泄漏。
- 环境版本锁定、数据完整性散列、重复实验误差条。
