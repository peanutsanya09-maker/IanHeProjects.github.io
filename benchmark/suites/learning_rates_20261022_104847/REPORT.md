# 学习率 benchmark 结果

已实际执行3组训练，每组5轮，并用各自最佳验证权重对完整3669张官方测试图片预测。完成时间：2026-10-22。

## 固定条件

U-Net结构不变，Adam，weight_decay=0，batch_size=2，256×256，float32，无数据增强、无调度器、无早停，从随机初始化开始，seed=0。
训练2944张，验证736张；相同数据划分。标签255不参与损失与指标。按全局前景验证Dice选择最佳轮次。
每组完整配置、代码快照、数据划分和权重分别保存在对应run目录；environment.txt记录环境，manifest.json记录实验清单。

## 结果

| 学习率 | 最佳轮次 | 验证Loss | 验证Dice | 测试Dice | 测试IoU | 训练及验证耗时（分钟） |
| --- | --- | --- | --- | --- | --- | --- |
| 0.001 | 5 | 0.2816 | 0.8209 | 0.8253 | 0.7025 | 9.70 |
| 0.0003 | 5 | 0.2432 | 0.8546 | 0.8603 | 0.7548 | 9.66 |
| 0.0001 | 5 | 0.1910 | 0.8929 | 0.8974 | 0.8139 | 9.62 |

精确数值见 results.csv；图见 learning_rate_bars.png 与可编辑矢量版本 learning_rate_bars.svg。
耗时包括训练、验证、权重保存，不含初始化；单组测试预测约60至64秒，另列 prediction_seconds。

## 结论与限制

- 本次固定5轮预算、单种子实验，0.0001的最佳验证Dice最高，比0.001高约0.0720（7.20个百分点）。
- 该配置在完整测试集的全局Dice为0.8974，但不是像素准确率；逐图平均Dice为0.8866，二者计算口径不同。
- 三组耗时接近，不能将几秒差异解释为学习率本身带来的稳定加速。
- 全局分数高不意味着每张图片都好；例如最佳配置的 japanese_chin_46 仍存在漏分，Dice约0.7924。
- 每组仅1次运行，没有误差条或统计显著性结论，不能推断充分收敛后仍是同一排名。
- 测试集成绩仅作描述，不用它重新选择超参数；后续决策仍使用验证集。
- 三组后期验证结果总体改善，但仅凭5轮不能排除更长训练时的过拟合。

## 文件结构

```text
benchmark/
  run_learning_rates.py       # 批量执行入口
  evaluate.py                 # 完整测试集预测和逐图指标
  runs/<suite>_<lr>/
    best.pth
    config.json
    split.json
    epochs.csv
    result.csv
    predictions/
      summary.csv
      per_image.csv
      *_comparison.png
      *_mask.png
  suites/learning_rates_20261022_104847/
    results.csv
    learning_rate_bars.png
    learning_rate_bars.svg
    manifest.json
    environment.txt
    *_train.log
    *_predict.log
```

预测最佳权重：

```powershell
python predict.py --model benchmark/runs/learning_rates_20261022_104847_lr_1e4/best.pth --output-dir outputs/lr_benchmark_best
```

根目录旧的 unet_pets.pth 未被覆盖；predict.py 不传 --model 时仍使用旧权重。

## 验证

已核对：三组除学习率和输出目录外配置一致；模型代码与划分一致；共15条epoch记录；每组3669个唯一测试图ID且顺序相同；共18张对比图；最佳权重SHA256与汇总相符；CSV最佳Dice与逐轮记录一致。已查看柱状图与预测示例。
