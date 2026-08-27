结果很清楚：TabPFN 确实学到了对 EAZY 有效的残差修正，但它不是对每个源都更好。它改善了约 63.4% 的源，并把灾难性异常源从 20 个减少到 15 个；主要风险是对少数原本已经非常准确的 EAZY 结果进行了错误修正。

## 一、总体统计结果

| 结果                                      | 含义                          |
| --------------------------------------- | --------------------------- |
| Blind-test sources = 476                | 完全用于最终评价的 476 个光谱确认源        |
| Improved = 302（63.45%）                  | TabPFN–EAZY 的误差小于 EAZY      |
| Worsened = 174（36.55%）                  | TabPFN 修正后误差反而增加            |
| Fixed EAZY outliers = 10                | 成功修复了 10 个 EAZY 灾难性异常源      |
| New outliers = 5                        | 错误修正导致 5 个新异常源              |
| Shared outliers = 10                    | 两种方法都没能正确预测                 |
| Direction agreement = 77.52%            | TabPFN 给出的修正方向有约 77.5% 是正确的 |
| Residual correlation = 0.903            | TabPFN 学到的修正量与真正需要的修正量高度相关  |
| Median correction = 0.0669              | 对普通源，TabPFN 通常只对 EAZY 做较小调整 |
| Median normalized improvement = 0.00536 | 典型源的归一化误差下降约 0.0054         |
|                                         |                             |

异常源数量可以直接核对：

\[ N_{\mathrm{outlier,EAZY}}=10_{\mathrm{fixed}}+10_{\mathrm{shared}}=20 \]\[ N_{\mathrm{outlier,hybrid}}=5_{\mathrm{new}}+10_{\mathrm{shared}}=15 \]

也就是说，TabPFN 修复了 10 个，却新增了 5 个，净减少 5 个异常源，即从 4.20% 降到 3.15%。

这里的核心不是“TabPFN 每次都正确”，而是：

> 有效修正带来的收益大于错误修正造成的损失。

---

## 二、左上图：TabPFN 学到的修正是否正确

横轴是每个源真正需要的 EAZY 对数残差修正：

\[ r_{\mathrm{true}} = \log(1+z_{\mathrm{spec}}) - \log(1+z_{\mathrm{EAZY}}) \]

纵轴是 TabPFN 实际给出的修正：

\[ \hat r_{\mathrm{TabPFN}} = \log(1+z_{\mathrm{TabPFN-EAZY}}) - \log(1+z_{\mathrm{EAZY}}) \]

虚线 \(y=x\) 代表理想情况：

\[ \hat r_{\mathrm{TabPFN}}=r_{\mathrm{true}} \]

也就是 TabPFN 给出的修正量与真实需要的修正完全相同。

- 绿色点：修正后误差减小。
- 橙色点：修正后误差增大。
- 点越接近虚线：修正量越准确。
- 点在原点附近：EAZY 本来已经比较准确，只需要很小的修正。
- 右上或左下的极端绿色点：EAZY 原来严重低估或高估红移，而 TabPFN 成功做了大幅修正。

这张图最重要的结果是：

\[ \rho(r_{\mathrm{true}},\hat r)=0.903 \]

这是很强的相关性，说明 TabPFN 不是随机扰动 EAZY，而是真正学到了光度特征与 EAZY 误差之间的规律。

但它不等于完全正确。原点附近的一部分橙色点说明：当 EAZY 原本已经准确时，TabPFN 有时会产生不必要的修正。

---

## 三、右上图：每个源到底改善还是恶化

横轴是 EAZY 的绝对归一化误差：

\[ \epsilon_{\mathrm{EAZY}} = \frac{|z_{\mathrm{EAZY}}-z_{\mathrm{spec}}|} {1+z_{\mathrm{spec}}} \]

纵轴是 TabPFN–EAZY 的误差：

\[ \epsilon_{\mathrm{hybrid}} = \frac{|z_{\mathrm{hybrid}}-z_{\mathrm{spec}}|} {1+z_{\mathrm{spec}}} \]

黑色虚线是：

\[ \epsilon_{\mathrm{hybrid}}=\epsilon_{\mathrm{EAZY}} \]

所以：

- 虚线下方：TabPFN 改善了预测。
- 虚线上方：TabPFN 使预测恶化。
- 离虚线越远：改善或恶化的程度越大。

蓝色竖线和绿色横线都是灾难性异常阈值：

\[ \epsilon=0.15 \]

因此图被划分成四种情况：

|位置|含义|
|---|---|
|左下|两种方法都不是异常源|
|右下|EAZY 是异常源，TabPFN 修复成功|
|右上|两种方法都是异常源|
|左上|EAZY 原本正常，TabPFN 制造了新异常源|

颜色也对应这些情况：

- 绿色圆点：成功修复的 EAZY outlier，共 10 个。
- 红色叉号：TabPFN 新引入的 outlier，共 5 个。
- 黑色方块：两种方法共有的 outlier，共 10 个。
- 紫色圆点：两种方法都没有达到灾难性异常标准。

大量紫色点位于虚线下方，与 302 个源改善的统计结果一致。

这张图还暴露出主要失败模式：部分源的 EAZY 误差本来接近零，TabPFN 却做了较大修正，使其越过 0.15 阈值。未来如果继续优化，方向应是控制这种过度修正，而不是重新训练一个完全独立的模型。

---

## 四、左下图：不同红移范围内谁更好

横轴是每个光谱红移区间的中位 \(z_{\mathrm{spec}}\)，纵轴是该区间的中位绝对归一化误差。

- 蓝线：EAZY。
- 绿线：TabPFN–EAZY。
- 绿线低于蓝线：hybrid 更好。
- 每个点上方的 `n`：该红移区间中的测试源数量。

结果显示：

- 在 \(z<1\) 区间，TabPFN–EAZY 明显比 EAZY 差。
- 在大约 \(1<z<8\) 的多数区间，TabPFN–EAZY 都优于 EAZY。
- 在 \(z\approx2\) 到 \(z\approx6\) 的主要样本区间，改善尤其稳定。
- 在 \(z\ge8\) 区间也有改善，但只有 9 个源，不能得出很强的普遍结论。
- \(z<1\) 也只有 22 个源，低红移恶化可能与训练样本分布、特征覆盖或模型过度修正有关。

这张图说明总体提升并非来自单一红移区间，但提升也不是所有区间均匀发生的。当前模型最可靠的贡献集中在样本较丰富的中等红移范围。

---

## 五、右下图：具体星系是怎样被修正的

每一行对应一个代表性源：

- 黑色菱形：真实光谱红移 \(z_{\mathrm{spec}}\)。
- 蓝色圆点：EAZY 预测。
- 绿色方块：TabPFN–EAZY 预测。
- 箭头：从 EAZY 预测移动到 hybrid 预测。
- 绿色箭头：修正后更接近真实值。
- 红色箭头：修正后离真实值更远。

### 成功修复的异常源

例如 `GS:315258`：

\[ z_{\mathrm{spec}}=3.482,\quad z_{\mathrm{EAZY}}=0.32,\quad z_{\mathrm{hybrid}}=3.400 \]

EAZY 把一个 \(z\approx3.5\) 的星系误认为低红移星系，而 TabPFN 几乎将其修正到真实位置：

\[ \epsilon: 0.7055\rightarrow0.0184 \]

`GS:245893` 也是类似情况。这两个源说明 TabPFN 可以利用光度、颜色及 EAZY 诊断信息识别部分严重的红移退化问题。

### 改善但仍然是异常源

`GN:1238092`：

\[ 0.41\rightarrow1.003,\qquad z_{\mathrm{spec}}=3.665 \]

方向正确，误差从 0.698 降至 0.571，但修正幅度仍然不足，因此仍属于灾难性异常源。

这说明“修正方向正确”不一定意味着“最终预测合格”。

### 新产生的异常源

`GN:1067323`：

\[ z_{\mathrm{spec}}=0.253,\quad z_{\mathrm{EAZY}}=0.25,\quad z_{\mathrm{hybrid}}=1.095 \]

EAZY 原本几乎完全正确，TabPFN 却做了一个没有必要的大幅正修正：

\[ \epsilon:0.0026\rightarrow0.6715 \]

这是当前模型最典型、也最值得在 limitations 中说明的失败模式：对原本可靠的模板拟合结果过度修正。

### 普通源的改善与伤害

最后四行没有达到灾难性异常标准，但展示了日常情况下的变化：

- `GS:303480`、`GN:1009377`：小幅修正后几乎落在真实红移上。
- `GS:309703`、`GS:329422`：原来的 EAZY 更准确，TabPFN 修正方向或幅度不合适。

---

## 六、这组图真正证明了什么

它回答的是“EAZY 和 TabPFN 各自贡献了什么”，而不是仅仅再次比较最终指标：

- EAZY 提供物理模板拟合得到的基础红移和稳定锚点。
- TabPFN 学习 EAZY 在什么光度和颜色条件下会产生系统误差。
- TabPFN 主要负责预测对数红移残差，而不是从零替代 EAZY。
- 77.5% 的修正方向正确，残差相关性达到 0.903。
- 63.4% 的测试源得到改善。
- 它修复了部分严重的 EAZY 红移退化，但也可能破坏原本准确的结果。

因此，最准确的一句话结论是：

> TabPFN learned a strongly correlated correction to the EAZY log-redshift residual, improving 63.4% of blind-test sources and reducing catastrophic outliers from 20 to 15, although occasional overcorrection introduced five new failures.

需要注意，这是一张“方法贡献图”，不是“特征重要性图”。它说明 EAZY 基线与 TabPFN 修正各自起什么作用，但不直接回答 F090W、颜色或信噪比中的哪一个特征最重要。

进度与作用：Task 5B 约完成 92%。模型训练、五方法验证、blind test 和方法贡献分析都已完成；现在只剩最终 Case Study 整理，包括 README、notebook 精简、项目结构、结果总结，以及 CV/RS 表述。