# Speaker Notes — Overfitting & Underfitting in Machine Learning
# 演讲稿 — 机器学习中的过拟合与欠拟合

**AIAA 2711 — Spring 2025 Presentation**

**Total time: ~9 minutes 50 seconds + Q&A buffer**
**总时长：约9分50秒 + Q&A缓冲**

| Speaker   | Slides  | Duration |
|-----------|---------|----------|
| Speaker 1 | 1–6b   | 4:30     |
| Speaker 2 | 7–10   | 3:20     |
| Speaker 3 | 11–14  | 2:00     |

---

## Slide 1 — Title (10s) — Speaker 1

**EN:** Good morning everyone. My name is [Speaker 1], and together with [Speaker 2] and [Speaker 3], we will present our study on **overfitting and underfitting** in machine learning. Our talk has three parts: theory, regularization, and experiments. Let's begin.

**中文：** 大家早上好。我是[Speaker 1]，和[Speaker 2]、[Speaker 3]一起为大家介绍机器学习中的**过拟合与欠拟合**。我们的演讲分三部分：理论、正则化和实验。让我们开始。

---

## Slide 2 — The Hook: Which Model Do You Trust? (50s) — Speaker 1

**EN:** [POINT TO FIGURE] Take a look at these three polynomial fits. The **same data points**, but three very different models.

On the left, **degree one** — a straight line. It clearly misses the curved pattern. [PAUSE 1s]

In the middle, **degree four**. It follows the general trend without chasing every point. [PAUSE 1s]

On the right, **degree fifteen**. It passes through every single training point, but look at those wild oscillations between them.

[INTERACTION] Let me ask for a quick show of hands. How many of you would trust the **degree-one** model to predict a new data point? [PAUSE 2s] How about **degree four**? [PAUSE 2s] And **degree fifteen**? [PAUSE 2s]

Most of you chose degree four — and your intuition is correct. But **why** is it the best choice?

This is the **fundamental tension** in machine learning: too simple versus too complex. The straight line **underfits** — it cannot capture the pattern. The degree-fifteen curve **overfits** — it memorizes noise. Today we will make this intuition **mathematically precise**.

**中文：** [指向图片] 请看这三个多项式拟合。**同样的数据点**，却有三种完全不同的模型。

左边是**1次多项式**——一条直线，明显没法捕捉弯曲的模式。

中间是**4次多项式**，它沿着整体趋势走，但不追逐每个点。

右边是**15次多项式**，它穿过每一个训练数据点，但看看点之间那些剧烈的震荡。

[互动] 让我快速做个调查。多少人会相信**1次**模型来预测新数据？举手。[停顿2秒] **4次**呢？[停顿2秒] **15次**呢？[停顿2秒]

大部分人选了4次——你们的直觉是对的。但**为什么**它是最好的选择？

这就是机器学习的**核心矛盾**：太简单 vs 太复杂。直线**欠拟合**——无法捕捉模式。15次曲线**过拟合**——记住了噪声。今天我们将用**数学**来精确解释这一现象。

---

## Slide 3 — Formal Definitions (40s) — Speaker 1

**EN:** Let us start with **formal definitions**.

**Training error** is the average loss over our training data. We compute it directly from the N samples we trained on.

**Test error** is the expected loss over the **true data distribution**. This is what we actually care about — how well the model generalizes.

Now, two key terms.

**Underfitting** means training error is **high**. The model is too simple to learn even the training data.

**Overfitting** means training error is **much lower** than test error. The model fits training data well but fails on new data.

The **generalization gap** — the difference between test error and training error — is our **diagnostic tool**. A large gap signals overfitting. Keep this in mind.

**中文：** 我们先给出**严格定义**。

**训练误差**是模型在训练数据上的平均损失，直接从N个训练样本计算。

**测试误差**是在**真实数据分布**上的期望损失。这才是我们真正关心的——泛化能力。

两个关键概念：

**欠拟合**意味着训练误差就**很高**。模型太简单，连训练数据都学不好。

**过拟合**意味着训练误差**远低于**测试误差。模型在训练数据上表现好，但在新数据上失败。

**泛化间隙**——测试误差与训练误差之差——是我们的**诊断工具**。间隙越大说明过拟合越严重。请记住这一点。

---

## Slide 4 — The Key Question (20s) — Speaker 1

**EN:** [POINT TO FIGURE] This plot shows what happens as we increase model complexity.

**Training error** decreases **monotonically** — more complex models always fit training data better.

But **test error** follows a **U-shape**. It first drops, then rises. The **minimum** sits around degree three to five.

So the question becomes: **why** does test error behave this way? The answer is the **bias-variance decomposition**. This uses expectation and variance from **Weeks nine and ten** of our course.

**中文：** [指向图片] 这张图展示了随模型复杂度增加的变化。

**训练误差**单调递减——更复杂的模型总能更好地拟合训练数据。

但**测试误差**呈**U型曲线**。先下降，再上升。**最小值**在3到5次附近。

问题来了：**为什么**测试误差是这样变化的？答案就是**偏差-方差分解**。这会用到第9、10周学过的期望和方差概念。

---

## Slide 5 — Bias-Variance Setup (40s) — Speaker 1

**EN:** Here is the setup.

We assume the data follows this model: **y equals f of x plus epsilon**. The true function is f of x. The noise epsilon is Gaussian with **mean zero** and **variance sigma squared**.

We train a model f-hat on a dataset D. Different training sets give us **different models**.

Our goal is to **decompose** the expected prediction error. Specifically, we want to break down the expectation over D of y minus f-hat, quantity squared.

Here is the critical point. The expectation is over **all possible training datasets D**. We are asking: on average, across all training sets we could draw, how far off is our model? This is what makes the decomposition powerful.

**中文：** 这是我们的基本设定。

假设数据遵循模型：**y = f(x) + ε**。f(x)是真实函数，噪声ε服从均值为零、方差为σ²的高斯分布。

我们在数据集D上训练模型f̂。不同的训练集会给出**不同的模型**。

我们的目标是**分解**预期预测误差，即对D取期望的(y - f̂)²。

关键点在于：这个期望是对**所有可能的训练数据集D**取的。我们在问：平均来看，跨越所有可能的训练集，模型偏离真相多远？这就是这个分解的核心力量。

---

## Slide 6a — Bias-Variance Derivation: Steps (50s) — Speaker 1

**EN:** This is the **core derivation** of our presentation. I will walk through it step by step — showing every term, so you can see exactly where each piece comes from.

[CLICK] **Step one.** We start by substituting y equals f plus epsilon into the squared error. When we expand the square, we get **three terms**:

The first term is f minus f-hat, quantity squared — the **model error** squared.

The second term is epsilon squared — the **noise** squared.

The third term is the **cross-term**: two times epsilon times f minus f-hat.

Now, why does this cross-term vanish? Let me write it out. Since epsilon is **independent** of f-hat, and f is a fixed function, we can factor the expectation. The expectation of epsilon is **zero** — it's zero-mean Gaussian noise. Zero times anything is zero. So the entire cross-term **disappears**.

And the expectation of epsilon squared is simply **sigma squared** by definition of variance.

[CLICK] This gives us our first key equation: Expected error equals the expected model error squared, **plus sigma squared**.

[CLICK] **Step two.** Now we need to decompose the model error term. The trick is to introduce a **shorthand**: let f-bar equal the expected value of f-hat over all datasets. Think of f-bar as what your model predicts **on average**.

We add and subtract f-bar inside the square. Expanding gives us **three terms** again:

The first is f minus f-bar, quantity squared — this is a **constant** because both f and f-bar are fixed.

The second is the expected squared deviation of f-hat from f-bar — how much the model **varies**.

The third is the **cross-term**: two times f minus f-bar, times the expectation of f-bar minus f-hat.

[CLICK] This cross-term: the expectation of f-bar minus f-hat equals f-bar minus E of f-hat. But f-bar **IS** E of f-hat — that's how we defined it! So f-bar minus f-bar equals **zero**. The cross-term vanishes.

**中文：** 这是我们演讲的**核心推导**。我会逐步展示每一项，让你们看清每个部分的来源。

[点击] **第一步。** 把 y = f + ε 代入平方误差展开，我们得到**三项**：

第一项是 (f - f̂)²——**模型误差**的平方。

第二项是 ε²——**噪声**的平方。

第三项是**交叉项**：2ε(f - f̂)。

这个交叉项为什么为零？让我写出来。因为 ε 和 f̂ **独立**，f 是固定函数，所以期望可以拆分。E[ε] = 0——零均值高斯噪声。零乘以任何东西都是零。所以整个交叉项**消失了**。

而 E[ε²] 就是方差的定义，等于 **σ²**。

[点击] 这给出第一个关键等式：期望误差 = 期望模型误差² + σ²。

[点击] **第二步。** 现在分解模型误差项。技巧是引入**简写符号**：令 f̄ = E_D[f̂]，即模型在所有数据集上的**平均预测**。

在平方内加减 f̄，展开又得到**三项**：

第一项 (f - f̄)² 是**常数**，因为 f 和 f̄ 都是固定的。

第二项是 f̂ 偏离 f̄ 的期望平方偏差——模型的**波动**程度。

第三项是**交叉项**：2(f - f̄) · E_D[f̄ - f̂]。

[点击] 这个交叉项：E_D[f̄ - f̂] = f̄ - E_D[f̂]。但 f̄ **就是** E_D[f̂]——这就是我们的定义！所以 f̄ - f̄ = **零**。交叉项消失。

---

## Slide 6b — Bias-Variance Result & Verification (40s) — Speaker 1

**EN:** [CLICK] Putting Steps one and two together, we arrive at the **fundamental result**:

**Expected Error equals Bias squared plus Variance plus sigma squared.**

[POINT TO FORMULA] Let me interpret each term.

**Bias squared** — f minus f-bar, quantity squared — measures how far the **average model** is from the true function. If your model class cannot represent the true function well, bias is high. That means **underfitting**.

**Variance** — the expected squared deviation of f-hat from f-bar — measures how much the model **changes** when you use a different training set. If the model is very sensitive to which particular data points it trains on, variance is high. That means **overfitting**.

**Sigma squared** is the **irreducible noise**. Even if you had the perfect model, the data itself is noisy. This is the floor — no model can go below it.

[POINT TO FIGURE] Now look at this figure on the right. We ran a **Monte Carlo simulation**: generated 200 independent datasets from the same true function, fit polynomials of each degree, and computed bias and variance empirically.

The **blue curve** shows bias squared — it decreases as degree increases. The **orange curve** shows variance — it increases. Their **sum** — the total error — follows the U-shape we predicted. The minimum sits right around degree **three to four**.

This is the bias-variance tradeoff in action. Theory predicts it, experiments confirm it.

That concludes the theoretical foundation. Now [Speaker 2] will show us how to **fix overfitting** using regularization.

**[HANDOFF to Speaker 2]**

**中文：** [点击] 把第一步和第二步合在一起，我们得到**核心结果**：

**期望误差 = 偏差² + 方差 + σ²**

[指向公式] 解释每一项：

**偏差²**——(f - f̄)²——度量**平均模型**离真实函数有多远。如果模型类别无法很好地表示真实函数，偏差就高，这意味着**欠拟合**。

**方差**——f̂ 偏离 f̄ 的期望平方偏差——度量换一组训练数据后模型**变化**多大。如果模型对具体数据点非常敏感，方差就高，这意味着**过拟合**。

**σ²** 是**不可约噪声**。即使有完美模型，数据本身有噪声。这是下限——没有模型能低于它。

[指向图片] 看右边这张图。我们做了**蒙特卡洛模拟**：从同一真实函数生成200个独立数据集，对每个多项式度数拟合，然后实证计算偏差和方差。

**蓝色曲线**是偏差²——随度数增加而递减。**橙色曲线**是方差——随度数增加而递增。它们的**和**——总误差——呈现我们预测的U型。最小值恰好在**3到4次**附近。

这就是偏差-方差权衡的实证。理论预测它，实验验证它。

理论基础讲完了。接下来由[Speaker 2]讲解如何用**正则化修复过拟合**。

**[交接给 Speaker 2]**

---

## Slide 8 — Regularization: Constraining Complexity (30s) — Speaker 2

**EN:** Thank you, [Speaker 1].

We just saw that **high variance** leads to overfitting. The natural fix is to **constrain the model's complexity**.

On the left, we have **ordinary least squares** — OLS. It minimizes the squared error with **no constraints**. The model is free to use any coefficient values.

On the right, we have **Ridge regression**. It adds an **L2 penalty** — alpha times the squared norm of the weights.

One quick note: the L2 norm here is the **Euclidean norm** from Week three. So this connects directly to material you already know.

**中文：** 谢谢[Speaker 1]。

我们刚看到**高方差**导致过拟合。自然的解决方法是**约束模型复杂度**。

左边是**普通最小二乘法**OLS，最小化平方误差，**没有任何约束**，系数可以取任意值。

右边是**Ridge回归**，添加了**L2惩罚项**——α乘以权重的平方范数。

提一下：这里的L2范数就是第三周学过的**欧几里德范数**，直接对应你们已有的知识。

---

## Slide 9 — Ridge: Closed-Form & Interpretations (80s) — Speaker 2

**EN:** Ridge regression has a **clean closed-form solution**.

**w-ridge** equals X-transpose-X **plus alpha-I**, inverse, times X-transpose-y.

Compare this to OLS, which is just X-transpose-X inverse, times X-transpose-y. The only difference is adding **alpha times the identity matrix**.

This small change has **two powerful interpretations** that connect to our course material.

**Interpretation one: Eigenvalue perspective**, from Week five.

Recall that X-transpose-X has eigenvalues lambda-one through lambda-p. Adding alpha-I **shifts** every eigenvalue from lambda-i to lambda-i **plus alpha**.

Why does this matter? When some eigenvalues are near zero, the OLS solution becomes **unstable** — small changes in data cause huge changes in coefficients. Adding alpha **stabilizes** the inversion. This directly reduces **variance**.

**Interpretation two: Lagrangian perspective**, from Week nine.

Ridge regression is **equivalent** to a constrained optimization problem. We minimize the squared error **subject to** the constraint that the squared norm of w is less than or equal to some threshold t.

The parameter alpha is exactly the **Lagrange multiplier** from the KKT conditions. Increasing alpha shrinks the constraint region, forcing **simpler** models.

The key insight: adding alpha-I trades a **small increase in bias** for a **large decrease in variance**.

**中文：** Ridge回归有一个**优美的闭式解**。

**w_ridge** = (X^TX + αI)^(-1) X^Ty。

和OLS对比，唯一区别就是多了**α乘以单位矩阵**。

这个小变化有**两个深刻的数学解释**，都和课程内容相关。

**解释一：特征值视角**，来自第五周。

回忆X^TX的特征值λ₁到λₚ。加上αI后，每个特征值从λᵢ变为λᵢ + α。

这为什么重要？当某些特征值接近零时，OLS解变得**不稳定**——数据的微小变化导致系数巨变。加上α**稳定了**矩阵求逆，直接**降低方差**。

**解释二：拉格朗日视角**，来自第九周。

Ridge等价于一个**约束优化问题**：最小化平方误差，约束条件是||w||² ≤ t。

参数α正好是KKT条件中的**拉格朗日乘子**。α增大，约束区域缩小，迫使模型更**简单**。

核心洞察：加上αI，用**小幅偏差增加**换取**大幅方差降低**。

---

## Slide 10 — Lasso & Geometric Comparison (60s) — Speaker 2

**EN:** Now let us compare Ridge with **Lasso**.

Lasso replaces the L2 penalty with an **L1 penalty**: alpha times the **sum of absolute values** of the weights.

The key difference is **geometric**.

The L2 constraint set is a **circle** — or a sphere in higher dimensions. The level curves of the loss function are ellipses. They typically meet the circle at a point where **no coordinate is exactly zero**. So Ridge **shrinks** all coefficients, but they stay **non-zero**.

The L1 constraint set is a **diamond**. Its corners sit on the coordinate axes. The level curves of the loss function almost always hit a **corner** first. At a corner, one or more coordinates are **exactly zero**.

[POINT TO FIGURE] You can see this in the coefficient plot. Ridge coefficients shrink smoothly toward zero. Lasso coefficients **drop to exactly zero** one by one.

This gives Lasso a unique advantage. Lasso is not just regularization — it is also **automatic feature selection**. It tells you which features matter and which can be discarded.

**中文：** 现在把Ridge和**Lasso**做对比。

Lasso把L2惩罚换成了**L1惩罚**：α乘以权重的**绝对值之和**。

关键区别在于**几何形状**。

L2约束集是**圆形**（高维中是球）。损失函数的等高线是椭圆。它们通常在**没有坐标恰好为零**的点与圆相交。所以Ridge让所有系数**收缩**，但不会变为零。

L1约束集是**菱形**。它的**角落在坐标轴上**。损失函数的等高线几乎总是先碰到**角落**。在角落处，一个或多个坐标**恰好为零**。

[指向图片] 从系数图可以看到：Ridge系数平滑地趋向零，而Lasso系数**逐个变为恰好零**。

这赋予Lasso独特优势。Lasso不仅是正则化，还是**自动特征选择**。它告诉你哪些特征重要、哪些可以丢弃。

---

## Slide 11 — Cross-Validation: Choosing α (30s) — Speaker 2

**EN:** So both Ridge and Lasso have a hyperparameter alpha. How do we choose the right value?

The standard answer is **K-fold cross-validation**. It works in four steps.

**Step one:** partition the data into K equal folds. **Step two:** for each fold k, train on the other K minus one folds and evaluate on fold k. **Step three:** average the K scores. **Step four:** select the alpha that minimizes this average CV error.

The beauty of cross-validation is that it gives an **unbiased estimate** of test error **without** needing a separate validation set. Every data point is used for both training and evaluation.

That covers the methods. Now [Speaker 3] will show whether the **experiments** confirm our theory.

**[HANDOFF to Speaker 3]**

**中文：** Ridge和Lasso都有超参数α。怎么选最优值？

标准答案是**K折交叉验证**，分四步。

**第一步：** 将数据分成K等份。**第二步：** 对每一份k，用剩余K-1份训练、在第k份上评估。**第三步：** 取K个分数的平均。**第四步：** 选使CV误差最小的α。

交叉验证的优美之处在于：**不需要**单独的验证集就能得到测试误差的**无偏估计**。每个数据点既用于训练又用于评估。

方法讲完了。接下来[Speaker 3]展示**实验是否验证了我们的理论**。

**[交接给 Speaker 3]**

---

## Slide 12 — Experimental Verification (50s) — Speaker 3

**EN:** Thank you, [Speaker 2].

We made **three predictions** from the theory, and we tested **all three** experimentally.

[POINT TO FIGURE] Let me walk through the table.

**Prediction one:** test error should be U-shaped with respect to complexity. We fit polynomials of increasing degree. The result: **confirmed**.

**Prediction two:** as complexity increases, bias should decrease and variance should increase. We ran a **Monte Carlo simulation** with 200 independently sampled datasets. The result: **confirmed**. The curves cross exactly where test error is minimized.

[POINT TO FIGURE] You can see this clearly in the bias-variance tradeoff plot. The **blue** curve is bias squared — it decreases. The **orange** curve is variance — it increases. Their **sum** follows the U-shape.

**Prediction three:** Ridge should shrink coefficients and reduce test error. The result: **confirmed**.

In short, the theory works. Every prediction matches the experiment.

**中文：** 谢谢[Speaker 2]。

我们从理论做了**三个预测**，并**全部**进行了实验验证。

[指向图片] 让我逐一说明。

**预测一：** 测试误差应随复杂度呈U型。我们拟合了不同次数的多项式，结果：**验证通过**。

**预测二：** 复杂度增加时，偏差应下降、方差应上升。我们用200个独立采样的数据集做了**蒙特卡洛模拟**，结果：**验证通过**。曲线恰好在测试误差最小处交叉。

[指向图片] 偏差-方差权衡图清楚展示：**蓝色**曲线是偏差²——递减；**橙色**曲线是方差——递增。它们的**和**形成U型。

**预测三：** Ridge应收缩系数并降低测试误差，结果：**验证通过**。

总结：理论成立，每个预测都与实验一致。

---

## Slide 13 — Deep Learning & Summary (40s) — Speaker 3

**EN:** You might wonder: does this only apply to polynomials?

[POINT TO FIGURE] This plot shows a **neural network** experiment. We trained networks of increasing size on the same synthetic data. The pattern is the **same**: wider networks fit training data better, but at some point, test error starts to rise.

The bias-variance tradeoff is a **universal principle**.

Let me summarize with **three key takeaways**.

**First:** the bias-variance decomposition gives us a **mathematical explanation** for why test error is U-shaped.

**Second:** regularization — Ridge with L2, Lasso with L1 — **controls complexity** through norm penalties.

**Third:** cross-validation is the **practical tool** for selecting the best balance point.

**中文：** 你们可能会问：这只适用于多项式吗？

[指向图片] 这张图展示了**神经网络**实验。我们在同样的数据上训练了不同规模的网络。模式**完全相同**：更大的网络拟合训练数据更好，但到一定程度后测试误差开始上升。

偏差-方差权衡是**普适原理**。

用**三个要点**总结：

**第一：** 偏差-方差分解为U型测试误差提供了**数学解释**。

**第二：** 正则化——Ridge的L2、Lasso的L1——通过范数惩罚**控制复杂度**。

**第三：** 交叉验证是选择最佳平衡点的**实用工具**。

---

## Slide 14 — Interactive Closing & Q&A (30s) — Speaker 3

**EN:** Before we open for questions, let me give you a **quick scenario**.

[INTERACTION] Suppose your model achieves **99 percent** training accuracy, but only **60 percent** test accuracy.

Two questions. **First:** is this overfitting or underfitting? **Second:** what would you do to fix it?

[PAUSE 5s] Take a moment to think.

[CLICK] Here is the answer. The generalization gap is 99 minus 60 — that is **39 percent**. Training accuracy is high, but test accuracy is much lower. This is clearly **overfitting**.

The fix? **Add regularization** or reduce model complexity. Exactly the tools we discussed today.

Thank you for your attention. We are happy to take questions.

**[END — Q&A]**

**中文：** 最后，进入Q&A之前，给大家一个**小测试**。

[互动] 假设你的模型达到了**99%**的训练准确率，但只有**60%**的测试准确率。

两个问题。**第一：** 这是过拟合还是欠拟合？**第二：** 你会怎么修复？

[停顿5秒] 思考一下。

[点击] 答案是：泛化间隙 = 99% - 60% = **39%**。训练准确率高但测试准确率远低于此，这明显是**过拟合**。

怎么修复？**加正则化**或降低模型复杂度。正是今天讲的方法。

感谢大家的关注，欢迎提问。

**[演讲结束 — Q&A]**
