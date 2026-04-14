# Ant-Game: AlphaZero 结合启发式剪枝策略文档

本文档旨在描述针对 2025-2026 第三十届智能体大赛（蚁洋陷役2）所实现的 AlphaZero 混合启发式策略（AlphaZero with Heuristic Pruning）。

## 1. 策略背景与设计动机

传统的深度强化学习（Deep Reinforcement Learning, DRL）方法（如纯正的 AlphaZero）虽然效果上限极高，但具有以下显著缺陷：
1. **训练算力需求海量**：从零开始探索成千上万的动作空间需要极长的探索周期和巨大的计算资源。
2. **训练极易崩溃（翻车）**：缺乏有效引导，在复杂游戏中可能迟迟无法学会有意义的战术。
3. **推理开销大**：每回合 10 秒的时间限制下，MCTS 展开次数可能不足以覆盖所有可能的动作。

为了在**有限算力**下冲击高排名（Top 20% / 四强），我们采用了一种**“混合模型”（Hybrid Model）**的思路：
**将预先编写的专家启发式规则（Heuristic Rules）作为动作空间的剪枝器（Action Pruner），结合 AlphaZero 的价值网络（Value Network）和策略网络（Policy Network）进行树搜索和自我对弈（Self-Play）。**

---

## 2. 核心机制详解

### 2.1 动作空间剪枝 (Action Pruning)
强化学习最怕的就是动作空间爆炸。Ant-Game 每回合合法动作可能多达几百种。我们通过引入混合模型中的专家知识（`custom_utils.py`），强行干预 MCTS 的 `_expand()` 阶段。
在每一次树搜索展开节点时，网络只会在以下被过滤出的高价值动作中进行打分：
- **建塔**：只在根据兵线形势（DEFEND/ATTACK/BALANCED）动态计算出的“最佳前线战略格”建塔。
- **升级塔**：只优先升级威胁热点区域内、且等级不到满级的塔。
- **释放超级武器**：
  - **闪电风暴**：当敌方兵线压境或敌方蚂蚁高密度集结时触发。
  - **EMP 轰炸**：当扫描到敌方塔阵密集（至少能瘫痪 2 级以上的塔总和）时触发。
  - **引力护盾 / 紧急回避**：当我方战斗蚂蚁冲锋至敌方半场深处、面临敌方炮火时触发保护。
- **升级基地**：经济富余且无前线直接威胁时尝试点科技。

### 2.2 价值目标缩放与真实胜负主导 (Value Target & Reward Scaling)
网络在初期的盲目探索中很难遇到基地被摧毁的情况。为了加快收敛，我们引入了稠密奖励（兵线压制分、经济分、塔数分）。
- **约束机制**：如果单纯使用混合模型的打分，分数可能累加到非常大，导致网络无法区分“中期场面占优”和“最终赢下比赛”。我们将所有稠密奖励强制缩放并约束在 `(-0.95, 0.95)` 之间（使用 `tanh(total/3.0) * 0.95`）。
- **终极目标**：在 `_value_target` 中，只有当游戏**真正分出胜负**时，网络才会得到绝对的 `+1.0` 或 `-1.0` 的强烈反馈。这种绝对的差值诱惑将驱使网络在后期学会“孤注一掷推平对手”。

### 2.3 MCTS 推理极致加速
为了满足 Saiblo 评测环境 **10秒/回合** 的严苛要求：
- 禁用了 ActionCatalog 内置的 O(N) 级别 1-step rollout 排序。
- 在 `ai_alphazero.py` 中引入了 **Time-Bound MCTS** 机制。只要运行时间接近阈值，直接阻断 MCTS 的深入探索，转而依靠已遍历节点中的最多访问次数出牌。

---

## 3. 架构与文件说明

| 文件路径 | 模块说明 |
| -------- | -------- |
| `AI/ai_alphazero.py` | 比赛 AI 的实际运行类 `AlphaZeroAgent`。包含模型加载、自定义 MCTS 推演（`custom_expand`）以及超时防护。 |
| `SDK/train_alphazero_custom.py` | 训练入口脚本。自定义了 `CustomPriorGuidedMCTS`，在自我对弈收集数据时使用与推理一致的动作剪枝和稠密奖励机制。 |
| `AI/custom_utils.py` | 存放用于启发式剪枝的专家规则和状态分析工具（由 Phase 2 混合模型遗留）。 |
| `checkpoints/ai_alphazero_latest.npz` | 训练出的神经网络权重文件（需通过 `train_alphazero_custom.py` 训练生成）。 |

---

## 4. 使用指南

### 4.1 启动训练 (Self-Play)
使用 `antgame` conda 环境执行训练：
```bash
conda run -n antgame python SDK/train_alphazero_custom.py --batches 10 --episodes 10 --iterations 50 --max-depth 5 --learning-rate 1e-3
```
- `--batches`: 网络更新次数。
- `--episodes`: 每次更新前收集的自我对弈局数。
- `--iterations`: MCTS 在训练时的搜索次数。

### 4.2 本地测试对局
验证训练出的模型对战其他 AI（如贪心 AI）：
```bash
conda run -n antgame python tools/run_local_match.py --ai0 alphazero --ai1 greedy --seed 7
```

### 4.3 打包提交
将 AI 代码及最新训练的模型权重打包成可提交的 `zip` 格式：
```bash
conda run -n antgame python AI/package_ai.py alphazero
```
生成的文件为 `AI/ai_alphazero.zip`，可直接上传到 Saiblo 评测平台。

---

## 5. 后续演进方向建议
1. **扩展剪枝动作池**：在 `custom_utils.py` 中加入更多的针对超级武器（EMP, 引力护盾等）的释放时机判断，供网络学习。
2. **调整 Mix 参数**：训练稳定后，逐渐增加 `--prior-mix` 和 `--value-mix` 的权重，让网络接管更多决策。
3. **增加训练规模**：如果有 GPU 或多核机器，可加大 `batches` 和 `episodes`，让模型泛化能力更强。