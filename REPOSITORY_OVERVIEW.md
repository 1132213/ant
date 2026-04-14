# Ant-Game 仓库概述

## 项目简介

这是 **2025-2026 第三十届智能体大赛 蚁洋陷役2** 的游戏代码仓库，是一个双人在六边形网格地图上对抗的策略游戏。

玩家通过操控基地，派遣蚂蚁进攻敌方基地，同时建造防御塔和使用超级武器来改变战局。

---

## 目录结构

```
Ant-Game/
├── AI/                    # AI策略实现目录
├── SDK/                   # 软件开发工具包
├── game/                  # C++游戏引擎源码
├── tests/                 # 测试套件
├── tools/                 # 辅助工具脚本
├── README.md              # 项目主文档
├── pytest.ini             # Pytest配置
├── zip.sh                 # 打包脚本
└── .gitlab-ci.yml         # CI/CD配置
```

---

## 核心模块详解

### 1. AI/ - AI策略模块

存放各种AI策略实现，是选手编写AI的主要目录。

| 文件 | 说明 |
|------|------|
| `main.py` | AI统一入口，负责协议通信，从打包后的`ai.py`导入AI类 |
| `common.py` | 基础代理类和会话管理 |
| `protocol.py` | 通信协议处理 |
| `ai_example.py` | 示例AI，最简参考实现 |
| `ai_random.py` | 随机策略AI |
| `ai_greedy.py` | 贪心策略AI（入口文件） |
| `ai_greedy/` | 贪心策略实现模块 |
| `ai_mcts.py` | MCTS蒙特卡洛树搜索AI |
| `zip_*.sh` | 各AI的打包脚本 |

**AI继承体系：**
- `BaseAgent` (基类) → `ExampleAgent` → 自定义AI
- 核心方法：`choose_bundle(state, player, bundles)` 返回 `ActionBundle`

---

### 2. SDK/ - 软件开发工具包

提供游戏规则、状态管理、训练环境等核心功能。

#### 2.1 SDK/backend/ - 后端引擎

| 文件 | 说明 |
|------|------|
| `core.py` | 核心后端实现，支持Python和Native C++后端 |
| `state.py` | 游戏状态管理（`BackendState`, `PythonBackendState`） |
| `engine.py` | 游戏引擎实现（`GameState`, `PublicRoundState`, `TurnResolution`） |
| `model.py` | 数据模型（`Operation`, `Tower`, `Ant`等） |
| `runtime.py` | 比赛运行时（`MatchRuntime`） |
| `forecast.py` | 预测模拟器（`ForecastSimulator`, `ForecastState`） |

#### 2.2 SDK/utils/ - 工具模块

| 文件 | 说明 |
|------|------|
| `actions.py` | 动作系统（`ActionBundle`, `ActionCatalog`） |
| `constants.py` | 游戏常量（塔属性、超级武器、地图尺寸等） |
| `features.py` | 特征提取器 |
| `geometry.py` | 几何计算（六边形距离、邻居计算等） |

#### 2.3 SDK/training/ - 训练环境

| 文件 | 说明 |
|------|------|
| `env.py` | 并行训练环境（`AntWarParallelEnv`） |
| `base.py` | 训练基类（`BaseSelfPlayTrainer`） |
| `alphazero.py` | AlphaZero训练框架 |
| `selfplay.py` | 自对弈训练器 |
| `policies.py` | 策略网络 |
| `logging_utils.py` | 训练日志工具 |

#### 2.4 SDK根目录

| 文件 | 说明 |
|------|------|
| `__init__.py` | SDK公共API导出 |
| `alphazero.py` | AlphaZero算法核心实现（`PolicyValueNet`, `PriorGuidedMCTS`） |
| `native_antwar.cpp` | Native C++加速模块 |
| `native_adapter.py` | Native后端适配器 |
| `train_example.py` | 训练示例脚本 |
| `train_mcts.py` | MCTS训练脚本 |

---

### 3. game/ - C++游戏引擎

原生C++实现的高性能游戏引擎，用于正式评测。

```
game/
├── src/                   # 源文件
│   ├── main.cpp          # 主入口
│   ├── game.cpp          # 游戏核心逻辑
│   ├── ant.cpp           # 蚂蚁逻辑
│   ├── building.cpp      # 建筑/防御塔逻辑
│   ├── map.cpp           # 地图逻辑
│   ├── coin.cpp          # 金币系统
│   ├── aco.cpp           # 信息素系统
│   ├── operation.cpp     # 操作处理
│   ├── output.cpp        # 输出处理
│   └── comm_judger.cpp   # 通信判题器
├── include/              # 头文件
│   ├── game.hpp          # 游戏类定义
│   ├── ant.h             # 蚂蚁类
│   ├── building.h        # 建筑类
│   ├── map.h             # 地图类
│   ├── player.h          # 玩家类
│   ├── item.h            # 道具类
│   ├── coin.h            # 金币类
│   ├── operation.h       # 操作类
│   ├── output.h          # 输出类
│   ├── pos.h             # 位置类
│   ├── comm_judger.h     # 通信类
│   └── json.hpp          # JSON库
├── lib/                  # 静态链接库目录
└── Makefile              # 编译配置
```

---

### 4. tests/ - 测试套件

| 文件 | 说明 |
|------|------|
| `conftest.py` | Pytest配置和fixtures |
| `test_engine.py` | 引擎测试 |
| `test_env_and_protocol.py` | 环境和协议测试 |
| `test_actions_and_ai.py` | 动作和AI测试 |
| `test_training.py` | 训练模块测试 |
| `test_packaging.py` | 打包测试 |
| `test_cpp_runtime.py` | C++运行时测试 |
| `fixtures/` | 测试数据fixtures |

---

### 5. tools/ - 辅助工具

| 文件 | 说明 |
|------|------|
| `run_local_match.py` | 本地对局运行脚本 |
| `setup_native.py` | Native后端编译脚本 |

---

## 游戏核心概念

### 地图系统
- 六边形网格地图（"even-q"坐标系）
- 地图尺寸：边长为10的正六边形
- 玩家0基地：(2, 9)，玩家1基地：(16, 9)
- 不同区域有不同功能（建造区、移动区）

### 防御塔系统
- 13种防御塔类型
- 3个升级等级
- 分支升级树：Basic → Heavy/Quick/Mortar/Producer → 高级塔

### 蚂蚁系统
- **兵种(kind)**：普通蚂蚁、战斗蚂蚁
- **行为(behavior)**：默认型、保守型、随机型、蛊惑型、免控型
- 风险感知寻路算法
- 信息素系统

### 超级武器
1. 闪电风暴 (Lightning Storm)
2. EMP轰炸 (EMP Blaster)
3. 引力护盾 (Deflectors)
4. 紧急回避 (Emergency Evasion)

---

## 开发指南

### 编写自定义AI

1. 创建 `AI/ai_xxx.py`
2. 继承 `BaseAgent`
3. 实现 `choose_bundle()` 方法
4. 创建对应的 `AI/zip_xxx.sh` 打包脚本

```python
from AI.common import BaseAgent
from SDK.utils.actions import ActionBundle
from SDK.backend import BackendState

class MyAI(BaseAgent):
    def choose_bundle(self, state: BackendState, player: int, 
                      bundles: list[ActionBundle] | None = None) -> ActionBundle:
        bundles = bundles or self.list_bundles(state, player)
        # 你的策略逻辑
        return bundles[0]

class AI(MyAI):
    pass
```

### 运行本地对局

```bash
python tools/run_local_match.py --ai0 greedy --ai1 mcts --seed 7
```

### 运行训练

```bash
# 示例训练
bash SDK/train_example.sh --seed 1 --max-actions 16

# MCTS训练
bash SDK/train_mcts.sh --episodes 2 --iterations 24 --max-depth 3 --seed 1
```

### 打包AI

```bash
# 打包示例AI
bash AI/zip_example.sh

# 打包贪心AI
bash AI/zip_greedy.sh

# 打包MCTS AI
bash AI/zip_mcts.sh
```

### 运行测试

```bash
pytest tests/
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 游戏引擎 | C++ |
| AI/SDK | Python 3 |
| 训练框架 | NumPy (可扩展PyTorch等) |
| 测试框架 | pytest |
| 构建工具 | make (C++), shell scripts |

---

## 通信协议

### 输入格式
- 初始化信息：`K M`（玩家ID和随机种子）
- 局面信息：回合数、塔信息、蚂蚁信息、金币、基地状态等

### 输出格式
- 四字节大端序长度前缀 + JSON数据
- 操作类型：建造塔(11)、升级塔(12)、降级塔(13)、超级武器(21-24)、基地升级(31-32)

---

## 关键常量

| 常量 | 值 |
|------|-----|
| 最大回合数 | 512 |
| AI每回合时间限制 | 10秒 |
| 初始金币 | 50 |
| 基础收入 | 每2回合3金币 |
| 普通蚂蚁最大寿命 | 64回合 |

---

## 文件依赖关系

```
AI/main.py
    ├── AI/common.py (BaseAgent, MatchSession)
    ├── AI/protocol.py (ProtocolSession)
    └── ai.py (打包后的AI类)

SDK/__init__.py
    ├── SDK/backend/ (游戏状态、引擎)
    ├── SDK/utils/ (动作、特征、常量)
    └── SDK/training/ (训练环境)

game/src/main.cpp
    └── game/include/*.h (所有头文件)
```

---

## 扩展资源

- [游戏规则详解](./README.md) - 完整游戏规则文档
- [AI示例](./AI/ai_example.py) - 最简AI实现参考
- [贪心AI实现](./AI/ai_greedy/) - 较完整的策略实现
- [MCTS AI实现](./AI/ai_mcts.py) - 蒙特卡洛树搜索实现

---

*文档生成时间：2026年4月5日*
