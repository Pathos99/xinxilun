# 6G空天通信课程作业 - 完整交付物说明

## 文件清单

### 1. 核心报告文件
| 文件名 | 格式 | 说明 |
|--------|------|------|
| **6G_Course_Project_Report.docx** | Word | **主报告文件**，包含完整项目内容、图片占位符、所有技术细节 |
| 6G_Final_Report_v2.md | Markdown | 补充版Markdown报告 |
| 6G_Space_Ground_Research_Report.md | Markdown | 原始研究报告 |
### 2. 代码文件
| 文件名 | 语言 | 说明 |
|--------|------|------|
| **matlab_figures.m** | MATLAB | **图表生成代码**，运行后生成10张专业图表 |
| **6g_satellite_simulation.py** | Python | **主仿真代码**，包含完整的6G空地通信仿 |
### 3. 数据与结果文件
| 文件名 | 格式 | 说明 |
|--------|------|------|
| simulation_results.csv | CSV | 仿真结果CSV格式 |
| simulation_results.png | PNG | Python自动生成的结果可视化 |
| 6G_Technical_Analysis_v2.png | PNG | 技术分析综合图 |

---

## 🔧 运行环境与依赖

### Python仿真代码 (6g_satellite_simulation.py)

#### 环境要求
```
Python >= 3.8
```

#### 依赖安装
```bash
pip install numpy pandas matplotlib scipy openpyxl
```

#### 输出文件
运行后将在当前目录生成：
- `simulation_results.csv` - CSV格式结果
- `simulation_results.png` - 4面板可视化图表

---

### MATLAB图表代码 (matlab_figures.m)

#### 环境要求
```
MATLAB R2020a 或更高版本
```
#### 输出文件 (10张图)
| 序号 | 文件名 | 内容 | 对应报告章节 |
|------|--------|------|--------------|
| 1 | fig01_beam_pattern.png | 多波束增益方向图 (Bessel函数) | Section 4.1 |
| 2 | fig02_dvbs2x_acm.png | DVB-S2X ACM频谱效率 | Section 4.2 |
| 3 | fig03_coding_selection.png | 6G编码选择 (按块长度) | Section 4.3 |
| 4 | fig04_interference_framework.png | IBI/ISI/LTI干扰框架 | Section 4.4 |
| 5 | fig05_slicing_qos.png | 网络切片QoS性能 | Section 7 |
| 6 | fig06_phased_array.png | 相控阵波束增益 | Section 4.1 |
| 7 | fig07_leo_constellation.png | LEO星座覆盖示意图 | Section 1 |
| 8 | fig08_architecture_mapping.png | 5G→6G架构映射 | Section 1 |
| 9 | fig09_channel_model.png | Shadowed-Rician信道模型 | Section 4 |
| 10 | fig10_simulation_results.png | 资源分配仿真结果 | Section 7 |

---

##  原始数据文件使用说明

### 竞赛原始数据文件
以下文件来自2025华数杯B题，用于仿真：

| 文件名 | 内容 | 在仿真中的用途 |
|--------|------|----------------|
| **channel_data.xlsx** | 单基站信道数据 | 问题1-2: 大规模衰减、小规模衰减、用户位置 |
| **BS1.xlsx** | 用户-BS1信道 | 问题3: 多基站干扰场景 |
| **BS2.xlsx** | 用户-BS2信道 | 问题3: 多基站干扰场景 |
| **BS3.xlsx** | 用户-BS3信道 | 问题3: 多基站干扰场景 |
| **taskflow.xlsx** | 用户任务到达 | 所有问题: 任务调度 |
| **MBS_1.xlsx** | 用户-宏基站信道 | 问题4-5: 异构网络 |
| **SBS_1.xlsx** | 用户-微基站1信道 | 问题4-5: 异构网络 |
| **SBS_2.xlsx** | 用户-微基站2信道 | 问题4-5: 异构网络 |
| **SBS_3.xlsx** | 用户-微基站3信道 | 问题4-5: 异构网络 |

### 数据结构
每个Excel文件包含多个Sheet:
- **大规模衰减**: 用户到基站的路径损耗 (dB)，随位置变化
- **小规模瑞丽衰减**: 复数信道系数，随时间变化
- **用户任务流**: 每个时间槽的任务到达概率

### 用户分类
- **U1-U10**: URLLC用户 (高可靠低时延)
- **e1-e20**: eMBB用户 (增强移动宽带)
- **m1-m40**: mMTC用户 (大规模机器通信)

---

##  Word报告使用说明

### 6GCourse_Project_Report.docx

这是主报告文件，包含：

1. **项目概述** - 背景、课程要求、架构映射
2. **数据文件说明** - 所有原始数据的详细说明
3. **文献综述** - 8篇论文的关键成果梳理
4. **技术框架** - 多波束、DVB-S2X、信道编码、干扰管理
5. **仿真代码说明** - Python代码结构、运行方法
6. **MATLAB代码说明** - 图表生成代码说明
7. **结果分析** - 仿真结果与性能评估
8. **结论** - 创新点与课程要求满足情况

---

## 论文关键成果速查

### 核心论文

#### Paper 6: IEEE COMST 2025 - 频谱共享与干扰管理
**核心贡献**: IBI/ISI/LTI三类干扰分类框架
- IBI (波束间干扰): 同卫星不同波束 → 波束跳变、预编码 (-35%)
- ISI (星间干扰): 不同卫星 → 排斥区、联合优化 (-28%)
- LTI (星地干扰): LEO与地面网络 → 频率配对、RIS (-55%)

#### Paper 7: IEEE Proc. 2024 - 6G信道编码趋势
**核心贡献**: 按块长度选择最优编码
- N ≤ 128: 卷积码 (Viterbi解码)
- 128 < N ≤ 512: Polar码 (SCL-8解码)
- N > 512: LDPC码 (BP-12迭代)

### 其他论文关键参数

| 论文 | 关键参数/公式 |
|------|--------------|
| Paper 1 | LDPC: BG1 (N=68, M=46), BG2 (N=52, M=42) |
| Paper 2 | 混合波束赋形: W_k = W_k^RF × W_k^BB |
| Paper 3 | 频率复用: 4-color (0.73Gb/s), Block-SVD (+120%) |
| Paper 4 | 星座: Starlink (42K卫星, 550-1325km) |
| Paper 5 | 三层管理: MEO (全局) → CH-LEO (区域) → SES (优先级) |

---

## 课程要求对照表

| 课程要求 | 报告章节 | 关键证据 |
|----------|----------|----------|
| 多波束技术 | Section 4.1 | Bessel增益模型, G_max=48dBi, θ_3dB=0.5° |
| 频率复用 | Section 4.4 | IBI/ISI/LTI框架, 干扰降低35-55% |
| 高波束增益 | Section 4.1 | 相控阵64-1024单元, 23-35 dBi |
| 高阶调制 | Section 4.2 | DVB-S2X 256APSK, 5.51 bit/s/Hz |
| 信道编码 | Section 4.3 | 块长度自适应选择 (IEEE Proc. 2024) |
