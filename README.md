# Piano Score Studio — Antigravity Plugin (v2.0)

一套面向**纯钢琴独奏音频（WAV / MP3 / FLAC）**的开源端到端 **Antigravity 插件**：自动完成音源体检选优、速度/拍号校准、谐波撞音审计、**人类学习版钢琴谱排谱（PDF / MusicXML）**、**双频段力度计划与两轮复查**，并调用开源 **Salamander Grand Piano V3**（16 层力度采样）渲染高音质校准音频与分轨力度踏板 MIDI。

---

## 一、「轻代码仓库 + 一键本地拉取开源资产」

| 层级 | 组成内容 | 精确体积 | 是否直接放进 GitHub 仓库 |
| :--- | :--- | :---: | :--- |
| **1. 本插件代码仓库** | `plugin.json`、`SKILL.md`、检查清单、Eval 门禁、8 个跨平台脚本、安装引导 | **约 `100–160 KB`** | **是**（纯代码与文档，克隆秒下） |
| **2. 高音质钢琴音色 + 渲染器** | `Salamander Grand Piano V3`（FLAC 16 层采样，CC-BY 3.0）+ `sfizz`（BSD） | **约 `725 MB`**（`713.78 MB` + `11.22 MB`） | 否（通过 `./install.ps1` 或 `setup_assets.py --download-salamander` 一键自动拉取） |
| **3. 音频转曲谱 AI 模型** | `MuScriptor medium`（1,171 MB，CC BY-NC 4.0）+ `beat_this`（77 MB） | **约 `1.25 GB`** | 否（受非商业/HF 协议约束，安装脚本引导使用者用个人 HF Token 下载） |
| **4. 制谱与 Python 运行时** | MuseScore 4（~433 MB）+ Python/PyTorch 依赖（~2.5–3.5 GB） | **约 `3.0–4.0 GB`** | 否（本地环境安装） |

> [!TIP]
> **如果你只想把成品音色和代码一起给人**：GitHub 仓库本身只有一百多 KB；其他人克隆后执行安装脚本，**只需再下载约 `725 MB` 的开源钢琴采样即可渲染出同等高音质钢琴曲**；若需要全自动“音频转乐谱 + 高音质渲染”，整机本地资产（含模型与 PyTorch）约占 `5–6 GB`。

---

## 二、目录结构

```text
piano-transcription-studio-plugin/
├── plugin.json                                      # Antigravity 插件清单
├── README.md                                        # 本说明文档
├── LICENSE                                          # 插件代码 MIT 协议
├── THIRD_PARTY_LICENSES.md                          # 第三方音色/模型/软件署名与许可说明
├── .gitignore                                       # 排除本地模型、采样、WAV、虚拟环境与密钥
├── config.example.json                              # 路径与渲染后端配置模板
├── requirements.txt                                 # Python 依赖清单
├── install.ps1                                      # Windows 一键本地部署脚本
├── install.sh                                       # macOS / Linux 一键本地部署脚本
├── rules/
│   └── AGENTS.md                                    # 插件启用时的常驻行为红线
└── skills/
    └── piano-score-studio/
        ├── SKILL.md                                 # 完整 6 阶段工作流与人类可读曲谱审核标准
        ├── references/
        │   ├── human_readable_score_checklist.md    # 8 维度“人类可读学习谱”过关标准清单
        │   ├── workflow_v2.md                       # 完整 SOP 规范
        │   └── eval_rubric_v2.json                  # 6 阶段 16 项机读 Eval 门禁规则
        └── scripts/
            ├── setup_assets.py                      # 环境自检 & Salamander V3 开源采样一键下载解压
            ├── stage0_audio_triage.py               # Stage 0：音频体检、多版本选优、首音检测
            ├── stage2_pitch_audit.py                # Stage 2：小二度撞音、延音幽灵音与调外音谐波审计
            ├── stage3_engrave_study_score.py        # Stage 3：通用学习版钢琴谱生成与 XML 审计
            ├── stage4_dynamics_plan.py              # Stage 4.1：LH(45-350Hz)/RH(350-3500Hz) 力度计划生成
            ├── stage4_render_calibrated.py          # Stage 4.2-4.4：力度标定、CC64 踏板与高音质渲染（支持 sfizz_render / NativeHost）
            ├── stage5_ab_excerpts.py                # Stage 5：开头/高潮/尾奏 12s 原音 vs 校准音 A/B 对照切片
            └── eval_gate_runner.py                  # 全流程机读 Eval 门禁评估器
```

---

## 三、开箱即用安装指南

### 第 1 步：克隆仓库并一键全自动部署

**Windows (PowerShell):**
```powershell
git clone https://github.com/JiananZhao/antigravity-piano-score-studio.git
cd antigravity-piano-score-studio
.\install.ps1 -InstallAll
```

**macOS / Linux:**
```bash
git clone https://github.com/JiananZhao/antigravity-piano-score-studio.git
cd antigravity-piano-score-studio
chmod +x install.sh && ./install.sh --install-all
```

#### 安装可选项一览

| Windows 开关 | macOS / Linux 开关 | 自动完成内容 |
| :--- | :--- | :--- |
| `-InstallAll` | `--install-all` | 一键启用以下全部可选项（最省心） |
| `-InstallTools` | `--install-tools` | 自动安装 **MuseScore 4**（Windows 用 `winget`，macOS 用 `brew`）与 **`sfizz_render`**（自动从 GitHub Releases / `brew` 安装到 `assets_local/sfizz/`） |
| `-DownloadSalamander` | `--download-salamander` | 自动下载并解压 CC-BY 3.0 **Salamander Grand Piano V3**（~714 MB）到 `assets_local/salamander/` |
| `-DownloadMuScriptor` | `--download-muscriptor` | 自动下载 **MuScriptor medium**（~1.17 GB）到 `models/medium/`（若尚未在 HuggingFace 点同意协议并登录，会打印 3 步引导并安全跳过，不中断其余安装） |
| *（默认必执行）* | *（默认必执行）* | 创建 `.venv` 并安装 `requirements.txt`、初始化并将检测到的路径**自动写入 `config.json`**、检查 CUDA GPU 与环境状态 |

> [!IMPORTANT]
> **唯一需要用户本人点一次鼠标的步骤**：`MuScriptor/muscriptor-medium`（CC BY-NC 4.0）在 HuggingFace 上设为受控协议仓库。如需启用“音频 → 乐谱”AI 转录功能，请使用者本人打开 [MuScriptor/muscriptor-medium](https://huggingface.co/MuScriptor/muscriptor-medium) 点一次 **Agree and access repository**，并在终端执行 `huggingface-cli login`（或设置 `HF_TOKEN`），随后即可直接运行 `--install-all` 或 `setup_assets.py --download-muscriptor --write-config` 全自动拉取完成。除此之外的所有工具、Python 环境、制谱软件、采样音色与配置写入均可 100% 全自动完成。

### 第 2 步：挂载为 Antigravity 插件
将本仓库目录复制（或创建软链接）到 Antigravity 全局插件目录即可自动生效：
- **Windows**：`%USERPROFILE%\.gemini\config\plugins\piano-score-studio\`
- **macOS / Linux**：`~/.gemini/config/plugins/piano-score-studio/`

在对话中直接让 Agent 执行如：
> “请使用 `piano-score-studio` 对 `input/` 下的钢琴音频完成选源、学习版制谱、两轮力度校准和 Salamander 高音质音频渲染。”
