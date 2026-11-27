# 钢琴音频转学习版曲谱与高音质音频工作流规范（v2.0）

版本：2.0 | 适用插件：`piano-score-studio` | 机读标准：`references/eval_rubric_v2.json`

## 1. 核心定位与硬约束
1. **双重交付**：同时产出人类学习版乐谱（`score_learning.pdf` / `.musicxml`）与经 Salamander Grand Piano V3（16 层采样）渲染校准的高音质 48 kHz/24-bit WAV 及分轨 MIDI。
2. **禁止占位符直出**：Stage 2 初稿允许 `velocity = 100` 占位，但**严禁直接用于最终音频渲染**；正式渲染前必须先产出 `dynamics_plan.md` 并完成两轮力度复查。
3. **两轮谱面复查**：Stage 3 必须通过 Round 1（XML 结构审计）+ Round 2（PDF 逐页视觉审查），满足 [human_readable_score_checklist.md](human_readable_score_checklist.md) 全部 8 个维度。
4. **先计划后渲染、两轮力度复查**：Stage 4 必须先运行 `stage4_dynamics_plan.py` 分析原音 `LH (45–350 Hz)` 与 `RH (350–3500 Hz)` 双频段包络，执行标定后依次完成 Round 1（静态 MIDI/XML 分层复查）与 Round 2（渲染后双频段声学闭环复查）。
5. **只读与安全**：不覆盖原始音频；所有密钥（如 `hf_token`）与本地大文件不进入 Git 仓库。

## 2. 六阶段流水线与命令行调用

- **Stage 0（音频体检与选源）**：`scripts/stage0_audio_triage.py`
- **Stage 1（速度/拍号/网格校准与转录）**：`scripts/stage1_tempo_and_transcribe.py`（`beat_this` + 圆统计 `|R|` 扫描 + 20s 分段漂移检验 + 低音/和声循环检验 + 亚拍网格选择 + `MuScriptor` 转录 → 输出 `stage1_calibration.json`、`score.mid`、`score.musicxml`）
- **Stage 2（撞音/幽灵音/调外音谐波审计）**：`scripts/stage2_pitch_audit.py`
- **Stage 3（学习版排谱 + 2 轮复查）**：`scripts/stage3_engrave_study_score.py` → `MuseScore4` 渲染 → 逐页视觉审查 PDF
- **Stage 4（力度计划 → 执行 → 2 轮复查）**：`scripts/stage4_dynamics_plan.py` → `scripts/stage4_render_calibrated.py`（支持 `sfizz_render` CLI 与 `NativeHost`）
- **Stage 5（A/B 切片与总门禁）**：`scripts/stage5_ab_excerpts.py` + `scripts/eval_gate_runner.py`
