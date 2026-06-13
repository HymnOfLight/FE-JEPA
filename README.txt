FE-JEPA 研究提案：中英双语包
=====================================

内容清单
--------
FE-JEPA_proposal_ZH.tex / .pdf   中文版（13 页）
FE-JEPA_proposal_EN.tex / .pdf   英文版（14 页）
两版逐节对应：摘要 / 引言 / 研究现状 / 空白分析与可行性 /
FE-JEPA 课题（含引理 1 完整证明与命题 2 草案）/ 执行计划
（阶段 0–4、E1–E5 预注册证伪实验、G0–G2 门控、指标、算力、
风险登记表）/ 预期贡献 / 局限。参考文献两版一致，保持英文原貌。

编译说明
--------
中文版：xelatex 编译两遍；依赖 fontspec + 系统字体
  "Noto Serif CJK SC"（Linux 常见预装）。文件刻意不依赖
  ctex/xeCJK；若本地装有 ctex，可按文件头注释换用
  \documentclass{ctexart} 并删除 fontspec 三行。
英文版：pdflatex 编译两遍即可。

参考文献核验状态
----------------
正式 bibliography 中的条目均已核验或为高置信度常识文献。
三篇 2026 预印本（AeroJEPA arXiv:2605.05586、Shape
arXiv:2604.22826、LGS arXiv:2602.11229）以脚注引用且
**作者名单未录入**，投稿前请在 arXiv 核验。文中提及的会议
截止日期一律以官方 CFP 为准。

生成日期：2026-06-12
