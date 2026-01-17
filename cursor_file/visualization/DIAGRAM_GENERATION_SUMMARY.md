# Bio-COT Framework Diagram Generation Summary

## 生成时间
2025-12-30 09:56

## 文件位置
`/data2/hmy/VLM_Caus_Rm_Mics/cursor_file/visualization/bio_cot_framework_diagram.pdf`

## 图表内容

### 1. 输入层
- **Clinical Data**: HPV, TCT, Age [B, 7]
- **Image Features**: OCT + Colposcopy [B, 512]
- **Center Labels**: Hospital ID [B]

### 2. 核心模块
- **Student Prior Network**: 轻量级MLP (7 → 256 → 512 → 768)
  - 输出: z_sem (语义锚点) [B, 768]
- **Dual Head Encoder**: 特征投影 (512 → 1536 → 768)
  - 输出: z_causal [B, 768] 和 z_noise [B, 768]
- **Noise Memory Bank**: 存储不同中心的噪声特征
  - 输出: z_noise_cf (反事实噪声) [B, 768]

### 3. 损失函数
- **Sinkhorn OT Loss**: L_ot = OT(z_causal, z_sem)
- **Consistency Loss**: L_consist = ||logits - logits_cf||
- **Adversarial Loss**: L_adv = CE(D(z_noise), center_id)
- **Classification Loss**: L_cls = CE(logits, labels)

### 4. 输出
- **Classifier**: MLP (768 → 768 → 384 → 2)
- **Logits**: 分类结果 [B, 2]

### 5. 创新点标注
- Innovation 1: Student Prior (Fast Training)
- Innovation 2: Sinkhorn OT (Flexible Alignment)
- Innovation 3: Memory Bank (Counterfactual)

## 设计特点

1. **清晰的布局**: 24x16 画布，避免重叠
2. **颜色编码**: 
   - 输入: 浅蓝色
   - Student Prior: 浅橙色
   - Encoder: 浅绿色
   - Memory Bank: 浅红色
   - Loss: 浅灰色
   - Output: 浅黄色
3. **箭头连接**: 
   - 实线箭头: 数据流
   - 虚线箭头: 损失计算
4. **英文标签**: 所有文字使用英文，字体完整
5. **公式标注**: 使用等宽字体显示维度信息

## 总损失公式
L = λ_cls·L_cls + λ_ot·L_ot + λ_consist·L_consist + λ_adv·L_adv

