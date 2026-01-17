import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.colors as mcolors

def draw_academic_roadmap():
    # 1. 设置画布 (16:9 宽屏适合论文横版插图)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')  # 隐藏坐标轴

    # ---------------------------------------------------------
    # 2. 定制配色方案 (根据您的要求)
    # ---------------------------------------------------------
    # 用户指定的背景色
    user_palette = {
        'purple_grey': '#E8E7EB',  # A: Foundation Models
        'blue_grey':   '#D5D9E3',  # B: Agents (and Arrow)
        'warm_grey':   '#ECE8E6',  # C: Federated
        'rose_grey':   '#E3DEDE',  # D: Physics
    }

    # 为了保证对比度，我们为边框生成稍微深一点的同色系颜色
    # 这里的逻辑是：让边框比背景深，看起来更精致专业
    colors = {
        # A. Foundation Models
        'A_bg': user_palette['purple_grey'],
        'A_border': '#9FA0B5',  # 深紫灰

        # B. Autonomous Agents
        'B_bg': user_palette['blue_grey'],
        'B_border': '#8C9BB5',  # 深蓝灰

        # C. Federated Learning
        'C_bg': user_palette['warm_grey'],
        'C_border': '#9E9892',  # 深暖灰

        # D. Physics-Informed
        'D_bg': user_palette['rose_grey'],
        'D_border': '#A89696',  # 深玫瑰灰

        # E. Regulatory (复用 Rose Grey 但加深一点区分，或者复用 Purple)
        'E_bg': user_palette['rose_grey'], 
        'E_border': '#B86B6B',  # 稍微强调一点红色调代表Regulatory/Warning

        # 通用设置
        'text_title': '#2F2F2F',  # 深炭色标题，比纯黑柔和
        'text_body':  '#454545',  # 正文颜色
        'arrow_fill': '#C4CCD9',  # 箭头颜色 (基于蓝灰色加深)
        'arrow_edge': '#8C9BB5'
    }

    # 设置字体 (优先使用 Arial, 如果没有则使用系统无衬线字体)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    # ---------------------------------------------------------
    # 3. 绘制中央时间轴箭头 (曲线上升)
    # ---------------------------------------------------------
    # 使用二次贝塞尔曲线，因为FancyArrowPatch不支持三次贝塞尔曲线
    verts = [
        (0.5, 0.5),   # 起点
        (8, 4.5),     # 控制点
        (15.5, 8.5),  # 终点
    ]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    path = Path(verts, codes)
    
    # 绘制箭头实体
    pp = patches.FancyArrowPatch(
        path=path, 
        arrowstyle='simple,head_length=20,head_width=20,tail_width=8',
        facecolor=colors['arrow_fill'], 
        edgecolor=colors['arrow_edge'],
        alpha=0.6, 
        zorder=0, 
        mutation_scale=1.5
    )
    ax.add_patch(pp)

    # 时间轴标签 (沿着箭头)
    ax.text(3, 1.0, "2025: Data & Federated Learning", fontsize=11, fontweight='bold', color='#555555', rotation=8)
    ax.text(8, 4.5, "2030: Convergence of Agentic AI", fontsize=11, fontweight='bold', color='#555555', rotation=55)
    ax.text(12.5, 8.0, "2035+: Global Ecosystem", fontsize=11, fontweight='bold', color='#555555', rotation=8)

    # ---------------------------------------------------------
    # 4. 绘制模块框的辅助函数
    # ---------------------------------------------------------
    def draw_box(x, y, width, height, title, content, bg_color, border_color):
        # 阴影效果 (可选，为了增加立体感，这里用很淡的灰色)
        shadow = patches.FancyBboxPatch(
            (x+0.05, y-0.05), width, height,
            boxstyle="round,pad=0.2,rounding_size=0.2",
            fc='#E0E0E0', alpha=0.5, zorder=9
        )
        ax.add_patch(shadow)

        # 主体方框
        box = patches.FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.2,rounding_size=0.2",
            edgecolor=border_color,
            facecolor=bg_color,
            linewidth=2,
            zorder=10
        )
        ax.add_patch(box)
        
        # 标题栏背景 (稍微深一点，让标题突出)
        # title_bg = patches.Rectangle((x, y+height-0.6), width, 0.6, color=border_color, alpha=0.1, zorder=11)
        # ax.add_patch(title_bg)

        # 标题文字
        ax.text(x + 0.2, y + height - 0.3, title, fontsize=12, fontweight='bold', color=colors['text_title'], zorder=12)
        
        # 正文内容
        ax.text(x + 0.25, y + height - 0.8, content, fontsize=10, color=colors['text_body'], va='top', linespacing=1.6, zorder=12)

    # ---------------------------------------------------------
    # 5. 填充内容 (基于您的论文部分)
    # ---------------------------------------------------------
    
    # A. Foundation Models (左上) -> Purple Grey
    content_a = (
        "• GMAI & Large Multi-modal Models\n"
        "• Zero-shot / Few-shot Learning\n"
        "• Semantic Prompting for Rare Cases\n"
        "• Overcoming Label Scarcity"
    )
    draw_box(0.5, 5.8, 4.5, 2.2, "A. Foundation Models & GMAI", content_a, colors['A_bg'], colors['A_border'])

    # B. Autonomous Agents (左中) -> Blue Grey
    content_b = (
        "• Agentic AI: Active Reasoning\n"
        "• Workflow Orchestration (EHR + OCT)\n"
        "• Dynamic Risk Profiling\n"
        "• Interactive Clinician Reports"
    )
    draw_box(0.5, 2.8, 4.5, 2.2, "B. Autonomous Diagnostic Agents", content_b, colors['B_bg'], colors['B_border'])

    # C. Federated Learning (左下) -> Warm Grey
    content_c = (
        "• Data Privacy Preserved\n"
        "• Federated Aggregation: θ ← Σ(wk · θk)\n"
        "• Self-Supervised Pretraining\n"
        "• Addressing Data Fragmentation"
    )
    draw_box(5.5, 0.5, 4.5, 2.2, "C. Federated Self-Supervised Learning", content_c, colors['C_bg'], colors['C_border'])

    # D. Physics-Informed (右上) -> Rose Grey
    content_d = (
        "• Physics-Informed Neural Networks (PINN)\n"
        "• Modeling Light Transport: s(z)\n"
        "• Reducing Hallucinations\n"
        "• Enhanced Interpretability"
    )
    draw_box(11.0, 5.8, 4.5, 2.2, "D. Physics-Informed Modeling", content_d, colors['D_bg'], colors['D_border'])

    # E. Regulatory & Edge (右下) -> Rose Grey (强调)
    content_e = (
        "• Prospective Clinical Validation\n"
        "• Edge AI: Quantization & Pruning\n"
        "• Portable Battery-Operated Devices\n"
        "• Secure Updates & Drift Adaptation"
    )
    draw_box(11.0, 2.8, 4.5, 2.2, "E. Regulatory & Edge Deployment", content_e, colors['E_bg'], colors['E_border'])

    # 6. 总标题
    ax.text(8, 8.8, "Roadmap: Future of Gynecologic OCT-AI Ecosystem", 
            fontsize=18, fontweight='bold', ha='center', color='#212121')

    # 7. 保存文件
    # PDF 是矢量格式，天然高清
    plt.tight_layout()
    plt.savefig('OCT_AI_Roadmap_Academic_Colors.pdf', format='pdf', dpi=300, bbox_inches='tight')
    
    # 也可以保存一份高清PNG用于预览
    plt.savefig('OCT_AI_Roadmap_Academic_Colors.png', format='png', dpi=300, bbox_inches='tight')
    
    print("成功生成: OCT_AI_Roadmap_Academic_Colors.pdf (高清矢量图)")

if __name__ == "__main__":
    draw_academic_roadmap()

