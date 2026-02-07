import os
from fpdf import FPDF
import matplotlib.pyplot as plt

# ----------------------------
# Directories
# ----------------------------
OUTPUT_DIR = "outputs"
GRAPH_DIR = os.path.join(OUTPUT_DIR, "graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# PDF Generation
# ----------------------------
def generate_pdf_report(output):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ----------------------------
    # Safe font
    # ----------------------------
    FONT_NAME = "Arial"
    FONT_SIZE_TITLE = 14
    FONT_SIZE_BODY = 12
    pdf.set_font(FONT_NAME, "", FONT_SIZE_BODY)

    # ----------------------------
    # Helpers
    # ----------------------------
    def wrap_text(text, max_chars_per_line=80):
        if not text:
            return [""]
        lines = []
        for paragraph in str(text).split("\n"):
            while paragraph:
                if len(paragraph) <= max_chars_per_line:
                    lines.append(paragraph)
                    break
                else:
                    lines.append(paragraph[:max_chars_per_line])
                    paragraph = paragraph[max_chars_per_line:]
        return lines

    def safe_text(text):
        try:
            return str(text).encode("ascii", errors="ignore").decode()
        except Exception:
            return str(text)

    def add_section(title, content):
        pdf.add_page()
        pdf.set_font(FONT_NAME, "", FONT_SIZE_TITLE)
        title_safe = safe_text(title)
        for line in wrap_text(title_safe):
            pdf.multi_cell(pdf.w - 20, 8, line)
        pdf.ln(4)

        pdf.set_font(FONT_NAME, "", FONT_SIZE_BODY)
        if isinstance(content, dict) and content:
            for k, v in content.items():
                line_safe = safe_text(f"{k}: {v}")
                for piece in wrap_text(line_safe):
                    pdf.multi_cell(pdf.w - 20, 6, piece)
        elif isinstance(content, str) and content.strip():
            line_safe = safe_text(content)
            for piece in wrap_text(line_safe):
                pdf.multi_cell(pdf.w - 20, 6, piece)
        elif content is None or (isinstance(content, (list, dict)) and len(content) == 0):
            pdf.multi_cell(pdf.w - 20, 6, "No data available.")
        else:
            pdf.multi_cell(pdf.w - 20, 6, safe_text(content))

    # ----------------------------
    # Add main sections
    # ----------------------------
    add_section("Problem Discovery", output.get("problem_discovery", {}))
    add_section("Predictions", output.get("predictions", {}))
    add_section("Recommendations", output.get("recommendations", {}))
    add_section("Self Critic", output.get("self_critic", {}))

    # -------------------------------------------------
    # ✅ NEW SECTION — Decision Intelligence
    # -------------------------------------------------
    add_section(
        "Decision Intelligence",
        output.get("decision_intelligence", {})
    )

    # ----------------------------
    # Add graphs with progress
    # ----------------------------
    graphs = output.get("graphs", [])
    total_graphs = len(graphs)

    if graphs:
        for i, graph_func in enumerate(graphs, start=1):
            print(f"📊 Generating graph {i}/{total_graphs}...")
            plt.figure()
            try:
                graph_func()
            except Exception as e:
                print(f"⚠️ Error creating graph {i}: {e}")
                plt.close()
                continue

            # Save graph to PNG
            graph_path = os.path.join(GRAPH_DIR, f"graph_{i}.png")
            plt.savefig(graph_path, bbox_inches="tight")
            plt.close()

            # Add graph to PDF
            pdf.add_page()
            try:
                pdf.image(graph_path, x=10, y=20, w=pdf.w - 20)
            except Exception as e:
                print(f"⚠️ Could not add graph {i} to PDF: {e}")

        print(f"✅ Graphs saved in folder: {GRAPH_DIR}")
    else:
        print("ℹ️ No graphs to add.")

    # ----------------------------
    # Save PDF
    # ----------------------------
    report_path = os.path.join(OUTPUT_DIR, "report.pdf")
    pdf.output(report_path)
    print(f"📄 PDF report saved at: {report_path}")

    return report_path
