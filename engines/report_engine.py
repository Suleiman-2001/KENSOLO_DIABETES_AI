import os
from fpdf import FPDF

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
        for line in wrap_text(safe_text(title)):
            pdf.multi_cell(pdf.w - 20, 8, line)
        pdf.ln(4)

        pdf.set_font(FONT_NAME, "", FONT_SIZE_BODY)
        if isinstance(content, dict) and content:
            for k, v in content.items():
                for piece in wrap_text(safe_text(f"{k}: {v}")):
                    pdf.multi_cell(pdf.w - 20, 6, piece)
        elif isinstance(content, str) and content.strip():
            for piece in wrap_text(safe_text(content)):
                pdf.multi_cell(pdf.w - 20, 6, piece)
        else:
            pdf.multi_cell(pdf.w - 20, 6, "No data available.")

    # ----------------------------
    # Add main sections
    # ----------------------------
    add_section("Problem Discovery", output.get("problem_discovery", {}))
    add_section("Predictions", output.get("predictions", {}))
    add_section("Recommendations", output.get("recommendations", {}))
    add_section("Self Critic", output.get("self_critic", {}))

    # ----------------------------
    # Add graphs from saved PNGs
    # ----------------------------
    graph_paths = output.get("graphs", [])
    if graph_paths:
        for i, path in enumerate(graph_paths, start=1):
            if os.path.exists(path):
                pdf.add_page()
                pdf.image(path, x=10, y=20, w=pdf.w - 20)
                print(f"📊 Added graph {i}: {path}")
            else:
                print(f"⚠️ Graph file not found: {path}")
        print(f"✅ All graphs embedded into PDF.")
    else:
        print("ℹ️ No graphs to embed in PDF.")

    # ----------------------------
    # Save PDF
    # ----------------------------
    report_path = os.path.join(OUTPUT_DIR, "report.pdf")
    pdf.output(report_path)
    print(f"📄 PDF report saved at: {report_path}")

    return report_path
