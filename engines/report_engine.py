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


def generate_pdf_report(output):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    FONT = "Arial"
    TITLE_SIZE = 14
    BODY_SIZE = 11

    pdf.set_font(FONT, size=BODY_SIZE)

    # ----------------------------
    # TEXT HELPERS
    # ----------------------------
    def clean_text(text):
        try:
            return str(text).encode("ascii", errors="ignore").decode()
        except:
            return str(text)

    def wrap(text, limit=90):
        lines = []
        for part in str(text).split("\n"):
            while len(part) > limit:
                lines.append(part[:limit])
                part = part[limit:]
            lines.append(part)
        return lines

    def add_section(title, content):
        pdf.add_page()
        pdf.set_font(FONT, size=TITLE_SIZE)

        pdf.multi_cell(0, 8, clean_text(title))
        pdf.ln(2)

        pdf.set_font(FONT, size=BODY_SIZE)

        # ---------------- dict handling ----------------
        if isinstance(content, dict):
            if not content:
                pdf.multi_cell(0, 6, "No data available.")
                return

            for k, v in content.items():
                if isinstance(v, (dict, list)):
                    line = f"{k}:"
                    for l in wrap(clean_text(line)):
                        pdf.multi_cell(0, 6, l)
                    if isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            sub_line = f"  - {sub_k}: {sub_v}"
                            for l in wrap(clean_text(sub_line)):
                                pdf.multi_cell(0, 6, l)
                    else:
                        for item in v[:50]:
                            item_line = f"  - {item}"
                            for l in wrap(clean_text(item_line)):
                                pdf.multi_cell(0, 6, l)
                else:
                    line = f"{k}: {v}"
                    for l in wrap(clean_text(line)):
                        pdf.multi_cell(0, 6, l)

        # ---------------- list handling ----------------
        elif isinstance(content, list):
            if not content:
                pdf.multi_cell(0, 6, "No data available.")
            else:
                for item in content[:200]:
                    for l in wrap(clean_text(str(item))):
                        pdf.multi_cell(0, 6, l)

        # ---------------- string handling ----------------
        elif isinstance(content, str):
            for l in wrap(clean_text(content)):
                pdf.multi_cell(0, 6, l)

        else:
            pdf.multi_cell(0, 6, clean_text(str(content)))

    def add_metric_block(title, metrics):
        if not metrics:
            return
        add_section(title, metrics)

    # ----------------------------
    # CORE SECTIONS
    # ----------------------------
    add_section("Experiment Summary", output.get("experiment_summary", {}))
    add_section("Problem Discovery", output.get("problem_discovery", {}))
    add_section("Target-Level Evaluation", output.get("predictions", {}))
    add_section("Recommendations", output.get("recommendations", {}))
    add_section("Model Leaderboard", output.get("model_leaderboard", []))
    add_section("Self Critic", output.get("self_critic", {}))
    add_section("Decision Intelligence", output.get("decisions", {}))
    add_section("Risk Scoring", output.get("risk_scoring", {}))
    add_section("Model Monitoring", output.get("model_monitoring", {}))

    # ----------------------------
    # GRAPHS
    # ----------------------------
    graphs = output.get("graphs", [])

    if graphs:
        for i, g in enumerate(graphs, start=1):
            print(f"📊 Rendering graph {i}/{len(graphs)}")

            try:
                plt.figure()

                # If callable graph
                if callable(g):
                    g()
                else:
                    continue

                path = os.path.join(GRAPH_DIR, f"graph_{i}.png")
                plt.savefig(path, bbox_inches="tight")
                plt.close()

                pdf.add_page()
                pdf.image(path, x=10, y=20, w=180)

            except Exception as e:
                print(f"⚠️ Graph {i} failed: {e}")
                plt.close()

    else:
        print("ℹ️ No graphs found.")

    # ----------------------------
    # SAVE PDF
    # ----------------------------
    report_path = os.path.join(OUTPUT_DIR, "report.pdf")
    pdf.output(report_path)

    print(f"📄 Report saved: {report_path}")
    return report_path