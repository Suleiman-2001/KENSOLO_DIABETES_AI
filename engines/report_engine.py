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
                line = f"{k}: {v}"
                for l in wrap(clean_text(line)):
                    pdf.multi_cell(0, 6, l)

        # ---------------- list handling ----------------
        elif isinstance(content, list):
            if not content:
                pdf.multi_cell(0, 6, "No data available.")
            else:
                for item in content[:200]:  # prevent overload
                    for l in wrap(clean_text(item)):
                        pdf.multi_cell(0, 6, l)

        # ---------------- string handling ----------------
        elif isinstance(content, str):
            for l in wrap(clean_text(content)):
                pdf.multi_cell(0, 6, l)

        else:
            pdf.multi_cell(0, 6, clean_text(str(content)))

    # ----------------------------
    # CORE SECTIONS
    # ----------------------------
    add_section("Problem Discovery", output.get("problem_discovery", {}))
    add_section("Predictions", output.get("predictions", {}))
    add_section("Recommendations", output.get("recommendations", {}))
    add_section("Self Critic", output.get("self_critic", {}))
    add_section("Decision Intelligence", output.get("decision_intelligence", {}))

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