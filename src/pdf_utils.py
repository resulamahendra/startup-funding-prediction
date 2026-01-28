from fpdf import FPDF

def generate_prediction_pdf(inputs, prediction, output_file="prediction_output.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Startup Funding Prediction Output", ln=True, align="C")
    pdf.ln(10)

    # Inputs
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Input Details:", ln=True)
    pdf.ln(3)

    pdf.set_font("Arial", size=11)
    for key, value in inputs.items():
        pdf.cell(0, 8, f"{key}: {value}", ln=True)

    pdf.ln(8)

    # Prediction
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Prediction Result:", ln=True)
    pdf.ln(3)

    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, prediction)

    pdf.output(output_file)
    return output_file
