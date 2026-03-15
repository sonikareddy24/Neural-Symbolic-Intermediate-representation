import os
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        # Background for header
        self.set_fill_color(30, 58, 138)  # Dark Blue
        self.rect(0, 0, 210, 30, 'F')
        
        # Title
        self.set_y(10)
        self.set_font('helvetica', 'B', 20)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, 'NS-IR Compiler: Learned Cost Modeling', ln=True, align='C')
        
        self.set_font('helvetica', 'I', 12)
        self.cell(0, 8, 'An automated, AI-driven approach to code optimization', ln=True, align='C')
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def section_title(self, title):
        self.set_font('helvetica', 'B', 16)
        self.set_text_color(30, 58, 138)  # Dark Blue
        self.cell(0, 10, title, ln=True)
        # Line below title
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.5)
        self.line(self.get_x(), self.get_y(), 200, self.get_y())
        self.ln(4)

    def section_body(self, text):
        self.set_font('helvetica', '', 11)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 6, text)
        self.ln(6)

    def bullet_point(self, title, text):
        self.set_font('helvetica', 'B', 11)
        self.set_text_color(30, 58, 138)
        self.cell(5, 6, chr(149)) # Bullet character
        self.cell(0, 6, title + ':', ln=True)
        
        self.set_font('helvetica', '', 11)
        self.set_text_color(60, 60, 60)
        self.set_x(15)
        self.multi_cell(0, 6, text)
        self.ln(3)

def create_pdf(filepath):
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)
    
    # --- 1. Introduction ---
    pdf.section_title("1. Introduction & Project Goal")
    pdf.section_body(
        "Modern software demands aggressive, architecture-specific compilers to bridge the gap between "
        "high-level programming languages and complex hardware execution. The primary goal of the NS-IR "
        "(Neural-Symbolic Intermediate Representation) Compiler project is to replace traditional, "
        "human-written compiler heuristics with a fast, data-driven AI cost-model. By accurately predicting "
        "execution speedups directly from program structure, NS-IR automates code optimization and massively "
        "accelerates software development cycles."
    )
    
    # --- 2. The Problem Statement ---
    pdf.section_title("2. The Problem Statement")
    pdf.bullet_point(
        "The Human Bottleneck",
        "Standard compilers like LLVM and GCC rely on immense catalogs of hand-written rules "
        "created by domain experts. These static rules fail to adapt dynamically to the subtleties of new hardware."
    )
    pdf.bullet_point(
        "Current AI Limitations",
        "Many recent auto-tuning solutions (such as AutoTVM) operate by literally compiling and executing "
        "thousands of variations of the code to find the best one. This execution-based searching often takes "
        "hours or days for a single program."
    )
    pdf.bullet_point(
        "The Missing Link",
        "To break this barrier, we require a highly perceptive AI model that can statically read an "
        "Intermediate Representation graph and instantly determine its optimal compilation trajectory without executing it."
    )
    
    # --- 3. Data and Representation ---
    pdf.section_title("3. Dataset & Pre-processing")
    pdf.bullet_point(
        "Scale and Generation",
        "The model is trained on a synthetic, expansive dataset of 50,000 code samples. This scale exposes the "
        "algorithm to a vast topology of program structures to prevent overfitting."
    )
    pdf.bullet_point(
        "Graph Extraction Strategy",
        "Rather than reading fragile text code, the model parses the code into semantic mathematical graphs. "
        "This explicitly maps out loops, mathematical operations, and data-flow dependencies."
    )
    pdf.bullet_point(
        "Vectorization",
        "The extracted graphs and optimization instructions are transformed into deeply embedded floating-point "
        "tensors (arrays), granting the deep learning model native mathematical access to the code context."
    )
    
    # --- 4. System Architecture ---
    pdf.add_page()
    pdf.section_title("4. System Architecture (Key Innovations)")
    pdf.section_body(
        "The upgraded AI framework uses a state-of-the-art Transformer encoder foundation, layered with three "
        "sophisticated modern mechanisms:"
    )
    pdf.bullet_point(
        "Cross-Attention Fusion (Core Novelty)",
        "Older systems blindly combine program structure with optimization steps. Our framework utilizes Cross-Attention, "
        "forcing the AI to calculate exactly which specific lines of code are impacted by particular transformations "
        "(e.g., how loop tiling affects a specific matrix multiplication). This unlocks deep contextual awareness."
    )
    pdf.bullet_point(
        "Rotary Positional Embeddings (RoPE)",
        "We bypassed traditional sequence limitations using RoPE. This allows the AI to correctly interpret the structure "
        "of massive enterprise programs, even if it was exclusively trained on small kernel snippets."
    )
    pdf.bullet_point(
        "Monte Carlo Uncertainty Modeling",
        "Using Monte Carlo Dropout, the network assesses its own predictive risk. It outputs both the expected "
        "speedup and a confidence interval. The compiler avoids applying extreme optimizations if the AI is uncertain."
    )

    # --- 5. Model Training ---
    pdf.section_title("5. Model Training and Dynamics")
    pdf.bullet_point(
        "Contrastive Clustering (NT-Xent)",
        "To amplify learning, the model groups programs with similar performance characteristics closer "
        "together in its internal 'brain' using a specialized contrastive loss approach, making it exceptionally "
        "accurate at recognizing optimization patterns."
    )
    pdf.bullet_point(
        "Cyclic Optimization",
        "During its 150-epoch training span across 5.8 million parameters, the network uses 'Cosine Annealing' "
        "to occasionally spike its own learning rate. This prevents the training cycle from stagnating in dead-ends, "
        "culminating in 'Stochastic Weight Averaging' for maximum stability."
    )
    
    # --- 6. Final Results ---
    pdf.section_title("6. Evaluation and Final Results")
    pdf.section_body(
        "The refined architecture delivered transformative capabilities when validated across robust benchmarking tests."
    )
    pdf.bullet_point(
        "Exceptional Accuracy",
        "The model reached an incredibly low Mean Absolute Percentage Error (MAPE) of just 2.28%. This represents "
        "a dramatic paradigm shift from earlier models which plateaued around 30% error rates."
    )
    pdf.bullet_point(
        "Sub-millisecond Latency",
        "Despite containing millions of parameters, the model computes optimal execution forecasts in under "
        "5 milliseconds, making it legitimately fast enough to integrate into Just-In-Time (JIT) compilation pipelines."
    )
    
    # --- 7. Conclusion ---
    pdf.section_title("7. Conclusion & Future Scope")
    pdf.section_body(
        "The NS-IR Neural Compiler framework successfully validates that AI Transformers can replace hand-crafted "
        "optimizations by deeply understanding code structure. The 2.28% validation accuracy effectively solves the "
        "latency issue associated with trial-and-error autotuners, ushering in the era of instant, AI-driven code acceleration.\n\n"
        "Going forward, this framework is slated to be integrated directly into physical ARM and x86 hardware backends for "
        "real-world, multi-task deployment (simultaneously managing execution speed, memory bandwidth, and power consumption constraints)."
    )

    pdf.output(filepath)

if __name__ == '__main__':
    destination = os.path.expanduser("~/Desktop/NS_IR_Extended_Project_Summary.pdf")
    create_pdf(destination)
    print(f"Extended summary PDF successfully generated at {destination}")
