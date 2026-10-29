# generate_pdf_report.py
"""
Script to generate reports/Activity_Report.pdf using ReportLab.
Produces an academic laboratory report for Machine Learning CIE 2.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and display 'Page X of Y'."""
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "Machine Learning CIE 2 — Activity 1: Dataset Preparation & EDA")
            self.setStrokeColor(colors.HexColor("#cccccc"))
            self.setLineWidth(0.5)
            self.line(54, 750, 558, 750)

        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_text)
        self.drawString(54, 36, "Academic Reference: https://github.com/Princy2129 | Course: Machine Learning (Unit I)")
        self.setStrokeColor(colors.HexColor("#cccccc"))
        self.setLineWidth(0.5)
        self.line(54, 46, 558, 46)
        self.restoreState()


def build_pdf():
    pdf_path = os.path.join("reports", "Activity_Report.pdf")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1a365d'),
        alignment=1, # Center
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#2b6cb0'),
        alignment=1,
        spaceAfter=12
    )

    meta_style = ParagraphStyle(
        'MetaBox',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#333333')
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1a365d'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#2b6cb0'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor('#222222'),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#222222'),
        leftIndent=14,
        spaceAfter=3
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#1a365d')
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#222222')
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("Machine Learning CIE 2 — Activity Report", title_style))
    story.append(Paragraph("Activity 1: Dataset Preparation Activity — Unit I: Foundations & Data Preparation", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1a365d'), spaceBefore=2, spaceAfter=8))

    meta_data = [
        [
            Paragraph("<b>Academic Level:</b> Second Year Engineering (B.E./B.Tech)", meta_style),
            Paragraph("<b>Course:</b> Machine Learning (CIE 2)", meta_style)
        ],
        [
            Paragraph("<b>Target Dataset:</b> Titanic Passenger Manifest (Seaborn)", meta_style),
            Paragraph("<b>Prepared Output:</b> <code>prepared_titanic.csv</code> (775 × 13)", meta_style)
        ],
        [
            Paragraph("<b>Faculty Reference:</b> <a href='https://github.com/Princy2129'>https://github.com/Princy2129</a>", meta_style),
            Paragraph("<b>Status:</b> Completed & Verified Reproducible", meta_style)
        ]
    ]
    t_meta = Table(meta_data, colWidths=[250, 254])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f0f4f8')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))

    # Section 1: Executive Summary & Objective
    story.append(Paragraph("1. Executive Summary & Objective", h1_style))
    story.append(Paragraph(
        "Real-world tabular datasets rarely present themselves ready for algorithmic ingestion. They consistently suffer "
        "from non-random missingness, duplicate observations, unstandardised text formats, redundant multi-collinear attributes, "
        "and unscaled continuous values. The goal of this activity is to build an end-to-end, scientifically defensible, "
        "and fully reproducible data preparation pipeline on the canonical Titanic passenger dataset without training any predictive model.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Key Deliverables:</b> An executed Jupyter Notebook (<code>dataset_preparation.ipynb</code>), an exported clean CSV "
        "(<code>prepared_titanic.csv</code>), full GitHub documentation (<code>README.md</code>), exact dependencies (<code>requirements.txt</code>), "
        "and this formal academic report.",
        body_style
    ))

    # Section 2: Dataset Profile & Schema
    story.append(Paragraph("2. Dataset Selection & Architectural Profile", h1_style))
    story.append(Paragraph(
        "The public dataset was sourced directly from the verified Seaborn repository: "
        "<code>https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv</code>. "
        "The raw manifest contains exactly <b>891 passenger records</b> across <b>15 features</b>.",
        body_style
    ))

    schema_table_data = [
        [Paragraph("Feature", table_header), Paragraph("Type", table_header), Paragraph("Role / Nature", table_header), Paragraph("Description & Engineering Treatment", table_header)],
        [Paragraph("survived", table_cell), Paragraph("int64", table_cell), Paragraph("Future Target", table_cell), Paragraph("0 = No, 1 = Yes (Binary ground truth; unscaled)", table_cell)],
        [Paragraph("pclass", table_cell), Paragraph("int64", table_cell), Paragraph("Ordinal Input", table_cell), Paragraph("Ticket class: 1st > 2nd > 3rd; preserved as ordinal integer", table_cell)],
        [Paragraph("sex", table_cell), Paragraph("object", table_cell), Paragraph("Nominal Input", table_cell), Paragraph("Biological sex; one-hot encoded to sex_male (0/1)", table_cell)],
        [Paragraph("age", table_cell), Paragraph("float64", table_cell), Paragraph("Continuous Input", table_cell), Paragraph("177 missing (19.87%); median imputed (28.0) & StandardScaled", table_cell)],
        [Paragraph("sibsp / parch", table_cell), Paragraph("int64", table_cell), Paragraph("Discrete Input", table_cell), Paragraph("Sibling/spouse and parent/child counts; preserved as integers", table_cell)],
        [Paragraph("fare", table_cell), Paragraph("float64", table_cell), Paragraph("Continuous Input", table_cell), Paragraph("Ticket price (£0 to £512.33); continuous StandardScaled", table_cell)],
        [Paragraph("embarked", table_cell), Paragraph("object", table_cell), Paragraph("Nominal Input", table_cell), Paragraph("2 missing; mode imputed ('S') & OHE into embarked_Q, S", table_cell)],
        [Paragraph("deck", table_cell), Paragraph("object", table_cell), Paragraph("Sparse / Drop", table_cell), Paragraph("688 missing (77.22%); dropped due to excessive missingness", table_cell)],
        [Paragraph("alive / class / embark_town", table_cell), Paragraph("object", table_cell), Paragraph("Redundant", table_cell), Paragraph("Exact string duplicates of survived, pclass, and embarked; dropped", table_cell)],
        [Paragraph("adult_male / alone", table_cell), Paragraph("bool", table_cell), Paragraph("Binary Input", table_cell), Paragraph("Boolean indicators converted to clean 0/1 integers", table_cell)]
    ]
    t_schema = Table(schema_table_data, colWidths=[90, 45, 85, 284])
    t_schema.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a365d')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_schema)

    # Section 3: Missing-Value & Data Quality Remediation
    story.append(Paragraph("3. Missing-Value Remediation & Data Quality Auditing", h1_style))
    story.append(Paragraph(
        "A complete audit revealed 867 total missing cells across four features. Blanket listwise deletion was rejected "
        "because discarding rows with any missing value would destroy 77.5% of the data. Instead, column-specific strategies were applied:",
        body_style
    ))
    story.append(Paragraph("• <b>Exclusion of <code>deck</code> (77.22% missing):</b> Attempting to impute cabin deck for 688 of 891 records would inject massive artificial bias. Dropping this feature preserves empirical fidelity.", bullet_style))
    story.append(Paragraph("• <b>Median Imputation for <code>age</code> (19.87% missing):</b> The age distribution is right-skewed with extreme values (infants to 80 years). The median (28.0 years) is resilient to outliers, whereas the mean (29.7 years) is distorted.", bullet_style))
    story.append(Paragraph("• <b>Mode Imputation for <code>embarked</code> / <code>embark_town</code> (0.22% missing):</b> Only two records were null. Southampton ('S') accounts for 72.4% of all boardings and was applied as the mode.", bullet_style))
    story.append(Paragraph("• <b>Deduplication:</b> 116 exact duplicate rows (identical passenger profiles) were removed to avoid inflating specific cohorts and preventing train-test evaluation bias, leaving 775 clean unique observations.", bullet_style))

    # Section 4: Exploratory Data Analysis & Visualizations
    story.append(PageBreak())
    story.append(Paragraph("4. Exploratory Data Analysis (EDA) & Visualizations", h1_style))
    story.append(Paragraph(
        "Three focused, publication-grade visualisations were generated using Matplotlib and Seaborn, capturing univariate and bivariate interactions:",
        body_style
    ))

    # Figure 1: Age
    fig_w = 400
    fig_h = 160
    if os.path.exists("reports/figures/age_distribution.png"):
        story.append(Image("reports/figures/age_distribution.png", width=fig_w, height=fig_h))
        story.append(Spacer(1, 2))
        story.append(Paragraph("<b>Figure 1:</b> Distribution of passenger ages in the cleaned dataset with KDE and central tendencies.", callout_style))
        story.append(Spacer(1, 6))

    # Figure 2: Survival
    if os.path.exists("reports/figures/survival_counts.png"):
        story.append(Image("reports/figures/survival_counts.png", width=fig_w, height=fig_h))
        story.append(Spacer(1, 2))
        story.append(Paragraph("<b>Figure 2:</b> Passenger survival counts showing 455 non-survivors (58.7%) vs. 320 survivors (41.3%).", callout_style))
        story.append(Spacer(1, 6))

    # Figure 3: Scatter
    story.append(PageBreak())
    if os.path.exists("reports/figures/age_vs_fare.png"):
        story.append(Image("reports/figures/age_vs_fare.png", width=fig_w, height=185))
        story.append(Spacer(1, 2))
        story.append(Paragraph("<b>Figure 3:</b> Passenger age vs. ticket fare stratified by survival status.", callout_style))
        story.append(Spacer(1, 8))

    story.append(Paragraph("Verified Empirical Observations from EDA:", h2_style))
    story.append(Paragraph("1. <b>Demographic Distribution:</b> Passenger ages range continuously from 0.42 to 80.0 years. The primary concentration lies in the 20–35 year range, with a secondary peak among young children under 8 years.", bullet_style))
    story.append(Paragraph("2. <b>Class Imbalance:</b> Non-survivors (455; 58.7%) outnumber survivors (320; 41.3%) by 135 individuals, establishing moderate baseline class imbalance for future predictive classification.", bullet_style))
    story.append(Paragraph("3. <b>Fare Stratification:</b> Over 75% of passengers paid under £35.00. High-fare tickets (>£100 up to £512.33) are almost exclusively populated by surviving passengers, indicating socioeconomic travel class strongly correlated with evacuation outcome.", bullet_style))

    # Section 5: Feature Engineering, Encoding & Scaling
    story.append(Paragraph("5. Feature Engineering, Categorical Encoding & Numerical Scaling", h1_style))
    story.append(Paragraph(
        "<b>Categorical Encoding:</b> Redundant string columns (<code>alive</code>, <code>class</code>, <code>embark_town</code>) were dropped to avoid perfect multicollinearity. "
        "Binary flags (<code>adult_male</code>, <code>alone</code>) were cast to integer 0/1. Nominal categorical variables (<code>sex</code>, <code>embarked</code>, <code>who</code>) "
        "were transformed using One-Hot Encoding with <code>drop_first=True</code>, dropping reference categories (female, Cherbourg, child) to eliminate the dummy variable trap.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Numerical Scaling:</b> Features <code>age</code> (range: 0.42–80 yrs) and <code>fare</code> (range: £0–£512.33) possess vastly disparate magnitudes. "
        "Distance-based algorithms (KNN, SVM) and gradient-based models (Logistic Regression, Neural Networks) are biased when unscaled features dominate distance calculations. "
        "Using <code>StandardScaler</code>, <code>age</code> and <code>fare</code> were standardized to mean 0.0 and standard deviation 1.0. "
        "Target <code>survived</code> and discrete/binary features were strictly excluded from scaling to preserve interpretability.",
        body_style
    ))

    # Section 6: Validation & Export
    story.append(Paragraph("6. Final Validation & Evaluation Criteria Mapping", h1_style))
    
    val_table_data = [
        [Paragraph("Metric / Stage", table_header), Paragraph("Raw Initial", table_header), Paragraph("Prepared Final", table_header), Paragraph("Status / Verification", table_header)],
        [Paragraph("Row Count", table_cell), Paragraph("891", table_cell), Paragraph("775", table_cell), Paragraph("116 duplicates removed; zero data corruption", table_cell)],
        [Paragraph("Column Count", table_cell), Paragraph("15", table_cell), Paragraph("13", table_cell), Paragraph("3 redundant dropped, 1 sparse dropped, OHE added", table_cell)],
        [Paragraph("Missing Cells", table_cell), Paragraph("867", table_cell), Paragraph("0", table_cell), Paragraph("Verified via assert df_scaled.isnull().sum().sum() == 0", table_cell)],
        [Paragraph("Duplicate Rows", table_cell), Paragraph("107 (raw)", table_cell), Paragraph("0", table_cell), Paragraph("Verified via assert df_scaled.duplicated().sum() == 0", table_cell)],
        [Paragraph("Age Scaled Mean / Std", table_cell), Paragraph("29.70 / 14.53", table_cell), Paragraph("0.00 / 1.00", table_cell), Paragraph("Zero-mean, unit-variance confirmed", table_cell)],
        [Paragraph("Fare Scaled Mean / Std", table_cell), Paragraph("32.20 / 49.69", table_cell), Paragraph("0.00 / 1.00", table_cell), Paragraph("Zero-mean, unit-variance confirmed", table_cell)],
    ]
    t_val = Table(val_table_data, colWidths=[130, 80, 80, 214])
    t_val.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a365d')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_val)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Mapping to Five Academic Evaluation Criteria:", h2_style))
    criteria_data = [
        [Paragraph("Evaluation Criterion", table_header), Paragraph("Weight", table_header), Paragraph("Implementation Evidence in Project", table_header)],
        [Paragraph("1. Dataset Selection & Understanding", table_cell), Paragraph("1 Mark", table_cell), Paragraph("Authoritative Seaborn URL ingested; shape (891, 15) documented; column semantics, types, and target identified.", table_cell)],
        [Paragraph("2. Missing-Value & Quality Handling", table_cell), Paragraph("1 Mark", table_cell), Paragraph("Audit table computed; deck dropped (77.2% null); age median-imputed; embarked mode-imputed; 116 duplicates purged.", table_cell)],
        [Paragraph("3. EDA & Visualisations", table_cell), Paragraph("1 Mark", table_cell), Paragraph("Three readable plots (Histogram, Bar chart, Scatter plot) generated with Matplotlib/Seaborn; empirical observations documented.", table_cell)],
        [Paragraph("4. Encoding & Appropriate Scaling", table_cell), Paragraph("1 Mark", table_cell), Paragraph("Redundancies eliminated; OHE with drop_first applied to nominals; StandardScaler applied to age & fare; target excluded.", table_cell)],
        [Paragraph("5. Prepared Dataset & Conclusion", table_cell), Paragraph("1 Mark", table_cell), Paragraph("Exported to prepared_titanic.csv (775 × 13); verified via reload & assertions; concise student-friendly conclusion provided.", table_cell)]
    ]
    t_crit = Table(criteria_data, colWidths=[150, 55, 299])
    t_crit.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2b6cb0')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_crit)
    story.append(Spacer(1, 8))

    # Section 7: Conclusion & Acknowledgement
    story.append(Paragraph("7. Academic Acknowledgement & Conclusion", h1_style))
    story.append(Paragraph(
        "This project was completed as part of the Machine Learning CIE 2 activity. Faculty reference and Machine Learning learning resources: "
        "<a href='https://github.com/Princy2129'>https://github.com/Princy2129</a>",
        body_style
    ))
    story.append(Paragraph(
        "<b>Conclusion:</b> All requirements for Unit I: Foundations & Data Preparation have been fully met. "
        "The resulting prepared dataset (775 rows, 13 features, 0 nulls, 0 duplicates) provides a standardized, mathematically robust foundation for future model development. "
        "In strict adherence to project scope, no machine learning prediction model was trained.",
        body_style
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Academic Report PDF successfully compiled to: '{pdf_path}'")

if __name__ == "__main__":
    build_pdf()
