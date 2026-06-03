import os
import subprocess
import sys

def install_and_import_reportlab():
    """Ensure reportlab is installed so we can generate PDF binaries programmatically."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    except ImportError:
        print("Installing reportlab for PDF generation...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
        
install_and_import_reportlab()

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def create_leave_policy(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1e1b4b'),
        spaceAfter=15
    )
    heading_style = ParagraphStyle(
        'DocSection',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#4338ca'),
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor('#1f2937'),
        spaceAfter=10
    )
    
    story = []
    
    # Title
    story.append(Paragraph("Acme Corp - Employee Leave Policy", title_style))
    story.append(Spacer(1, 10))
    
    # Introduction
    story.append(Paragraph(
        "This document details the leave allowances, accrual rules, and approval workflows "
        "for all full-time employees at Acme Corp. Please read this policy carefully. Any "
        "exceptions require written approval from the Chief Human Resources Officer.",
        body_style
    ))
    story.append(Spacer(1, 10))
    
    # Section 1: Annual Leave
    story.append(Paragraph("1. Annual Paid Leave Allowance", heading_style))
    story.append(Paragraph(
        "All permanent full-time employees are entitled to 25 days of paid annual leave per calendar year. "
        "Leave accrues monthly at a rate of 2.08 days per full month of employment. Employees can roll over "
        "up to 5 unused annual leave days into the next calendar year. Any additional accrued, unused leave "
        "beyond 5 days will expire on December 31st of each year. Newly hired employees are eligible to take "
        "paid leave after completing their 3-month probation period.",
        body_style
    ))
    
    # Section 2: Sick Leave
    story.append(Paragraph("2. Sick Leave Policy", heading_style))
    story.append(Paragraph(
        "Acme Corp provides 10 days of fully paid sick leave per calendar year. Sick leave is designed for "
        "personal illness, medical appointments, or caring for immediate family members who are unwell. "
        "Employees must notify their direct manager by 9:00 AM on the day of absence. For any sick leave "
        "exceeding 3 consecutive working days, employees must submit a valid medical certificate or doctor's "
        "note to HR upon returning to work. Unused sick leave cannot be rolled over to the next year and "
        "will not be paid out upon resignation or termination.",
        body_style
    ))
    
    # Section 3: Parental Leave
    story.append(Paragraph("3. Parental Leave", heading_style))
    story.append(Paragraph(
        "To support new parents, Acme Corp offers comprehensive parental leave policies:\n"
        "• Maternity Leave: Eligible birthing parents receive 16 weeks of fully paid maternity leave. This "
        "leave can start up to 4 weeks before the expected date of delivery.\n"
        "• Paternity Leave: Non-birthing parents receive 4 weeks of fully paid paternity leave, which must be "
        "taken within the first 12 months of the child's birth or adoption.\n"
        "All parental leave requests must be submitted to HR at least 60 days before the planned start date.",
        body_style
    ))
    
    # Section 4: Public Holidays
    story.append(Paragraph("4. Corporate Holidays", heading_style))
    story.append(Paragraph(
        "Acme Corp observes 11 standard public holidays, including New Year's Day, Memorial Day, "
        "Independence Day, Labor Day, Thanksgiving Day (and the day after), and Christmas Day. A full "
        "schedule of holidays is published on the intranet on December 1st for the upcoming year.",
        body_style
    ))
    
    doc.build(story)
    print(f"Created Leave Policy PDF at {output_path}")

def create_employee_handbook(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1e1b4b'),
        spaceAfter=15
    )
    heading_style = ParagraphStyle(
        'DocSection',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#4338ca'),
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor('#1f2937'),
        spaceAfter=10
    )
    
    story = []
    
    # Title
    story.append(Paragraph("Acme Corp - Employee Handbook", title_style))
    story.append(Spacer(1, 10))
    
    # Introduction
    story.append(Paragraph(
        "Welcome to Acme Corp! This handbook contains general guidelines on our work environment, dress code, "
        "remote work policies, and employee benefits. We are committed to fostering a workplace of innovation, "
        "inclusivity, and professional growth.",
        body_style
    ))
    story.append(Spacer(1, 10))
    
    # Code of Conduct
    story.append(Paragraph("1. Code of Conduct and Workplace Professionalism", heading_style))
    story.append(Paragraph(
        "All employees are expected to maintain the highest standards of professional conduct. Acme Corp maintains "
        "a zero-tolerance policy for harassment, discrimination, or bullying of any kind. Employees must protect "
        "the confidentiality of Acme's proprietary information and customer data. Personal social media usage must "
        "not conflict with work responsibilities or represent Acme Corp without authorization.",
        body_style
    ))
    
    # Remote Work
    story.append(Paragraph("2. Flexible and Remote Work Guidelines", heading_style))
    story.append(Paragraph(
        "Acme Corp supports flexible working arrangements. All office-based staff are eligible for a hybrid work "
        "schedule, allowing up to 2 days of remote work per week. Remote work days must be coordinated with and "
        "approved by your direct manager to ensure team coverage. Core collaboration hours are 10:00 AM to 4:00 PM "
        "EST, during which all employees (remote or in-office) should be online and responsive on Slack and email. "
        "Acme will provide a stipend of $500 for home office equipment setup upon joining.",
        body_style
    ))
    
    # Dress Code
    story.append(Paragraph("3. Dress Code Standards", heading_style))
    story.append(Paragraph(
        "Our standard dress code is business casual. This includes collared shirts, blouses, slacks, chinos, and "
        "closed-toe professional shoes. Fridays are designated as Casual Friday, where employees are welcome to wear "
        "jeans (without rips) and clean, casual t-shirts. On days when clients are visiting the office, employees "
        "meeting them should dress in standard business attire.",
        body_style
    ))
    
    # Health Benefits
    story.append(Paragraph("4. Healthcare and Dental Benefits", heading_style))
    story.append(Paragraph(
        "We care about your health. Acme Corp offers comprehensive healthcare insurance through Blue Cross Blue Shield. "
        "The plan covers preventative care at 100%, and prescription drug co-pays at $15 for generic and $35 for brand names. "
        "Dental and vision insurance is fully covered by Acme Corp, providing checkups twice a year and $2,000 annual "
        "orthodontia coverage per employee. Coverage begins on the first day of the month following your start date.",
        body_style
    ))
    
    doc.build(story)
    print(f"Created Employee Handbook PDF at {output_path}")

if __name__ == "__main__":
    # Define document folder
    scratch_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(os.path.dirname(scratch_dir), "sample_docs")
    os.makedirs(docs_dir, exist_ok=True)
    
    create_leave_policy(os.path.join(docs_dir, "acme_leave_policy.pdf"))
    create_employee_handbook(os.path.join(docs_dir, "acme_employee_handbook.pdf"))
