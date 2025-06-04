from pix2tex.cli import LatexOCR
from PIL import Image
import re
def clean_latex(latex):
    # 주요 스타일 제거
    latex = re.sub(r'\\(mathbf|mathbb|mathfrak|mathsf|textbf|scriptstyle|displaystyle)\{(.*?)\}', r'\2', latex)
    return latex


def classify_formula(latex):
    latex = clean_latex(latex)

    if any(op in latex for op in ['\\le', '\\ge', '<', '>']):
        return '부등식(Inequality)'
    elif '\\in' in latex or '\\cup' in latex or '\\cap' in latex or '\\forall' in latex or '\\exists' in latex:
        return '집합(Set Theory)'
    elif '\\vec' in latex or '\\mathbf' in latex:
        return '벡터(Vector Expression)'
    elif '\\sqrt' in latex:
        return '루트(Square Root)'
    elif re.search(r'\^[{]?[a-zA-Z0-9]+[}]?', latex):  # x^2, a^{n+1}
        return '지수(Exponent)'
    elif '+' in latex or '-' in latex:
        return '덧셈/뺄셈(Addition/Subtraction)'
    elif '*' in latex or '\\cdot' in latex or '\\times' in latex:
        return '곱셈(Multiplication)'
    elif '\\frac' in latex or '/' in latex:
        return '분수(Fraction)'
    elif '\\sum' in latex or '\\prod' in latex:
        return '합/곱(Summation/Product)'
    elif '\\int' in latex:
        return '적분(Integral)'
    elif '\\lim' in latex:
        return '극한(Limit)'
    elif re.search(r'd[xy]', latex) or '\\partial' in latex:
        return '미분(Differentiation)'
    elif '\\tan' in latex or '\\sin' in latex or '\\cos' in latex:
        return '삼각함수(Trigonometric Function)'
    elif '\\begin{matrix}' in latex or '\\begin{bmatrix}' in latex:
        return '행렬(Matrix)'
    elif '=' in latex:
        return '방정식(Equation)'
    elif re.search(r'[a-zA-Z]\^(\d+|[a-zA-Z])', latex):
        return '다항식(Polynomial)'
    elif '\\log' in latex or '\\ln' in latex or 'e^' in latex:
        return '로그/지수함수(Logarithmic/Exponential)'
    elif re.search(r'\|.*\|', latex):
        return '절댓값(Absolute Value)'
    elif '\\begin{cases}' in latex:
        return '케이스 함수(Piecewise Function)'
    elif '\\binom' in latex:
        return '조합/이항계수(Combinatorics)'
    elif '\\equiv' in latex or '\\mod' in latex or '\\pmod' in latex:
        return '합동/정수론(Congruence/Number Theory)'
    elif latex.strip() == '' or re.fullmatch(r'[^\w\s]+', latex):
        return '미정의(Undefined)'


# 모델 초기화
model = LatexOCR()
img = Image.open("images/0007.jpg")
latex = model(img)

# 분류
category = classify_formula(latex)
print(f"LaTeX: {latex}")
print(f"Category: {category}")
