"""
md_to_notebook.py — 강의노트 .md 파일을 Colab .ipynb 로 변환.
- ![...](images/xxx.png) 마크다운 이미지 마커를 만나면:
  - IMAGE_MAP 에 등록된 파일이면 -> 해당 인터랙티브 함수를 호출하는 코드 셀로 교체
  - 등록되지 않았으면 -> 이미지 마크다운을 그대로 유지한 마크다운 셀 (정적 이미지)
- 그 외 모든 텍스트는 마크다운 셀로 그대로 보존한다 (H2 헤딩 기준으로 셀 분할).
"""
import re
import os
import nbformat as nbf

REPO = "/home/claude/work/repo"
RAW_BASE = "https://raw.githubusercontent.com/gabraxas/LVs-and-Policy/main/lecture/_shared/course_interactive.py"

SETUP_CODE = f'''# ▶ 실행 전 준비 — 이 셀을 먼저 실행하세요 (약 10~20초 소요)
!pip install -q ipywidgets
!apt-get -qq install -y fonts-nanum > /dev/null 2>&1

import urllib.request
urllib.request.urlretrieve(
    "{RAW_BASE}",
    "course_interactive.py")

from course_interactive import *
setup_korean_font()
print("준비 완료 — 아래 셀들을 순서대로 실행하며 강의를 진행하세요.")
'''

IMAGE_MAP = {
    # week02
    "02-orbital-velocity-vs-altitude.png": "orbital_velocity_explorer()",
    "03-hohmann-transfer.png": "hohmann_transfer_explorer()",
    "05-delta-v-budget.png": "delta_v_budget_calculator()",
    "07-rocket-equation-tyranny.png": "rocket_equation_explorer()",
    "08-propellant-density-isp.png": "propellant_isp_explorer()",
    "09-staging-diminishing-returns.png": "staging_calculator()",
    # week03
    "02-nozzle-expansion-regimes.png": "thrust_nozzle_explorer()",
    "03-propellant-isp-comparison.png": "propellant_isp_explorer()",
    "06-density-isp-tradeoff.png": "propellant_isp_explorer()  # 위와 동일 위젯 재사용 — 밀도비추력까지 함께 표시됨",
    "08-cycle-performance-vs-risk.png": "engine_cycle_comparison()",
    # week04
    "01-max-q-throttle-bucket.png": "max_q_explorer()",
    "02-ascent-trajectory-phases.png": "ascent_trajectory_explorer()",
}

IMG_RE = re.compile(r'!\[[^\]]*\]\(images/([^)]+)\)')
HEADING_RE = re.compile(r'^(##+) ', re.MULTILINE)


def split_by_heading(text):
    """H2/H3 헤딩 기준으로 텍스트를 청크로 분할 (각 청크는 헤딩 포함 다음 헤딩 전까지)."""
    idxs = [m.start() for m in HEADING_RE.finditer(text)]
    if not idxs or idxs[0] != 0:
        idxs = [0] + idxs
    idxs.append(len(text))
    chunks = []
    for i in range(len(idxs) - 1):
        chunk = text[idxs[i]:idxs[i + 1]].strip("\n")
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def split_markdown(text):
    """텍스트를 이미지 마커 기준으로 (markdown, image_filename) 조각들로 분할."""
    parts = []
    last = 0
    for m in IMG_RE.finditer(text):
        before = text[last:m.start()].strip("\n")
        if before.strip():
            parts.append(("md", before))
        parts.append(("img", m.group(1)))
        last = m.end()
    tail = text[last:].strip("\n")
    if tail.strip():
        parts.append(("md", tail))
    return parts


def build_notebook(week_num, md_path, out_path, heading_widgets=None):
    """
    heading_widgets: list of (heading_substring, label, call_code) — 해당 부분 문자열을 포함하는
    헤딩 청크 뒤에 인터랙티브 위젯 셀을 삽입한다 (이미지가 없는 주차용).
    """
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    nb = nbf.v4.new_notebook()
    cells = []

    title_line = text.split("\n", 1)[0].lstrip("# ").strip()
    badge = (f"# {title_line}\n\n"
             f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
             f"(https://colab.research.google.com/github/gabraxas/LVs-and-Policy/blob/main/"
             f"lecture/week{week_num:02d}/week{week_num:02d}.ipynb)\n\n"
             f"> 이 노트북은 GitHub에 저장되고 Colab에서 실행됩니다. 위 배지를 눌러 Colab에서 열거나, "
             f"아래 첫 코드 셀부터 순서대로 실행하세요.")
    cells.append(nbf.v4.new_markdown_cell(badge))
    cells.append(nbf.v4.new_code_cell(SETUP_CODE))

    heading_widgets = heading_widgets or []
    used_widgets = set()

    for chunk in split_by_heading(text):
        for kind, content in split_markdown(chunk):
            if kind == "md":
                cells.append(nbf.v4.new_markdown_cell(content))
            else:
                fname = content
                if fname in IMAGE_MAP:
                    call = IMAGE_MAP[fname]
                    code = f"# {fname} 대신 인터랙티브 위젯으로 대체됨\n{call}"
                    cells.append(nbf.v4.new_code_cell(code))
                else:
                    raw_url = (f"https://raw.githubusercontent.com/gabraxas/LVs-and-Policy/main/"
                               f"lecture/week{week_num:02d}/images/{fname}")
                    cells.append(nbf.v4.new_markdown_cell(f"![diagram]({raw_url})"))

        # heading 기반 위젯 삽입 (이미지가 없는 주차)
        heading_line = chunk.split("\n", 1)[0]
        for key, label, call in heading_widgets:
            if key in heading_line and (key, label) not in used_widgets:
                cells.append(nbf.v4.new_markdown_cell(f"**[인터랙티브] {label}**"))
                cells.append(nbf.v4.new_code_cell(call))
                used_widgets.add((key, label))

    nb["cells"] = cells
    nb["metadata"] = {
        "colab": {"provenance": [], "name": f"week{week_num:02d}.ipynb"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"wrote {out_path}  ({len(cells)} cells)")


if __name__ == "__main__":
    jobs = [
        (1, f"{REPO}/lecture/week01/week01-intro-and-industry-landscape.md",
         f"{REPO}/lecture/week01/week01.ipynb",
         [("2.2 16주간", "궤도속도 탐색기 — 고도를 바꿔보며 원궤도속도·탈출속도 확인",
           "orbital_velocity_explorer()"),
          ("2.2 16주간", "$/kg 사다리 — 발사체 세대별 kg당 비용 비교", "cost_per_kg_ladder()")]),
        (2, f"{REPO}/lecture/week02/week02-rocket-flight-fundamentals.md",
         f"{REPO}/lecture/week02/week02.ipynb", None),
        (3, f"{REPO}/lecture/week03/week03-propulsion-systems.md",
         f"{REPO}/lecture/week03/week03.ipynb", None),
        (4, f"{REPO}/lecture/week04/week04-launch-vehicle-systems-design.md",
         f"{REPO}/lecture/week04/week04.ipynb", None),
    ]
    for week_num, md_path, out_path, extra in jobs:
        build_notebook(week_num, md_path, out_path, extra)

