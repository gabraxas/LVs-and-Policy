"""
course_interactive.py
우주수송정책과 발사체 기술 — 강의용 인터랙티브 계산기 모음 (1~4주차 공용)

사용법 (Colab 노트북 첫 셀):
    !pip install -q ipywidgets
    !apt-get -qq install -y fonts-nanum > /dev/null 2>&1
    import urllib.request
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/gabraxas/LVs-and-Policy/main/lecture/_shared/course_interactive.py",
        "course_interactive.py")
    from course_interactive import *
    setup_korean_font()
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import ipywidgets as widgets
from ipywidgets import interact, FloatSlider, IntSlider, Dropdown, fixed

# ---------- 공통 색상/스타일 ----------
INK = "#1D1D1F"
INK_MUTED = "#555555"
PRIMARY = "#0066CC"
AMBER = "#C77700"
GREEN = "#1E8E5A"
RED = "#C0392B"
PURPLE = "#7C3AED"
GRID = "#E0E0E0"

MU_EARTH = 3.986e5   # km^3/s^2
R_EARTH = 6378.0     # km
G0 = 9.80665         # m/s^2


def setup_korean_font():
    """Colab 환경에서 한글 폰트(나눔고딕)를 설치하고 matplotlib에 등록한다.
    노트북의 첫 코드 셀에서 반드시 한 번 호출할 것 — 호출하지 않으면 그래프의 한글이 깨져 보인다."""
    import subprocess
    import os
    import matplotlib.font_manager as fm
    try:
        subprocess.run(["apt-get", "-qq", "install", "-y", "fonts-nanum"], check=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        font_dir = "/usr/share/fonts/truetype/nanum"
        found = False
        if os.path.isdir(font_dir):
            for f in os.listdir(font_dir):
                if f.endswith(".ttf"):
                    fm.fontManager.addfont(os.path.join(font_dir, f))
                    found = True
        if found:
            plt.rcParams["font.family"] = "NanumGothic"
            plt.rcParams["axes.unicode_minus"] = False
            print("한글 폰트 설정 완료 (NanumGothic). 이후 그래프부터 한글이 정상 표시됩니다.")
        else:
            print("나눔고딕 폰트 파일을 찾지 못했습니다. 그래프의 한글이 깨져 보일 수 있습니다.")
    except Exception as e:
        print(f"한글 폰트 자동 설치 실패: {e}")
        print("Colab이 아닌 로컬 환경이라면 이 경고는 무시하거나, 시스템에 설치된 한글 폰트를 "
              "plt.rcParams['font.family']에 직접 지정하세요.")


def _style(ax, grid=True):
    ax.set_facecolor("white")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#B0B0B0")
    if grid:
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=INK_MUTED, labelsize=10)


# ============================================================
# 1. 궤도속도 탐색기 (1·2주차 §1.2)
# ============================================================
def _plot_orbital_velocity(altitude_km):
    h = np.linspace(150, 40000, 600)
    r = R_EARTH + h
    v = np.sqrt(MU_EARTH / r)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(h, v, color=PRIMARY, linewidth=2.4, zorder=3)
    _style(ax)
    ax.set_xscale("log")
    ax.set_xlabel("고도 (km, 로그축)")
    ax.set_ylabel("원궤도 속도 (km/s)")

    r_sel = R_EARTH + altitude_km
    v_sel = np.sqrt(MU_EARTH / r_sel)
    v_esc = np.sqrt(2) * v_sel
    ax.plot(altitude_km, v_sel, "o", color=AMBER, markersize=10, zorder=5)
    ax.axvline(altitude_km, color=AMBER, linestyle="--", linewidth=1, alpha=0.6)

    refs = [(200, "LEO"), (700, "SSO"), (20200, "MEO"), (35786, "GEO")]
    for hh, name in refs:
        ax.plot(hh, np.sqrt(MU_EARTH / (R_EARTH + hh)), "x", color=INK_MUTED, markersize=6)
        ax.annotate(name, (hh, np.sqrt(MU_EARTH / (R_EARTH + hh))), fontsize=8, color=INK_MUTED,
                    xytext=(0, 8), textcoords="offset points", ha="center")

    ax.set_title(f"고도 {altitude_km:,.0f} km → 원궤도속도 {v_sel:.2f} km/s · 탈출속도 {v_esc:.2f} km/s")
    plt.tight_layout()
    plt.show()

    print(f"[선택 고도 {altitude_km:,.0f} km]")
    print(f"  원궤도 속도  v_c  = sqrt(mu/r) = {v_sel:.3f} km/s")
    print(f"  탈출 속도    v_esc = sqrt(2)*v_c = {v_esc:.3f} km/s  (원궤도속도의 약 {v_esc/v_sel:.2f}배)")
    print(f"  비궤도에너지 ε = -mu/(2a) = {-MU_EARTH/(2*r_sel):.2f} MJ/kg")


def orbital_velocity_explorer():
    """슬라이더로 고도를 바꿔가며 원궤도속도·탈출속도가 어떻게 변하는지 확인한다."""
    interact(_plot_orbital_velocity,
             altitude_km=FloatSlider(value=400, min=160, max=40000, step=50,
                                      description="고도(km)", readout_format=",.0f",
                                      style={"description_width": "80px"}, layout={"width": "500px"}))


# ============================================================
# 2. 호만 전이 탐색기 (2주차 §1.3)
# ============================================================
def _plot_hohmann(alt_leo_km, alt_geo_km):
    r_leo = R_EARTH + alt_leo_km
    r_geo = R_EARTH + alt_geo_km
    v_leo = np.sqrt(MU_EARTH / r_leo)
    v_geo = np.sqrt(MU_EARTH / r_geo)
    a_t = (r_leo + r_geo) / 2
    v_p = np.sqrt(MU_EARTH * (2 / r_leo - 1 / a_t))
    v_a = np.sqrt(MU_EARTH * (2 / r_geo - 1 / a_t))
    dv1 = v_p - v_leo
    dv2 = v_geo - v_a

    # normalized drawing scale
    r1, r2 = 1.0, 1.0 + 1.6 * (r_geo - r_leo) / (r_geo)
    r2 = max(r2, 1.3)
    e_t = (r2 - r1) / (r2 + r1)
    a_draw = (r1 + r2) / 2
    b_draw = a_draw * np.sqrt(1 - e_t ** 2)
    center = -(r2 - a_draw)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    th = np.linspace(0, 2 * np.pi, 400)
    ax.add_patch(Circle((0, 0), r1, facecolor="none", edgecolor=PRIMARY, linewidth=2))
    ax.add_patch(Circle((0, 0), r2, facecolor="none", edgecolor=GREEN, linewidth=2))
    ell_x = center + a_draw * np.cos(th)
    ell_y = b_draw * np.sin(th)
    keep = ell_y >= -0.02
    ax.plot(ell_x[keep], ell_y[keep], "--", color=AMBER, linewidth=2)
    ax.plot(-r1, 0, "o", color=PRIMARY, markersize=8)
    ax.plot(r2, 0, "o", color=GREEN, markersize=8)
    ax.annotate(f"Δv1={dv1:.2f}km/s", (-r1, 0), xytext=(-r1 - 0.5, 0.3), fontsize=10, color=AMBER)
    ax.annotate(f"Δv2={dv2:.2f}km/s", (r2, 0), xytext=(r2 + 0.1, 0.3), fontsize=10, color=GREEN)
    ax.text(-r1, -0.25, f"고도{alt_leo_km:,.0f}km", fontsize=9, color=PRIMARY, ha="center")
    ax.text(r2, -0.25, f"고도{alt_geo_km:,.0f}km", fontsize=9, color=GREEN, ha="center")
    ax.axis("off")
    lim = r2 + 1.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim * 0.6, lim * 0.6)
    ax.set_title(f"호만 전이 합계 Δv = {dv1+dv2:.3f} km/s")
    plt.tight_layout()
    plt.show()

    print(f"근지점(출발) 속도 v_c1 = {v_leo:.3f} km/s,  전이궤도 근지점속도 = {v_p:.3f} km/s  →  Δv1 = {dv1:.3f} km/s")
    print(f"원지점(도착) 속도 v_c2 = {v_geo:.3f} km/s,  전이궤도 원지점속도 = {v_a:.3f} km/s  →  Δv2 = {dv2:.3f} km/s")
    print(f"합계 Δv = {dv1 + dv2:.3f} km/s (궤도경사각 변경 제외)")


def hohmann_transfer_explorer():
    """출발/도착 궤도 고도를 슬라이더로 바꾸며 호만 전이 Δv를 계산한다."""
    interact(_plot_hohmann,
             alt_leo_km=FloatSlider(value=200, min=160, max=2000, step=10,
                                     description="출발 고도(km)", style={"description_width": "100px"}, layout={"width": "500px"}),
             alt_geo_km=FloatSlider(value=35786, min=2000, max=42000, step=100,
                                     description="도착 고도(km)", readout_format=",.0f",
                                     style={"description_width": "100px"}, layout={"width": "500px"}))


# ============================================================
# 3. Δv 예산 계산기 (2주차 §1.6)
# ============================================================
def _plot_delta_v_budget(target_alt_km, gravity_loss, other_loss, rotation_bonus):
    v_orbit = np.sqrt(MU_EARTH / (R_EARTH + target_alt_km))
    cats = ["궤도속도", "중력손실", "기타손실", "자전보너스", "요구 Δv"]
    deltas = [v_orbit, gravity_loss, other_loss, -rotation_bonus, None]
    colors = [PRIMARY, "#D97706", "#8B4513", GREEN, INK]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cum = 0
    for i in range(4):
        v = deltas[i]
        bottom = cum if v >= 0 else cum + v
        height = abs(v)
        ax.bar(i, height, bottom=bottom, width=0.6, color=colors[i], zorder=3, edgecolor="white")
        ax.text(i, bottom + height + 0.1, f"{v:+.2f}", ha="center", fontsize=10, fontweight="bold")
        cum += v
    ax.bar(4, cum, width=0.6, color=colors[4], zorder=3, edgecolor="white")
    ax.text(4, cum + 0.1, f"{cum:.2f} km/s", ha="center", fontsize=11, fontweight="bold")

    ax.set_xticks(range(5))
    ax.set_xticklabels(cats, fontsize=10)
    _style(ax, grid=False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_ylabel("누적 Δv (km/s)")
    ax.set_ylim(0, cum + 1.5)
    ax.set_title(f"목표고도 {target_alt_km:,.0f}km 임무 — 실제 요구 Δv = {cum:.3f} km/s")
    plt.tight_layout()
    plt.show()


def delta_v_budget_calculator():
    """궤도 고도와 각종 손실을 슬라이더로 조절해 실제 요구 Δv를 계산한다."""
    interact(_plot_delta_v_budget,
             target_alt_km=FloatSlider(value=200, min=160, max=2000, step=10,
                                        description="목표고도(km)", style={"description_width": "100px"}, layout={"width": "500px"}),
             gravity_loss=FloatSlider(value=1.35, min=0.5, max=2.5, step=0.05,
                                       description="중력손실(km/s)", style={"description_width": "100px"}, layout={"width": "500px"}),
             other_loss=FloatSlider(value=0.35, min=0.0, max=1.0, step=0.05,
                                     description="기타손실(km/s)", style={"description_width": "100px"}, layout={"width": "500px"}),
             rotation_bonus=FloatSlider(value=0.35, min=0.0, max=0.465, step=0.01,
                                         description="자전보너스(km/s)", style={"description_width": "100px"}, layout={"width": "500px"}))


# ============================================================
# 4. 로켓방정식 탐색기 (2주차 §3.2)
# ============================================================
def _plot_rocket_equation(isp_s, delta_v_km_s):
    c = isp_s * G0 / 1000  # km/s
    dv_over_c = np.linspace(0, 6, 300)
    MR_curve = np.exp(dv_over_c)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(dv_over_c, MR_curve, color=PRIMARY, linewidth=2.4, zorder=3)
    _style(ax)
    ax.set_xlabel("Δv / c")
    ax.set_ylabel("질량비 MR = m0/mf")

    x_sel = delta_v_km_s / c
    y_sel = np.exp(x_sel)
    ax.plot(x_sel, y_sel, "o", color=AMBER, markersize=10, zorder=5)
    ax.set_ylim(0, max(30, y_sel * 1.3))
    ax.set_title(f"Isp={isp_s}s, Δv={delta_v_km_s}km/s → MR={y_sel:.2f}")
    plt.tight_layout()
    plt.show()

    prop_frac = 1 - 1 / y_sel
    print(f"유효배기속도 c = Isp*g0 = {c:.3f} km/s")
    print(f"질량비 MR = exp(Δv/c) = {y_sel:.3f}")
    print(f"추진제 질량분율 = 1 - 1/MR = {prop_frac*100:.1f}%")
    print(f"→ 이륙질량의 {prop_frac*100:.1f}%가 추진제, 나머지 {(1-prop_frac)*100:.1f}%로 구조+탑재체를 감당해야 함")


def rocket_equation_explorer():
    """비추력과 요구 Δv를 슬라이더로 바꾸며 질량비·추진제 질량분율을 확인한다."""
    interact(_plot_rocket_equation,
             isp_s=FloatSlider(value=300, min=200, max=465, step=5,
                                description="Isp(s)", style={"description_width": "80px"}, layout={"width": "500px"}),
             delta_v_km_s=FloatSlider(value=9.4, min=1, max=15, step=0.1,
                                       description="요구Δv(km/s)", style={"description_width": "80px"}, layout={"width": "500px"}))


# ============================================================
# 5. 탑재중량비·다단화 계산기 (2주차 §3.4, §4.3)
# ============================================================
def _plot_staging(isp_s, epsilon, delta_v_km_s, n_stages):
    c = isp_s * G0 / 1000
    dv_per_stage = delta_v_km_s / n_stages
    MR_stage = np.exp(dv_per_stage / c)
    lam_stage = (1 / MR_stage - epsilon) / (1 - epsilon)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    if lam_stage <= 0:
        ax.text(0.5, 0.5, "λ < 0  →  이 조건에서는 실현 불가능\n(구조계수 대비 요구 Δv가 너무 큼)",
                ha="center", va="center", fontsize=14, color=RED, transform=ax.transAxes, fontweight="bold")
        ax.axis("off")
        lam_total = None
    else:
        lam_total = lam_stage ** n_stages
        stages_x = list(range(1, n_stages + 1))
        cum_lams = [lam_stage ** k for k in stages_x]
        ax.bar(stages_x, [c * 100 for c in cum_lams], color=PRIMARY, width=0.5, zorder=3, edgecolor="white")
        for x, v in zip(stages_x, cum_lams):
            ax.text(x, v * 100 + 0.05, f"{v*100:.2f}%", ha="center", fontsize=10, fontweight="bold")
        _style(ax)
        ax.set_xticks(stages_x)
        ax.set_xlabel("단수 (균등 배분 가정)")
        ax.set_ylabel("탑재중량비 λ_total (%)")
        ax.set_title(f"Isp={isp_s}s, ε={epsilon}, Δv={delta_v_km_s}km/s → {n_stages}단 λ_total={lam_total*100:.2f}%")
    plt.tight_layout()
    plt.show()

    print(f"단별 배분 Δv = {dv_per_stage:.3f} km/s,  단별 질량비 MR = {MR_stage:.3f}")
    if lam_stage <= 0:
        print("단별 λ가 음수입니다 — 구조계수(ε)를 낮추거나 단수를 늘려보세요.")
    else:
        print(f"단별 탑재중량비 λ_stage = {lam_stage*100:.2f}%")
        print(f"{n_stages}단 총 탑재중량비 λ_total = λ_stage^{n_stages} = {lam_total*100:.3f}%")


def staging_calculator():
    """구조계수·비추력·요구Δv·단수를 슬라이더로 바꾸며 탑재중량비를 계산한다. (1단=SSTO 실현 가능성 확인용)"""
    interact(_plot_staging,
             isp_s=FloatSlider(value=330, min=250, max=465, step=5,
                                description="Isp(s)", style={"description_width": "90px"}, layout={"width": "480px"}),
             epsilon=FloatSlider(value=0.08, min=0.04, max=0.14, step=0.005,
                                  description="구조계수 ε", readout_format=".3f",
                                  style={"description_width": "90px"}, layout={"width": "480px"}),
             delta_v_km_s=FloatSlider(value=9.4, min=3, max=14, step=0.1,
                                       description="요구Δv(km/s)", style={"description_width": "90px"}, layout={"width": "480px"}),
             n_stages=IntSlider(value=2, min=1, max=5, step=1,
                                 description="단수", style={"description_width": "90px"}, layout={"width": "480px"}))


# ============================================================
# 6. 추력·노즐 팽창 탐색기 (3주차 §1.2~2.2)
# ============================================================
def _plot_nozzle(pc_MPa, expansion_ratio, altitude_km):
    # simplified isentropic-ish relation for illustration (not exact gas dynamics)
    pa = 101.3 * np.exp(-altitude_km / 8.0)  # kPa, rough atmosphere
    pc_kPa = pc_MPa * 1000
    pe_kPa = pc_kPa / (expansion_ratio ** 1.2)  # illustrative relation

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["연소실 Pc", "출구 Pe", "외기압 Pa"]
    vals = [pc_kPa, pe_kPa, pa]
    colors_b = [INK, PRIMARY, AMBER]
    ax.bar(labels, vals, color=colors_b, width=0.5, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.02, f"{v:,.1f} kPa", ha="center", fontsize=10, fontweight="bold")
    ax.set_yscale("log")
    _style(ax, grid=False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)

    if pe_kPa < pa:
        regime = "과팽창 (Pe < Pa) — 지상에서 흐름 박리 위험"
        rc = RED
    elif abs(pe_kPa - pa) / pa < 0.15:
        regime = "최적 팽창에 근접 (Pe ≈ Pa)"
        rc = GREEN
    else:
        regime = "부족팽창 (Pe > Pa) — 배기 에너지 낭비"
        rc = AMBER
    ax.set_title(f"고도 {altitude_km:,.0f}km, ε={expansion_ratio:.0f} → {regime}", color=rc, fontsize=12)
    plt.tight_layout()
    plt.show()
    print(f"연소실압 Pc = {pc_kPa:,.0f} kPa,  추정 출구압 Pe = {pe_kPa:,.1f} kPa,  외기압 Pa(고도 {altitude_km:.0f}km) = {pa:.2f} kPa")
    print(f"→ {regime}")
    print("※ Pe 계산은 강의용 근사 관계식이며 실제 노즐 설계값과는 차이가 있다.")


def thrust_nozzle_explorer():
    """연소실압·팽창비·고도를 슬라이더로 바꾸며 과팽창/최적/부족팽창 상태를 확인한다."""
    interact(_plot_nozzle,
             pc_MPa=FloatSlider(value=10, min=1, max=35, step=1,
                                 description="연소실압(MPa)", style={"description_width": "100px"}, layout={"width": "480px"}),
             expansion_ratio=FloatSlider(value=16, min=5, max=200, step=1,
                                          description="팽창비 ε", style={"description_width": "100px"}, layout={"width": "480px"}),
             altitude_km=FloatSlider(value=0, min=0, max=100, step=1,
                                      description="고도(km)", style={"description_width": "100px"}, layout={"width": "480px"}))


# ============================================================
# 7. 추진제 비교 탐색기 (3주차 §1.4, §2.4)
# ============================================================
_PROPELLANTS = {
    "LOX/RP-1 (케로신)":  dict(isp=340, tc=3600, mw=22, density=1.02, color=PRIMARY),
    "LOX/CH4 (메탄)":     dict(isp=360, tc=3500, mw=20, density=0.83, color=GREEN),
    "LOX/LH2 (수소)":     dict(isp=450, tc=3000, mw=10, density=0.36, color=PURPLE),
    "고체(APCP)":         dict(isp=265, tc=3000, mw=25, density=1.80, color="#8B5E3C"),
}


def _plot_propellant(name):
    p = _PROPELLANTS[name]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    ax = axes[0]
    names = list(_PROPELLANTS.keys())
    isps = [_PROPELLANTS[n]["isp"] for n in names]
    colors = [_PROPELLANTS[n]["color"] if n == name else "#D9D9D9" for n in names]
    ax.bar(names, isps, color=colors, width=0.55, zorder=3)
    ax.set_ylabel("진공 Isp (s)")
    _style(ax, grid=False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(axis="x", labelrotation=20)
    ax.set_title("추진제별 비추력")

    ax2 = axes[1]
    for n in names:
        pn = _PROPELLANTS[n]
        rho_isp = pn["density"] * pn["isp"]
        c = pn["color"] if n == name else "#D9D9D9"
        ax2.scatter(pn["isp"], pn["density"], s=rho_isp * 2.2, color=c, edgecolor="white", linewidth=1.2, zorder=4)
    sel_rho_isp = p["density"] * p["isp"]
    ax2.annotate(f"{name}\nρ·Isp≈{sel_rho_isp:.0f}", (p["isp"], p["density"]), xytext=(10, 10),
                 textcoords="offset points", fontsize=9.5, fontweight="bold")
    _style(ax2)
    ax2.set_xlabel("진공 Isp (s)")
    ax2.set_ylabel("밀도 (g/cm³)")
    ax2.set_title("밀도비추력 (원 크기 = ρ·Isp)")

    plt.tight_layout()
    plt.show()
    print(f"[{name}] Tc≈{p['tc']}K, 배기 평균분자량 M≈{p['mw']}, Isp≈{p['isp']}s, ρ≈{p['density']}g/cm³, ρ·Isp≈{sel_rho_isp:.0f}")
    print("직관: Isp ∝ sqrt(Tc/M) — 온도보다 분자량이 Isp를 더 크게 좌우한다.")


def propellant_isp_explorer():
    """추진제 조합을 선택해 비추력·밀도비추력을 비교한다."""
    interact(_plot_propellant, name=Dropdown(options=list(_PROPELLANTS.keys()), description="추진제",
                                              style={"description_width": "80px"}))


# ============================================================
# 8. 엔진 사이클 비교 탐색기 (3주차 §3.6)
# ============================================================
_CYCLES = {
    "가압식": dict(risk=1, perf=1, color=INK_MUTED, ex="착륙선·추력기"),
    "전기펌프": dict(risk=2, perf=2.5, color=GREEN, ex="Rutherford"),
    "가스발생기": dict(risk=3, perf=4, color=PRIMARY, ex="Merlin, 누리호"),
    "팽창기": dict(risk=3.3, perf=5.6, color="#7C9A3C", ex="RL10, Vinci"),
    "다단연소": dict(risk=4.3, perf=7.7, color=AMBER, ex="RD-180, RS-25"),
    "전량연소(FFSC)": dict(risk=5, perf=9.3, color=RED, ex="Raptor"),
}


def _plot_cycle(name):
    fig, ax = plt.subplots(figsize=(8, 6))
    for n, c in _CYCLES.items():
        col = c["color"] if n == name else "#D9D9D9"
        size = 550 if n == name else 350
        ax.scatter(c["risk"], c["perf"], s=size, color=col, edgecolor="white", linewidth=1.6, zorder=4)
        ax.annotate(f"{n}\n({c['ex']})", (c["risk"], c["perf"]), xytext=(0, 14 if n != name else 18),
                    textcoords="offset points", ha="center", fontsize=9.5,
                    fontweight="bold" if n == name else "normal")
    _style(ax)
    ax.set_xlabel("개발 난이도·비용·일정 위험 →")
    ax.set_ylabel("성능(연소실 압력·Isp) →")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"선택: {name}")
    plt.tight_layout()
    plt.show()
    c = _CYCLES[name]
    print(f"[{name}] 대표 엔진: {c['ex']}")
    print("성능이 높을수록 개발 난이도·비용·일정 위험도 함께 커진다 — '무엇을 살 것인가'의 문제.")


def engine_cycle_comparison():
    """엔진 사이클을 선택해 성능-개발위험 트레이드오프에서의 위치를 확인한다."""
    interact(_plot_cycle, name=Dropdown(options=list(_CYCLES.keys()), description="사이클",
                                         style={"description_width": "80px"}))


# ============================================================
# 9. Max-Q / 스로틀 버킷 탐색기 (4주차 §1.2)
# ============================================================
def _plot_max_q(bucket_depth_pct, bucket_width_s):
    t = np.linspace(0, 140, 400)
    v = 10 * t - 0.02 * t ** 2
    rho = np.exp(-t / 45)
    q = 0.5 * rho * v ** 2
    q = q / q.max() * 34
    t_maxq = t[np.argmax(q)]

    throttle = np.ones_like(t) * 100
    mask = (t > t_maxq - bucket_width_s) & (t < t_maxq + bucket_width_s * 0.8)
    throttle[mask] = 100 - bucket_depth_pct * np.exp(-((t[mask] - t_maxq) ** 2) / (2 * (bucket_width_s / 2) ** 2))

    fig, ax1 = plt.subplots(figsize=(8.5, 5.5))
    ax1.plot(t, q, color=PRIMARY, linewidth=2.4)
    ax1.set_xlabel("발사 후 경과시간 (s)")
    ax1.set_ylabel("동압 q (kPa)", color=PRIMARY)
    ax1.tick_params(axis="y", labelcolor=PRIMARY)
    _style(ax1)
    ax2 = ax1.twinx()
    ax2.plot(t, throttle, color=GREEN, linewidth=2, linestyle="--")
    ax2.set_ylabel("엔진 스로틀 (%)", color=GREEN)
    ax2.tick_params(axis="y", labelcolor=GREEN)
    ax2.set_ylim(50, 105)
    ax1.set_title(f"스로틀 버킷 깊이 {bucket_depth_pct:.0f}%p, 폭 ±{bucket_width_s:.0f}s")
    plt.tight_layout()
    plt.show()
    print("스로틀을 더 깊이/넓게 줄일수록 구조 하중은 줄지만 중력손실(추가 Δv 요구)이 커진다 — 트레이드오프.")


def max_q_explorer():
    """스로틀 버킷의 깊이와 폭을 슬라이더로 바꾸며 Max-Q 대응 전략을 시험한다."""
    interact(_plot_max_q,
             bucket_depth_pct=FloatSlider(value=30, min=0, max=50, step=1,
                                           description="버킷 깊이(%p)", style={"description_width": "100px"}, layout={"width": "480px"}),
             bucket_width_s=FloatSlider(value=25, min=5, max=45, step=1,
                                         description="버킷 폭(s)", style={"description_width": "100px"}, layout={"width": "480px"}))


# ============================================================
# 10. 상승궤적(중력선회) 탐색기 (4주차 §2.1)
# ============================================================
def _plot_ascent(pitch_deg, pitch_start_alt):
    t1 = np.linspace(0, 1, 60)
    y1 = t1 * pitch_start_alt
    x1 = np.zeros_like(t1)

    ang_max = np.radians(90 - pitch_deg)
    t2 = np.linspace(0, 1, 100)
    ang = t2 * ang_max
    r_turn = 30
    x2 = r_turn * np.sin(ang)
    y2 = pitch_start_alt + r_turn * (1 - np.cos(ang)) * 0.9

    t3 = np.linspace(0, 1, 100)
    x3 = x2[-1] + t3 * 70
    y3 = y2[-1] + t3 * 20 * np.cos(ang_max)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot(x1, y1, color=RED, linewidth=3)
    ax.plot(x2, y2, color=AMBER, linewidth=3)
    ax.plot(x3, y3, color=PRIMARY, linewidth=3)
    ax.axhline(0, color="#B0B0B0", linewidth=1)
    _style(ax)
    ax.set_xlabel("사거리 (상대 스케일)")
    ax.set_ylabel("고도 (상대 스케일)")
    ax.set_title(f"피치오버 각도 {pitch_deg}°, 시작고도 {pitch_start_alt:.0f} — "
                 f"{'급격한 선회 (조향손실↑)' if pitch_deg > 15 else '완만한 선회 (표준 중력선회)'}")
    plt.tight_layout()
    plt.show()
    print("피치오버 각도가 클수록 궤적이 빨리 눕지만 조향손실(구조 하중)이 커진다.")
    print("작을수록 중력선회가 부드럽지만 궤적이 눕는 데 시간이 걸려 중력손실이 커질 수 있다.")


def ascent_trajectory_explorer():
    """초기 피치오버 각도와 시작 고도를 슬라이더로 바꾸며 상승궤적의 형태 변화를 확인한다."""
    interact(_plot_ascent,
             pitch_deg=FloatSlider(value=5, min=1, max=30, step=1,
                                    description="피치오버(°)", style={"description_width": "100px"}, layout={"width": "480px"}),
             pitch_start_alt=FloatSlider(value=15, min=5, max=30, step=1,
                                          description="시작고도", style={"description_width": "100px"}, layout={"width": "480px"}))


# ============================================================
# 11. $/kg 사다리 (1주차 §2.2)
# ============================================================
_COST_LADDER = [
    ("우주왕복선 (STS)", 55000, "#8B5E3C"),
    ("Falcon 9 (전용)", 2700, PRIMARY),
    ("라이드셰어 (소형위성)", 7000, AMBER),
    ("Starship (목표)", 100, GREEN),
]


def _plot_cost_ladder(highlight):
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [c[0] for c in _COST_LADDER]
    vals = [c[1] for c in _COST_LADDER]
    colors = [c[2] if c[0] == highlight else "#D9D9D9" for c in _COST_LADDER]
    ax.bar(names, vals, color=colors, width=0.55, zorder=3)
    ax.set_yscale("log")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.15, f"${v:,}/kg", ha="center", fontsize=10, fontweight="bold")
    _style(ax, grid=False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(axis="x", labelrotation=15)
    ax.set_ylabel("$/kg (로그축)")
    ax.set_title("발사 kg당 비용의 궤적")
    plt.tight_layout()
    plt.show()


def cost_per_kg_ladder():
    """발사체 세대를 선택해 $/kg이 어떻게 하락해 왔는지 비교한다."""
    interact(_plot_cost_ladder, highlight=Dropdown(options=[c[0] for c in _COST_LADDER],
                                                    description="비교 대상", style={"description_width": "80px"}))


# ============================================================
# 12. 몬테카를로 위험비교 탐색기 (5주차 §SSTO/TSTO 방법론, Greenberg Ch.4 §VI)
# ============================================================
def _plot_monte_carlo_risk(perf_uncertainty_pct, cost_sensitivity, n_sim):
    rng = np.random.default_rng(42)
    # ① 달성 성능(설계점=100 기준, %) 표본추출 — 정규분포 후 [70,115]로 절단(최소허용~설계점 초과 캡)
    perf = rng.normal(100, perf_uncertainty_pct, n_sim)
    perf = np.clip(perf, 70, 115)

    # ② 설계점 미달분(shortfall)이 클수록 탑재체 여유 축소 → 비용 급증 (2차 다항식 민감도, 슬라이드8)
    shortfall = np.clip(100 - perf, 0, None)
    cost_multiplier = 1.0 + cost_sensitivity * (shortfall / 20) ** 2
    base_cost = 100.0  # 임의단위(억원) — 실제 설계평가 수치 아님
    pvlcc = base_cost * cost_multiplier

    m = pvlcc.mean()
    sigma = pvlcc.std()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    ax = axes[0]
    ax.hist(perf, bins=30, color=PRIMARY, alpha=0.85, zorder=3)
    _style(ax)
    ax.axvline(100, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel("달성 성능 (설계점=100 기준, %)")
    ax.set_ylabel("빈도")
    ax.set_title("① 성능 표본분포 (Monte Carlo)")

    ax = axes[1]
    ax.hist(pvlcc, bins=30, color=AMBER, alpha=0.85, zorder=3)
    _style(ax)
    ax.axvline(m, color=RED, linewidth=2, label=f"m = {m:.1f}")
    ax.axvline(m + sigma, color=RED, linestyle="--", linewidth=1, label=f"m±σ = {sigma:.1f}")
    ax.axvline(m - sigma, color=RED, linestyle="--", linewidth=1)
    ax.set_xlabel("생애주기비용 현재가치 PVLCC (임의단위)")
    ax.set_ylabel("빈도")
    ax.set_title("② 비용 사후분포 → (m, σ) 산출")
    ax.legend(fontsize=8, frameon=False)

    plt.tight_layout()
    plt.show()

    print(f"[{n_sim:,}회 반복]")
    print(f"  기대비용(m)  = {m:.2f}")
    print(f"  표준편차(σ)  = {sigma:.2f}   (변동계수 σ/m = {sigma/m:.1%})")
    print("  → 성능 불확실성이 클수록, 비용민감도계수가 클수록 σ(위험)가 커집니다.")
    print("  → 이 (m, σ) 한 쌍이 아키텍처 간 위험-기댓값 비교의 기본 단위입니다.")


def monte_carlo_risk_explorer():
    """SSTO/TSTO 방법론의 핵심인 몬테카를로 절차를 직접 실행 — 성능 불확실성·비용민감도를 바꿔가며
    (m, σ)가 어떻게 달라지는지 확인한다."""
    interact(_plot_monte_carlo_risk,
             perf_uncertainty_pct=FloatSlider(value=8, min=1, max=20, step=1,
                                               description="성능 불확실성(%p)",
                                               style={"description_width": "120px"}, layout={"width": "480px"}),
             cost_sensitivity=FloatSlider(value=1.5, min=0.2, max=4.0, step=0.1,
                                           description="비용 민감도계수",
                                           style={"description_width": "120px"}, layout={"width": "480px"}),
             n_sim=IntSlider(value=2000, min=200, max=5000, step=200,
                              description="반복횟수(MAXR)",
                              style={"description_width": "120px"}, layout={"width": "480px"}))


# ============================================================
# 13. (m, σ) 위험-기댓값 프론티어 탐색기 (5주차 §DECISION FRAME)
# ============================================================
_ARCHITECTURES = {
    "기존/개량 ELV": (100, 8, INK_MUTED),
    "TSTO": (85, 22, PRIMARY),
    "SSTO": (72, 35, AMBER),
    "HRST(마그레브, 참고용)": (60, 30, PURPLE),
}


def _plot_risk_frontier(highlight, show_frontier):
    fig, ax = plt.subplots(figsize=(7.5, 6))

    pts = list(_ARCHITECTURES.items())
    for name, (m, s, c) in pts:
        color = c if name == highlight else "#D9D9D9"
        size = 220 if name == highlight else 140
        ax.scatter(m, s, s=size, color=color, zorder=4, edgecolor="white", linewidth=1.2)
        ax.annotate(name, (m, s), fontsize=9, color=INK,
                    xytext=(8, 8), textcoords="offset points",
                    fontweight="bold" if name == highlight else "normal")

    if show_frontier:
        # 지배되지 않는 점(더 낮은 m에서 더 낮은 σ가 없는 점)을 m 오름차순으로 찾아 프론티어 구성
        sorted_pts = sorted(pts, key=lambda kv: kv[1][0])
        frontier = []
        min_sigma_so_far = float("inf")
        for name, (m, s, c) in sorted_pts:
            if s < min_sigma_so_far:
                frontier.append((m, s))
                min_sigma_so_far = s
        fx = [p[0] for p in frontier]
        fy = [p[1] for p in frontier]
        ax.plot(fx, fy, "--", color=GREEN, linewidth=1.6, zorder=2, label="최적 대안 프론티어")
        ax.legend(fontsize=9, frameon=False, loc="upper right")

    _style(ax)
    ax.set_xlabel("기대현재가치비용 m (임의단위 — 왼쪽일수록 비용 낮음)")
    ax.set_ylabel("표준편차 σ (위험 — 아래쪽일수록 위험 낮음)")
    ax.set_title(f"위험-기댓값 평면 — 선택: {highlight}  (좌하단이 우월)")
    plt.tight_layout()
    plt.show()

    m, s, _ = _ARCHITECTURES[highlight]
    print(f"[{highlight}]  m = {m}   σ = {s}")
    print("동일 위험(σ)이면 m이 낮은 대안이 우월, 동일 m이면 σ가 낮은 대안이 우월합니다.")
    print("프론티어 위 대안들 사이의 최종 선택은 의사결정자의 위험선호(risk appetite)에 달려 있습니다.")


def risk_frontier_explorer():
    """ELV·SSTO·TSTO(·HRST)를 (m, σ) 평면에 놓고 최적 대안 프론티어와 위험-기댓값 상충관계를 확인한다."""
    interact(_plot_risk_frontier,
             highlight=Dropdown(options=list(_ARCHITECTURES.keys()), value="TSTO",
                                 description="아키텍처 선택", style={"description_width": "100px"}),
             show_frontier=widgets.Checkbox(value=True, description="프론티어 표시"))


# ============================================================
# 14. 우주왕복선 재사용 경제성 탐색기 (6주차 §PART III, 개념설명용 근사모델)
# ============================================================
def _plot_shuttle_economics(annual_flights, refurb_cost_musd, program_years):
    # 개념 설명을 위한 근사 모델 — 실제 프로그램 원가 자료와 다를 수 있음
    dev_cost = 30e9          # 개발비 (임의 근사, $)
    orbiter_unit_cost = 2e9  # 기체(오비터) 단가 (임의 근사, $)
    n_orbiters = 4
    reuses_per_orbiter = 25  # 오비터당 설계 재사용 횟수 가정
    ops_cost_per_flight = 80e6  # 발사장·관제 등 비행당 고정 운용비 (임의 근사)

    flights_axis = np.arange(1, 51)
    total_flights = flights_axis * program_years
    orbiter_amort = (orbiter_unit_cost * n_orbiters) / (reuses_per_orbiter * n_orbiters)
    cost_curve = (dev_cost / total_flights) + orbiter_amort + refurb_cost_musd * 1e6 + ops_cost_per_flight

    sel_total_flights = annual_flights * program_years
    sel_cost = dev_cost / sel_total_flights + orbiter_amort + refurb_cost_musd * 1e6 + ops_cost_per_flight

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(flights_axis, cost_curve / 1e6, color=PRIMARY, linewidth=2.4, zorder=3)
    _style(ax)
    ax.set_yscale("log")
    ax.set_xlabel("연간 비행 횟수 (가정)")
    ax.set_ylabel("회당 비용 ($M, 로그축)")

    ax.plot(annual_flights, sel_cost / 1e6, "o", color=AMBER, markersize=11, zorder=5)
    ax.axvline(annual_flights, color=AMBER, linestyle="--", linewidth=1, alpha=0.6)

    ax.axvline(4.5, color=RED, linestyle=":", linewidth=1.4)
    ax.annotate("실제 평균(연 4.5회)", (4.5, cost_curve.max() / 1e6 * 0.7), fontsize=8, color=RED,
                xytext=(6, 0), textcoords="offset points")

    ax.set_title(f"연 {annual_flights}회 비행 시 회당 비용 ≈ ${sel_cost/1e6:,.0f}M "
                 f"(정비비 ${refurb_cost_musd:.0f}M/회 가정)")
    plt.tight_layout()
    plt.show()

    print(f"[가정] 개발비 ${dev_cost/1e9:.0f}B · 오비터 {n_orbiters}대 · 오비터당 재사용 {reuses_per_orbiter}회 · "
          f"프로그램 기간 {program_years}년")
    print(f"[선택] 연 {annual_flights}회 비행 → 총 {sel_total_flights:,.0f}회 → 회당 비용 ${sel_cost/1e6:,.0f}M")
    print("  → 개발비 상환분은 비행 횟수가 늘수록 급격히 줄지만, 정비비·운용비는 줄지 않는다.")
    print("  → 실제 셔틀의 평균 비행 빈도(연 4.5회 안팎)는 곡선의 왼쪽(고비용 구간)에 위치했다는 점이 핵심.")
    print("  ※ 위 수치는 개념 설명을 위한 근사 모델이며, 실제 프로그램 원가 자료와 다를 수 있습니다.")


def shuttle_reuse_economics_explorer():
    """슬라이드14의 재사용 경제학 공식(회당비용 = 개발비/총비행수 + 기체비/재사용횟수 + 정비비 + 운용비)을
    직접 조작 — 약속된 발사 빈도와 실제 발사 빈도의 격차가 회당 비용에 미치는 영향을 확인한다."""
    interact(_plot_shuttle_economics,
             annual_flights=IntSlider(value=10, min=1, max=50, step=1,
                                       description="연간 비행 횟수",
                                       style={"description_width": "110px"}, layout={"width": "480px"}),
             refurb_cost_musd=FloatSlider(value=300, min=50, max=800, step=25,
                                           description="정비비($M/회)",
                                           style={"description_width": "110px"}, layout={"width": "480px"}),
             program_years=IntSlider(value=30, min=10, max=40, step=5,
                                      description="프로그램 기간(년)",
                                      style={"description_width": "110px"}, layout={"width": "480px"}))
