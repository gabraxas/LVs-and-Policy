"""
course_interactive.py
우주수송정책과 발사체 기술 — 강의용 인터랙티브 계산기 모음 (1~7주차 공용)

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

# ============================================================
# 15. FAA Part 450 연혁 타임라인 (7주차 §PART3, 정적 다이어그램)
# ============================================================
def faa_part450_timeline():
    """1984년 CSLA부터 2026년 Part 450 전면시행까지의 미국 발사허가 법제 연혁을
    타임라인으로 보여준다 (정적 다이어그램 — 슬라이더 없음)."""
    events = [
        ("1984", "상업우주발사법(CSLA)\n제정 — 민간발사 허가체계 출발", PRIMARY),
        ("1988", "정부 배상책임\n지원(Indemnification) 도입", PRIMARY),
        ("2006", "발사장(Launch Site)\n허가 규정 정비", PRIMARY),
        ("2021", "Part 450 규정 제정\n(4개 규정→성능기반 단일화)", AMBER),
        ("2026.3", "구 규정(legacy license)\n완전 폐지, 전면 시행", RED),
    ]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    xs = list(range(len(events)))
    ax.plot(xs, [0] * len(xs), color="#B0B0B0", linewidth=2, zorder=1)
    for x, (yr, label, color) in zip(xs, events):
        ax.scatter(x, 0, s=220, color=color, zorder=3, edgecolor="white", linewidth=1.5)
        ax.annotate(yr, (x, 0), fontsize=11, fontweight="bold", color=INK,
                    xytext=(0, 20), textcoords="offset points", ha="center")
        ax.annotate(label, (x, 0), fontsize=8.5, color=INK_MUTED,
                    xytext=(0, -22), textcoords="offset points", ha="center", va="top")
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.set_ylim(-1.4, 1.1)
    ax.axis("off")
    ax.set_title("미국 상업발사 허가법제 연혁 — CSLA(1984) → Part 450 전면시행(2026.3.9)",
                  fontsize=12, color=INK, pad=6)
    plt.tight_layout()
    plt.show()
    print("42년에 걸쳐 4개 개별 규정(ELV/RLV/발사장/재진입)이 하나의 성능기반 규정으로 통합되었습니다.")
    print("전환 기간 동안 신규·기존 사업자 모두 legacy license에서 Part 450으로 이관을 완료해야 했습니다.")


# ============================================================
# 16. 한국 우주법제 연혁 타임라인 (7주차 §PART2, 정적 다이어그램)
# ============================================================
def korea_space_law_timeline():
    """2005년 우주개발진흥법 제정부터 2026년 3대 입법과제까지, 한국 발사허가·손해배상 법제의
    최근 개편 흐름을 타임라인으로 보여준다 (정적 다이어그램 — 슬라이더 없음)."""
    events = [
        ("2005", "우주개발진흥법\n제정 (제11조 발사허가)", PRIMARY),
        ("2007", "우주손해배상법\n제정 (무과실책임)", PRIMARY),
        ("2024下", "발사허가 표준절차 개선\n+ 중장기 발사면허제도", AMBER),
        ("2025", "시험비행 사전신고·\n예비평가 신설", AMBER),
        ("2026", "3대 입법과제\n(우주항공기본법 등)", RED),
    ]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    xs = list(range(len(events)))
    ax.plot(xs, [0] * len(xs), color="#B0B0B0", linewidth=2, zorder=1)
    for x, (yr, label, color) in zip(xs, events):
        ax.scatter(x, 0, s=220, color=color, zorder=3, edgecolor="white", linewidth=1.5)
        ax.annotate(yr, (x, 0), fontsize=11, fontweight="bold", color=INK,
                    xytext=(0, 20), textcoords="offset points", ha="center")
        ax.annotate(label, (x, 0), fontsize=8.5, color=INK_MUTED,
                    xytext=(0, -22), textcoords="offset points", ha="center", va="top")
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.set_ylim(-1.4, 1.1)
    ax.axis("off")
    ax.set_title("한국 우주법제 연혁 — 우주개발진흥법(2005) → 3대 입법과제(2026)",
                  fontsize=12, color=INK, pad=6)
    plt.tight_layout()
    plt.show()
    print("21년 만에 '개발 지원' 중심 법제에서 '민간 산업화 전제' 법제로 구조가 재편되고 있습니다.")
    print("중장기 발사면허제도는 미국 Part 450의 포트폴리오 라이선스와 같은 방향의 개편입니다.")


# ============================================================
# 17. 3단 책임구조 탐색기 (7주차 §PART4 WHY INSURANCE)
# ============================================================
def _plot_liability_tiers(total_loss_musd, mpl_musd, gov_cap_musd):
    tier1 = min(total_loss_musd, mpl_musd)
    tier2 = min(max(total_loss_musd - mpl_musd, 0), max(gov_cap_musd - mpl_musd, 0))
    tier3 = max(total_loss_musd - gov_cap_musd, 0)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1, 1.6]})

    ax = axes[0]
    labels = ["① 오퍼레이터\n보험(MPL까지)", "② 정부배상\n(제한적)", "③ 국가\n무제한책임"]
    vals = [tier1, tier2, tier3]
    colors = [PRIMARY, AMBER, RED]
    bottoms = [0, tier1, tier1 + tier2]
    for i, (v, c, b) in enumerate(zip(vals, colors, bottoms)):
        ax.bar(0, v, bottom=b, color=c, width=0.5, zorder=3,
               label=f"{labels[i]}: ${v:,.0f}M" if v > 0 else None)
    _style(ax, grid=True)
    ax.set_xlim(-0.6, 1.3)
    ax.set_xticks([])
    ax.set_ylabel("배상액 ($M)")
    ax.axhline(mpl_musd, color=PRIMARY, linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(gov_cap_musd, color=AMBER, linestyle="--", linewidth=1, alpha=0.7)
    ax.set_title(f"총 손실 ${total_loss_musd:,.0f}M의 배분")
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.05, 1.0))

    ax = axes[1]
    stages = ["MPL\n(오퍼레이터 보험한도)", "정부배상 상한", "손실 발생 시나리오"]
    ax.barh(0, mpl_musd, color=PRIMARY, height=0.5, zorder=3, label="① MPL")
    ax.barh(0, gov_cap_musd - mpl_musd, left=mpl_musd, color=AMBER, height=0.5, zorder=3, label="② 정부배상 구간")
    ax.barh(0, max(total_loss_musd - gov_cap_musd, 0) if total_loss_musd > gov_cap_musd else 0,
            left=gov_cap_musd, color=RED, height=0.5, zorder=3, label="③ 국가무제한 구간(예시)")
    ax.axvline(total_loss_musd, color=INK, linewidth=2.2, zorder=5)
    ax.annotate(f"실제 손실\n${total_loss_musd:,.0f}M", (total_loss_musd, 0.42), fontsize=9,
                color=INK, ha="center", fontweight="bold")
    _style(ax, grid=False)
    ax.set_yticks([])
    ax.set_xlabel("책임 한도 스케일 ($M)")
    ax.set_title("어느 구간에 손실이 위치하는가")
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    plt.tight_layout()
    plt.show()

    print(f"[가정] MPL(보험한도) = ${mpl_musd:,.0f}M · 정부배상 상한 = ${gov_cap_musd:,.0f}M · "
          f"총 손실 = ${total_loss_musd:,.0f}M")
    print(f"  ① 오퍼레이터 보험 부담: ${tier1:,.0f}M")
    print(f"  ② 정부배상(제한적) 부담: ${tier2:,.0f}M")
    print(f"  ③ 국가 무제한책임 부담: ${tier3:,.0f}M")
    if tier3 > 0:
        print("  → 손실이 정부배상 상한마저 초과 — 책임협약상 국가책임에는 법정 상한이 없으므로 "
              "이론상 초과분은 전액 국가(납세자) 부담입니다.")
    else:
        print("  → 이번 시나리오는 정부배상 상한 이내에서 흡수됩니다.")


def liability_tiers_explorer():
    """국제법상 국가의 무과실 책임이 ①오퍼레이터 보험 → ②정부의 제한적 배상 → ③국가의 무제한책임
    순으로 재분배되는 3단 구조를, 손실 규모를 직접 조작해 확인한다."""
    interact(_plot_liability_tiers,
             total_loss_musd=FloatSlider(value=300, min=0, max=2000, step=50,
                                          description="총 손실액($M)",
                                          style={"description_width": "120px"}, layout={"width": "480px"}),
             mpl_musd=FloatSlider(value=200, min=50, max=500, step=10,
                                   description="MPL 보험한도($M)",
                                   style={"description_width": "120px"}, layout={"width": "480px"}),
             gov_cap_musd=FloatSlider(value=1500, min=500, max=3000, step=100,
                                       description="정부배상 상한($M)",
                                       style={"description_width": "120px"}, layout={"width": "480px"}))


# ============================================================
# 18. 기대사상자수(Ec)·최대예상손실(MPL) 계산기 (7주차 §PART4 MATH 1)
# ============================================================
def _plot_ec_mpl(fail_prob_pct, casualty_area_km2, pop_density, vsl_musd, property_damage_musd):
    P = fail_prob_pct / 100
    Ec = P * casualty_area_km2 * pop_density  # 기대사상자수(개념적 단순화 모델)
    casualty_value = Ec * vsl_musd
    mpl = casualty_value + property_damage_musd

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    rhos = np.linspace(0, 500, 200)
    ecs = P * casualty_area_km2 * rhos
    ax.plot(rhos, ecs, color=PRIMARY, linewidth=2.2, zorder=3)
    ax.axvline(pop_density, color=AMBER, linestyle="--", linewidth=1.3)
    ax.plot(pop_density, Ec, "o", color=AMBER, markersize=10, zorder=5)
    _style(ax)
    ax.set_xlabel("낙하구역 인구밀도 ρ (명/km²) — 해상=0")
    ax.set_ylabel("기대사상자수 Ec")
    ax.set_title(f"Ec = P × A_c × ρ  →  Ec = {Ec:.3f}")

    ax = axes[1]
    bars = ["인명피해\n환산액\n(Ec×VSL)", "재산피해\n추정액", "MPL\n(합산)"]
    vals = [casualty_value, property_damage_musd, mpl]
    colors = [PRIMARY, AMBER, RED]
    b = ax.bar(bars, vals, color=colors, zorder=3, width=0.55)
    _style(ax)
    ax.set_ylabel("$M")
    ax.axhspan(100, 500, color=GREEN, alpha=0.08, zorder=0)
    ax.annotate("FAA 통상 MPL 구간\n($100M~$500M)", xy=(2.35, 480), fontsize=7.5, color=GREEN, ha="left")
    for rect, v in zip(b, vals):
        ax.annotate(f"${v:,.0f}M", (rect.get_x() + rect.get_width() / 2, v), fontsize=9,
                    ha="center", va="bottom", color=INK)
    ax.set_title(f"MPL = Ec×VSL + 재산피해 = ${mpl:,.0f}M")

    plt.tight_layout()
    plt.show()

    print(f"[입력] 실패확률 P={fail_prob_pct:.1f}%, 사상면적 A_c={casualty_area_km2:.2f}km², "
          f"인구밀도 ρ={pop_density:.0f}명/km², VSL=${vsl_musd:.1f}M/인, 재산피해=${property_damage_musd:.0f}M")
    print(f"[산출] Ec = {Ec:.4f}  →  인명피해 환산액 ${casualty_value:,.1f}M  →  MPL ≈ ${mpl:,.0f}M")
    print("  → 해상 발사(ρ≈0)는 인명피해 환산액이 0에 수렴 — 재산피해만으로 MPL이 결정되는 이유입니다.")
    print("  → 내륙·근접 도심 낙하구역일수록 ρ가 커져 MPL이 슬라이드9의 상단($5억)에 근접합니다.")


def ec_mpl_explorer():
    """제3자 배상책임보험의 필요금액을 정하는 기대사상자수(Ec) 방법론을 직접 조작 —
    실패확률·사상면적·낙하구역 인구밀도가 MPL(최대예상손실)에 미치는 영향을 확인한다."""
    interact(_plot_ec_mpl,
             fail_prob_pct=FloatSlider(value=3.0, min=1.0, max=8.0, step=0.5,
                                        description="실패확률(%)",
                                        style={"description_width": "120px"}, layout={"width": "480px"}),
             casualty_area_km2=FloatSlider(value=1.0, min=0.2, max=5.0, step=0.2,
                                            description="사상면적 Ac(km²)",
                                            style={"description_width": "120px"}, layout={"width": "480px"}),
             pop_density=FloatSlider(value=50, min=0, max=500, step=10,
                                      description="인구밀도 ρ(명/km²)",
                                      style={"description_width": "120px"}, layout={"width": "480px"}),
             vsl_musd=FloatSlider(value=10.0, min=5.0, max=15.0, step=0.5,
                                   description="VSL($M/인)",
                                   style={"description_width": "120px"}, layout={"width": "480px"}),
             property_damage_musd=FloatSlider(value=120, min=20, max=300, step=10,
                                               description="재산피해 추정($M)",
                                               style={"description_width": "120px"}, layout={"width": "480px"}))


# ============================================================
# 19. 순보험료 → 영업보험료 계산기 (7주차 §PART4 MATH 2)
# ============================================================
def _plot_premium_waterfall(mpl_musd, fail_prob_pct, risk_margin_pct, expense_ratio_pct,
                             reinsurance_cost_pct, profit_margin_pct):
    P = fail_prob_pct / 100
    pure_premium = P * mpl_musd
    loaded = pure_premium * (1 + risk_margin_pct / 100 + expense_ratio_pct / 100)
    denom = 1 - reinsurance_cost_pct / 100 - profit_margin_pct / 100
    denom = max(denom, 0.05)
    gross_premium = loaded / denom
    rate_on_line = gross_premium / mpl_musd * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    stages = ["순보험료\n(P×MPL)", "+리스크마진", "+사업비율", "÷(1−재보험\n−이윤율)"]
    stage_vals = [pure_premium,
                  pure_premium * (1 + risk_margin_pct / 100),
                  loaded,
                  gross_premium]
    colors = [PRIMARY, "#5B9BD5", AMBER, RED]
    ax.bar(stages, stage_vals, color=colors, zorder=3, width=0.6)
    _style(ax)
    for i, v in enumerate(stage_vals):
        ax.annotate(f"${v:,.1f}M", (i, v), fontsize=9, ha="center", va="bottom", color=INK)
    ax.set_ylabel("$M")
    ax.set_title(f"순보험료 ${pure_premium:,.1f}M → 영업보험료 ${gross_premium:,.1f}M")

    ax = axes[1]
    ax.bar(["요율(Rate on Line)"], [rate_on_line], color=AMBER, width=0.4, zorder=3)
    ax.axhspan(1.5, 2.0, color=GREEN, alpha=0.12, zorder=0)
    ax.annotate("슬라이드13 통상 TPL 요율\n(MPL의 1.5~2.0%)", xy=(0.28, 1.9), fontsize=8, color=GREEN)
    _style(ax)
    ax.set_ylabel("영업보험료 / MPL (%)")
    ax.set_ylim(0, max(rate_on_line * 1.3, 3))
    ax.set_title(f"= {rate_on_line:.2f}%")

    plt.tight_layout()
    plt.show()

    print(f"[입력] MPL=${mpl_musd:,.0f}M, 실패확률={fail_prob_pct:.1f}%, 리스크마진={risk_margin_pct:.0f}%, "
          f"사업비율={expense_ratio_pct:.0f}%, 재보험비용률={reinsurance_cost_pct:.0f}%, 이윤율={profit_margin_pct:.0f}%")
    print(f"[산출] 순보험료 ${pure_premium:,.2f}M → 영업보험료 ${gross_premium:,.2f}M "
          f"(MPL의 {rate_on_line:.2f}%)")
    if rate_on_line > 2.0:
        print("  → 슬라이드13의 통상 요율대(1.5~2.0%)를 상회 — 신형 기체·경화시장 국면을 가정한 결과일 수 있습니다.")
    elif rate_on_line < 1.5:
        print("  → 통상 요율대보다 낮음 — 성숙 기체·연화시장(soft market) 가정에 해당합니다.")
    else:
        print("  → 슬라이드13이 제시하는 통상 요율대(1.5~2.0%) 안에 위치합니다.")


def insurance_premium_explorer():
    """순보험료(P×MPL)에서 리스크마진·사업비율·재보험비용률·이윤율을 반영한 영업보험료까지의
    산정 절차를 워터폴로 확인하고, 결과 요율을 시장 통상 요율대(1.5~2.0%)와 비교한다."""
    interact(_plot_premium_waterfall,
             mpl_musd=FloatSlider(value=250, min=100, max=500, step=10,
                                   description="MPL($M)",
                                   style={"description_width": "130px"}, layout={"width": "460px"}),
             fail_prob_pct=FloatSlider(value=1.5, min=1.0, max=6.0, step=0.5,
                                        description="실패확률(%)",
                                        style={"description_width": "130px"}, layout={"width": "460px"}),
             risk_margin_pct=FloatSlider(value=10, min=0, max=60, step=5,
                                          description="리스크마진(%)",
                                          style={"description_width": "130px"}, layout={"width": "460px"}),
             expense_ratio_pct=FloatSlider(value=8, min=5, max=25, step=1,
                                            description="사업비율(%)",
                                            style={"description_width": "130px"}, layout={"width": "460px"}),
             reinsurance_cost_pct=FloatSlider(value=8, min=0, max=20, step=1,
                                               description="재보험비용률(%)",
                                               style={"description_width": "130px"}, layout={"width": "460px"}),
             profit_margin_pct=FloatSlider(value=6, min=0, max=15, step=1,
                                            description="이윤율(%)",
                                            style={"description_width": "130px"}, layout={"width": "460px"}))


# ============================================================
# 20. 발사보험 시장 경화(hardening) 탐색기 (7주차 §PART4 MARKET)
# ============================================================
def _plot_insurance_market_cycle(loss_ratio_pct, base_capacity_musd):
    hardening = max(0, (loss_ratio_pct - 100) / 100)
    rate_multiplier = 1 + hardening * 0.8
    capacity = base_capacity_musd * (1 - min(hardening * 0.3, 0.5))

    loss_ratios = np.linspace(50, 250, 200)
    hardening_curve = np.clip((loss_ratios - 100) / 100, 0, None)
    rate_curve = 1 + hardening_curve * 0.8
    capacity_curve = base_capacity_musd * (1 - np.clip(hardening_curve * 0.3, 0, 0.5))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    ax.plot(loss_ratios, rate_curve, color=RED, linewidth=2.2, zorder=3)
    ax.plot(loss_ratio_pct, rate_multiplier, "o", color=RED, markersize=10, zorder=5)
    ax.axvline(179, color=INK_MUTED, linestyle=":", linewidth=1.2)
    ax.annotate("2023년 실적\n(179%)", (179, 1.1), fontsize=8, color=INK_MUTED)
    _style(ax)
    ax.set_xlabel("손해율 (청구/보험료, %)")
    ax.set_ylabel("요율 배수 (기준=1.0)")
    ax.set_title(f"손해율 {loss_ratio_pct:.0f}% → 요율 ×{rate_multiplier:.2f}")

    ax = axes[1]
    ax.plot(loss_ratios, capacity_curve, color=PRIMARY, linewidth=2.2, zorder=3)
    ax.plot(loss_ratio_pct, capacity, "o", color=PRIMARY, markersize=10, zorder=5)
    _style(ax)
    ax.set_xlabel("손해율 (%)")
    ax.set_ylabel("단일리스크 조달가능 용량($M)")
    ax.axhspan(300, 325, color=GREEN, alpha=0.12, zorder=0)
    ax.annotate("2025~26 실측 구간\n($300~325M)", xy=(160, 330), fontsize=8, color=GREEN)
    ax.set_title(f"조달가능 용량 ≈ ${capacity:,.0f}M")

    plt.tight_layout()
    plt.show()

    print(f"[입력] 손해율={loss_ratio_pct:.0f}%, 기준용량=${base_capacity_musd:,.0f}M")
    print(f"[산출] 요율 배수 ×{rate_multiplier:.2f} · 조달가능 용량 ≈ ${capacity:,.0f}M")
    print("  → 손해율이 100%를 넘는 해(2023년 179%)의 이듬해는 요율이 오르고 용량이 줄어드는 "
          "'경화(hardening)' 국면이 나타납니다.")
    print("  → 용량 부족은 대형 임무(다수 위성 라이드셰어 등)의 보험 미가입(무보험 발사) 유인으로 이어집니다.")


def insurance_market_cycle_explorer():
    """2023년 손해율 179%(보험료 $557M vs 청구 $995M) 이후 나타난 발사보험 시장의
    경화(hardening) 메커니즘 — 손해율이 요율과 조달가능 용량에 미치는 영향을 단순 모델로 확인한다."""
    interact(_plot_insurance_market_cycle,
             loss_ratio_pct=FloatSlider(value=179, min=50, max=250, step=5,
                                         description="손해율(%)",
                                         style={"description_width": "120px"}, layout={"width": "480px"}),
             base_capacity_musd=FloatSlider(value=400, min=300, max=500, step=10,
                                             description="연화시장 기준용량($M)",
                                             style={"description_width": "120px"}, layout={"width": "480px"}))

# ============================================================
# 21. 미국 우주수송 법제 변천 타임라인 (9주차 §PART1, 정적 다이어그램)
# ============================================================
def us_space_law_timeline():
    """1958년 NASA법(국가독점)부터 2026년 상업화 행정명령·Part 450 전면시행까지,
    미국 우주수송 법제가 단계적으로 개방되어 온 40년의 흐름을 보여준다 (정적 다이어그램)."""
    events = [
        ("1958", "NASA법\n국가독점 출발", INK_MUTED),
        ("1984", "CSLA\n민간발사 '허용'", PRIMARY),
        ("1986", "챌린저 참사\n왕복선 상업탑재 금지→실질개방", RED),
        ("2004", "CSLAA\n유인 상업비행 규제체계", PRIMARY),
        ("2006", "COTS 개시\n정부 앵커계약 모델", AMBER),
        ("2015", "CSLCA\n우주자원 소유권 인정", PRIMARY),
        ("2021", "Part 450\n발사허가 통합·간소화", AMBER),
        ("2025~26", "Golden Dome ·\n상업화 행정명령", RED),
    ]
    fig, ax = plt.subplots(figsize=(13, 3.6))
    xs = list(range(len(events)))
    ax.plot(xs, [0] * len(xs), color="#B0B0B0", linewidth=2, zorder=1)
    for i, (x, (yr, label, color)) in enumerate(zip(xs, events)):
        ax.scatter(x, 0, s=200, color=color, zorder=3, edgecolor="white", linewidth=1.5)
        y_off = 24 if i % 2 == 0 else -24
        va = "bottom" if i % 2 == 0 else "top"
        ax.annotate(yr, (x, 0), fontsize=10.5, fontweight="bold", color=INK,
                    xytext=(0, y_off), textcoords="offset points", ha="center", va=va)
        ax.annotate(label, (x, 0), fontsize=8, color=INK_MUTED,
                    xytext=(0, y_off + (14 if i % 2 == 0 else -14)), textcoords="offset points",
                    ha="center", va=va)
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.set_ylim(-2.0, 2.0)
    ax.axis("off")
    ax.set_title("미국 우주수송 법제 40년 — '허용 → 위기가 강제한 개방 → 새 영역 규제화'의 반복",
                  fontsize=12, color=INK, pad=8)
    plt.tight_layout()
    plt.show()
    print("10~15년마다 '허용 → 위기가 강제한 실질개방 → 새 영역 규제화'의 사이클이 반복됩니다.")
    print("2025~26년의 안보확장(Golden Dome)·규제완화 행정명령도 이 사다리의 최신 단입니다.")


# ============================================================
# 22. Cost-Plus vs 고정가 마일스톤 계약 탐색기 (9주차 §PART2 COTS/ROLE SHIFT)
# ============================================================
def _plot_contract_comparison(target_cost_musd, cost_overrun_pct, fee_pct, milestone_margin_pct):
    actual_cost = target_cost_musd * (1 + cost_overrun_pct / 100)
    fee = target_cost_musd * fee_pct / 100
    fixed_price = target_cost_musd * (1 + milestone_margin_pct / 100)

    overruns = np.linspace(0, 100, 200)
    actual_costs = target_cost_musd * (1 + overruns / 100)

    gov_cost_plus = actual_costs + fee
    gov_fixed = np.full_like(overruns, fixed_price)

    contractor_profit_cost_plus = np.full_like(overruns, fee)
    contractor_profit_fixed = fixed_price - actual_costs

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.plot(overruns, gov_cost_plus, color=RED, linewidth=2.2, label="Cost-Plus (구모델)", zorder=3)
    ax.plot(overruns, gov_fixed, color=PRIMARY, linewidth=2.2, label="고정가 마일스톤 (COTS)", zorder=3)
    ax.plot(cost_overrun_pct, actual_cost + fee, "o", color=RED, markersize=9, zorder=5)
    ax.plot(cost_overrun_pct, fixed_price, "o", color=PRIMARY, markersize=9, zorder=5)
    ax.set_facecolor("white")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel("실제비용 초과율(%)")
    ax.set_ylabel("정부 지불액($M)")
    ax.set_title("정부 부담액 — 초과분을 누가 흡수하는가")
    ax.legend(fontsize=8.5, frameon=False)

    ax = axes[1]
    ax.plot(overruns, contractor_profit_cost_plus, color=RED, linewidth=2.2, label="Cost-Plus (구모델)", zorder=3)
    ax.plot(overruns, contractor_profit_fixed, color=PRIMARY, linewidth=2.2, label="고정가 마일스톤 (COTS)", zorder=3)
    ax.axhline(0, color=INK_MUTED, linestyle=":", linewidth=1)
    ax.plot(cost_overrun_pct, fee, "o", color=RED, markersize=9, zorder=5)
    ax.plot(cost_overrun_pct, fixed_price - actual_cost, "o", color=PRIMARY, markersize=9, zorder=5)
    ax.set_facecolor("white")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel("실제비용 초과율(%)")
    ax.set_ylabel("기업 손익($M)")
    ax.set_title("기업 손익 — 누가 리스크를 지는가")
    ax.legend(fontsize=8.5, frameon=False)

    plt.tight_layout()
    plt.show()

    print(f"[입력] 목표비용=${target_cost_musd:,.0f}M, 실제비용 초과율={cost_overrun_pct:.0f}%, "
          f"cost-plus 수수료율={fee_pct:.0f}%, 고정가 마진율={milestone_margin_pct:.0f}%")
    print(f"[Cost-Plus] 정부지불 = ${actual_cost + fee:,.1f}M (실제비용을 그대로 반영) · 기업이익 = ${fee:,.1f}M (항상 보장)")
    print(f"[고정가 마일스톤] 정부지불 = ${fixed_price:,.1f}M (고정) · 기업손익 = ${fixed_price - actual_cost:,.1f}M")
    if fixed_price - actual_cost < 0:
        print("  → 초과율이 커지면 고정가 계약에서는 기업이 손실을 직접 부담합니다 — COTS가 SpaceX에 큰 리스크였던 이유입니다.")
    else:
        print("  → 이 초과율에서는 고정가 계약에서도 기업이 이익을 유지합니다.")
    print("  → Cost-Plus는 정부가 예산초과 리스크를 흡수, 고정가 마일스톤은 기업이 리스크를 흡수 — "
          "COTS가 '정부 예산 예측가능성'과 '기업의 비용절감 유인'을 동시에 얻은 이유입니다.")


def cost_plus_vs_fixed_price_explorer():
    """Apollo~Shuttle 시대의 Cost-Plus 계약과, COTS 이후의 고정가 마일스톤 계약이
    비용 초과 리스크를 정부와 기업 사이에 어떻게 다르게 배분하는지 직접 조작해 비교한다."""
    interact(_plot_contract_comparison,
             target_cost_musd=FloatSlider(value=300, min=50, max=1000, step=50,
                                           description="목표비용($M)",
                                           style={"description_width": "120px"}, layout={"width": "480px"}),
             cost_overrun_pct=FloatSlider(value=25, min=0, max=100, step=5,
                                           description="실제비용 초과율(%)",
                                           style={"description_width": "120px"}, layout={"width": "480px"}),
             fee_pct=FloatSlider(value=10, min=5, max=20, step=1,
                                  description="Cost-Plus 수수료율(%)",
                                  style={"description_width": "120px"}, layout={"width": "480px"}),
             milestone_margin_pct=FloatSlider(value=10, min=0, max=30, step=5,
                                               description="고정가 마진율(%)",
                                               style={"description_width": "120px"}, layout={"width": "480px"}))


# ============================================================
# 23. 우주투자 자본집중도 (9주차 §PART3 TREND, 정적 다이어그램)
# ============================================================
def investment_concentration_chart():
    """2025~26년 미국 우주투자 통계 — 발사서비스가 딜 건수 비중보다 자본 비중이 훨씬 크고,
    $50M 이상 메가라운드가 소수 건수로 자본 대부분을 차지하는 자본집중 구조를 보여준다
    (정적 다이어그램 — 슬라이드13 수치 기반)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    cats = ["딜 건수 비중", "자본 비중"]
    vals = [17, 30]
    ax.bar(cats, vals, color=[PRIMARY, AMBER], width=0.5, zorder=3)
    for i, v in enumerate(vals):
        ax.annotate(f"{v}%", (i, v), fontsize=12, ha="center", va="bottom", fontweight="bold", color=INK)
    ax.set_facecolor("white")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_ylabel("%")
    ax.set_ylim(0, 40)
    ax.set_title("발사서비스 부문 — 딜 건수 대비 자본 쏠림")

    ax = axes[1]
    cats2 = ["딜 건수 비중\n($50M+ 라운드)", "자본 비중\n($50M+ 라운드)"]
    vals2 = [45, 92]
    ax.bar(cats2, vals2, color=[PRIMARY, RED], width=0.5, zorder=3)
    for i, v in enumerate(vals2):
        ax.annotate(f"{v}%", (i, v), fontsize=12, ha="center", va="bottom", fontweight="bold", color=INK)
    ax.set_facecolor("white")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.set_title("메가라운드($50M+) — 건수는 45%, 자본은 92%")

    plt.tight_layout()
    plt.show()

    print("발사서비스는 딜 건수 비중(17%)보다 자본 비중(30%)이 훨씬 큽니다 — 소수의 대형 라운드에 자본이 집중되는")
    print("자본집약 산업 특성을 보여줍니다 (12주차 학습곡선·발사빈도 논의와 연결).")
    print("$50M 이상 라운드는 딜 건수의 45%에 불과하지만 자본의 92%를 차지 — 사실상 '메가라운드 시장'입니다.")


# ============================================================
# 24. Golden Dome 예산 브레이크다운 (9주차 §PART4 BUDGET, 정적 다이어그램)
# ============================================================
def golden_dome_budget_chart():
    """Golden Dome 미사일방어 구상이 견인한 FY2026~27 우주군 예산 규모를 보여준다
    (정적 다이어그램 — 슬라이드17 수치 기반, 추정치는 발표기관에 따라 편차가 있음에 유의)."""
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    labels = ["FY26 우주군\n기본예산", "SBI 프로토타입\n계약총액(12개사)", "FY27 Golden Dome\n요청예산", "FY27 중\n기본예산분"]
    values = [26.3, 3.2, 17.5, 0.398]
    colors = [PRIMARY, AMBER, RED, "#D9A441"]
    bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)
    for rect, v in zip(bars, values):
        ax.annotate(f"${v:,.1f}B", (rect.get_x() + rect.get_width() / 2, v), fontsize=10,
                    ha="center", va="bottom", fontweight="bold", color=INK)
    ax.set_facecolor("white")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_ylabel("$B (십억 달러)")
    ax.set_title("Golden Dome이 견인한 우주군 예산 — FY26 대비 +40% 증액")
    plt.tight_layout()
    plt.show()

    print("FY2026 우주군 기본예산은 전년 대비 약 +40% 증액되었고, 이 증가분의 상당 부분이 Golden Dome 관련입니다.")
    print("CBO는 20년 총 비용을 약 $1.2조로 추정하지만 국방부는 이 추정치에 이의를 제기 — 추정치 간 편차가 큽니다.")
    print("SBI 프로토타입 계약은 12개사에 Other Transaction Authority(OTA)로 발주되어, COTS의 '경쟁유지' 철학이")
    print("안보 조달에도 이식된 사례로 해석됩니다.")

# ============================================================
# 25. Brander-Spencer 전략적 무역정책 탐색기 (10주차 §PART1 ECONOMICS)
# ============================================================
def _plot_brander_spencer(market_size, demand_slope, marginal_cost, subsidy):
    A, b, c, s = market_size, demand_slope, marginal_cost, subsidy
    X = A - c
    s_grid = np.linspace(0, c * 0.95, 200)

    qH = (X + 2 * s_grid) / (3 * b)
    qF = (X - s_grid) / (3 * b)
    qF = np.clip(qF, 0, None)
    price = A - b * (qH + qF)

    firm_profit_H = (price - c + s_grid) * qH        # 보조금 포함 기업 회계이윤
    national_welfare_H = (price - c) * qH             # 보조금 비용 차감한 국가후생
    profit_F = (price - c) * qF                        # 해외기업 이윤

    idx_sel = np.argmin(np.abs(s_grid - s))
    idx_opt = np.argmax(national_welfare_H)
    s_opt = s_grid[idx_opt]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.plot(s_grid, firm_profit_H, color=AMBER, linewidth=2, label="자국기업 회계이윤(보조금 포함)", zorder=3)
    ax.plot(s_grid, national_welfare_H, color=PRIMARY, linewidth=2.4, label="자국 국가후생(보조금 비용 차감)", zorder=3)
    ax.plot(s_grid, profit_F, color=RED, linewidth=2, linestyle="--", label="해외기업 이윤", zorder=3)
    ax.axvline(s_opt, color=PRIMARY, linestyle=":", linewidth=1.3)
    ax.plot(s_grid[idx_sel], national_welfare_H[idx_sel], "o", color=PRIMARY, markersize=10, zorder=5)
    ax.annotate(f"후생최대화 보조금\ns*≈{s_opt:.1f}", (s_opt, national_welfare_H[idx_opt]), fontsize=8,
                color=PRIMARY, xytext=(8, -18), textcoords="offset points")
    ax.set_facecolor("white")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel("자국기업 보조금 s (단위당)")
    ax.set_ylabel("이윤 / 후생")
    ax.set_title("보조금이 이윤을 '이전'시키는가")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    ax = axes[1]
    ax.plot(s_grid, qH, color=PRIMARY, linewidth=2.2, label="자국기업 생산량 qH", zorder=3)
    ax.plot(s_grid, qF, color=RED, linewidth=2.2, label="해외기업 생산량 qF", zorder=3)
    ax.axvline(s, color=INK_MUTED, linestyle=":", linewidth=1.2)
    ax.set_facecolor("white")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel("자국기업 보조금 s (단위당)")
    ax.set_ylabel("생산량")
    ax.set_title("보조금이 시장점유율을 이전시킨다")
    ax.legend(fontsize=8.5, frameon=False)

    plt.tight_layout()
    plt.show()

    print(f"[입력] 시장규모 A={A}, 수요기울기 b={b}, 한계비용 c={c}, 보조금 s={s}")
    print(f"[산출@s={s}] 자국생산 qH={qH[idx_sel]:.2f} · 해외생산 qF={qF[idx_sel]:.2f} · 시장가격={price[idx_sel]:.2f}")
    print(f"  자국기업 회계이윤 = {firm_profit_H[idx_sel]:.2f} · 자국 국가후생(보조금 비용 차감) = {national_welfare_H[idx_sel]:.2f} · "
          f"해외기업 이윤 = {profit_F[idx_sel]:.2f}")
    print(f"  → 국가후생을 극대화하는 보조금은 s*≈{s_opt:.1f}입니다. s=0(자유무역)보다 국가후생이 "
          f"{'높습니다' if national_welfare_H[idx_opt] > national_welfare_H[0] else '낮습니다'} — "
          "이것이 Brander-Spencer 모형이 자유무역의 표준 결론을 뒤집는 지점입니다.")
    print("  주의: 이 결과는 '정부가 시장구조·비용함수를 정확히 안다'는 강한 가정에 의존합니다 — "
          "슬라이드23의 공공선택론 비판이 겨냥하는 지점이 바로 이 가정입니다.")


def brander_spencer_explorer():
    """소수 기업만 생존 가능한 과점시장(복점)에서, 자국기업에 대한 보조금이 해외기업의 시장점유율을
    '이전'시켜 국가후생을 높일 수 있다는 Brander-Spencer(1985) 전략적 무역정책 모형을
    Cournot 복점 게임으로 직접 조작해 확인한다."""
    interact(_plot_brander_spencer,
             market_size=FloatSlider(value=100, min=60, max=200, step=10,
                                      description="시장규모 A",
                                      style={"description_width": "120px"}, layout={"width": "480px"}),
             demand_slope=FloatSlider(value=1.0, min=0.5, max=2.0, step=0.1,
                                       description="수요기울기 b",
                                       style={"description_width": "120px"}, layout={"width": "480px"}),
             marginal_cost=FloatSlider(value=40, min=20, max=60, step=5,
                                        description="한계비용 c",
                                        style={"description_width": "120px"}, layout={"width": "480px"}),
             subsidy=FloatSlider(value=10, min=0, max=40, step=2,
                                  description="보조금 s",
                                  style={"description_width": "120px"}, layout={"width": "480px"}))


# ============================================================
# 26. 5개국 정책동인 레이더 (10주차 §PART5·6, 정적 다이어그램)
# ============================================================
def national_strategy_radar():
    """유럽·일본·중국·인도·러시아 5개국의 발사체 전략을 '경제/안보/위신' 세 렌즈의 비중으로
    겹쳐 보여준다. 점수는 강의 슬라이드17~21의 질적 서술을 예시적으로 수치화한 것으로,
    정밀한 계량치가 아니라 상대적 비중을 직관적으로 비교하기 위한 것이다 (정적 다이어그램)."""
    countries = {
        "유럽":   {"경제": 2, "안보": 3, "위신": 4, "color": PRIMARY},
        "일본":   {"경제": 4, "안보": 3, "위신": 2, "color": AMBER},
        "중국":   {"경제": 4, "안보": 5, "위신": 5, "color": RED},
        "인도":   {"경제": 3, "안보": 1, "위신": 5, "color": GREEN},
        "러시아": {"경제": 1, "안보": 2, "위신": 2, "color": "#8E6C8A"},
    }
    axes_labels = ["경제", "안보", "위신"]
    n = len(axes_labels)
    angles = [i / n * 2 * np.pi for i in range(n)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.2, 7.2), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=12)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8, color=INK_MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)

    for name, scores in countries.items():
        vals = [scores[a] for a in axes_labels]
        vals += vals[:1]
        ax.plot(angles, vals, color=scores["color"], linewidth=2, label=name, zorder=3)
        ax.fill(angles, vals, color=scores["color"], alpha=0.06, zorder=2)

    ax.set_title("5개국 발사체 전략의 1·2차 동인 비중 (예시적 스코어링)", fontsize=12, color=INK, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10, frameon=False)
    plt.tight_layout()
    plt.show()

    print("중국은 세 렌즈 모두에서 높은 점수 — '경제+안보 완전결합, 위신(우주굴기)'이라는 슬라이드19의 서술과 일치합니다.")
    print("인도는 위신이 압도적으로 높고 안보는 낮음 — 안보 앵커 없이 순수 상업·위신 동력만으로 움직이는 구조입니다.")
    print("러시아는 세 축 모두 낮게 나타나는데, 이는 '경로의존·매몰비용'이라는 제4의 동인이 이 프레임워크 바깥에서")
    print("작동하고 있음을 시사합니다 — 세 렌즈로 설명되지 않는 예외 사례라는 것 자체가 중요한 발견입니다.")


# ============================================================
# 27. 비교산업 3렌즈 히트맵 (10주차 §PART4, 정적 다이어그램)
# ============================================================
def comparative_industries_heatmap():
    """조선·반도체·전투기·원자력·발사체 5개 산업이 '경제 스필오버·안보 외부성·국가위신'
    세 렌즈에 얼마나 강하게 걸쳐 있는지를 히트맵으로 비교한다. 점수는 슬라이드13~15의
    질적 서술을 예시적으로 수치화한 것이다 (정적 다이어그램)."""
    industries = ["조선", "반도체", "전투기", "원자력", "발사체"]
    lenses = ["경제(스필오버)", "안보(외부성)", "위신"]
    scores = np.array([
        [3, 4, 2],   # 조선
        [4, 5, 3],   # 반도체
        [2, 5, 3],   # 전투기
        [3, 4, 2],   # 원자력
        [4, 5, 4],   # 발사체
    ])

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    im = ax.imshow(scores, cmap="YlOrRd", vmin=0, vmax=5, aspect="auto")

    ax.set_xticks(range(len(lenses)))
    ax.set_xticklabels(lenses, fontsize=10)
    ax.set_yticks(range(len(industries)))
    ax.set_yticklabels(industries, fontsize=11)

    for i in range(len(industries)):
        for j in range(len(lenses)):
            v = scores[i, j]
            txt_color = "white" if v >= 4 else INK
            ax.text(j, i, str(v), ha="center", va="center", fontsize=13,
                    fontweight="bold", color=txt_color)

    for i, name in enumerate(industries):
        if name == "발사체":
            ax.add_patch(plt.Rectangle((-0.5, i - 0.5), len(lenses), 1, fill=False,
                                        edgecolor=INK, linewidth=2.5, zorder=5))

    ax.set_title("5개 산업 × 3개 렌즈 — 발사체가 유일하게 세 축 모두에서 높은 산업",
                  fontsize=11.5, color=INK, pad=10)
    plt.tight_layout()
    plt.show()

    print("발사체(굵은 테두리)는 경제·안보·위신 세 렌즈 모두에서 높은 점수를 받는 유일한 산업입니다.")
    print("반도체·전투기는 안보 축에서, 조선·원자력은 경제·안보 결합에서 강하지만, 위신 축까지 모두 강한 것은")
    print("발사체가 유일합니다 — 슬라이드15가 '두 축이 동시에 작동하는 유일한 산업'이라 지목하는 이유입니다.")

# ============================================================
# 28. 스타십 반복시험(IFT) 캠페인 타임라인 (13주차 §PART1 TEST CAMPAIGN, 정적 다이어그램)
# ============================================================
def starship_test_campaign_timeline():
    """2023년 IFT-1부터 2026년 IFT-12(V3)까지, 스타십의 반복시험(iterate-and-fly) 개발
    캠페인이 실패를 데이터로 축적해 온 흐름을 보여준다 (정적 다이어그램 — 2026.7 기준)."""
    events = [
        ("2023\nIFT-1~2", "초기 실패기\n발사대 파손·단분리 실패\n(33기 동시점화 데이터 확보)", RED),
        ("2024\nIFT-3~6", "이정표 달성기\n준궤도 재진입·인도양 연착수\n첫 부스터 타워 캐치 성공", AMBER),
        ("2025\nIFT-7~11", "V2 성숙기\n블록2 상단·재비행 부스터\n성공·실패 혼재", AMBER),
        ("2026.5\nIFT-12(V3)", "V3 진입기\n상단 상승 성공·궤도상 재점화 실패\n부스트백 실패", PRIMARY),
        ("2026.7~\nFlight 13~14", "궤도 진입 시도\n(예고)\n'the big one'", GREEN),
    ]
    fig, ax = plt.subplots(figsize=(13, 3.8))
    xs = list(range(len(events)))
    ax.plot(xs, [0] * len(xs), color="#B0B0B0", linewidth=2, zorder=1)
    for i, (x, (yr, label, color)) in enumerate(zip(xs, events)):
        ax.scatter(x, 0, s=200, color=color, zorder=3, edgecolor="white", linewidth=1.5)
        y_off = 22 if i % 2 == 0 else -22
        va = "bottom" if i % 2 == 0 else "top"
        ax.annotate(yr, (x, 0), fontsize=10, fontweight="bold", color=INK,
                    xytext=(0, y_off), textcoords="offset points", ha="center", va=va)
        ax.annotate(label, (x, 0), fontsize=7.8, color=INK_MUTED,
                    xytext=(0, y_off + (16 if i % 2 == 0 else -16)), textcoords="offset points",
                    ha="center", va=va)
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.set_ylim(-2.2, 2.2)
    ax.axis("off")
    ax.set_title("스타십 반복시험(Iterate-and-Fly) 캠페인 — 12회 비행, 아직 완전 궤도 비행은 없음(2026.7 기준)",
                  fontsize=11.5, color=INK, pad=8)
    plt.tight_layout()
    plt.show()
    print("전통적 개발이라면 용납되지 않았을 실패들이 설계 수정의 원료가 되는 것이 반복시험 방법론의 핵심입니다.")
    print("2026.7 현재까지 12회 비행 모두 준궤도 프로파일 — 완전한 궤도 비행은 아직 실증되지 않았습니다.")


# ============================================================
# 29. 스타십 재사용 횟수별 $/kg 탐색기 (13주차 §PART2 COST DISCONTINUITY)
# ============================================================
def _plot_starship_reuse_cost(reuse_count):
    n_grid = np.linspace(1, 70, 200)
    cost_grid = 280 * n_grid ** (-0.6325)
    cost_sel = 280 * reuse_count ** (-0.6325)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.plot(n_grid, cost_grid, color=PRIMARY, linewidth=2.4, zorder=3)
    ax.axvspan(6, 6, color=AMBER, alpha=0)
    ax.axhspan(78, 94, color=AMBER, alpha=0.12, zorder=0)
    ax.annotate("부분재사용(6회)\n$78~94/kg", xy=(9, 88), fontsize=8, color=AMBER)
    ax.axhspan(13, 32, color=GREEN, alpha=0.12, zorder=0)
    ax.annotate("고재사용(20~70회)\n$13~32/kg", xy=(30, 40), fontsize=8, color=GREEN)
    ax.plot(reuse_count, cost_sel, "o", color=RED, markersize=11, zorder=5)
    ax.set_yscale("log")
    ax.set_facecolor("white")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel("동일 기체 누적 재사용 횟수")
    ax.set_ylabel("$/kg (로그축, 원가 추정)")
    ax.set_title(f"재사용 {reuse_count:.0f}회 시 원가 ≈ ${cost_sel:.0f}/kg")

    ax = axes[1]
    gens = ["우주왕복선\n(STS)", "Falcon 9\n(재사용)", f"Starship\n({reuse_count:.0f}회 재사용)"]
    vals = [55000, 2700, cost_sel]
    colors = ["#8B5E3C", PRIMARY, RED]
    bars = ax.bar(gens, vals, color=colors, width=0.55, zorder=3)
    ax.set_yscale("log")
    for rect, v in zip(bars, vals):
        ax.annotate(f"${v:,.0f}/kg", (rect.get_x() + rect.get_width() / 2, v), fontsize=9,
                    ha="center", va="bottom", fontweight="bold", color=INK)
    ax.set_facecolor("white")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_ylabel("$/kg (로그축)")
    ax.set_title("세대 간 격차는 '배수'가 아니라 '자릿수'")

    plt.tight_layout()
    plt.show()

    print(f"[재사용 {reuse_count:.0f}회] 원가 추정 ≈ ${cost_sel:.0f}/kg (설계상 잠재력 — 실증 전 목표치)")
    print(f"  우주왕복선(STS) 대비 약 {55000/cost_sel:,.0f}배, Falcon 9(재사용) 대비 약 {2700/cost_sel:,.0f}배 낮은 원가 수준입니다.")
    print("  주의: 원가와 '가격'은 별개입니다 — 성숙기 회당 가격은 $10M 이하 목표로 제시되는데, 이는 원가가 아니라")
    print("  시장지배력에 따른 마진의 함수입니다(12주차 PpF 논리). 모든 수치는 실증 전 추정치로 다룰 것.")


def starship_reuse_cost_explorer():
    """스타십의 $/kg 원가가 동일 기체의 누적 재사용 횟수에 따라 어떻게 하락하는지,
    그리고 그 하락이 이전 세대 발사체 대비 얼마나 '자릿수' 단위의 격차를 만드는지 확인한다.
    (모든 수치는 강의 슬라이드8 추정치에 기반한 설계상 잠재력 — 실증 전 값)"""
    interact(_plot_starship_reuse_cost,
             reuse_count=FloatSlider(value=6, min=1, max=70, step=1,
                                      description="재사용 횟수",
                                      style={"description_width": "120px"}, layout={"width": "480px"}))


# ============================================================
# 30. 궤도 급유 탱커 발사 횟수 계산기 (13주차 §PART1 ORBITAL REFILLING)
# ============================================================
def _plot_orbital_refueling(target_prop_t, tanker_delivered_t, boiloff_loss_pct):
    effective_per_flight = tanker_delivered_t * (1 - boiloff_loss_pct / 100)
    n_flights = int(np.ceil(target_prop_t / max(effective_per_flight, 1)))

    loss_grid = np.linspace(0, 30, 100)
    eff_grid = tanker_delivered_t * (1 - loss_grid / 100)
    n_grid = np.ceil(target_prop_t / np.maximum(eff_grid, 1))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    labels = [f"탱커 #{i+1}" for i in range(min(n_flights, 15))]
    vals = [effective_per_flight] * min(n_flights, 15)
    ax.bar(labels, vals, color=PRIMARY, width=0.6, zorder=3)
    ax.set_facecolor("white")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(axis="x", labelrotation=60, labelsize=7)
    ax.set_ylabel("선체당 실질 이송량(t)")
    extra = f" (총 {n_flights}회 중 15회만 표시)" if n_flights > 15 else ""
    ax.set_title(f"필요 탱커 발사 횟수 = {n_flights}회{extra}")

    ax = axes[1]
    ax.plot(loss_grid, n_grid, color=RED, linewidth=2.2, zorder=3)
    ax.plot(boiloff_loss_pct, n_flights, "o", color=RED, markersize=10, zorder=5)
    ax.set_facecolor("white")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel("극저온 추진제 손실률(boil-off, %)")
    ax.set_ylabel("필요 탱커 발사 횟수")
    ax.set_title("손실률이 커질수록 탱커 왕복이 급증")

    plt.tight_layout()
    plt.show()

    print(f"[입력] 목표 추진제량={target_prop_t:.0f}t, 탱커 1회당 이송량={tanker_delivered_t:.0f}t, "
          f"boil-off 손실률={boiloff_loss_pct:.0f}%")
    print(f"[산출] 선체당 실질 이송량 = {effective_per_flight:.1f}t → 필요 탱커 발사 횟수 = {n_flights}회")
    print("  → 슬라이드7의 'HLS 기준 약 10회'는 손실률이 낮고 이송량이 큰 낙관적 가정에 해당합니다.")
    print("  → boil-off 손실률이 커질수록 필요 탱커 횟수가 비선형적으로 늘어나며, 이것이 궤도 급유가")
    print("  '미시연 상태에서 가장 큰 일정 리스크'로 꼽히는 이유입니다 (슬라이드11 회의론자의 장부).")


def orbital_refueling_calculator():
    """목표 추진제량·탱커 1회당 이송량·극저온 boil-off 손실률을 조작해, 달·화성급 임무에
    필요한 궤도상 탱커 발사 횟수가 어떻게 달라지는지 확인한다."""
    interact(_plot_orbital_refueling,
             target_prop_t=FloatSlider(value=1200, min=500, max=2000, step=100,
                                        description="목표 추진제량(t)",
                                        style={"description_width": "130px"}, layout={"width": "460px"}),
             tanker_delivered_t=FloatSlider(value=120, min=50, max=200, step=10,
                                             description="탱커 1회당 이송량(t)",
                                             style={"description_width": "130px"}, layout={"width": "460px"}),
             boiloff_loss_pct=FloatSlider(value=10, min=0, max=30, step=2,
                                           description="Boil-off 손실률(%)",
                                           style={"description_width": "130px"}, layout={"width": "460px"}))


# ============================================================
# 31. 스타십 시나리오 스트레스 테스트 (13주차 WORKSHOP 연계)
# ============================================================
def _plot_scenario_stress_test(team_price_per_kg, scenario):
    if scenario == "A: 성숙 (2030)":
        band_low, band_high = 50, 150
        color = GREEN
    else:
        band_low, band_high = 1500, 3000
        color = RED

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.axhspan(band_low, band_high, color=color, alpha=0.15, zorder=0)
    ax.annotate(f"스타십 시나리오 가격대\n${band_low:,}~{band_high:,}/kg", xy=(0.5, (band_low + band_high) / 2),
                fontsize=9, color=color, ha="center", fontweight="bold")
    ax.axhline(team_price_per_kg, color=INK, linewidth=2.4, zorder=3)
    ax.annotate(f"우리 팀 벤치마크\n${team_price_per_kg:,.0f}/kg", xy=(0.5, team_price_per_kg),
                fontsize=9, color=INK, ha="center", va="bottom", fontweight="bold",
                xytext=(0, 8), textcoords="offset points")
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yscale("log")
    ax.set_ylabel("$/kg (로그축)")
    ax.set_facecolor("white")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_title(f"우리 팀 가격 벤치마크 vs 스타십 {scenario}")
    plt.tight_layout()
    plt.show()

    if team_price_per_kg > band_high:
        verdict = "우리 팀 가격이 스타십 가격대보다 확연히 높습니다 — 정면경쟁 대신 스타십이 못 주는 가치(궤도·일정·안보 요건 등)에서 차별화가 필요합니다."
    elif team_price_per_kg < band_low:
        verdict = "우리 팀 가격이 스타십 가격대보다 낮습니다 — 비현실적인 가정이 아닌지 원가구조(12주차 M4)를 재점검할 필요가 있습니다."
    else:
        verdict = "우리 팀 가격이 스타십 가격대와 겹칩니다 — 정면경쟁 구간이므로 차별화 요소가 없다면 시장에서 밀릴 위험이 큽니다."

    print(f"[시나리오] {scenario} — 가격대 ${band_low:,}~{band_high:,}/kg")
    print(f"[우리 팀] ${team_price_per_kg:,.0f}/kg")
    print(f"  → {verdict}")


def scenario_stress_test_explorer():
    """우리 팀의 발사서비스 가격 벤치마크를, 스타십 '성숙(A)' 또는 '지연(B)' 시나리오의
    예상 가격대와 비교해 생존 가능성을 점검한다 — 워크숍(보고서 ⑥ 리스크 분석) 연계 도구."""
    interact(_plot_scenario_stress_test,
             team_price_per_kg=FloatSlider(value=2500, min=50, max=5000, step=50,
                                            description="우리 팀 가격($/kg)",
                                            style={"description_width": "130px"}, layout={"width": "460px"}),
             scenario=Dropdown(options=["A: 성숙 (2030)", "B: 지연 (2030)"],
                                description="시나리오",
                                style={"description_width": "80px"}))

# ============================================================
# 32. 비즈니스 모델 캔버스(BMC) 뷰어 (11주차 §CASE STUDIES, 3사 비교)
# ============================================================
_BMC_DATA = {
    "SpaceX": {
        "flag": False,
        "blocks": {
            "핵심 파트너": "NASA(COTS/CRS)\n美 국방부(NSSL)\nFAA",
            "핵심 활동": "설계·제작·발사\n전과정 내재화\n재사용 회수·정비",
            "핵심 자원": "Merlin/Raptor 엔진\n재사용 Falcon 9 함대\n자체 발사장",
            "가치 제안": "압도적 저가($/kg)\n높은 빈도·일정신뢰성\n재사용 실증 신뢰도",
            "고객 관계": "장기 대량계약\n온라인 예약(라이드셰어)",
            "채널": "직판·정부조달\nTransporter 웹예약",
            "고객 세그먼트": "NASA·국방부\n상업 위성사업자\n군집운영사·Starlink",
            "비용 구조": "개발비 + 대량생산 고정비\n발사운영비 — 학습곡선 하락",
            "수익원": "발사서비스 대금\n정부계약, Starlink 매출",
        },
    },
    "Rocket Lab": {
        "flag": False,
        "blocks": {
            "핵심 파트너": "NASA·美 우주군\n뉴질랜드/美 정부\n위성부품 피인수기업",
            "핵심 활동": "Electron 전용발사\nNeutron 개발\n위성버스 제조",
            "핵심 자원": "마히아 발사장\nRutherford 엔진\nPhoton 플랫폼",
            "가치 제안": "맞춤 궤도·일정\n원스톱 위성제작(End-to-end)",
            "고객 관계": "임무단위 밀착관리\n정부 장기 프로그램",
            "채널": "직판·정부조달\n우주시스템 교차판매",
            "고객 세그먼트": "소형위성 상업사\n美 정부·군\n위성제작 고객",
            "비용 구조": "Electron 생산·운영비\n($/kg 열위) + Neutron 개발비",
            "수익원": "전용발사(프리미엄)\n위성시스템 매출(다수 차지)",
        },
    },
    "Virgin Orbit (파산, 2023)": {
        "flag": True,
        "blocks": {
            "핵심 파트너": "Virgin 그룹\n보잉 747 개조 파트너",
            "핵심 활동": "공중발사\nLauncherOne",
            "핵심 자원": "개조 747 '코스믹 걸'\n공중발사 기술",
            "가치 제안": "'어디서든 발사' 유연성\n→ 시장은 결국 '싼 가격' 원함",
            "고객 관계": "건별 임무 계약",
            "채널": "직판·정부 영업",
            "고객 세그먼트": "소형위성 운영사\n자국발사 희망국\n→ 라이드셰어와 중복",
            "비용 구조": "항공기+로켓 이중구조\n저빈도(총 6회) 간접비 회수 불능",
            "수익원": "매출 미미(누적손실 >$10억)\n→ 2023년 파산",
        },
        "weak_blocks": {"가치 제안", "고객 세그먼트", "비용 구조", "수익원"},
    },
}


def _draw_bmc_block(ax, x, y, w, h, title, text, weak=False):
    face = "#FCEBD5" if weak else "#F5F5F5"
    edge = AMBER if weak else "#B0B0B0"
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=1.6, zorder=2))
    ax.text(x + w / 2, y + h - 0.05, title, fontsize=8.5, fontweight="bold", color=INK,
            ha="center", va="top", zorder=3)
    ax.text(x + w / 2, y + h / 2 - 0.06, text, fontsize=6.8, color=INK_MUTED,
            ha="center", va="center", zorder=3, linespacing=1.4)


def _plot_bmc_canvas(company):
    data = _BMC_DATA[company]
    blocks = data["blocks"]
    weak = data.get("weak_blocks", set())

    fig, ax = plt.subplots(figsize=(13, 6.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    col_w = 2.0
    left_x = 0
    center_x = 2 * col_w
    right_x = 5 * col_w
    top_h = 4.0
    row_h = top_h / 3

    left_titles = ["핵심 파트너", "핵심 활동", "핵심 자원"]
    for i, t in enumerate(left_titles):
        y = top_h - row_h * (i + 1)
        _draw_bmc_block(ax, left_x, y, col_w, row_h, t, blocks[t], t in weak)

    _draw_bmc_block(ax, center_x, 0, col_w * 3, top_h, "가치 제안", blocks["가치 제안"], "가치 제안" in weak)

    right_titles = ["고객 관계", "채널", "고객 세그먼트"]
    for i, t in enumerate(right_titles):
        y = top_h - row_h * (i + 1)
        _draw_bmc_block(ax, right_x, y, col_w, row_h, t, blocks[t], t in weak)

    bottom_h = 6 - top_h - 0.15
    _draw_bmc_block(ax, 0, top_h + 0.15, col_w * 4, bottom_h, "비용 구조", blocks["비용 구조"], "비용 구조" in weak)
    _draw_bmc_block(ax, col_w * 4, top_h + 0.15, col_w * 4, bottom_h, "수익원", blocks["수익원"], "수익원" in weak)

    title_suffix = " — 주황색 = 실패의 구조적 원인" if data["flag"] else ""
    ax.set_title(f"{company} — 비즈니스 모델 캔버스{title_suffix}", fontsize=13, color=INK, pad=10)
    plt.tight_layout()
    plt.show()

    if data["flag"]:
        print("가치제안이 고객의 실제 구매기준(가격)과 어긋났고, 고객층은 라이드셰어와 겹쳤으며,")
        print("비용구조는 이중으로 무거웠습니다 — 세 블록의 동시 실패가 파산으로 이어진 구조입니다.")
    else:
        print(f"{company}는 오른쪽(시장) 블록과 왼쪽(역량) 블록이 서로를 강화하는 정합적 구조를 보여줍니다.")


def bmc_canvas_viewer():
    """SpaceX·Rocket Lab·Virgin Orbit 세 기업의 비즈니스 모델 캔버스(9블록)를 선택해
    나란히 비교한다. Virgin Orbit은 실패로 이어진 취약 블록이 주황색으로 표시된다."""
    interact(_plot_bmc_canvas,
             company=Dropdown(options=list(_BMC_DATA.keys()), description="기업 선택",
                               style={"description_width": "80px"}, layout={"width": "320px"}))


# ============================================================
# 33. 단위경제·손익분기 발사횟수 탐색기 (11주차 §UNIT ECONOMICS)
# ============================================================
def _plot_breakeven(dev_musd, man1_musd, ops1_musd, learning_rate_pct, price_per_flight_musd):
    b = np.log2(learning_rate_pct / 100)
    n = np.arange(1, 201)
    unit_cost = (man1_musd + ops1_musd) * n.astype(float) ** b
    cpf = dev_musd / n + unit_cost
    margin_per_flight = price_per_flight_musd - cpf

    cumulative_cost = dev_musd + np.cumsum(unit_cost)
    cumulative_revenue = n * price_per_flight_musd
    cumulative_margin = cumulative_revenue - cumulative_cost

    breakeven_idx = np.argmax(cumulative_margin >= 0) if np.any(cumulative_margin >= 0) else -1
    na_star = n[breakeven_idx] if breakeven_idx >= 0 else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.plot(n, cpf, color=RED, linewidth=2.2, label="발사당 원가 CpF(n)", zorder=3)
    ax.axhline(price_per_flight_musd, color=PRIMARY, linewidth=2, linestyle="--", label="발사가 PpF", zorder=3)
    ax.fill_between(n, cpf, price_per_flight_musd, where=(cpf < price_per_flight_musd),
                     color=GREEN, alpha=0.12, zorder=1)
    ax.set_facecolor("white")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel("누적 발사(생산) 횟수 n")
    ax.set_ylabel("$M")
    ax.set_title("발사당 원가 CpF가 가격 PpF 아래로 내려오는 시점")
    ax.legend(fontsize=8.5, frameon=False)

    ax = axes[1]
    ax.plot(n, cumulative_margin, color=PRIMARY, linewidth=2.2, zorder=3)
    ax.axhline(0, color=INK_MUTED, linestyle=":", linewidth=1.2)
    if na_star is not None:
        ax.plot(na_star, cumulative_margin[breakeven_idx], "o", color=RED, markersize=11, zorder=5)
        ax.annotate(f"Na*≈{na_star}회", (na_star, cumulative_margin[breakeven_idx]), fontsize=9,
                    color=RED, xytext=(8, 10), textcoords="offset points", fontweight="bold")
    ax.set_facecolor("white")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel("누적 발사(생산) 횟수 n")
    ax.set_ylabel("누적 손익($M)")
    ax.set_title("손익분기 발사횟수 Na*")

    plt.tight_layout()
    plt.show()

    print(f"[입력] 개발비 DEV=${dev_musd:.0f}M, 초기 생산비 MAN1=${man1_musd:.0f}M, 초기 운영비 OPS1=${ops1_musd:.0f}M, "
          f"학습률={learning_rate_pct:.0f}%, 발사가 PpF=${price_per_flight_musd:.0f}M")
    if na_star is not None:
        print(f"[산출] 손익분기 발사횟수 Na* ≈ {na_star}회 — 이 횟수를 넘어서면 누적으로 흑자 전환됩니다.")
    else:
        print("[산출] 200회 이내에서 손익분기에 도달하지 못합니다 — 가격을 올리거나 개발비·학습률을 개선해야 합니다.")
    print("  → 학습률(%)을 낮추면(더 가파른 학습곡선) 원가 하락이 빨라져 Na*가 앞당겨집니다 — 슬라이드11이")
    print("  강조하는 '학습곡선·LpA 효과를 반영해 Na*가 앞당겨지는 논리'가 바로 이 메커니즘입니다.")
    print("  주의: 이는 12주차 TRANSCOST 정식 모형의 단순화된 근사입니다 — 세부 CER은 12주차에서 다룹니다.")


def unit_economics_breakeven_explorer():
    """발사 1회 마진 = PpF - CpF, CpF = DEV/n + MANₙ + OPSₙ 구조에서, 개발비·초기 생산비·
    학습률·발사가를 직접 조작해 손익분기 발사횟수 Na*가 어떻게 달라지는지 확인한다."""
    interact(_plot_breakeven,
             dev_musd=FloatSlider(value=400, min=100, max=1000, step=50,
                                   description="개발비 DEV($M)",
                                   style={"description_width": "140px"}, layout={"width": "460px"}),
             man1_musd=FloatSlider(value=15, min=5, max=30, step=1,
                                    description="초기 생산비 MAN1($M)",
                                    style={"description_width": "140px"}, layout={"width": "460px"}),
             ops1_musd=FloatSlider(value=5, min=2, max=15, step=1,
                                    description="초기 운영비 OPS1($M)",
                                    style={"description_width": "140px"}, layout={"width": "460px"}),
             learning_rate_pct=FloatSlider(value=85, min=75, max=95, step=1,
                                            description="학습률(%)",
                                            style={"description_width": "140px"}, layout={"width": "460px"}),
             price_per_flight_musd=FloatSlider(value=30, min=10, max=80, step=5,
                                                description="발사가 PpF($M)",
                                                style={"description_width": "140px"}, layout={"width": "460px"}))


# ============================================================
# 34. 번레이트·활주로(runway) 계산기 (11주차 §UNIT ECONOMICS 현금흐름)
# ============================================================
def _plot_runway(cash_on_hand_musd, monthly_burn_musd, milestone_months):
    runway_months = cash_on_hand_musd / max(monthly_burn_musd, 0.1)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(["활주로(runway)"], [runway_months], color=PRIMARY if runway_months >= milestone_months else RED,
            height=0.4, zorder=3)
    ax.axvline(milestone_months, color=AMBER, linewidth=2.2, linestyle="--", zorder=4)
    ax.annotate(f"다음 마일스톤까지\n{milestone_months:.0f}개월", xy=(milestone_months, 0.6),
                fontsize=9, color=AMBER, ha="center", fontweight="bold")
    ax.annotate(f"{runway_months:.1f}개월", xy=(runway_months, 0), fontsize=11,
                ha="left" if runway_months < milestone_months else "right",
                va="center", color=INK, fontweight="bold",
                xytext=(6, 0), textcoords="offset points")
    ax.set_facecolor("white")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel("개월")
    ax.set_xlim(0, max(runway_months, milestone_months) * 1.3)
    ax.set_title("보유현금 활주로 vs 다음 자금조달 마일스톤까지 필요 기간")
    plt.tight_layout()
    plt.show()

    print(f"[입력] 보유현금=${cash_on_hand_musd:.0f}M, 월 소진율(burn rate)=${monthly_burn_musd:.1f}M/월, "
          f"다음 마일스톤까지={milestone_months:.0f}개월")
    print(f"[산출] 활주로 = {runway_months:.1f}개월")
    if runway_months >= milestone_months:
        print(f"  → 활주로가 마일스톤 도달 기간보다 {runway_months - milestone_months:.1f}개월 여유가 있습니다 — 생존 가능 구간입니다.")
    else:
        print(f"  → 활주로가 마일스톤 도달 기간보다 {milestone_months - runway_months:.1f}개월 부족합니다 — 자금경색(cash crunch) 위험입니다.")
        print("  → 번레이트를 줄이거나, 마일스톤 이전에 브릿지 투자·선수금 확보가 필요합니다.")


def funding_runway_calculator():
    """보유현금과 월간 현금소진율(burn rate)로 활주로(runway)를 계산하고,
    다음 기술·자금조달 마일스톤까지 생존 가능한지 점검한다."""
    interact(_plot_runway,
             cash_on_hand_musd=FloatSlider(value=80, min=10, max=300, step=10,
                                            description="보유현금($M)",
                                            style={"description_width": "140px"}, layout={"width": "460px"}),
             monthly_burn_musd=FloatSlider(value=6, min=1, max=25, step=1,
                                            description="월 소진율($M/월)",
                                            style={"description_width": "140px"}, layout={"width": "460px"}),
             milestone_months=FloatSlider(value=14, min=6, max=36, step=2,
                                           description="마일스톤까지(개월)",
                                           style={"description_width": "140px"}, layout={"width": "460px"}))


# ============================================================
# 35. 자금조달 여정 사다리 (11주차 §FUNDING JOURNEY, 정적 다이어그램)
# ============================================================
def funding_journey_ladder():
    """VC·엔젤 투자에서 정부 앵커계약, 상장을 거쳐 영업현금흐름 자립까지, 발사체 기업의
    4단계 자금조달 여정과 각 단계의 전환 조건을 보여준다 (정적 다이어그램)."""
    stages = [
        ("1단계", "VC·엔젤 투자", "개념설계~엔진 시제품\n기술 스토리와 팀으로 조달", INK_MUTED),
        ("2단계", "정부 앵커계약", "COTS형 마일스톤 계약\n'정부가 첫 고객' 신용 획득", PRIMARY),
        ("3단계", "상장(IPO/SPAC)", "대규모 자본 조달\n※ 2021 SPAC 붐 다수 몰락", AMBER),
        ("4단계", "영업현금흐름 자립", "백로그 기반 안정 매출\n흑자 전환 — 극소수만 도달", GREEN),
    ]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    xs = list(range(len(stages)))
    ax.plot(xs, [0] * len(xs), color="#B0B0B0", linewidth=2, zorder=1)
    for x, (num, label, detail, color) in zip(xs, stages):
        ax.scatter(x, 0, s=220, color=color, zorder=3, edgecolor="white", linewidth=1.5)
        ax.annotate(f"{num}\n{label}", (x, 0), fontsize=10, fontweight="bold", color=INK,
                    xytext=(0, 22), textcoords="offset points", ha="center")
        ax.annotate(detail, (x, 0), fontsize=7.6, color=INK_MUTED,
                    xytext=(0, -26), textcoords="offset points", ha="center", va="top")
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.set_ylim(-1.6, 1.6)
    ax.axis("off")
    ax.set_title("발사체 기업의 자금조달 여정 — 각 단계 전환의 조건은 '기술 마일스톤 달성'",
                  fontsize=12, color=INK, pad=8)
    plt.tight_layout()
    plt.show()
    print("엔진 연소시험 성공 → 시리즈 투자, 궤도 도달 → 정부계약 자격, 상업발사 연속 성공 → 상장·대형계약.")
    print("자금조달 계획 없는 개발일정은 공상이고, 기술 마일스톤 없는 자금계획은 공수표입니다.")

# ============================================================
# 36. CER 개발비 계산기 (12주차 §PART2 CER)
# ============================================================
_CER_COMPONENTS = {
    "액체 추진 소모성 1단": {"a": 100, "x": 0.555, "mass_label": "무추진제 건조질량"},
    "날개형(유익) 재사용 1단": {"a": 1442, "x": 0.326, "mass_label": "엔진 제외 건조질량"},
}


def _plot_cer_dev_cost(component, mass_t, f1, f2, f3, wyr_price_musd):
    params = _CER_COMPONENTS[component]
    a, x = params["a"], params["x"]
    dev_wyr = f1 * f2 * f3 * a * mass_t ** x
    dev_musd = dev_wyr * wyr_price_musd

    mass_grid = np.linspace(max(mass_t * 0.2, 0.2), mass_t * 3, 200)
    dev_grid_wyr = f1 * f2 * f3 * a * mass_grid ** x

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(mass_grid, dev_grid_wyr, color=PRIMARY, linewidth=2.2, zorder=3)
    ax.plot(mass_t, dev_wyr, "o", color=RED, markersize=11, zorder=5)
    ax.annotate(f"M={mass_t:,.1f}t\n{dev_wyr:,.0f} WYr", (mass_t, dev_wyr), fontsize=9,
                color=RED, xytext=(10, 10), textcoords="offset points", fontweight="bold")
    ax.set_facecolor("white")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel(f"{params['mass_label']} (t, 톤)")
    ax.set_ylabel("개발비 (Work-Year, WYr)")
    ax.set_title(f"{component} — C_Dev = f1·f2·f3·a·M^{x}  (지수<1: 규모의 경제)")
    plt.tight_layout()
    plt.show()

    print(f"[입력] 구성요소={component}, 질량 M={mass_t:,.1f}t, f1={f1:.2f}, f2={f2:.2f}, f3={f3:.2f}, "
          f"WYr 단가=${wyr_price_musd:.2f}M")
    print(f"[산출] 개발비 = {dev_wyr:,.0f} WYr ≈ ${dev_musd:,.1f}M (WYr 단가 환산)")
    print("  → 지수 x < 1이므로, 질량이 2배가 되어도 개발비는 2배보다 적게 늘어납니다 — 대형화에도")
    print("  '규모의 경제'가 작동한다는 것이 CER의 직관입니다. 계수(a, x)는 예시치이며, 실무 적용 시")
    print("  최신 TRANSCOST 매뉴얼 표를 반드시 참조할 것.")


def cer_development_cost_calculator():
    """비용추정관계식(CER) C_Dev = f1·f2·f3·a·M^x 을 직접 조작해, 질량과 보정계수가
    개발비(Work-Year 단위)에 어떻게 반영되는지 확인한다. (12주차 슬라이드6 예시 계수 기반)"""
    interact(_plot_cer_dev_cost,
             component=Dropdown(options=list(_CER_COMPONENTS.keys()), description="구성요소",
                                 style={"description_width": "80px"}, layout={"width": "320px"}),
             mass_t=FloatSlider(value=5.0, min=0.5, max=30.0, step=0.5,
                                 description="건조질량(t)",
                                 style={"description_width": "120px"}, layout={"width": "460px"}),
             f1=FloatSlider(value=1.0, min=0.5, max=2.0, step=0.1,
                             description="f1 기술성숙도",
                             style={"description_width": "120px"}, layout={"width": "460px"}),
             f2=FloatSlider(value=1.0, min=0.5, max=2.0, step=0.1,
                             description="f2 기술계수",
                             style={"description_width": "120px"}, layout={"width": "460px"}),
             f3=FloatSlider(value=1.0, min=0.5, max=2.0, step=0.1,
                             description="f3 팀 경험",
                             style={"description_width": "120px"}, layout={"width": "460px"}),
             wyr_price_musd=FloatSlider(value=0.25, min=0.1, max=0.5, step=0.05,
                                         description="WYr 단가($M)",
                                         style={"description_width": "120px"}, layout={"width": "460px"}))


# ============================================================
# 37. Crawford 학습곡선 탐색기 (12주차 §PART2 LEARNING CURVE)
# ============================================================
def _plot_learning_curve(t1_musd, learning_rate_pct, highlight_n):
    b = np.log2(learning_rate_pct / 100)
    n_grid = np.arange(1, 129)
    u_grid = t1_musd * n_grid.astype(float) ** b
    u_highlight = t1_musd * highlight_n ** b

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for p, style, lbl in [(95, ":", "95%"), (90, "--", "90%"), (85, "-.", "85%")]:
        bb = np.log2(p / 100)
        ax.plot(n_grid, t1_musd * n_grid.astype(float) ** bb, color=INK_MUTED, linewidth=1,
                 linestyle=style, alpha=0.6, zorder=2, label=f"참고: p={p}%")
    ax.plot(n_grid, u_grid, color=PRIMARY, linewidth=2.4, zorder=3, label=f"현재 p={learning_rate_pct:.0f}%")
    ax.plot(highlight_n, u_highlight, "o", color=RED, markersize=11, zorder=5)
    ax.annotate(f"{highlight_n}호기\n${u_highlight:.2f}M\n(T1의 {u_highlight/t1_musd*100:.0f}%)",
                (highlight_n, u_highlight), fontsize=8.5, color=RED, xytext=(10, 10),
                textcoords="offset points", fontweight="bold")
    ax.set_facecolor("white")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel("누적 생산 순번 n")
    ax.set_ylabel("단위비용 Uₙ ($M)")
    ax.set_title(f"Crawford 학습곡선 — Uₙ = T1 · n^b,  b=ln({learning_rate_pct:.0f}%)/ln2")
    ax.legend(fontsize=7.5, frameon=False)
    plt.tight_layout()
    plt.show()

    print(f"[입력] T1(1호기 비용)=${t1_musd:.1f}M, 학습률 p={learning_rate_pct:.0f}%")
    print(f"[산출] {highlight_n}호기 단위비용 ≈ ${u_highlight:.2f}M (T1의 {u_highlight/t1_musd*100:.0f}%)")
    print("  → '많이 만드는 자가 싸게 만든다' — 대량생산·고빈도 사업자의 원가 우위는 기술이 아니라")
    print("  이 산수에서 나옵니다. 학습률을 82~96% 범위에서 잘못 가정하면 생산비 추정이 크게 왜곡될 수")
    print("  있으므로(RAND), 보고서에서 학습률 가정을 반드시 명시할 것.")


def learning_curve_explorer():
    """Crawford 단위 학습곡선 Uₙ = T1·n^b (b=ln(p)/ln2)을 직접 조작해, 학습률 p와 누적생산
    순번 n이 단위비용을 어떻게 낮추는지 확인한다. (12주차 슬라이드7~8 연계)"""
    interact(_plot_learning_curve,
             t1_musd=FloatSlider(value=50, min=10, max=200, step=10,
                                  description="T1(1호기 비용, $M)",
                                  style={"description_width": "150px"}, layout={"width": "460px"}),
             learning_rate_pct=FloatSlider(value=90, min=82, max=96, step=1,
                                            description="학습률 p(%)",
                                            style={"description_width": "150px"}, layout={"width": "460px"}),
             highlight_n=IntSlider(value=8, min=2, max=128, step=1,
                                    description="강조할 순번 n",
                                    style={"description_width": "150px"}, layout={"width": "460px"}))


# ============================================================
# 38. 발사빈도(LpA)·간접운영비 탐색기 (12주차 §PART2 OPERATIONS/LAUNCH RATE)
# ============================================================
def _plot_lpa_ioc(lpa, outsourcing_share):
    lpa_grid = np.linspace(1, 60, 200)
    ioc_index_grid = 40 * outsourcing_share + 22.5 * lpa_grid ** (-0.379)
    ioc_at_1 = 40 * outsourcing_share + 22.5 * 1 ** (-0.379)
    ioc_sel = 40 * outsourcing_share + 22.5 * lpa ** (-0.379)
    pct_of_baseline = ioc_sel / ioc_at_1 * 100

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(lpa_grid, ioc_index_grid, color=RED, linewidth=2.2, zorder=3)
    ax.plot(lpa, ioc_sel, "o", color=RED, markersize=11, zorder=5)
    ax.annotate(f"LpA={lpa:.0f}\n간접비 지수={ioc_sel:.1f}\n(LpA=1 대비 {pct_of_baseline:.0f}%)",
                (lpa, ioc_sel), fontsize=8.5, color=RED, xytext=(10, 10),
                textcoords="offset points", fontweight="bold")
    ax.axhline(ioc_at_1, color=INK_MUTED, linestyle=":", linewidth=1)
    ax.annotate(f"LpA=1 기준선 = {ioc_at_1:.1f}", (45, ioc_at_1), fontsize=8, color=INK_MUTED,
                va="bottom")
    ax.set_facecolor("white")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel("연간발사횟수 LpA")
    ax.set_ylabel("간접운영비(IOC) 지수")
    ax.set_title("발사빈도가 오를수록 회당 간접비가 급락한다")
    plt.tight_layout()
    plt.show()

    print(f"[입력] LpA={lpa:.0f}, 외주비중 S={outsourcing_share:.1f}")
    print(f"[산출] 간접비 지수 = {ioc_sel:.1f} (LpA=1 대비 {pct_of_baseline:.0f}%)")
    print("  → LpA가 1에서 50으로 늘면 회당 간접비 지수가 대략 100→23 수준으로 급락합니다(외주비중=0 기준).")
    print("  → 학습곡선(생산비 하락)과 LpA(운영비 하락)는 곱으로 작용 — '많이 만들고 자주 쏘는' 사업자만")
    print("  이중의 원가 우위를 누립니다. 목표 LpA와 그 근거(수주 계획) 없는 원가 추정은 신뢰받지 못합니다.")


def lpa_indirect_cost_explorer():
    """간접운영비(IOC)가 연간발사횟수(LpA)의 음의 거듭제곱으로 하락하는 구조
    (IOC = 40·S + 22.5·LpA^-0.379)를 직접 조작해 확인한다. (12주차 슬라이드9~10 연계)"""
    interact(_plot_lpa_ioc,
             lpa=FloatSlider(value=10, min=1, max=60, step=1,
                              description="연간발사횟수 LpA",
                              style={"description_width": "130px"}, layout={"width": "460px"}),
             outsourcing_share=FloatSlider(value=0.0, min=0.0, max=1.0, step=0.1,
                                            description="외주비중 S",
                                            style={"description_width": "130px"}, layout={"width": "460px"}))


# ============================================================
# 39. CpF → PpF 종합 계산기 (12주차 §PART2 PRICING)
# ============================================================
def _plot_cpf_to_ppf(dev_musd, na, man_n_musd, ops_n_musd, margin_pct):
    cpf = dev_musd / na + man_n_musd + ops_n_musd
    ppf = cpf * (1 + margin_pct / 100)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    stages = ["개발비 상각\n(DEV/Na)", "생산비\n(MANₙ)", "운영비\n(OPSₙ)", "CpF\n(원가 합계)", "PpF\n(가격)"]
    cum1 = dev_musd / na
    cum2 = cum1 + man_n_musd
    cum3 = cum2 + ops_n_musd
    vals = [cum1, man_n_musd, ops_n_musd, cpf, ppf]
    bottoms = [0, cum1, cum2, 0, 0]
    colors = [PRIMARY, AMBER, "#5B9BD5", RED, GREEN]
    for i, (v, bt, c) in enumerate(zip(vals, bottoms, colors)):
        if i < 3:
            ax.bar(stages[i], v, bottom=bt, color=c, width=0.55, zorder=3)
        else:
            ax.bar(stages[i], v, color=c, width=0.55, zorder=3)
    for i, v in enumerate([cum3, cum3, cum3, cpf, ppf]):
        pass
    ax.annotate(f"${cpf:.2f}M", (3, cpf), fontsize=9, ha="center", va="bottom", fontweight="bold", color=INK)
    ax.annotate(f"${ppf:.2f}M", (4, ppf), fontsize=9, ha="center", va="bottom", fontweight="bold", color=INK)
    ax.set_facecolor("white")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_ylabel("$M")
    ax.set_title(f"CpF = DEV/Na + MANₙ + OPSₙ = \\${cpf:.2f}M  →  PpF = CpF×(1+{margin_pct:.0f}%) = \\${ppf:.2f}M")
    plt.tight_layout()
    plt.show()

    print(f"[입력] 개발비 DEV=${dev_musd:.0f}M, 계획 발사횟수 Na={na:.0f}, 생산비 MANₙ=${man_n_musd:.1f}M, "
          f"운영비 OPSₙ=${ops_n_musd:.1f}M, 이윤율={margin_pct:.0f}%")
    print(f"[산출] CpF=${cpf:.2f}M → PpF=${ppf:.2f}M")
    print("  참고 — Drenthe et al.(2017) 검증 사례: Falcon 1 개발비 추정 $96M vs 실측 $90M(오차 +7%),")
    print("  Falcon 9 개발비 추정 $419M vs 실측 $372M(오차 +13%) — 표준 CER은 수직계열화 상업기업의")
    print("  비용을 과대추정하는 경향이 있다는 한계를 함께 고려할 것.")


def cpf_to_ppf_calculator():
    """발사원가 CpF = DEV/Na + MANₙ + OPSₙ 에서 이윤율을 반영한 발사가격 PpF까지의
    산정 절차를 워터폴로 확인한다 — BMC '비용구조·수익원' 블록의 정량화. (12주차 슬라이드11)"""
    interact(_plot_cpf_to_ppf,
             dev_musd=FloatSlider(value=400, min=50, max=1000, step=50,
                                   description="개발비 DEV($M)",
                                   style={"description_width": "140px"}, layout={"width": "460px"}),
             na=FloatSlider(value=20, min=5, max=100, step=5,
                             description="계획 발사횟수 Na",
                             style={"description_width": "140px"}, layout={"width": "460px"}),
             man_n_musd=FloatSlider(value=15, min=2, max=50, step=1,
                                     description="생산비 MANₙ($M)",
                                     style={"description_width": "140px"}, layout={"width": "460px"}),
             ops_n_musd=FloatSlider(value=5, min=1, max=30, step=1,
                                     description="운영비 OPSₙ($M)",
                                     style={"description_width": "140px"}, layout={"width": "460px"}),
             margin_pct=FloatSlider(value=8, min=0, max=30, step=1,
                                     description="이윤율(%)",
                                     style={"description_width": "140px"}, layout={"width": "460px"}))


# ============================================================
# 40. 라이드셰어 vs 전용발사 가격 비교 (12주차 §PART3 RIDESHARE, 정적 다이어그램)
# ============================================================
def rideshare_vs_dedicated_chart():
    """2019년 라이드셰어 도입 전후, 소형위성 발사의 실효가격이 어떻게 재편됐는지
    (라이드셰어 vs 전용 소형발사) 비교한다. (정적 다이어그램 — 슬라이드12 수치 기반)"""
    fig, ax = plt.subplots(figsize=(9, 5))
    categories = ["라이드셰어 도입 전\n(~2018)", "라이드셰어 초기\n(2019~)", "라이드셰어 현재\n(2026 초)", "전용 소형발사\n(Electron 등)"]
    low = [15000, 2000, 5000, 20000]
    high = [30000, 5000, 9000, 40000]
    mid = [(l + h) / 2 for l, h in zip(low, high)]
    colors = [INK_MUTED, GREEN, AMBER, RED]

    for i, (c, l, h, m, col) in enumerate(zip(categories, low, high, mid, colors)):
        ax.plot([i, i], [l, h], color=col, linewidth=8, alpha=0.35, zorder=2, solid_capstyle="round")
        ax.plot(i, m, "o", color=col, markersize=12, zorder=3)
        ax.annotate(f"${l:,}~{h:,}/kg", (i, h), fontsize=9, ha="center", va="bottom",
                    fontweight="bold", color=col, xytext=(0, 6), textcoords="offset points")

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_yscale("log")
    ax.set_facecolor("white")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_ylabel("실효가격 $/kg (로그축)")
    ax.set_title("시장은 '저가 합승(라이드셰어)'과 '고가 전용'으로 양분 — 중간지대가 소멸")
    plt.tight_layout()
    plt.show()

    print("가격 5배 하락의 원천은 신기술이 아니라 '남의 학습곡선에 합승'하는 구조입니다.")
    print("2026년 초 기준 라이드셰어는 약 $7,000/kg 수준으로 재상승(수요 증가·인플레이션 반영)했지만,")
    print("전용 소형발사 대비 여전히 수 분의 1 수준입니다 — 전용발사는 '내 궤도·내 일정' 가치에 집중해야 생존합니다.")


# ============================================================
# 41. TAM-SAM-SOM 퍼널 계산기 (12주차 WORKSHOP ① 수요 정량화)
# ============================================================
def _plot_tam_sam_som(tam_flights, sam_pct, som_pct):
    sam_flights = tam_flights * sam_pct / 100
    som_flights = sam_flights * som_pct / 100

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"TAM\n{tam_flights:.0f}회/년", f"SAM\n{sam_flights:.1f}회/년", f"SOM\n{som_flights:.2f}회/년"]
    vals = [tam_flights, sam_flights, som_flights]
    colors = [INK_MUTED, AMBER, GREEN]
    widths = [1.0, sam_flights / tam_flights if tam_flights else 0, som_flights / tam_flights if tam_flights else 0]

    for i, (lbl, v, c, w) in enumerate(zip(labels, vals, colors, widths)):
        ax.barh(2 - i, w, height=0.6, color=c, zorder=3, left=(1 - w) / 2)
        ax.annotate(lbl, (0.5, 2 - i), fontsize=10, ha="center", va="center", fontweight="bold", color="white",
                    zorder=4)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 2.5)
    ax.axis("off")
    ax.set_title(f"TAM {tam_flights:.0f}회 → SAM {sam_flights:.1f}회({sam_pct:.0f}%) → SOM {som_flights:.2f}회({som_pct:.0f}%)",
                 fontsize=11.5, color=INK)
    plt.tight_layout()
    plt.show()

    print(f"[입력] TAM(전체 발사수요)={tam_flights:.0f}회/년, SAM 비중={sam_pct:.0f}%, SOM 비중={som_pct:.0f}%")
    print(f"[산출] SAM={sam_flights:.1f}회/년, SOM={som_flights:.2f}회/년")
    print("  → SOM이 곧 우리 팀의 현실적 목표 LpA입니다 — 이 값을 위 CpF/PpF·손익분기 계산기의")
    print("  발사빈도·계획 발사횟수 가정과 반드시 정합시킬 것. 근거 출처(위성 발사계획, 정부 중기계획)를 명시할 것.")


def tam_sam_som_funnel():
    """전체 발사수요(TAM)에서 목표 세그먼트로 좁힌 유효시장(SAM), 현실적 수주 가능분(SOM)까지
    깔때기 구조로 정량화한다 — 사업기획보고서 ③ 시장분석의 출발점. (12주차 워크숍 ①)"""
    interact(_plot_tam_sam_som,
             tam_flights=FloatSlider(value=200, min=20, max=500, step=10,
                                      description="TAM(회/년)",
                                      style={"description_width": "120px"}, layout={"width": "460px"}),
             sam_pct=FloatSlider(value=15, min=1, max=100, step=1,
                                  description="SAM 비중(%)",
                                  style={"description_width": "120px"}, layout={"width": "460px"}),
             som_pct=FloatSlider(value=10, min=1, max=100, step=1,
                                  description="SOM 비중(%)",
                                  style={"description_width": "120px"}, layout={"width": "460px"}))
