from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


# =========================================================
# Streamlit 기본 설정
# =========================================================

st.set_page_config(
    page_title="학습 공백 위험지수 결과 대시보드",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main .block-container { padding-top: 2rem; padding-bottom: 4rem; }
    .calendar-grid {
        display: grid;
        grid-template-columns: repeat(7, minmax(0, 1fr));
        gap: 8px;
    }
    .calendar-head {
        text-align: center;
        font-weight: 800;
        color: #475569;
        font-size: 14px;
        padding: 6px 0;
    }
    .calendar-day {
        min-height: 92px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        padding: 10px;
        font-size: 13px;
        overflow: hidden;
    }
    .study { background: #ecfdf5; border-color: #bbf7d0; }
    .rest { background: #fff1f2; border-color: #fecdd3; }
    .vacation { background: #f1f5f9; border-color: #cbd5e1; }
    .empty { background: #f8fafc; border-color: #f1f5f9; }
    .day-top {
        display: flex;
        justify-content: space-between;
        gap: 6px;
        align-items: center;
    }
    .badge {
        display: inline-block;
        border-radius: 999px;
        padding: 2px 8px;
        font-size: 11px;
        color: white;
        font-weight: 800;
    }
    .badge-study { background: #10b981; }
    .badge-rest { background: #f43f5e; }
    .badge-vacation { background: #64748b; }
    .score {
        margin-top: 7px;
        color: #475569;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .event {
        margin-top: 7px;
        color: #64748b;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        line-height: 1.35;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 경로 설정
# =========================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

CORE_SUMMARY_PATH = DATA_DIR / "core_summary.csv"
SCHOOL_SUMMARY_PATH = DATA_DIR / "school_summary.csv"
VACATION_CHECK_PATH = DATA_DIR / "vacation_check.csv"
DETAIL_INDEX_PATH = DATA_DIR / "detail_index.csv"


# =========================================================
# 데이터 로드 함수
# =========================================================

@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾지 못했습니다: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_detail_part(relative_path: str) -> pd.DataFrame:
    path = DATA_DIR / relative_path

    if not path.exists():
        raise FileNotFoundError(f"상세 parquet 파일을 찾지 못했습니다: {path}")

    df = pd.read_parquet(path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


def prepare_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "학교표시명" not in df.columns:
        df["학교표시명"] = df["학교명"].astype(str) + " (" + df["학교과정명"].astype(str) + ")"

    numeric_cols = [
        "분석일수",
        "휴업일수",
        "수업일수",
        "휴업일비율",
        "평균기억점수",
        "평균위험점수",
        "최대위험점수",
        "최저기억점수",
        "고위험일수",
        "고위험일비율",
        "최종위험지수",
        "방학일수",
        "추론방학일수",
        "위험백분위",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# =========================================================
# 시각화 함수
# =========================================================

def make_memory_figure(one_school: pd.DataFrame, school_name: str) -> go.Figure:
    df = one_school.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in ["memory_score", "risk_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["memory_plot"] = df["memory_score"].where(~df["is_vacation_day"], np.nan)
    df["risk_plot"] = df["risk_score"].where(~df["is_vacation_day"], np.nan)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["memory_plot"],
            mode="lines",
            name="기억점수",
            connectgaps=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["risk_plot"],
            mode="lines",
            name="위험점수",
            connectgaps=False,
        )
    )

    if "is_main_rest_day" in df.columns:
        rest = df[df["is_main_rest_day"]]
        fig.add_trace(
            go.Scatter(
                x=rest["date"],
                y=rest["memory_score"],
                mode="markers",
                name="휴업일",
                marker=dict(size=5),
            )
        )

    fig.update_layout(
        title=f"{school_name} 날짜별 기억점수·위험점수 변화",
        xaxis_title="날짜",
        yaxis_title="점수",
        yaxis=dict(range=[0, 100]),
        height=440,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=20, r=20, t=70, b=20),
    )

    return fig


def make_calendar_html(one_school: pd.DataFrame, month: str) -> str:
    df = one_school.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    month_dt = pd.to_datetime(month + "-01")
    year = month_dt.year
    mon = month_dt.month

    month_df = df[(df["date"].dt.year == year) & (df["date"].dt.month == mon)].copy()
    by_date = {
        d.strftime("%Y-%m-%d"): row
        for d, row in month_df.set_index("date").iterrows()
    }

    first = pd.Timestamp(year=year, month=mon, day=1)
    last_day = (first + pd.offsets.MonthEnd(0)).day
    start_weekday = (first.weekday() + 1) % 7  # 일요일=0
    weekdays = ["일", "월", "화", "수", "목", "금", "토"]

    def safe_bool(value):
        if isinstance(value, bool):
            return value
        if pd.isna(value):
            return False
        return str(value).lower() in ["true", "1", "yes"]

    cells = ""

    for w in weekdays:
        cells += f'<div class="calendar-head">{w}</div>'

    for _ in range(start_weekday):
        cells += '<div class="calendar-day empty"></div>'

    for day in range(1, last_day + 1):
        date_str = f"{year}-{mon:02d}-{day:02d}"
        row = by_date.get(date_str)

        if row is None:
            cells += f'<div class="calendar-day empty"><b>{day}</b></div>'
            continue

        is_vacation = safe_bool(row.get("is_vacation_day", False))
        is_rest = safe_bool(row.get("is_main_rest_day", False))

        memory_score = row.get("memory_score", np.nan)
        risk_score = row.get("risk_score", np.nan)

        if is_vacation:
            cls = "vacation"
            badge = '<span class="badge badge-vacation">방학</span>'
            score = "계산 제외"
        elif is_rest:
            cls = "rest"
            badge = '<span class="badge badge-rest">휴업</span>'
            if pd.notna(memory_score) and pd.notna(risk_score):
                score = f"기억 {float(memory_score):.1f} · 위험 {float(risk_score):.1f}"
            else:
                score = "점수 없음"
        else:
            cls = "study"
            badge = '<span class="badge badge-study">수업</span>'
            if pd.notna(memory_score) and pd.notna(risk_score):
                score = f"기억 {float(memory_score):.1f} · 위험 {float(risk_score):.1f}"
            else:
                score = "점수 없음"

        event = str(row.get("행사명_목록", ""))
        if event == "nan":
            event = ""

        cells += f"""
        <div class="calendar-day {cls}">
            <div class="day-top">
                <b>{day}</b>{badge}
            </div>
            <div class="score" title="{score}">{score}</div>
            <div class="event" title="{event}">{event or '-'}</div>
        </div>
        """

    return f"""
    <!doctype html>
    <html lang="ko">
    <head>
      <meta charset="utf-8" />
      <style>
        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            padding: 0;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: #0f172a;
            background: white;
        }}

        .calendar-grid {{
            display: grid;
            grid-template-columns: repeat(7, minmax(0, 1fr));
            gap: 8px;
            width: 100%;
        }}

        .calendar-head {{
            text-align: center;
            font-weight: 800;
            color: #475569;
            font-size: 14px;
            padding: 6px 0;
        }}

        .calendar-day {{
            min-height: 96px;
            border-radius: 16px;
            border: 1px solid #e5e7eb;
            padding: 10px;
            font-size: 13px;
            overflow: hidden;
        }}

        .study {{
            background: #ecfdf5;
            border-color: #bbf7d0;
        }}

        .rest {{
            background: #fff1f2;
            border-color: #fecdd3;
        }}

        .vacation {{
            background: #f1f5f9;
            border-color: #cbd5e1;
        }}

        .empty {{
            background: #f8fafc;
            border-color: #f1f5f9;
        }}

        .day-top {{
            display: flex;
            justify-content: space-between;
            gap: 6px;
            align-items: center;
        }}

        .badge {{
            display: inline-block;
            border-radius: 999px;
            padding: 2px 8px;
            font-size: 11px;
            color: white;
            font-weight: 800;
            white-space: nowrap;
        }}

        .badge-study {{
            background: #10b981;
        }}

        .badge-rest {{
            background: #f43f5e;
        }}

        .badge-vacation {{
            background: #64748b;
        }}

        .score {{
            margin-top: 7px;
            color: #475569;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .event {{
            margin-top: 7px;
            color: #64748b;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
            line-height: 1.35;
        }}
      </style>
    </head>
    <body>
      <div class="calendar-grid">
        {cells}
      </div>
    </body>
    </html>
    """
def get_zoom_axis(series: pd.Series, min_span: float = 1.0, pad_ratio: float = 0.18):
    """
    평균 위험지수 값들이 서로 비슷할 때 차이가 잘 보이도록 축 범위를 자동 확대한다.
    단, 그래프가 과도하게 확대되지 않도록 최소 표시 폭과 여백을 제한한다.
    """
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return [0.0, 1.0], 0.1

    v_min = float(values.min())
    v_max = float(values.max())
    data_span = v_max - v_min

    if data_span == 0:
        center = v_min
        half_span = min_span / 2
        axis_min = center - half_span
        axis_max = center + half_span
    else:
        span = max(data_span, min_span)
        padding = span * pad_ratio
        center = (v_min + v_max) / 2
        axis_min = center - span / 2 - padding
        axis_max = center + span / 2 + padding

    axis_min = max(0.0, axis_min)
    axis_span = axis_max - axis_min

    if axis_span <= 0.5:
        dtick = 0.05
    elif axis_span <= 1.5:
        dtick = 0.1
    elif axis_span <= 3:
        dtick = 0.2
    elif axis_span <= 6:
        dtick = 0.5
    else:
        dtick = 1.0

    return [axis_min, axis_max], dtick

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="학교별요약", index=False)
    return output.getvalue()


# =========================================================
# 데이터 파일 확인
# =========================================================

missing_files = [
    path for path in [
        CORE_SUMMARY_PATH,
        SCHOOL_SUMMARY_PATH,
        VACATION_CHECK_PATH,
        DETAIL_INDEX_PATH,
    ]
    if not path.exists()
]

if missing_files:
    st.error("필수 데이터 파일이 없습니다.")
    for path in missing_files:
        st.write(f"- `{path}`")
    st.stop()


# =========================================================
# 데이터 로드
# =========================================================

core_summary = prepare_summary_df(load_csv(CORE_SUMMARY_PATH))
school_summary = prepare_summary_df(load_csv(SCHOOL_SUMMARY_PATH))
vacation_check = load_csv(VACATION_CHECK_PATH)
detail_index = load_csv(DETAIL_INDEX_PATH)

if "학교표시명" not in core_summary.columns:
    core_summary["학교표시명"] = core_summary["학교명"].astype(str) + " (" + core_summary["학교과정명"].astype(str) + ")"

if "학교표시명" not in school_summary.columns:
    school_summary["학교표시명"] = school_summary["학교명"].astype(str) + " (" + school_summary["학교과정명"].astype(str) + ")"


# =========================================================
# 앱 화면
# =========================================================

st.title("📚 2025-03-01 ~ 2026-02-28 학사일정 기반 학습 공백 위험지수 결과 바로보기")
st.caption(
    "무거운 원본 CSV 분석은 로컬에서 미리 수행했고, 이 사이트는 저장된 결과를 읽어 "
    "학교별 그래프와 달력을 보여줍니다."
)


# =========================================================
# 요약 카드
# =========================================================

excluded_count = len(school_summary) - len(core_summary)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("전체 학교-과정", f"{len(school_summary):,}")
col2.metric("핵심 분석 대상", f"{len(core_summary):,}")
col3.metric("제외", f"{excluded_count:,}")
col4.metric("평균 위험지수", f"{core_summary['최종위험지수'].mean():.2f}")

if "휴업일비율" in core_summary.columns and "최종위험지수" in core_summary.columns:
    corr_value = core_summary["휴업일비율"].corr(core_summary["최종위험지수"])
    col5.metric("휴업일-위험 상관", f"{corr_value:.3f}")
else:
    col5.metric("휴업일-위험 상관", "없음")


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📌 전체 결과",
        "📊 전체 그래프",
        "🏫 학교별 상세",
        "💾 다운로드",
    ]
)


# =========================================================
# 탭 1. 전체 결과
# =========================================================

with tab1:
    st.subheader("전체 분석 결과")

    st.write(
        """
        이 대시보드는 학교별 학사일정에서 방학을 제외한 기간의 수업일·휴업일 배치를 분석하고,
        날짜별 기억점수와 위험점수를 계산한 결과를 보여줍니다.
        """
    )

    show_cols = [
        "시도교육청명",
        "학교명",
        "학교과정명",
        "분석일수",
        "휴업일수",
        "수업일수",
        "휴업일비율",
        "평균위험점수",
        "고위험일비율",
        "최종위험지수",
        "위험등급",
        "방학일수",
        "상대위험등급",
    ]
    show_cols = [c for c in show_cols if c in core_summary.columns]

    st.dataframe(
        core_summary.sort_values("최종위험지수", ascending=False)[show_cols].head(500),
        use_container_width=True,
        height=540,
    )

    with st.expander("방학일수 데이터 점검", expanded=False):
        if "방학일수" in vacation_check.columns:
            c1, c2, c3 = st.columns(3)
            c1.metric("방학일수 10일 미만", f"{(vacation_check['방학일수'] < 10).sum():,}")
            c2.metric("방학일수 140일 이상", f"{(vacation_check['방학일수'] >= 140).sum():,}")
            c3.metric("방학일수 최대값", f"{vacation_check['방학일수'].max():.0f}일")
            st.dataframe(vacation_check["방학일수"].describe().to_frame("방학일수"), use_container_width=True)
        else:
            st.info("vacation_check.csv에 방학일수 열이 없습니다.")


# =========================================================
# 탭 2. 전체 그래프
# =========================================================

with tab2:
    st.subheader("📊 전체 분석 결과")

    required_cols = ["시도교육청명", "학교과정명", "분석단위ID", "최종위험지수", "휴업일비율", "방학일수", "학교표시명"]
    missing_cols = [c for c in required_cols if c not in core_summary.columns]

    if missing_cols:
        st.error(f"그래프를 그리는 데 필요한 열이 없습니다: {missing_cols}")
        st.stop()

    g1, g2 = st.columns(2)

    region_summary = (
        core_summary.groupby("시도교육청명")
        .agg(
            학교수=("분석단위ID", "count"),
            평균최종위험지수=("최종위험지수", "mean"),
            평균휴업일비율=("휴업일비율", "mean"),
        )
        .reset_index()
        .sort_values("평균최종위험지수", ascending=False)
    )

    course_summary = (
        core_summary.groupby("학교과정명")
        .agg(
            학교수=("분석단위ID", "count"),
            평균최종위험지수=("최종위험지수", "mean"),
            평균휴업일비율=("휴업일비율", "mean"),
        )
        .reset_index()
        .sort_values("평균최종위험지수", ascending=False)
    )

    region_x_range, region_x_dtick = get_zoom_axis(
        region_summary["평균최종위험지수"],
        min_span=1.0,
    )

    course_y_range, course_y_dtick = get_zoom_axis(
        course_summary["평균최종위험지수"],
        min_span=1.0,
    )

    region_fig_height = min(520, max(380, 24 * len(region_summary) + 150))
    course_fig_height = 420

    with g1:
        region_fig = px.bar(
            region_summary,
            x="평균최종위험지수",
            y="시도교육청명",
            orientation="h",
            title="시도교육청별 평균 위험지수",
            labels={
                "평균최종위험지수": "평균 위험지수",
                "시도교육청명": "시도교육청",
            },
            text="평균최종위험지수",
        )

        region_fig.update_traces(
            texttemplate="%{x:.2f}",
            textposition="outside",
            cliponaxis=False,
        )

        region_fig.update_xaxes(
            range=region_x_range,
            dtick=region_x_dtick,
            tickformat=".2f",
            title="평균 위험지수",
        )

        region_fig.update_yaxes(
            categoryorder="array",
            categoryarray=region_summary["시도교육청명"].iloc[::-1].tolist(),
            title="시도교육청",
        )

        region_fig.update_layout(
            height=region_fig_height,
            margin=dict(l=10, r=60, t=60, b=40),
        )

        st.plotly_chart(region_fig, use_container_width=True)

    with g2:
        course_fig = px.bar(
            course_summary,
            x="학교과정명",
            y="평균최종위험지수",
            title="학교과정별 평균 위험지수",
            labels={
                "학교과정명": "학교과정",
                "평균최종위험지수": "평균 위험지수",
            },
            text="평균최종위험지수",
        )

        course_fig.update_traces(
            texttemplate="%{y:.2f}",
            textposition="outside",
            cliponaxis=False,
        )

        course_fig.update_yaxes(
            range=course_y_range,
            dtick=course_y_dtick,
            tickformat=".2f",
            title="평균 위험지수",
        )

        course_fig.update_xaxes(
            title="학교과정",
        )

        course_fig.update_layout(
            height=course_fig_height,
            margin=dict(l=10, r=35, t=60, b=40),
        )

        st.plotly_chart(course_fig, use_container_width=True)

    st.caption(
        "※ 위 두 그래프는 평균 위험지수 차이가 작게 보이는 문제를 줄이기 위해 값 축을 자동 확대했습니다. "
        "막대의 절대 길이보다 막대 끝 수치와 축 눈금을 함께 비교하세요."
    )

    s1, s2 = st.columns(2)

    with s1:
        fig_rest = px.scatter(
            core_summary,
            x="휴업일비율",
            y="최종위험지수",
            color="학교과정명",
            hover_name="학교표시명",
            title="휴업일비율과 최종위험지수",
            labels={
                "휴업일비율": "휴업일비율(%)",
                "최종위험지수": "최종위험지수",
                "학교과정명": "학교과정",
            },
        )
        st.plotly_chart(fig_rest, use_container_width=True)

    with s2:
        fig_vacation = px.scatter(
            core_summary,
            x="방학일수",
            y="최종위험지수",
            color="학교과정명",
            hover_name="학교표시명",
            title="방학일수와 최종위험지수",
            labels={
                "방학일수": "방학일수",
                "최종위험지수": "최종위험지수",
                "학교과정명": "학교과정",
            },
        )
        st.plotly_chart(fig_vacation, use_container_width=True)

# =========================================================
# 탭 3. 학교별 상세
# =========================================================

with tab3:
    st.subheader("학교 검색 및 상세 분석")

    f1, f2, f3, f4 = st.columns([2.2, 1.2, 1.2, 1.4])

    with f1:
        query = st.text_input("학교명 검색", placeholder="예: 거창대성고, 대구일과학고")

    with f2:
        regions = ["전체"] + sorted(core_summary["시도교육청명"].dropna().unique().tolist())
        region = st.selectbox("시도교육청", regions)

    with f3:
        courses = ["전체"] + sorted(core_summary["학교과정명"].dropna().unique().tolist())
        course = st.selectbox("학교과정", courses)

    with f4:
        sort_option = st.selectbox(
            "정렬",
            [
                "최종위험지수 높은순",
                "최종위험지수 낮은순",
                "휴업일비율 높은순",
                "방학일수 많은순",
                "학교명 가나다순",
            ],
        )

    filtered = core_summary.copy()

    if query.strip():
        filtered = filtered[
            filtered["학교명"].str.contains(query.strip(), case=False, na=False)
            | filtered["학교표시명"].str.contains(query.strip(), case=False, na=False)
        ]

    if region != "전체":
        filtered = filtered[filtered["시도교육청명"] == region]

    if course != "전체":
        filtered = filtered[filtered["학교과정명"] == course]

    if sort_option == "최종위험지수 높은순":
        filtered = filtered.sort_values("최종위험지수", ascending=False)
    elif sort_option == "최종위험지수 낮은순":
        filtered = filtered.sort_values("최종위험지수", ascending=True)
    elif sort_option == "휴업일비율 높은순":
        filtered = filtered.sort_values("휴업일비율", ascending=False)
    elif sort_option == "방학일수 많은순":
        filtered = filtered.sort_values("방학일수", ascending=False)
    else:
        filtered = filtered.sort_values("학교명", ascending=True)

    st.caption(f"검색 결과: {len(filtered):,}개")

    show_cols = [
        "시도교육청명",
        "학교명",
        "학교과정명",
        "분석일수",
        "휴업일수",
        "수업일수",
        "휴업일비율",
        "평균위험점수",
        "고위험일비율",
        "최종위험지수",
        "위험등급",
        "방학일수",
        "상대위험등급",
    ]
    show_cols = [c for c in show_cols if c in filtered.columns]

    st.dataframe(filtered[show_cols].head(500), use_container_width=True, height=340)

    if filtered.empty:
        st.warning("조건에 맞는 학교가 없습니다.")
        st.stop()

    selected_label = st.selectbox(
        "상세 분석을 볼 학교를 선택하세요.",
        filtered["학교표시명"].tolist(),
    )

    selected_row = filtered[filtered["학교표시명"] == selected_label].iloc[0]
    selected_id = selected_row["분석단위ID"]

    match = detail_index[detail_index["분석단위ID"] == selected_id]

    if match.empty:
        st.error("선택한 학교의 날짜별 상세 데이터 위치를 찾지 못했습니다.")
        st.stop()

    detail_file = match["detail_file"].iloc[0]

    try:
        detail_part = load_detail_part(detail_file)
    except Exception as e:
        st.error(f"상세 데이터 파일을 읽지 못했습니다: {detail_file}")
        st.exception(e)
        st.stop()

    one_school = detail_part[detail_part["분석단위ID"] == selected_id].copy()

    if one_school.empty:
        st.error("선택한 학교의 날짜별 상세 데이터가 비어 있습니다.")
        st.stop()

    one_school["date"] = pd.to_datetime(one_school["date"], errors="coerce")

    st.subheader(f"📌 {selected_row['학교표시명']} 상세 분석")

    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("최종위험지수", f"{selected_row['최종위험지수']:.2f}")
    d2.metric("휴업일비율", f"{selected_row['휴업일비율']:.2f}%")
    d3.metric("분석일수", f"{int(selected_row['분석일수']):,}일")
    d4.metric("방학일수", f"{int(selected_row['방학일수']):,}일")
    d5.metric("고위험일비율", f"{selected_row['고위험일비율']:.2f}%")

    st.plotly_chart(
        make_memory_figure(one_school, selected_row["학교표시명"]),
        use_container_width=True,
    )

    months = pd.date_range(
        one_school["date"].min(),
        one_school["date"].max(),
        freq="MS",
    ).strftime("%Y-%m").tolist()

    calendar_month = st.selectbox("달력 월 선택", months)
    components.html(make_calendar_html(one_school, calendar_month), height=720, scrolling=True)

    with st.expander("선택 학교 날짜별 데이터 보기", expanded=False):
        detail_cols = [
            "date",
            "is_vacation_day",
            "is_main_rest_day",
            "is_study_day",
            "memory_score",
            "risk_score",
            "p_value",
            "k_value",
            "calculation_type",
            "행사명_목록",
        ]
        detail_cols = [c for c in detail_cols if c in one_school.columns]
        st.dataframe(one_school[detail_cols], use_container_width=True, height=400)


# =========================================================
# 탭 4. 다운로드
# =========================================================

with tab4:
    st.subheader("결과 다운로드")

    csv_bytes = core_summary.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "핵심 분석 결과 CSV 다운로드",
        data=csv_bytes,
        file_name="core_summary.csv",
        mime="text/csv",
    )

    excel_bytes = to_excel_bytes(core_summary)
    st.download_button(
        "핵심 분석 결과 엑셀 다운로드",
        data=excel_bytes,
        file_name="학습공백_위험지수_분석결과.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )