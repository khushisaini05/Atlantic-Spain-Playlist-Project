import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Atlantic Spain – Music Lifecycle Dashboard",
    page_icon="🎵",
    layout="wide",
)

# ─────────────────────────────────────────────
# LOAD & PREPROCESS DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Atlantic_Spain.csv")
    # Parse multiple date formats safely
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")
    df["is_explicit"] = df["is_explicit"].astype(str).str.upper().map({"TRUE": True, "FALSE": False}).fillna(False)
    df["duration_min"] = df["duration_ms"] / 60000
    return df

@st.cache_data
def build_lifecycle(df):
    """Build per-song lifecycle metrics."""
    rows = []
    for (song, artist), grp in df.groupby(["song", "artist"]):
        grp = grp.sort_values("date")
        entry = grp["date"].min()
        exit_ = grp["date"].max()
        days  = (exit_ - entry).days + 1
        peak  = grp["position"].min()
        peak_date = grp.loc[grp["position"].idxmin(), "date"]
        time_to_peak = (peak_date - entry).days
        avg_pos = grp["position"].mean()
        avg_pop = grp["popularity"].mean()
        is_exp  = grp["is_explicit"].iloc[0]
        alb_type = grp["album_type"].iloc[0]
        tot_tracks = grp["total_tracks"].iloc[0]
        dur_min  = grp["duration_min"].mean()

        # Lifecycle stage
        if days <= 7:
            stage = "New Entry"
        elif peak <= 10:
            stage = "Peak Phase"
        elif avg_pos <= 20:
            stage = "Growth Phase"
        elif avg_pos <= 35:
            stage = "Mature Phase"
        else:
            stage = "Decline Phase"

        rows.append({
            "song": song, "artist": artist,
            "entry_date": entry, "exit_date": exit_,
            "days_on_playlist": days, "peak_position": peak,
            "time_to_peak_days": time_to_peak,
            "avg_position": avg_pos, "avg_popularity": avg_pop,
            "is_explicit": is_exp, "album_type": alb_type,
            "total_tracks": tot_tracks, "duration_min": dur_min,
            "lifecycle_stage": stage,
        })
    return pd.DataFrame(rows)

@st.cache_data
def daily_churn(df):
    """Compute daily entry / exit counts."""
    lc = build_lifecycle(df)
    entries = lc.groupby("entry_date").size().reset_index(name="entries")
    exits   = lc.groupby("exit_date").size().reset_index(name="exits")
    dates   = pd.DataFrame({"date": pd.date_range(df["date"].min(), df["date"].max())})
    churn   = dates.merge(entries.rename(columns={"entry_date":"date"}), on="date", how="left")
    churn   = churn.merge(exits.rename(columns={"exit_date":"date"}), on="date", how="left")
    churn   = churn.fillna(0)
    churn["churn_rate"] = (churn["entries"] + churn["exits"]) / 50 * 100
    return churn

df_raw = load_data()
df_lc  = build_lifecycle(df_raw)
df_churn = daily_churn(df_raw)

# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────
st.sidebar.title("🎛️ Filters")

date_min, date_max = df_raw["date"].min().date(), df_raw["date"].max().date()
date_range = st.sidebar.date_input("Date Range", value=(date_min, date_max), min_value=date_min, max_value=date_max)

stages = st.sidebar.multiselect(
    "Lifecycle Stage",
    options=["New Entry","Growth Phase","Peak Phase","Mature Phase","Decline Phase"],
    default=["New Entry","Growth Phase","Peak Phase","Mature Phase","Decline Phase"]
)

explicit_opt = st.sidebar.radio("Content Maturity", ["All", "Explicit Only", "Clean Only"])
album_opt    = st.sidebar.radio("Album Type", ["All", "Single", "Album"])

# Apply filters to lifecycle df
lc = df_lc.copy()
if len(date_range) == 2:
    lc = lc[(lc["entry_date"].dt.date >= date_range[0]) & (lc["exit_date"].dt.date <= date_range[1])]
if stages:
    lc = lc[lc["lifecycle_stage"].isin(stages)]
if explicit_opt == "Explicit Only":
    lc = lc[lc["is_explicit"] == True]
elif explicit_opt == "Clean Only":
    lc = lc[lc["is_explicit"] == False]
if album_opt != "All":
    lc = lc[lc["album_type"] == album_opt.lower()]

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🎵 Atlantic Recording Corporation")
st.subheader("Spain Top 50 – Content Maturity, Release Lifecycle & Playlist Rotation Analysis")
st.caption(f"Dataset: {df_raw['date'].min().date()} → {df_raw['date'].max().date()} | {df_raw['song'].nunique()} unique songs | {len(df_raw)} daily records")

# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Avg Days on Playlist",    f"{lc['days_on_playlist'].mean():.1f}")
k2.metric("Avg Entry-to-Peak",       f"{lc['time_to_peak_days'].mean():.1f} days")
k3.metric("Avg Churn Rate",          f"{df_churn['churn_rate'].mean():.1f}%")
k4.metric("Peak Positions ≤10",      f"{(lc['peak_position'] <= 10).sum()}")
explicit_score = lc[lc["is_explicit"]==True]["days_on_playlist"].mean()
clean_score    = lc[lc["is_explicit"]==False]["days_on_playlist"].mean()
k5.metric("Explicit Lifecycle Score", f"{explicit_score:.1f}d", delta=f"{explicit_score-clean_score:+.1f} vs Clean")
single_days = lc[lc["album_type"]=="single"]["days_on_playlist"].mean()
album_days  = lc[lc["album_type"]=="album"]["days_on_playlist"].mean()
ratio = single_days/album_days if album_days else 0
k6.metric("Single/Album Longevity Ratio", f"{ratio:.2f}x")

st.divider()

# ─────────────────────────────────────────────
# TAB LAYOUT
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📅 Lifecycle Timeline",
    "🔄 Playlist Churn",
    "🎭 Stage Distribution",
    "🔞 Content Maturity",
    "💿 Release Strategy"
])

# ══════════════════════════════════════════════
# TAB 1 – LIFECYCLE TIMELINE
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### Song Lifecycle Timeline (Top 40 by days)")
    top_songs = lc.nlargest(40, "days_on_playlist")
    fig = px.timeline(
        top_songs,
        x_start="entry_date", x_end="exit_date",
        y="song", color="lifecycle_stage",
        hover_data=["artist","peak_position","days_on_playlist","album_type"],
        color_discrete_map={
            "New Entry":"#4CAF50","Growth Phase":"#2196F3",
            "Peak Phase":"#FF9800","Mature Phase":"#9C27B0","Decline Phase":"#F44336"
        },
        title="Song Stay Duration on Spain Top 50"
    )
    fig.update_layout(height=650, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Days on Playlist Distribution")
        fig2 = px.histogram(lc, x="days_on_playlist", nbins=30, color="lifecycle_stage",
                            color_discrete_map={"New Entry":"#4CAF50","Growth Phase":"#2196F3",
                                                "Peak Phase":"#FF9800","Mature Phase":"#9C27B0","Decline Phase":"#F44336"})
        fig2.update_layout(bargap=0.1)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("#### Entry-to-Peak Speed vs Peak Position")
        fig3 = px.scatter(lc, x="time_to_peak_days", y="peak_position",
                          color="lifecycle_stage", size="days_on_playlist",
                          hover_data=["song","artist"],
                          color_discrete_map={"New Entry":"#4CAF50","Growth Phase":"#2196F3",
                                              "Peak Phase":"#FF9800","Mature Phase":"#9C27B0","Decline Phase":"#F44336"})
        fig3.update_yaxes(autorange="reversed")
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 – PLAYLIST CHURN
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### Daily Playlist Rotation & Churn Analysis")

    fig_churn = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=("Daily Entries vs Exits", "Daily Churn Rate (%)"))
    fig_churn.add_trace(go.Bar(x=df_churn["date"], y=df_churn["entries"],
                                name="Entries", marker_color="#4CAF50"), row=1, col=1)
    fig_churn.add_trace(go.Bar(x=df_churn["date"], y=df_churn["exits"],
                                name="Exits", marker_color="#F44336"), row=1, col=1)
    fig_churn.add_trace(go.Scatter(x=df_churn["date"], y=df_churn["churn_rate"],
                                    name="Churn Rate %", line=dict(color="#FF9800", width=2)), row=2, col=1)
    fig_churn.update_layout(height=550, barmode="group")
    st.plotly_chart(fig_churn, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Unique Songs", df_lc["song"].nunique())
    col2.metric("Avg Daily Entries", f"{df_churn['entries'].mean():.1f}")
    col3.metric("Avg Daily Exits",   f"{df_churn['exits'].mean():.1f}")

    # Monthly churn
    df_churn["month"] = df_churn["date"].dt.to_period("M").astype(str)
    monthly = df_churn.groupby("month")[["entries","exits","churn_rate"]].mean().reset_index()
    st.markdown("#### Monthly Average Rotation")
    fig_m = px.bar(monthly, x="month", y=["entries","exits"],
                   barmode="group", color_discrete_sequence=["#4CAF50","#F44336"])
    st.plotly_chart(fig_m, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 – LIFECYCLE STAGE DISTRIBUTION
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### Lifecycle Stage Distribution")
    col1, col2 = st.columns(2)

    with col1:
        stage_counts = lc["lifecycle_stage"].value_counts().reset_index()
        stage_counts.columns = ["stage","count"]
        fig_pie = px.pie(stage_counts, names="stage", values="count",
                         color="stage",
                         color_discrete_map={"New Entry":"#4CAF50","Growth Phase":"#2196F3",
                                              "Peak Phase":"#FF9800","Mature Phase":"#9C27B0","Decline Phase":"#F44336"},
                         title="Songs by Lifecycle Stage")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        stage_life = lc.groupby("lifecycle_stage")["days_on_playlist"].mean().reset_index()
        fig_bar = px.bar(stage_life, x="lifecycle_stage", y="days_on_playlist",
                         color="lifecycle_stage",
                         color_discrete_map={"New Entry":"#4CAF50","Growth Phase":"#2196F3",
                                              "Peak Phase":"#FF9800","Mature Phase":"#9C27B0","Decline Phase":"#F44336"},
                         title="Avg Days on Playlist per Stage")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### Popularity Score by Stage")
    fig_box = px.box(lc, x="lifecycle_stage", y="avg_popularity",
                     color="lifecycle_stage",
                     color_discrete_map={"New Entry":"#4CAF50","Growth Phase":"#2196F3",
                                          "Peak Phase":"#FF9800","Mature Phase":"#9C27B0","Decline Phase":"#F44336"})
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("#### Top 10 Longest-Staying Songs")
    top10 = lc.nlargest(10, "days_on_playlist")[
        ["song","artist","days_on_playlist","peak_position","lifecycle_stage","is_explicit","album_type"]
    ].reset_index(drop=True)
    top10.index += 1
    st.dataframe(top10, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 – CONTENT MATURITY (EXPLICIT vs CLEAN)
# ══════════════════════════════════════════════
with tab4:
    st.markdown("### Explicit vs Clean Content Lifecycle Comparison")

    col1, col2 = st.columns(2)
    with col1:
        exp_days = lc.groupby("is_explicit")["days_on_playlist"].mean().reset_index()
        exp_days["label"] = exp_days["is_explicit"].map({True:"Explicit", False:"Clean"})
        fig_e1 = px.bar(exp_days, x="label", y="days_on_playlist",
                        color="label", color_discrete_map={"Explicit":"#E91E63","Clean":"#03A9F4"},
                        title="Avg Days on Playlist")
        st.plotly_chart(fig_e1, use_container_width=True)

    with col2:
        exp_peak = lc.groupby("is_explicit")["peak_position"].mean().reset_index()
        exp_peak["label"] = exp_peak["is_explicit"].map({True:"Explicit", False:"Clean"})
        fig_e2 = px.bar(exp_peak, x="label", y="peak_position",
                        color="label", color_discrete_map={"Explicit":"#E91E63","Clean":"#03A9F4"},
                        title="Avg Peak Position (lower = better)")
        fig_e2.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_e2, use_container_width=True)

    lc["explicit_label"] = lc["is_explicit"].map({True:"Explicit", False:"Clean"})
    fig_e3 = px.histogram(lc, x="days_on_playlist", color="explicit_label",
                           barmode="overlay", opacity=0.7,
                           color_discrete_map={"Explicit":"#E91E63","Clean":"#03A9F4"},
                           title="Lifecycle Length Distribution: Explicit vs Clean")
    st.plotly_chart(fig_e3, use_container_width=True)

    st.markdown("#### Stage Breakdown by Content Type")
    stage_exp = lc.groupby(["lifecycle_stage","explicit_label"]).size().reset_index(name="count")
    fig_e4 = px.bar(stage_exp, x="lifecycle_stage", y="count", color="explicit_label",
                    barmode="group", color_discrete_map={"Explicit":"#E91E63","Clean":"#03A9F4"})
    st.plotly_chart(fig_e4, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 5 – RELEASE STRATEGY (SINGLE vs ALBUM)
# ══════════════════════════════════════════════
with tab5:
    st.markdown("### Single vs Album Track Release Strategy")

    col1, col2, col3 = st.columns(3)
    s_days = lc[lc["album_type"]=="single"]["days_on_playlist"].mean()
    a_days = lc[lc["album_type"]=="album"]["days_on_playlist"].mean()
    s_peak = lc[lc["album_type"]=="single"]["peak_position"].mean()
    a_peak = lc[lc["album_type"]=="album"]["peak_position"].mean()
    col1.metric("Single – Avg Days",  f"{s_days:.1f}")
    col2.metric("Album – Avg Days",   f"{a_days:.1f}")
    col3.metric("Longevity Ratio",    f"{s_days/a_days:.2f}x (Single/Album)")

    col1, col2 = st.columns(2)
    with col1:
        alb_life = lc.groupby("album_type")["days_on_playlist"].mean().reset_index()
        fig_a1 = px.bar(alb_life, x="album_type", y="days_on_playlist",
                        color="album_type", color_discrete_map={"single":"#FF9800","album":"#3F51B5"},
                        title="Avg Days on Playlist")
        st.plotly_chart(fig_a1, use_container_width=True)

    with col2:
        alb_stage = lc.groupby(["lifecycle_stage","album_type"]).size().reset_index(name="count")
        fig_a2 = px.bar(alb_stage, x="lifecycle_stage", y="count", color="album_type",
                        barmode="group", color_discrete_map={"single":"#FF9800","album":"#3F51B5"},
                        title="Stage Distribution by Release Type")
        st.plotly_chart(fig_a2, use_container_width=True)

    st.markdown("#### Duration vs Retention: Does song length matter?")
    fig_dur = px.scatter(lc, x="duration_min", y="days_on_playlist",
                         color="album_type", trendline="ols",
                         hover_data=["song","artist"],
                         color_discrete_map={"single":"#FF9800","album":"#3F51B5"},
                         title="Song Duration (min) vs Days on Playlist")
    st.plotly_chart(fig_dur, use_container_width=True)

    st.markdown("#### Album Size vs Lifecycle Stability")
    fig_tracks = px.scatter(lc, x="total_tracks", y="days_on_playlist",
                             color="album_type", size="avg_popularity",
                             hover_data=["song","artist"],
                             color_discrete_map={"single":"#FF9800","album":"#3F51B5"},
                             title="Total Tracks in Album vs Days on Playlist")
    st.plotly_chart(fig_tracks, use_container_width=True)

    st.markdown("#### Top Artists by Avg Days on Playlist")
    artist_lc = lc.groupby("artist").agg(
        avg_days=("days_on_playlist","mean"),
        songs=("song","count"),
        avg_peak=("peak_position","mean")
    ).reset_index().sort_values("avg_days", ascending=False).head(20)
    fig_art = px.bar(artist_lc, x="avg_days", y="artist", orientation="h",
                     color="avg_peak", color_continuous_scale="RdYlGn_r",
                     title="Top 20 Artists by Avg Playlist Stay")
    fig_art.update_layout(height=550, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_art, use_container_width=True)

st.divider()
st.caption("Atlantic Recording Corporation | Spain Top 50 Analytics | Built with Streamlit & Plotly")
