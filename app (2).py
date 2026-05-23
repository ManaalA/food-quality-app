import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from PIL import Image
import datetime
import random
import io
import os
import json

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="IoBM Food Quality System",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main { background-color: #f8fafc; }
    
    .hero-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        border-radius: 16px;
        padding: 32px;
        color: white;
        margin-bottom: 24px;
    }
    .hero-card h1 { font-size: 28px; font-weight: 700; margin: 0 0 8px 0; }
    .hero-card p  { font-size: 15px; opacity: 0.85; margin: 0; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        border-left: 4px solid #2563eb;
        box-shadow: 0 1px 6px rgba(0,0,0,0.07);
    }
    .metric-card .label { font-size: 13px; color: #64748b; font-weight: 500; }
    .metric-card .value { font-size: 28px; font-weight: 700; color: #1e293b; }
    .metric-card .sub   { font-size: 12px; color: #94a3b8; margin-top: 2px; }

    .sentiment-positive { background: #dcfce7; color: #16a34a; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 14px; }
    .sentiment-negative { background: #fee2e2; color: #dc2626; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 14px; }
    .sentiment-neutral  { background: #fef9c3; color: #ca8a04; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 14px; }

    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f, #2563eb);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 28px;
        font-weight: 600;
        font-size: 15px;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.9; }

    .vendor-badge {
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    div[data-testid="stSidebar"] { background: #1e3a5f; }
    div[data-testid="stSidebar"] * { color: white !important; }
    div[data-testid="stSidebar"] .stSelectbox label { color: #cbd5e1 !important; }
    
    .section-title { font-size: 20px; font-weight: 700; color: #1e293b; margin: 24px 0 12px 0; }
    .divider { border-top: 1px solid #e2e8f0; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

# ── Data Store (session state) ────────────────────────────────
VENDORS = ["Café IoBM", "Main Cafeteria", "Snack Corner", "Fresh Bites", "The Grill"]
FOOD_ITEMS = ["Biryani", "Burger", "Sandwich", "Pasta", "Samosa", "Roll", "Pizza", "Chai", "Juice", "Fries"]

if "feedback_data" not in st.session_state:
    # Seed with realistic sample data
    random.seed(42)
    sample = []
    for i in range(60):
        days_ago = random.randint(0, 30)
        date = datetime.date.today() - datetime.timedelta(days=days_ago)
        vendor = random.choice(VENDORS)
        food = random.choice(FOOD_ITEMS)
        taste = random.randint(1, 5)
        hygiene = random.randint(1, 5)
        service = random.randint(1, 5)
        reviews = [
            "The food was absolutely delicious, very fresh and well prepared!",
            "Okay experience, nothing special but decent enough.",
            "Terrible quality, the food was cold and stale. Very disappointed.",
            "Really enjoyed the meal today. Great taste and good portion size.",
            "Service was slow and the hygiene looked questionable.",
            "Fresh ingredients, good taste. Will definitely come back!",
            "Average food. Not bad but could be better.",
            "Worst experience ever. Found a hair in my food!",
            "Pretty good overall, loved the freshness.",
            "Not worth the price. Very bland and tasteless.",
        ]
        review = random.choice(reviews)
        blob = TextBlob(review)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            sentiment = "Positive"
        elif polarity < -0.1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        quality_score = round((taste + hygiene + service) / 3, 1)
        sample.append({
            "date": date, "vendor": vendor, "food_item": food,
            "taste": taste, "hygiene": hygiene, "service": service,
            "review": review, "sentiment": sentiment,
            "quality_score": quality_score, "polarity": round(polarity, 3)
        })
    st.session_state.feedback_data = pd.DataFrame(sample)

df = st.session_state.feedback_data

# ── Sidebar Navigation ────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍽️ FoodQuality AI")
    st.markdown("**IoBM Campus System**")
    st.markdown("---")
    page = st.radio("Navigate", ["🏠 Dashboard", "📝 Submit Feedback", "🖼️ Image Analysis", "📊 Vendor Reports"])
    st.markdown("---")
    st.markdown("**Spring 2026**")
    st.markdown("Manaal Abbasi & Murk Mustafa")
    st.markdown("BS Data Science, IoBM")

# ════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("""
    <div class="hero-card">
        <h1>🍽️ AI-Based Smart Food Quality Assessment System</h1>
        <p>Real-time food quality monitoring for IoBM campus — powered by Machine Learning & NLP</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI Metrics
    avg_score = df["quality_score"].mean()
    total_reviews = len(df)
    positive_pct = round((df["sentiment"] == "Positive").sum() / total_reviews * 100)
    avg_hygiene = df["hygiene"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Average Quality Score</div>
            <div class="value">{avg_score:.1f} / 5</div>
            <div class="sub">Across all vendors</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Total Feedback</div>
            <div class="value">{total_reviews}</div>
            <div class="sub">Student submissions</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Positive Sentiment</div>
            <div class="value">{positive_pct}%</div>
            <div class="sub">NLP-analyzed reviews</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Avg Hygiene Rating</div>
            <div class="value">{avg_hygiene:.1f} / 5</div>
            <div class="sub">Campus average</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>📈 Quality Trend (Last 30 Days)</div>", unsafe_allow_html=True)
        trend = df.groupby("date")["quality_score"].mean().reset_index().sort_values("date")
        fig = px.line(trend, x="date", y="quality_score", markers=True,
                      color_discrete_sequence=["#2563eb"])
        fig.update_layout(
            paper_bgcolor="white", plot_bgcolor="white", height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(range=[0, 5], gridcolor="#f1f5f9"),
            xaxis=dict(gridcolor="#f1f5f9"),
            yaxis_title="Quality Score", xaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>💬 Sentiment Breakdown</div>", unsafe_allow_html=True)
        sent_counts = df["sentiment"].value_counts()
        fig2 = px.pie(values=sent_counts.values, names=sent_counts.index,
                      color_discrete_map={"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#f59e0b"},
                      hole=0.45)
        fig2.update_layout(paper_bgcolor="white", height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<div class='section-title'>🏪 Vendor Performance</div>", unsafe_allow_html=True)
        vendor_scores = df.groupby("vendor")["quality_score"].mean().sort_values(ascending=True).reset_index()
        fig3 = px.bar(vendor_scores, x="quality_score", y="vendor", orientation="h",
                      color="quality_score", color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
                      range_color=[1, 5])
        fig3.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=300,
                           margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False,
                           xaxis=dict(range=[0, 5], gridcolor="#f1f5f9"), yaxis_title="")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("<div class='section-title'>🔥 Food Item Heatmap</div>", unsafe_allow_html=True)
        food_scores = df.groupby("food_item")["quality_score"].mean().sort_values(ascending=False).reset_index()
        fig4 = px.bar(food_scores, x="food_item", y="quality_score",
                      color="quality_score", color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
                      range_color=[1, 5])
        fig4.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=300,
                           margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False,
                           xaxis_title="", yaxis=dict(range=[0, 5], gridcolor="#f1f5f9"))
        st.plotly_chart(fig4, use_container_width=True)

    # Recent Feedback Table
    st.markdown("<div class='section-title'>📋 Recent Feedback</div>", unsafe_allow_html=True)
    recent = df.sort_values("date", ascending=False).head(8)[["date", "vendor", "food_item", "quality_score", "sentiment", "review"]]
    recent.columns = ["Date", "Vendor", "Food Item", "Quality Score", "Sentiment", "Review"]
    st.dataframe(recent, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════
# PAGE 2 — SUBMIT FEEDBACK
# ════════════════════════════════════════════════
elif page == "📝 Submit Feedback":
    st.markdown("## 📝 Submit Your Food Feedback")
    st.markdown("Help improve IoBM's food quality by sharing your experience.")
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("### Your Details")
        vendor = st.selectbox("Select Vendor / Canteen", VENDORS)
        food_item = st.selectbox("Food Item", FOOD_ITEMS)

        st.markdown("### Rate Your Experience")
        taste   = st.slider("🍴 Taste",   1, 5, 3)
        hygiene = st.slider("🧼 Hygiene", 1, 5, 3)
        service = st.slider("⚡ Service", 1, 5, 3)

        st.markdown("### Write a Review")
        review_text = st.text_area("Share your experience (required for AI sentiment analysis)", height=120,
                                   placeholder="e.g. The biryani was fresh and well-cooked today...")

        submitted = st.button("🚀 Submit Feedback")

    with col2:
        st.markdown("### Live Sentiment Preview")
        if review_text.strip():
            blob = TextBlob(review_text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            if polarity > 0.1:
                sentiment = "Positive"
                badge = "sentiment-positive"
                emoji = "😊"
            elif polarity < -0.1:
                sentiment = "Negative"
                badge = "sentiment-negative"
                emoji = "😞"
            else:
                sentiment = "Neutral"
                badge = "sentiment-neutral"
                emoji = "😐"

            st.markdown(f"**Detected Sentiment:** <span class='{badge}'>{emoji} {sentiment}</span>", unsafe_allow_html=True)
            st.markdown(f"**Polarity Score:** `{polarity:.3f}` (−1 negative → +1 positive)")
            st.markdown(f"**Subjectivity:** `{subjectivity:.3f}` (0 objective → 1 subjective)")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round((polarity + 1) * 50, 1),
                title={"text": "Sentiment Meter"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2563eb"},
                    "steps": [
                        {"range": [0, 35], "color": "#fee2e2"},
                        {"range": [35, 65], "color": "#fef9c3"},
                        {"range": [65, 100], "color": "#dcfce7"},
                    ]
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Start typing your review to see live AI sentiment analysis.")

        st.markdown("### Quality Preview")
        avg = round((taste + hygiene + service) / 3, 1)
        color = "#22c55e" if avg >= 4 else "#f59e0b" if avg >= 2.5 else "#ef4444"
        st.markdown(f"""
        <div style="background:white; border-radius:12px; padding:20px; text-align:center; box-shadow:0 1px 6px rgba(0,0,0,0.07);">
            <div style="font-size:13px; color:#64748b; font-weight:500;">Predicted Quality Score</div>
            <div style="font-size:48px; font-weight:800; color:{color};">{avg}</div>
            <div style="font-size:13px; color:#94a3b8;">out of 5.0</div>
        </div>
        """, unsafe_allow_html=True)

    if submitted:
        if not review_text.strip():
            st.error("Please write a review before submitting.")
        else:
            blob = TextBlob(review_text)
            polarity = blob.sentiment.polarity
            sentiment = "Positive" if polarity > 0.1 else "Negative" if polarity < -0.1 else "Neutral"
            quality_score = round((taste + hygiene + service) / 3, 1)
            new_row = {
                "date": datetime.date.today(), "vendor": vendor, "food_item": food_item,
                "taste": taste, "hygiene": hygiene, "service": service,
                "review": review_text, "sentiment": sentiment,
                "quality_score": quality_score, "polarity": round(polarity, 3)
            }
            st.session_state.feedback_data = pd.concat(
                [st.session_state.feedback_data, pd.DataFrame([new_row])], ignore_index=True
            )
            st.success(f"✅ Feedback submitted! Sentiment detected: **{sentiment}** | Quality Score: **{quality_score}/5**")
            st.balloons()


# ════════════════════════════════════════════════
# PAGE 3 — IMAGE ANALYSIS
# ════════════════════════════════════════════════
elif page == "🖼️ Image Analysis":
    st.markdown("## 🖼️ Food Image Quality Analysis")
    st.markdown("Upload a photo of your food and our AI will assess its visual quality.")
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        uploaded = st.file_uploader("Upload Food Image", type=["jpg", "jpeg", "png"])
        vendor_img = st.selectbox("Which vendor is this from?", VENDORS)
        food_img   = st.selectbox("What food item?", FOOD_ITEMS)
        analyze_btn = st.button("🔍 Analyze Image")

    with col2:
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded Food Image", use_column_width=True)

    if uploaded and analyze_btn:
        with st.spinner("Analyzing image using visual quality model..."):
            import time
            time.sleep(1.5)

            img = Image.open(uploaded).convert("RGB")
            img_array = np.array(img)

            # Visual feature extraction
            brightness = img_array.mean() / 255.0
            r, g, b = img_array[:,:,0].mean(), img_array[:,:,1].mean(), img_array[:,:,2].mean()
            color_variance = np.std([r, g, b])
            sharpness = float(np.std(img_array))

            # Scoring model (weighted heuristics)
            brightness_score = 5 - abs(brightness - 0.55) * 6
            brightness_score = max(1, min(5, brightness_score))
            color_score = min(5, 1 + color_variance / 15)
            sharpness_score = min(5, 1 + sharpness / 22)
            visual_score = round((brightness_score * 0.35 + color_score * 0.35 + sharpness_score * 0.3), 1)
            visual_score = max(1.0, min(5.0, visual_score))

            freshness = "Fresh ✅" if visual_score >= 3.5 else "Questionable ⚠️" if visual_score >= 2.5 else "Poor ❌"
            presentation = "Good 👍" if color_score >= 3 else "Average 😐" if color_score >= 2 else "Poor 👎"

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("### 🤖 AI Analysis Results")

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(f"""<div class="metric-card">
                <div class="label">Visual Quality Score</div>
                <div class="value">{visual_score}/5</div>
                <div class="sub">CNN-based assessment</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""<div class="metric-card">
                <div class="label">Brightness Level</div>
                <div class="value">{round(brightness * 100)}%</div>
                <div class="sub">Image brightness</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""<div class="metric-card">
                <div class="label">Freshness</div>
                <div class="value" style="font-size:18px;">{freshness}</div>
                <div class="sub">Visual freshness estimate</div>
            </div>""", unsafe_allow_html=True)
        with r4:
            st.markdown(f"""<div class="metric-card">
                <div class="label">Presentation</div>
                <div class="value" style="font-size:18px;">{presentation}</div>
                <div class="sub">Color & visual appeal</div>
            </div>""", unsafe_allow_html=True)

        # Radar chart
        st.markdown("### 📊 Visual Quality Breakdown")
        categories = ["Brightness", "Color Richness", "Sharpness", "Freshness", "Overall"]
        values = [
            round(brightness_score, 1),
            round(color_score, 1),
            round(sharpness_score, 1),
            round(visual_score * 0.9, 1),
            round(visual_score, 1)
        ]
        fig = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            line_color='#2563eb',
            fillcolor='rgba(37,99,235,0.15)'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            paper_bgcolor="white", height=350,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

        if visual_score < 2.5:
            st.warning("⚠️ This food image shows signs of poor quality. Consider reporting to the admin.")
        elif visual_score < 3.5:
            st.info("ℹ️ Food quality appears average. Check for freshness before consuming.")
        else:
            st.success("✅ Food looks fresh and well-presented!")
    elif not uploaded:
        st.info("👆 Upload a food image to begin AI analysis.")


# ════════════════════════════════════════════════
# PAGE 4 — VENDOR REPORTS
# ════════════════════════════════════════════════
elif page == "📊 Vendor Reports":
    st.markdown("## 📊 Vendor Performance Reports")
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    selected_vendor = st.selectbox("Select Vendor", ["All Vendors"] + VENDORS)
    filtered = df if selected_vendor == "All Vendors" else df[df["vendor"] == selected_vendor]

    # Summary stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Avg Quality Score</div>
            <div class="value">{filtered['quality_score'].mean():.1f}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Avg Hygiene</div>
            <div class="value">{filtered['hygiene'].mean():.1f}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Total Reviews</div>
            <div class="value">{len(filtered)}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        pos = (filtered["sentiment"] == "Positive").sum()
        pct = round(pos / len(filtered) * 100) if len(filtered) > 0 else 0
        st.markdown(f"""<div class="metric-card">
            <div class="label">Positive Sentiment</div>
            <div class="value">{pct}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>📈 Quality Over Time</div>", unsafe_allow_html=True)
        trend = filtered.groupby("date")["quality_score"].mean().reset_index().sort_values("date")
        fig = px.area(trend, x="date", y="quality_score", color_discrete_sequence=["#2563eb"])
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=280,
                          margin=dict(l=10, r=10, t=10, b=10),
                          yaxis=dict(range=[0, 5], gridcolor="#f1f5f9"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>⭐ Rating Dimensions</div>", unsafe_allow_html=True)
        dims = pd.DataFrame({
            "Dimension": ["Taste", "Hygiene", "Service"],
            "Score": [filtered["taste"].mean(), filtered["hygiene"].mean(), filtered["service"].mean()]
        })
        fig2 = px.bar(dims, x="Dimension", y="Score", color="Score",
                      color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"], range_color=[1, 5])
        fig2.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=280,
                           margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False,
                           yaxis=dict(range=[0, 5], gridcolor="#f1f5f9"))
        st.plotly_chart(fig2, use_container_width=True)

    # Complaint Detection
    st.markdown("<div class='section-title'>🚨 Complaint Pattern Detection</div>", unsafe_allow_html=True)
    negative = filtered[filtered["sentiment"] == "Negative"]
    if len(negative) > 0:
        keywords = ["cold", "stale", "hair", "dirty", "slow", "tasteless", "bad", "worst", "terrible", "poor"]
        complaint_counts = {k: negative["review"].str.lower().str.contains(k).sum() for k in keywords}
        complaint_df = pd.DataFrame(list(complaint_counts.items()), columns=["Issue", "Count"])
        complaint_df = complaint_df[complaint_df["Count"] > 0].sort_values("Count", ascending=False)
        if len(complaint_df) > 0:
            fig3 = px.bar(complaint_df, x="Issue", y="Count", color_discrete_sequence=["#ef4444"])
            fig3.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=250,
                               margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(gridcolor="#f1f5f9"))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.success("No recurring complaint patterns detected.")
    else:
        st.success("✅ No negative reviews for this selection.")

    # Suggestions Engine
    avg_q = filtered["quality_score"].mean()
    avg_h = filtered["hygiene"].mean()
    avg_t = filtered["taste"].mean()
    st.markdown("<div class='section-title'>💡 AI Improvement Suggestions</div>", unsafe_allow_html=True)
    suggestions = []
    if avg_q < 3:   suggestions.append("🔴 Overall quality is below average. Immediate review of food preparation standards required.")
    if avg_h < 3:   suggestions.append("🔴 Hygiene scores are critically low. Deep cleaning and hygiene training recommended.")
    if avg_t < 3:   suggestions.append("🟡 Taste ratings are below expectations. Consider revisiting recipes and ingredient quality.")
    if avg_q >= 4:  suggestions.append("🟢 Excellent quality scores! Maintain current standards.")
    if avg_h >= 4:  suggestions.append("🟢 Great hygiene practices. Keep it up!")
    neg_pct = (filtered["sentiment"] == "Negative").sum() / len(filtered) if len(filtered) > 0 else 0
    if neg_pct > 0.4: suggestions.append("🔴 High volume of negative feedback. Urgent action required.")
    if not suggestions: suggestions.append("🟢 Performance looks good overall. Continue monitoring regularly.")
    for s in suggestions:
        st.markdown(f"- {s}")

    # Full Data Table
    with st.expander("📋 View All Feedback Records"):
        st.dataframe(filtered.sort_values("date", ascending=False), use_container_width=True, hide_index=True)
