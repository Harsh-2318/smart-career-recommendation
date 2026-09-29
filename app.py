import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Smart Career Recommendation System",
    page_icon="🎓",
    layout="wide"
)



# CUSTOM CSS


st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 30px;
}

.card {
    padding: 25px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.result-card {
    padding: 30px;
    border-radius: 18px;
    background-color: #eef7ff;
    text-align: center;
    margin-top: 25px;
}

.career {
    font-size: 32px;
    font-weight: 700;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)



# TITLE

st.markdown(
    '<div class="title">🎓 Smart Career Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Career Prediction System</div>',
    unsafe_allow_html=True
)



# DATASET

data = {
    "Coding":        [9,7,8,7,8,6,3,4,2,5,3,2,1,4,5],
    "Maths":         [9,8,8,6,7,5,4,5,3,8,4,2,2,3,5],
    "Communication": [5,7,5,6,5,6,9,8,7,5,8,6,5,4,5],
    "Leadership":    [4,5,4,5,4,5,8,9,8,4,7,4,3,3,4],
    "Business":      [3,4,2,3,2,3,8,9,7,3,8,4,2,2,3],
    "Finance":       [4,3,2,2,3,2,5,7,8,9,4,2,1,1,2],
    "Technology":    [9,8,9,8,8,7,4,3,2,4,5,8,2,3,9],
    "Government":    [2,2,1,1,1,2,3,2,9,2,3,1,1,1,1],
    "Management":    [4,5,4,5,4,5,8,9,7,4,8,3,2,2,3],
    "ProblemSolving":[9,8,9,7,8,8,5,5,4,6,5,4,3,3,7],
    "Research":      [8,6,9,5,5,4,3,3,4,5,4,2,2,2,6],
    "Gaming":        [3,4,3,4,5,4,2,2,2,2,3,9,2,2,5],
    "Photography":   [2,2,1,1,2,1,3,2,2,2,3,2,9,8,2],
    "Robotics":      [7,5,8,4,5,4,2,2,1,2,3,3,2,9,8],

    "Career": [
        "Data Scientist",
        "Data Analyst",
        "ML Engineer",
        "Cloud Engineer",
        "Software Developer",
        "Cyber Security Analyst",
        "Business Analyst",
        "Entrepreneur",
        "Government Officer",
        "Financial Analyst",
        "Digital Marketer",
        "Game Developer",
        "Photographer",
        "Robotics Engineer",
        "AI Engineer"
    ]
}



# CREATE DATAFRAME

df = pd.DataFrame(data)


# =========================================================
# LABEL ENCODING
# =========================================================

encoder = LabelEncoder()

df["Career_Label"] = encoder.fit_transform(df["Career"])


# =========================================================
# FEATURES AND TARGET
# =========================================================

X = df.drop(["Career", "Career_Label"], axis=1)

y = df["Career_Label"]


# =========================================================
# TRAIN KNN MODEL
# =========================================================

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X, y)


# =========================================================
# STUDENT INFORMATION
# =========================================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("👤 Student Information")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input(
        "Student Name",
        placeholder="Enter your name"
    )

with col2:
    age = st.number_input(
        "Age",
        min_value=15,
        max_value=60,
        value=20
    )

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# SKILLS AND INTERESTS
# =========================================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("⭐ Rate Your Skills & Interests")

st.write(
    "Rate each skill from **1 to 10**, where 1 means very low "
    "and 10 means very high."
)


# ---------------------------------------------------------
# COLUMN 1
# ---------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    coding = st.slider(
        "💻 Coding Skills",
        1, 10, 5
    )

    maths = st.slider(
        "📐 Mathematics Skills",
        1, 10, 5
    )

    communication = st.slider(
        "🗣️ Communication Skills",
        1, 10, 5
    )

    leadership = st.slider(
        "👑 Leadership Skills",
        1, 10, 5
    )

    business = st.slider(
        "💼 Business Interest",
        1, 10, 5
    )

    finance = st.slider(
        "💰 Finance Interest",
        1, 10, 5
    )

    technology = st.slider(
        "⚙️ Technology Interest",
        1, 10, 5
    )


# ---------------------------------------------------------
# COLUMN 2
# ---------------------------------------------------------

with col2:

    government = st.slider(
        "🏛️ Government Job Interest",
        1, 10, 5
    )

    management = st.slider(
        "📊 Management Skills",
        1, 10, 5
    )

    problem_solving = st.slider(
        "🧩 Problem Solving Skills",
        1, 10, 5
    )

    research = st.slider(
        "🔬 Research Interest",
        1, 10, 5
    )

    gaming = st.slider(
        "🎮 Gaming Interest",
        1, 10, 5
    )

    photography = st.slider(
        "📷 Photography Interest",
        1, 10, 5
    )

    robotics = st.slider(
        "🤖 Robotics Interest",
        1, 10, 5
    )

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PREDICTION BUTTON
# =========================================================

col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    predict_button = st.button(
        "🔮 Predict My Career",
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    if name.strip() == "":
        st.warning("⚠️ Please enter your name.")

    else:

        # Create user dataframe
        user_data = pd.DataFrame([{

            "Coding": coding,
            "Maths": maths,
            "Communication": communication,
            "Leadership": leadership,
            "Business": business,
            "Finance": finance,
            "Technology": technology,
            "Government": government,
            "Management": management,
            "ProblemSolving": problem_solving,
            "Research": research,
            "Gaming": gaming,
            "Photography": photography,
            "Robotics": robotics

        }])


        # Make prediction
        prediction = model.predict(user_data)

        predicted_label = prediction[0]

        predicted_career = encoder.inverse_transform(
            [predicted_label]
        )[0]


        # Get probabilities
        probabilities = model.predict_proba(user_data)[0]

        top_indices = probabilities.argsort()[::-1][:3]


        # =================================================
        # RESULT
        # =================================================

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.subheader("🎯 Career Prediction")

        st.write(f"### Hello, {name}!")

        st.write(f"**Age:** {age}")

        st.markdown(
            f'<div class="career">🏆 {predicted_career}</div>',
            unsafe_allow_html=True
        )

        st.write(
            "Based on your skills and interests, "
            "the ML model recommends the above career."
        )

        st.markdown("</div>", unsafe_allow_html=True)


        # =================================================
        # TOP 3 CAREERS
        # =================================================

        st.subheader("📊 Top Career Recommendations")

        for rank, index in enumerate(top_indices, start=1):

            career_name = encoder.inverse_transform(
                [index]
            )[0]

            probability = probabilities[index] * 100

            st.write(
                f"**{rank}. {career_name}** — "
                f"{probability:.2f}%"
            )

            st.progress(float(probabilities[index]))


        # =================================================
        # STUDENT REPORT
        # =================================================

        st.subheader("📋 Student Report")

        report = pd.DataFrame({
            "Information": [
                "Student Name",
                "Age",
                "Predicted Career"
            ],
            "Value": [
                name,
                age,
                predicted_career
            ]
        })

        st.table(report)

        st.success(
            f"🎉 Thank you, {name}! Best of luck for your future!"
        )

