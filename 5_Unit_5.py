import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, classification_report
import statsmodels.formula.api as smf
from utils import app_header, quiz_block, ensure_seed, progress_sidebar, figure_show

st.set_page_config(page_title="Unit 5 — Statistical Modeling", layout="wide")
progress_sidebar(active_unit=5)
with st.sidebar:
    seed = st.number_input("Seed", value=42, step=1)
ensure_seed(seed)

app_header("UNIT 5: Statistical Modeling + Libraries",
           "Regression (scratch + sklearn), ANOVA, GLM, Logistic Regression, Hypothesis Tests.")

st.subheader("5.1 Simple Linear Regression — Scratch vs sklearn")
n = st.slider("n", 40, 300, 120, 10)
x = np.linspace(0, 10, n)
noise = st.slider("Noise σ", 0.0, 4.0, 2.0, 0.1)
y = 4.0 + 1.8*x + np.random.normal(0, noise, size=n)
X = np.c_[np.ones(n), x]
beta = np.linalg.pinv(X.T @ X) @ X.T @ y
yhat = X @ beta
lr = LinearRegression().fit(x.reshape(-1,1), y)
yhat_skl = lr.predict(x.reshape(-1,1))
fig, ax = plt.subplots(); ax.scatter(x, y, s=12); ax.plot(x, yhat); ax.plot(x, yhat_skl); ax.set_title("Simple Linear Regression"); figure_show(fig)
st.write("Scratch β:", np.round(beta,4)); st.write("sklearn intercept, slope:", float(lr.intercept_), float(lr.coef_[0])); st.write("R² (scratch):", float(r2_score(y, yhat)))

st.subheader("5.2 ANOVA — Compare 3+ Group Means")
grpA = np.random.normal(50, 5, size=30); grpB = np.random.normal(53, 5, size=30); grpC = np.random.normal(56, 5, size=30)
F, p = stats.f_oneway(grpA, grpB, grpC); st.write(f"ANOVA F={F:.3f}, p={p:.4g}")
fig, ax = plt.subplots(); ax.boxplot([grpA,grpB,grpC], labels=["A","B","C"]); ax.set_title("Drug Response by Group"); figure_show(fig)

st.subheader("5.3 GLM (OLS with Continuous + Categorical) — statsmodels")
n = 300
experience = np.random.uniform(0, 20, size=n)
gender = np.random.choice(["Male","Female"], size=n)
salary = 30000 + 2000*experience + (gender=="Male")*1500 + np.random.normal(0, 3000, size=n)
df_glm = pd.DataFrame({"salary": salary, "experience": experience, "gender": gender})
model = smf.ols("salary ~ experience + C(gender)", data=df_glm).fit()
st.write(model.summary())

st.subheader("5.4 Logistic Regression — Pass/Fail vs Study Hours")
n = 400
hours = np.random.uniform(0, 10, size=n)
logit = -2.0 + 0.6*hours
prob = 1/(1+np.exp(-logit))
passed = (np.random.rand(n) < prob).astype(int)
clf = LogisticRegression(); clf.fit(hours.reshape(-1,1), passed)
hgrid = np.linspace(0, 10, 200); probs = clf.predict_proba(hgrid.reshape(-1,1))[:,1]
fig, ax = plt.subplots(); ax.scatter(hours, passed, s=8, alpha=0.5); ax.plot(hgrid, probs); ax.set_title("Pass Probability vs Study Hours"); figure_show(fig)
pred = clf.predict(hours.reshape(-1,1)); st.write("Confusion matrix:"); st.write(confusion_matrix(passed, pred)); st.text(classification_report(passed, pred))

st.subheader("5.5 Hypothesis Testing — t-test & Chi-square")
x1 = np.random.normal(10, 2, size=40); x2 = np.random.normal(11, 2, size=40)
t_stat, p_val = stats.ttest_ind(x1, x2, equal_var=False)
st.write(f"Two-sample t-test: t={t_stat:.3f}, p={p_val:.4g}")
table = np.array([[30,10],[45,15]])
chi2, p, dof, exp = stats.chi2_contingency(table)
st.write(f"Chi-square: χ²={chi2:.3f}, p={p:.4g}, dof={dof}")
st.write("Expected counts under independence:"); st.dataframe(pd.DataFrame(exp, columns=["Pass","Fail"], index=["Low prep","High prep"]))

quiz_block(
    ["What can logistic regression model that linear regression cannot?",
     "In ANOVA, what does a small p-value indicate?",
     "Why include categorical variables in a GLM?"],
    ["Class probabilities in [0,1] for binary outcomes.",
     "At least one group mean differs significantly.",
     "Capture systematic group differences alongside continuous predictors."]
)