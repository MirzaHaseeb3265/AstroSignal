# AstroSignal 🌌
### AI-Assisted Discovery in Astronomical Data

**Discover the unusual in our universe with AI-powered astronomical data exploration.**

AstroSignal is an interactive platform that helps users explore real astronomical data and identify unusual stars, galaxies, and quasars for further investigation.

Using a sample of 5,000 records from the Sloan Digital Sky Survey (SDSS), AstroSignal combines machine learning, astronomical colour features, anomaly ranking, and transparent data-quality checks to make astronomical exploration more accessible.

🔭 **Live Application:** https://astrosignal.streamlit.app/

---

## 🚀 What We Built

AstroSignal transforms astronomical catalogue data into an interactive discovery experience. Users can explore objects, examine measurements, rank potential anomalies, and investigate why particular objects may deserve further attention.

The platform is designed for both beginners exploring astronomy and users interested in data-driven scientific investigation.

## ⚙️ How It Works

### 1. Data Preparation
We use a sample of 5,000 astronomical records from SDSS, including stars, galaxies, and quasars, with measurements such as celestial coordinates, photometric magnitudes, and redshift.

### 2. Feature Engineering
We derive colour indices from five photometric bands:

- `u-g`
- `g-r`
- `r-i`
- `i-z`

These features provide additional information about differences in an object's measured brightness across wavelengths.

### 3. Machine Learning
AstroSignal uses three complementary anomaly-detection approaches:

- **Isolation Forest:** Identifies observations that can be isolated relatively easily through random partitions.
- **Local Outlier Factor (LOF):** Identifies observations whose local density differs from that of their neighbours.
- **One-Class SVM:** Learns a boundary around typical observations and identifies potential outliers.

Each model provides its own anomaly-related results, offering different perspectives on potentially unusual objects.

### 4. Candidate Ranking and Quality Checks
The application presents anomaly scores, within-class percentiles, and data-quality flags, including checks for extreme colours and negative redshift measurements.

### 5. Candidate Investigation
Users can inspect an object's measurements, coordinates, scores, and supporting evidence to understand why it was flagged for further investigation.

## 🔬 What Makes AstroSignal Different?

AstroSignal does more than display astronomical measurements. It connects machine-learning-based ranking with accessible explanations and data-quality checks, helping users move from a large catalogue to individual candidates worth examining.

Our aim is to make the process of exploring potentially unusual astronomical objects more transparent and understandable.

## 📊 Results and Scientific Caution

AstroSignal can surface high-ranked candidates for further investigation. However, an unusual score does not prove that an object represents a new or rare astronomical phenomenon. Measurement errors, data-processing artefacts, and other effects may also produce unusual results.

We therefore present flagged objects as **candidates for investigation, not confirmed discoveries**.

## 🌍 Why It Matters

Astronomical surveys contain vast amounts of information. AstroSignal demonstrates how machine learning and interactive visual exploration can help people navigate this data, prioritize observations for closer inspection, and develop a better understanding of astronomical discovery workflows.

## 🔭 Future Work

- Expand the analysis to larger astronomical datasets.
- Evaluate the models using known examples of unusual objects.
- Improve candidate explanations and validation.
- Investigate disagreements between anomaly-detection models.
- Integrate additional public astronomical data sources.

## 📚 Data Source

**Sloan Digital Sky Survey (SDSS):** https://www.sdss.org/

**SDSS Data Release 17:** https://www.sdss.org/dr17/

## 👥 Team

**AstroSignal** — Built for the LLM Hackathon.

Combining astronomy, machine learning, and accessible scientific exploration to help uncover what deserves a closer look.

---

**Live Demo:** https://astrosignal.streamlit.app/

**GitHub Repository:** [Add your repository link here]

*AstroSignal — Making astronomical data more accessible, one candidate at a time.* 🌌
