# Datheon Machine Learning & Data Science Internship

This repository contains the complete four-task submission for the **Datheon Machine Learning & Data Science Internship** (`Datheon_Machine_Learning_&_Data_Science_Internship`), covering data exploration, preprocessing, regression, classification, and natural language processing (NLP) sentiment analysis, along with all key discoveries and real-world challenges encountered across every task.

---

## Repository Structure

~~~text
Datheon_Machine_Learning_&_Data_Science_Internship/
├── README.md
├── task_1_<exploratory_and_baseline_workflow>/
│   └── task_1.ipynb
├── task_2_california_housing/
│   └── california_housing_regression.ipynb
├── task_3_customer_churn/
│   └── customer_churn_classification.ipynb
└── task_4_nlp_sentiment_classification/
    └── nlp_sentiment_classification.ipynb
~~~

---

## Task-by-Task Detailed Walkthrough: Discoveries & Problems Faced

### Task 1: Foundational Data Exploration, Preprocessing & Baseline Modeling

#### Key Discoveries
* **Structured EDA Before Modeling:** Inspecting dataset dimensions, column names, data types, missing values, and target/feature distributions upfront prevents blind assumptions and helps catch data quality issues early.
* **Clean, Reproducible Preprocessing:** Organizing data cleaning, feature preparation, and model evaluation into clear, modular steps makes experiments repeatable and easier to audit.
* **Baseline Benchmarking:** Establishing a clean baseline model first provides a clear reference point to verify whether feature adjustments or alternative algorithms actually improve results rather than adding unnecessary complexity.
* **Clear Cell-by-Cell Documentation:** Pairing technical operations with clear explanations makes every notebook easier to follow from raw input to final takeaway.

#### Problems Faced & How They Were Addressed
* **Keeping Transformations Consistent & Leak-Free:** Ensuring preprocessing steps are applied cleanly without leaking information between training and evaluation required strict ordering of data preparation and model fitting.
* **Avoiding Messy, Monolithic Notebooks:** Breaking distinct operations (inspection, preprocessing, modeling, and evaluation) into separate, well-labeled steps made debugging faster and improved overall readability.

---

### Task 2: Regression Modeling on California Housing Data

#### Key Discoveries
* **Continuous Target Modeling Across Housing Features:** Analyzed relationships between housing/demographic attributes and continuous house prices using regression models (including Ordinary Least Squares and regularized linear models such as Ridge and Lasso where applicable).
* **Target Price Capping / Ceiling Effect:** Discovered that house prices in the dataset are capped at an upper limit, creating an artificial ceiling where high-value properties get clipped rather than scaling freely—which can compress upper-end predictions and affect residual patterns.
* **Impact of Feature Scaling & Regularization:** Features on different numeric scales or correlated predictors can cause unstable weights or overfitting in unregularized linear models, whereas scaling and regularization (`alpha` tuning) help control model complexity and coefficient behavior.
* **Multi-Metric & Visual Evaluation:** Evaluating regression performance requires looking at error metrics and residual/prediction diagnostics together rather than relying on a single score.

#### Problems Faced & How They Were Addressed
* **Price Capping Distortion:** Because upper-end prices hit a cap, models can underpredict or flatten near the ceiling and show systematic residuals on high-priced homes; noting this cap during EDA and error inspection prevented misinterpreting ceiling behavior as pure model failure.
* **Scale Mismatches & Overfitting Risk:** Wrapping preprocessing (such as scaling) and regression estimators cleanly inside pipelines and tuning regularization parameters helped prevent uneven feature dominance and over-reliance on noise.
* **Distinguishing Fit from Generalization:** Comparing model performance across splits and checking prediction errors highlighted where models overfit versus where linear assumptions had natural limits.

### Task 3: Binary Classification for Customer Churn

#### Key Discoveries
* **Churn Drivers & Classification Behavior:** Modeled binary customer churn to see how account, billing, usage, and customer profile features separate churning from retained customers.
* **Comparing Classifier Families:** Comparing linear/probabilistic classifiers (such as Logistic Regression) against tree-based classifiers (such as Decision Trees) showed differences in decision boundaries, interpretability, and sensitivity to feature patterns.
* **Importance of Confusion Matrices & Per-Class Metrics:** Relying solely on overall accuracy can hide whether a model misses the positive churn class or triggers too many false positives; precision, recall, F1-score, and confusion matrices provide a much clearer view.

#### Problems Faced & How They Were Addressed
* **Class Imbalance & Error Trade-Offs:** Evaluating models through confusion matrices and per-class metrics made it possible to inspect false positives vs. false negatives directly rather than being misled by headline accuracy.
* **Consistent Pipeline Execution:** Keeping data transformations and estimators unified in Scikit-Learn pipelines prevented mismatched preprocessing between training and inference.

---

### Task 4: NLP Sentiment Classification on Customer Reviews (`sentiment_reviews.csv`)

#### Key Discoveries
* **Dataset Structure & Class Imbalance:** Worked with `sentiment_reviews.csv` (`review_id`, `review_text`, `sentiment`), which contains **10 reviews** total: **5 `positive`**, **4 `negative`**, and only **1 `neutral`**.
* **Pipeline-Ready Text Cleaning with Negation Preservation:** Built a custom `clean_text()` function using regex (`re`) and `NLTK` that:
  1. Strips HTML tags (`<[^>]+>`) and non-alphabetic characters (`[^a-z\s]`).
  2. Converts text to lowercase and tokenizes with `word_tokenize`.
  3. Removes English stopwords while explicitly **preserving `not` and `no`** so negations do not get wiped out.
  4. Applies `WordNetLemmatizer` to normalize words to base forms.
* **TF-IDF + Dual-Model Comparison (`MultinomialNB` & `LogisticRegression`):** Built end-to-end `Pipeline` objects using `TfidfVectorizer(max_features=5000)` paired with `MultinomialNB()` and `LogisticRegression(max_iter=1000)`.
* **Full-Dataset Fit Metrics & Confusion Matrices:** When fitted across all 10 reviews (`X_full`, `y_full`), both `MultinomialNB` and `LogisticRegression` achieved **0.90 accuracy**, correctly classifying all **4 `negative`** reviews (`1.00` precision, `1.00` recall, `1.00` F1) and all **5 `positive`** reviews (`0.83` precision, `1.00` recall, `0.91` F1), while getting **0.00** (precision, recall, F1) on the single **`neutral`** review because that 1 neutral row was predicted as `positive`.
* **Custom Real-Time Inference (`predict_sentiment()`):** Implemented `predict_sentiment(phrase, pipeline=...)` to clean new input text and return the predicted sentiment along with class probabilities from both `LogisticRegression` and `MultinomialNB`:
  * `"The product quality is great and works smoothly"` -> cleaned to `"product quality great work smoothly"` -> predicted **`POSITIVE`** by both models (`LogisticRegression`: `negative: 0.2925, neutral: 0.085, positive: 0.6225`; `MultinomialNB`: `negative: 0.2734, neutral: 0.0848, positive: 0.6418`).
  * `"Terrible experience, stopped working and no support"` -> cleaned to `"terrible experience stopped working no support"` -> predicted **`NEGATIVE`** by both models (`LogisticRegression`: `negative: 0.4802, neutral: 0.091, positive: 0.4288`; `MultinomialNB`: `negative: 0.4945, neutral: 0.094, positive: 0.4115`).
  * `"Decent price for average performance"` -> cleaned to `"decent price average performance"` -> predicted **`POSITIVE`** by both models (`LogisticRegression`: `negative: 0.3616, neutral: 0.204, positive: 0.4344`; `MultinomialNB`: `negative: 0.3624, neutral: 0.2134, positive: 0.4242`), showing higher `neutral` probability than the extreme phrases (~`0.20–0.21` vs. ~`0.08–0.09`) but still pulled toward `positive`.

#### Problems Faced & How They Were Addressed
* **Extreme Small Dataset (10 Rows) & Single-Sample Class (`neutral` = 1):** Attempting a standard `train_test_split` on such a tiny dataset produced tiny test sets (for example, a 3-row test split with 2 `negative` and 1 `positive` samples yielded `0.33` accuracy), and `StratifiedKFold` failed with `Smallest class has fewer than 2 samples` because `neutral` had only 1 example. To avoid split/fold crashes and noisy 3-row test scores on a 10-row file, we documented the sample limitation clearly and fitted both pipelines directly on the full dataset for inspection and live inference.
* **Lack of Training Data for the `neutral` Class:** With only 1 `neutral` sample out of 10 rows, neither classifier had enough examples to learn a distinct `neutral` boundary, resulting in `0.00` accuracy/F1 on the `neutral` review and causing moderate/borderline reviews to lean toward the majority `positive` class.
* **Cluttered Raw Dictionary & `np.str_(...)` Output:** Early inference outputs printed raw dictionaries containing `np.str_(...)` wrappers and dense inline text, making model comparisons hard to read. Converting predictions/keys to clean Python `str`, rounding probabilities, and formatting line-by-line console output made the results crisp and easy to scan.

---

## Tools & Technologies Used
* **Programming & Data Manipulation:** Python, Pandas, NumPy
* **Machine Learning & NLP:** Scikit-Learn (`Pipeline`, `TfidfVectorizer`, `MultinomialNB`, `LogisticRegression`, `classification_report`, `confusion_matrix`), NLTK (`word_tokenize`, `stopwords`, `WordNetLemmatizer`), Regular Expressions (`re`)
* **Visualization:** Matplotlib, Seaborn

---

## Limitations & Future Work
* **Larger, Balanced Datasets:** Expanding small datasets—especially adding many more `neutral` reviews in Task 4—would enable proper train/test splits and cross-validation to evaluate true out-of-sample generalization.
* **Feature & Model Extensions:** Exploring n-grams (`ngram_range=(1, 2)`), class-imbalance handling, and hyperparameter tuning on larger datasets can further improve boundary separation on nuanced or neutral text.
