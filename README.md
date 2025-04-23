# Seasonal-HMMs-for-ENSO-detection
Developed a probabilistic framework using Hidden Markov Models to analyze and predict ENSO climate patterns through multivariate time-series analysis of oceanographic data.
ENSO HMM Analysis
This repository contains a Python implementation of a Hidden Markov Model (HMM) to study the El Niño-Southern Oscillation (ENSO) dynamics using Sea Surface Temperature (SST) anomalies (ONI) and Warm Water Volume (WWV) anomalies. The HMM identifies latent climate regimes (El Niño, La Niña, Neutral) and analyzes their transition dynamics, statistical regularities, and temporal coherence, as outlined in the project objectives. The code processes data across 12 seasonal windows (DJF–NDJ) and includes state alignment, comprehensive analyses (transition matrices, emission parameters, change points, clustering), and visualizations.
Project Overview
The HMM is applied to climatic time series (ONI and WWV) to uncover probabilistic dependencies and latent structures in ENSO phases, not for precise weather forecasting. Key features include:

Data Processing: Handles ONI and WWV anomalies for ~43 years (1981–2023), with imputation for missing values.
HMM Training: Uses a 3-state Gaussian HMM with full covariance, aligned to ENSO phases via majority voting.
Analyses:
Transition matrices for state dynamics.
Emission means and covariances for ONI/WWV patterns.
Change-point detection for regime shifts (e.g., 1982–83, 1997–98 El Niño).
Clustering of years by ONI/WWV similarity.
WWV correlations with NDJ WWV.


Visualizations: Time series, posterior probabilities, confusion matrices, and emission distributions.
Performance: Achieves accuracies from 0.4884 (NDJ) to 0.6364 (FMA), with SON and ASO performing best (0.6047, 0.5116).

This implementation improves upon an earlier version with poor accuracies (e.g., DJF: 0.2326, OND: 0.0930) by incorporating robust state alignment and comprehensive outputs.
Repository Structure
enso-hmm-analysis/
├── data/
│   ├── index.csv          # ONI data for 12 seasonal windows
│   └── anomaly.csv        # WWV anomaly data
├── hmm_outputs/           # Output directory for results and visualizations
│   ├── predictions_{period}.csv
│   ├── clusters_{period}.csv
│   ├── enso_hmm_states_{period}.png
│   ├── state_probs_{period}.png
│   ├── confusion_matrix_{period}.png
│   └── emission_dist_{period}.png
├── src/
│   └── enso_hmm.py        # Main script for HMM training and analysis
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── summary.txt            # Summary of HMM results for all periods

Prerequisites

Python: Version 3.8 or higher
Dependencies: Listed in requirements.txt
Hardware: Standard CPU, ~4GB RAM for processing ~43 years of data
Optional: Matplotlib for visualizing outputs (saved automatically)

Installation

Clone the Repository:
git clone https://github.com/your-username/enso-hmm-analysis.git
cd enso-hmm-analysis


Create a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt

Required packages include:

numpy
pandas
scikit-learn
hmmlearn
matplotlib
seaborn


Prepare Data:

Place index.csv (ONI) and anomaly.csv (WWV) in the data/ directory.
Ensure index.csv has columns Year, DJF_ONI, ..., NDJ_ONI.
Ensure anomaly.csv has columns Year, DJF_WWV, ..., NDJ_WWV.
Missing values are imputed with 0; verify data integrity before running.



Usage

Run the Main Script:
python src/enso_hmm.py

The script:

Loads and preprocesses data from data/index.csv and data/anomaly.csv.
Trains a 3-state Gaussian HMM for each seasonal window (DJF–NDJ).
Aligns HMM states to ENSO phases (El Niño, La Niña, Neutral) using majority voting.
Generates analyses (transition matrices, emission parameters, change points, clustering).
Saves results and visualizations to hmm_outputs/.


Output Files:

CSV Files:
predictions_{period}.csv: Predicted states and true states for each year.
clusters_{period}.csv: Clustered years by ONI/WWV similarity.


Visualizations:
enso_hmm_states_{period}.png: Time series of true vs. predicted states.
state_probs_{period}.png: Posterior probabilities for state assignments.
confusion_matrix_{period}.png: Confusion matrix for model performance.
emission_dist_{period}.png: ONI/WWV emission distributions.


Summary:
summary.txt: Aggregated results (accuracies, matrices, change points, clustering) for all periods.




Example Output (SON):

Accuracy: 0.6047
State Map: {0: 0, 1: 1, 2: 2} (El Niño, La Niña, Neutral)
Key Change Points: 1982–83, 1997–98, 2015–16 (El Niño events)
Clustering: Groups years into Neutral-dominated and La Niña/El Niño clusters
WWV-NDJ Correlation: 0.3533



Key Features

State Alignment: Uses align_states to map HMM states to ENSO phases based on majority voting, fixing misalignments in the original code (e.g., DJF Neutral ONI=1.6834).
Comprehensive Analysis:
Transition dynamics: Captures Neutral persistence (e.g., SON: 0.7273) and El Niño → La Niña oscillations (e.g., DJF: 0.999999973).
Statistical regularities: ONI/WWV means align with ENSO phases in SON, ASO, FMA (e.g., SON El Niño ONI=1.1609).
Temporal coherence: Detects regime shifts (e.g., 1982–83, 1997–98) and Neutral persistence (2–3 years in SON).


Robustness: Handles missing data, ensures covariance stability with a 1e-6 diagonal term, and supports small datasets (~43 years).
Visualizations: Generates interpretable plots for model validation and analysis.

Limitations

Markov Assumption: First-order HMM may miss long-range ENSO dependencies, affecting volatile periods (e.g., DJF, FMA).
Neutral Detection: Periods like DJF and NDJ struggle to detect Neutral states due to ONI/WWV overlap (e.g., DJF state map {0: 1, 1: 1, 2: 0}).
WWV Patterns: Inconsistent WWV means (e.g., SON El Niño WWV=0.7582, expected negative) suggest data or preprocessing issues.
Sample Size: Limited data (~43 years) may reduce statistical robustness, especially for rare El Niño events.

Recommendations for Improvement

Enhance Neutral Detection:

Increase n_iter (e.g., 200) or use covariance_type='diag':model = hmm.GaussianHMM(n_components=3, covariance_type='diag', n_iter=200, random_state=42)


Normalize ONI/WWV data to reduce overlap:from sklearn.preprocessing import StandardScaler
observations = StandardScaler().fit_transform(observations)




Validate WWV Data:

Check anomaly.csv for preprocessing errors (e.g., incorrect scaling or imputation).
Consider training HMM with ONI alone if WWV underperforms (e.g., MJJ correlation: -0.0244).


Refine Clustering:

Use silhouette scores to optimize n_clusters:from sklearn.metrics import silhouette_score
score = silhouette_score(observations, labels)


Try DBSCAN for non-spherical clusters.


Cross-Validation:

Split data into training/test sets to evaluate generalization:from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(observations, test_size=0.2, random_state=42)




Visualize Results:

Inspect enso_hmm_states_{period}.png to verify state transitions (e.g., 1997–98 El Niño).
Analyze state_probs_{period}.png for state uncertainty during volatile periods (e.g., DJF).



Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m 'Add your feature').
Push to the branch (git push origin feature/your-feature).
Open a Pull Request.

Please include tests and documentation for new features.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or issues, please open an issue on GitHub or contact the maintainer at shaheenalirv@gmail.com.

Acknowledgments

Built with hmmlearn, scikit-learn, and Matplotlib.
Inspired by research on ENSO dynamics and HMM applications in climate science.
Thanks to the climate data providers for ONI and WWV datasets.

