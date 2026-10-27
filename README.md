# 🚔 Crime Data Analysis System

Python project jo crime data ko clean karta hai, analyse karta hai, hotspots dhundta hai,
arrest predict karta hai aur future crime forecast karta hai. Ek interactive dashboard bhi hai.

## Features
| Module | Kya karta hai |
|---|---|
| `src/data_generator.py` | 20,000 realistic synthetic records (dirty data ke saath) |
| `src/preprocess.py` | Duplicates, missing values, invalid rows, text cleaning, feature engineering |
| `src/analysis.py` | Crime type/district/time analysis, arrest rates, victim profile, **district risk score** |
| `src/models.py` | **KMeans hotspots**, **RandomForest arrest predictor**, **seasonal trend forecast** |
| `src/visualize.py` | 10 charts (bar, trend, heatmaps, hotspot map, feature importance) |
| `main.py` | Poori pipeline ek command mein + `outputs/report.txt` |
| `app.py` | Streamlit dashboard: filters, KPIs, map, predictor |

## Setup & Run
```bash
pip install -r requirements.txt
python main.py            # charts -> outputs/, report -> outputs/report.txt
streamlit run app.py      # interactive dashboard
```

## Apna real dataset use karna
CSV mein ye columns hone chahiye (naam same rakho ya rename karo):
`date, hour, crime_type, district, latitude, longitude, location_type, victim_age, victim_gender, arrest`
(`arrest` = 1/0). Phir: `python main.py --data your_file.csv`
Free sources: Kaggle (Chicago / LA / India crime), data.gov.in, NCRB.

## Project structure
```
crime_analysis/
├── main.py  app.py  requirements.txt  README.md
├── src/  data_generator.py preprocess.py analysis.py models.py visualize.py
├── data/      (CSV files)
└── outputs/   (charts + report)
```

## Viva / Report ke liye key points
- **Problem**: crime patterns samajhna taaki police resources sahi jagah lage.
- **Pipeline**: Collect → Clean → Feature engineering → EDA → ML → Visualise → Dashboard.
- **Algorithms**: KMeans (unsupervised, hotspots), Random Forest (classification), Linear Regression + seasonality (forecast).
- **Metrics**: accuracy, precision/recall, confusion matrix, R².
- **Limitations**: synthetic data; real data mein bias aur under-reporting hoti hai.
- **Future scope**: real-time feed, LSTM/Prophet forecast, folium maps, FIR text NLP.
