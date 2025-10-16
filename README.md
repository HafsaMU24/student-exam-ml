# Student Exam ML Project

Detta projekt undersöker hur studenters livsstilsvanor påverkar deras akademiska prestationer.  
Datasetet består av 1000 simulerade studentposter med information om studietid, sömn, kost, närvaro, stressnivå och andra faktorer.

##  Projektets syfte
Målet är att förutsäga studenters exam_score baserat på deras vanor och bakgrundsfaktorer med hjälp av maskininlärning.

##  Innehåll
- **student_project.ipynb** – Jupyter Notebook med:
  - Explorativ dataanalys (EDA)  
  - Visualiseringar (histogram, boxplots, scatterplots, heatmap)  
  - Förbehandling och feature engineering  
  - Modellträning och jämförelse (Linear, Ridge, Lasso, Random Forest)  
  - Export av bästa modell till `.joblib`
- **app.py**– Flask-API som laddar den tränade modellen och gör prediktioner via en REST-endpoint (`/predict`).
- (student_model.joblib) – Sparad modell (bundle: scaler + model + feature-order).
- (student_habits_performance.csv) – Dataset som användes för träning.

##  Hur du kör projektet
1. Klona repot:
   git clone https://github.com/HafsaMU24/student-exam-ml.git
   cd student-exam-ml
2. Starta API:
   python app.py
3. Testa API:t i Postman eller via curl:
   POST http://127.0.0.1:5000/predict och andra endapoint(GET)

## Modeller och resultat
 - Flera regressionsmodeller tränades och jämfördes.
 - Random Forest presterade bäst med högst R² och lägst RMSE.
 - Modellen sparades och används i API:t för prediktioner.
