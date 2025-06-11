# Scam-HamDetector
# Real-Time Scam Detector

This project is built to detect scams in **SMS**, **Emails**, and **URLs** using different machine learning models. It's designed to work in real-time and can be integrated into other systems or used as a standalone tool which we are currently working on.

## What It Does
- Detects spam or scam in SMS using an ensembling model with both CNN & LSTM with an accuracy of 97.05%
- Detects scam emails using an ensembled model with BERT-based, RoBERTa, & Random Forest with an accuracy of 99% 
- Classifies URLs as phishing or safe using XGBoost, Logistic Regression, & Random Forest with an accuract of 95.78%

## Datasets used

- **SMS Model**: Trained on the UCI SMS spam datasets
- **Email Model**: Trained on a Email Phishing Dataset sourced from the OpenPhish repository
- **URL Model**: Trained on multiple phishing repositories such as PhishTank and OpenPhish

Each model takes the input (text or URL), preprocesses it, runs prediction, and outputs if it's a scam or not.

## What we have in store for future...!
- We Plan on developing a dashboard where users can copy and paste the text/email/url and know whether it's safe or malicious.
- By integrating the code with the dashboard with the help of frontend & backend it can help many oldage people who doesn't know and fall as a prey in scammers hand.
- And We are also planning to find the location of the scammer or atlest pitch in the idea of finding the location of the scammer and then forwarding the details of the scammer to the respective authorities mail IDs. 

## So To Do
- Add a web interface using Streamlit or Flask
- Combine all models into one dashboard
- Possibly deploy as an API

## How to Use
1. Clone the repository:
   git clone https://github.com/havyasree30/real-time-scam-detector.git
   cd real-time-scam-detector
   
2. Install the requirements:
   pip install -r requirements.txt
   
3. Run the models:
- SMS:
  ```
  python sms_model/run_sms_model.py
  ```
- Email:
  ```
  python email_model/run_email_model.py
  ```
- URL:
  ```
  python url_model/run_url_classifier.py
  ```

  
## Scam-HamDetector/
<pre> ```
├── Emailspam/
├── SMS-spam/
├── URL-spam/
├── ResultPictures/
├── README.md
├── requirements.txt
├── CONTRIBUTING.md
``` </pre>

## Contact
GitHub: [@havyasree30](https://github.com/havyasree30)  
LinkedIn: [Havyasree Polasam](https://www.linkedin.com/in/havyasree-polasam-6127b7271/)
