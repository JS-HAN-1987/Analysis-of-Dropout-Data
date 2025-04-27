import base64
import json
import os
from google.oauth2.service_account import Credentials
import gspread
import pandas as pd
import numpy as np
import xgboost
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from googleapiclient.discovery import build


json_file_path = 'service_account.json'
spreadsheet_url = "https://docs.google.com/spreadsheets/d/1DhfQFFR9gSV7plLLGgrqmNaohfbYW3Q9Fm_vuli8czI/edit?usp=sharing"

# service_account.json 파일을 직접 읽기
try:
    with open(json_file_path) as f:
        service_account_info = json.load(f)

    gc = gspread.service_account_from_dict(service_account_info)
    doc = gc.open_by_url(spreadsheet_url)
    print("성공적으로 스프레드시트에 연결되었습니다.")

except Exception as e:
    print(f"Google Sheets 연결 실패: {str(e)}")
    exit()

# 인증
try:
    gc = gspread.service_account_from_dict(service_account_info)
    doc = gc.open_by_url(
        'https://docs.google.com/spreadsheets/d/1DhfQFFR9gSV7plLLGgrqmNaohfbYW3Q9Fm_vuli8czI/edit?usp=sharing')
    print("성공적으로 스프레드시트에 연결되었습니다.")

except Exception as e:
    print(f"Google Sheets 연결 실패: {str(e)}")
    exit()

# 첫 번째 워크시트 제외 나머지 워크시트 삭제
worksheets = doc.worksheets()
for worksheet in worksheets[1:]:  # 첫 번째 워크시트를 제외한 나머지
    print(f"워크시트 {worksheet.title} 삭제")
    doc.del_worksheet(worksheet)

# 1번째 시트 선택
worksheet = doc.get_worksheet(0)  # 0번 인덱스 = 첫 번째 시트

# 모든 데이터 가져오기
data = worksheet.get_all_records()

# pandas DataFrame으로 변환
df = pd.DataFrame(data)


# '년도' 열 제거 (학업중단율과 무관하다고 가정)
df = df.drop(columns=['년도'])

# 결측치 확인 및 처리
#print("결측치 수:")
#print(df.isnull().sum())

# 결측치가 있는 경우 평균값으로 대체
#data = data.fillna(data.mean())


# 특성과 타겟 분리
X = df.drop(columns=['학업중단율'])
y = df['학업중단율']

# 데이터 분할 (훈련 80%, 테스트 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost 모델 훈련
model = xgboost.XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

# 모델 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# SHAP 값 계산
explainer = shap.Explainer(model)
shap_values = explainer(X)



# SHAP 데이터 생성
shap_df = pd.DataFrame({
    'feature': X.columns,
    'shap_importance': np.abs(shap_values.values).mean(axis=0)
}).sort_values(by='shap_importance', ascending=False)
# 기존 shap_df 준비


# 2번째 시트 선택 (없으면 새로 만들기)
try:
    worksheet = doc.get_worksheet(1)  # 인덱스 1 = 두 번째 시트
except:
    worksheet = doc.add_worksheet(title="SHAP Importance", rows="100", cols="20")

# 기존 데이터 삭제
worksheet.clear()

# 데이터프레임을 리스트로 변환
data = [shap_df.columns.tolist()] + shap_df.values.tolist()

# 시트에 데이터 업로드
worksheet.update('A1', data)


# 구글 시트 API 인증
credentials = Credentials.from_service_account_file(
    json_file_path,
    scopes=['https://www.googleapis.com/auth/spreadsheets']
)

service = build('sheets', 'v4', credentials=credentials)

spreadsheet_id = spreadsheet_url.split("/d/")[1].split("/")[0]

# 차트 생성 요청
chart_request = {
    "requests": [
        {
            "addChart": {
                "chart": {
                    "spec": {
                        "title": "Feature Importance for School Dropout Rate",
                        "basicChart": {
                            "chartType": "BAR",
                            "legendPosition": "NO_LEGEND",
                            "axis": [
                                {"position": "BOTTOM_AXIS", "title": "SHAP Importance"},
                                {"position": "LEFT_AXIS", "title": "Feature"}
                            ],
                            "domains": [
                                {"domain": {"sourceRange": {"sources": [{"sheetId": worksheet.id, "startRowIndex": 1, "endRowIndex": len(shap_df)+1, "startColumnIndex": 0, "endColumnIndex": 1}]}}}
                            ],
                            "series": [
                                {"series": {"sourceRange": {"sources": [{"sheetId": worksheet.id, "startRowIndex": 1, "endRowIndex": len(shap_df)+1, "startColumnIndex": 1, "endColumnIndex": 2}]}}}
                            ],
                            "headerCount": 0
                        }
                    },
                    "position": {
                        "overlayPosition": {
                            "anchorCell": {
                                "sheetId": worksheet.id,
                                "rowIndex": 0,  # 차트 시작할 위치 (0번째 행)
                                "columnIndex": 3  # 차트 시작할 위치 (D열부터)
                            },
                            "offsetXPixels": 10,
                            "offsetYPixels": 10
                        }
                    }
                }
            }
        }
    ]
}

# 요청 보내기
service.spreadsheets().batchUpdate(
    spreadsheetId=spreadsheet_id,
    body=chart_request
).execute()




# SHAP 상위 5개 특징 추출
shap_df = pd.DataFrame({
    'feature': X.columns,
    'shap_importance': np.abs(shap_values.values).mean(axis=0)
}).sort_values(by='shap_importance', ascending=False)
top_features = shap_df.head(5)['feature'].tolist()

for feature in top_features:
    title = f"Depend_{feature}"
    rows_needed = len(X) + 2
    cols_needed = 10   # 충분히 큰 값으로 설정

    try:
        sheet = doc.worksheet(title)
        sheet.clear()
        # 기존 시트가 작다면 크기 조정
        sheet.resize(rows=rows_needed, cols=cols_needed)
    except gspread.exceptions.WorksheetNotFound:
        sheet = doc.add_worksheet(
            title=title,
            rows=rows_needed,
            cols=cols_needed
        )

    # --- 이하 데이터 쓰기 및 차트 추가 코드는 그대로 ---
    x_vals = X[feature].tolist()
    y_vals = shap_values.values[:, X.columns.get_loc(feature)].tolist()
    header = [[f"{feature} value", "SHAP value"]]
    rows = [[x_vals[i], y_vals[i]] for i in range(len(x_vals))]
    sheet.update('A1:B1', header)
    sheet.update(f'A2:B{len(rows)+1}', rows)

    chart_req = {
        "addChart": {
            "chart": {
                "spec": {
                    "title": f"{feature} Dependence",
                    "basicChart": {
                        "chartType": "SCATTER",
                        "legendPosition": "NO_LEGEND",
                        "axis": [
                            {"position": "BOTTOM_AXIS", "title": f"{feature} value"},
                            {"position": "LEFT_AXIS", "title": "SHAP value"}
                        ],
                        "domains": [{
                            "domain": {
                                "sourceRange": {
                                    "sources": [{
                                        "sheetId": sheet.id,
                                        "startRowIndex": 1,
                                        "endRowIndex": len(X)+1,
                                        "startColumnIndex": 0,
                                        "endColumnIndex": 1
                                    }]
                                }
                            }
                        }],
                        "series": [{
                            "series": {
                                "sourceRange": {
                                    "sources": [{
                                        "sheetId": sheet.id,
                                        "startRowIndex": 1,
                                        "endRowIndex": len(X)+1,
                                        "startColumnIndex": 1,
                                        "endColumnIndex": 2
                                    }]
                                }
                            }
                        }],
                        "headerCount": 1
                    }
                },
                "position": {
                    "overlayPosition": {
                        "anchorCell": {
                            "sheetId": sheet.id,
                            "rowIndex": 0,
                            "columnIndex": 3   # 이제 D열(3)이 유효합니다
                        },
                        "offsetXPixels": 10,
                        "offsetYPixels": 10
                    }
                }
            }
        }
    }

    service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests":[chart_req]}
    ).execute()