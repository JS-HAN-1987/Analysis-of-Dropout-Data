from gspread_dataframe import set_with_dataframe
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import json
from google.oauth2.service_account import Credentials
import gspread
from googleapiclient.discovery import build
from scipy.interpolate import interp1d

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


# 1번째 시트 선택
worksheet = doc.get_worksheet(0)  # 0번 인덱스 = 첫 번째 시트

# 모든 데이터 가져오기
data = worksheet.get_all_records()

# pandas DataFrame으로 변환
df = pd.DataFrame(data)


# 연도를 float으로 변환
df['년도'] = df['년도'].astype(float)

# 보간용 새로운 연도 만들기
new_years = np.arange(df['년도'].min(), df['년도'].max() + 0.1, 0.1)

# 결과 저장
interpolated_df = pd.DataFrame({'년도': new_years})

# 각 변수에 대해 보간
for col in df.columns:
    if col == '년도':
        continue
    # 보간 함수 만들기 (선형 / cubic 선택 가능)
    f = interp1d(df['년도'], df[col], kind='cubic')  # 또는 kind='linear'
    interpolated_df[col] = f(new_years)

# 확인
print(interpolated_df.head())

# 5. 엑셀로 저장
#output_file = 'interpolated_smart_data.xlsx'
#interpolated_df.to_excel(output_file, index=False, engine='openpyxl')
#print(f"보간된 데이터가 '{output_file}'로 저장되었습니다.")

# 구글 시트 API 인증
credentials = Credentials.from_service_account_file(
    json_file_path,
    scopes=['https://www.googleapis.com/auth/spreadsheets']
)

service = build('sheets', 'v4', credentials=credentials)
spreadsheet_id = spreadsheet_url.split("/d/")[1].split("/")[0]





df = interpolated_df


# 예측 대상: 학업중단율
target_col = '학업중단율'
feature_cols = df.columns.drop(['년도', target_col])

# 정규화
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_x.fit_transform(df[feature_cols])
y_scaled = scaler_y.fit_transform(df[[target_col]])


# 시퀀스 생성 함수 (예: 3년치로 다음 해 예측)
def create_sequences(X, y, window_size=3):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i+window_size])
        ys.append(y[i+window_size])
    return np.array(Xs), np.array(ys)

# 3개 연도씩 묶어서 시퀀스 생성
X_seq, y_seq = create_sequences(X_scaled, y_scaled, window_size=3)

# 결과 확인
print("X_seq.shape:", X_seq.shape)  # 예: (23, 3, 30)
print("y_seq.shape:", y_seq.shape)  # 예: (23, 1)


# RNN 모델 구성
model = Sequential([
    SimpleRNN(64, activation='tanh', input_shape=(X_seq.shape[1], X_seq.shape[2])),
    Dense(32, activation='relu'),
    Dense(1)  # 출력: 학업중단율
])

# 모델 컴파일
model.compile(optimizer='adam', loss='mse')

# 모델 학습
history = model.fit(X_seq, y_seq, epochs=100, verbose=0)
# 예측값 생성 (정규화된 상태)
y_pred_scaled = model.predict(X_seq)

# 원래 값으로 복원 (역정규화)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_seq)

# 결과 DataFrame

results_df = pd.DataFrame({
    "Actual": y_true.flatten(),
    "Predicted": y_pred.flatten()
})
print(results_df.head())

start_year = 1999
end_year = 2023
n_samples = (end_year - start_year)*10 - 2
years = np.linspace(start_year, end_year, n_samples)

# 1. 새 시트 만들기
try:
    # 기존 시트가 존재하는지 확인
    worksheet_result = doc.worksheet("RNN")
    # 기존 데이터 클리어
    doc.del_worksheet( worksheet_result)
except gspread.exceptions.WorksheetNotFound:
    a=1

# 시트가 없으면 새로 생성
worksheet_result = doc.add_worksheet(title="RNN", rows=100, cols=20)



# 2. importance_df를 새 시트에 쓰기
#set_with_dataframe(worksheet_result, results_df)

# 년도 데이터를 results_df에 추가
results_df.insert(0, "Year", years)  # 첫 번째 열로 년도 추가
# 시트에 데이터 쓰기 (년도 포함)
set_with_dataframe(worksheet_result, results_df)

# 3. 차트 추가
spreadsheet_id = doc.id
sheet_id = worksheet_result.id

# 구글 시트 API 인증
credentials = Credentials.from_service_account_file(
    json_file_path,
    scopes=['https://www.googleapis.com/auth/spreadsheets']
)

service = build('sheets', 'v4', credentials=credentials)
spreadsheet_id = spreadsheet_url.split("/d/")[1].split("/")[0]

requests = [{
    "addChart": {
        "chart": {
            "spec": {
                "title": "Actual vs Predicted 학업중단율",
                "basicChart": {
                    "chartType": "LINE",
                    "legendPosition": "BOTTOM_LEGEND",
                    "axis": [
                        {"position": "BOTTOM_AXIS", "title": "Year"},
                        {"position": "LEFT_AXIS", "title": "학업중단율"}
                    ],
                    "domains": [{
                        "domain": {
                            "sourceRange": {
                                "sources": [{
                                    "sheetId": sheet_id,
                                    "startRowIndex": 0,  # 헤더 포함
                                    "endRowIndex": len(results_df)+1,
                                    "startColumnIndex": 0,  # Year 열 (A열)
                                    "endColumnIndex": 1
                                }]
                            }
                        }
                    }],
                    "series": [
                        {
                            "series": {
                                "sourceRange": {
                                    "sources": [{
                                        "sheetId": sheet_id,
                                        "startRowIndex": 0,  # 헤더 포함
                                        "endRowIndex": len(results_df)+1,
                                        "startColumnIndex": 1,  # Actual 열 (B열)
                                        "endColumnIndex": 2
                                    }]
                                }
                            },
                            "targetAxis": "LEFT_AXIS",
                            "type": "LINE",
                            "pointStyle": {
                                "shape": "CIRCLE",
                                "size": 6
                            }
                        },
                        {
                            "series": {
                                "sourceRange": {
                                    "sources": [{
                                        "sheetId": sheet_id,
                                        "startRowIndex": 0,  # 헤더 포함
                                        "endRowIndex": len(results_df)+1,
                                        "startColumnIndex": 2,  # Predicted 열 (C열)
                                        "endColumnIndex": 3
                                    }]
                                }
                            },
                            "targetAxis": "LEFT_AXIS",
                            "type": "LINE",
                            "pointStyle": {
                                "shape": "DIAMOND",
                                "size": 6
                            }
                        }
                    ],
                    "headerCount": 1  # 헤더 행 포함
                }
            },
            "position": {
                "overlayPosition": {
                    "anchorCell": {
                        "sheetId": sheet_id,
                        "rowIndex": 0,
                        "columnIndex": 3  # D열에 차트 배치
                    }
                }
            }
        }
    }
}]

# 차트 요청 실행
response = service.spreadsheets().batchUpdate(
    spreadsheetId=spreadsheet_id,
    body={"requests": requests}
).execute()








mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")

# 기준 성능 (원래 데이터로 예측한 MSE)
baseline_mse = mean_squared_error(y_true, y_pred)

# 변수별 중요도 저장 리스트
feature_importance = []

# 전체 feature 개수
num_features = X_seq.shape[2]

# 반복: 한 변수씩 섞어보기
for i in range(num_features):
    X_permuted = X_seq.copy()

    # 해당 변수만 무작위로 섞기 (모든 시퀀스에 대해)
    for j in range(X_seq.shape[0]):
        np.random.shuffle(X_permuted[j, :, i])

    # 섞은 데이터로 예측
    y_permuted_pred = model.predict(X_permuted)
    y_permuted_pred = scaler_y.inverse_transform(y_permuted_pred)

    # 성능 계산 (MSE)
    permuted_mse = mean_squared_error(y_true, y_permuted_pred)

    # 중요도 = 성능 하락 정도
    importance = permuted_mse - baseline_mse
    feature_importance.append(importance)


importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

# 부호 제거: 절댓값으로 바꾸기
importance_df['Importance'] = importance_df['Importance'].abs()

# 중요도 기준 내림차순 정렬
importance_df = importance_df.sort_values(by='Importance', ascending=False)

mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error: {mse:.6f}")
print(importance_df)  # 중요도 높은 상위 10개


# 1. 새 시트 만들기
try:
    # 기존 시트가 존재하는지 확인
    worksheet_importance = doc.worksheet("Feature Importance")
    # 기존 데이터 클리어
    doc.del_worksheet(worksheet_importance)
except gspread.exceptions.WorksheetNotFound:
    a = 1

worksheet_importance = doc.add_worksheet(title="Feature Importance", rows=100, cols=20)


# 2. importance_df를 새 시트에 쓰기
set_with_dataframe(worksheet_importance, importance_df)

# 3. 차트 추가
spreadsheet_id = doc.id
sheet_id = worksheet_importance.id
requests = [{
    "addChart": {
        "chart": {
            "spec": {
                "title": "Feature Importance (Permutation)",
                "basicChart": {
                    "chartType": "BAR",
                    "legendPosition": "RIGHT_LEGEND",
                    "axis": [
                        {"position": "BOTTOM_AXIS", "title": "Importance (MSE Increase)"},
                        {"position": "LEFT_AXIS", "title": "Features"}
                    ],
                    "domains": [{
                        "domain": {
                            "sourceRange": {
                                "sources": [{
                                    "sheetId": sheet_id,
                                    "startRowIndex": 0,  # 헤더 포함
                                    "endRowIndex": len(importance_df)+1,
                                    "startColumnIndex": 0,  # 변수명 열 (A열)
                                    "endColumnIndex": 1
                                }]
                            }
                        }
                    }],
                    "series": [
                        {
                            "series": {
                                "sourceRange": {
                                    "sources": [{
                                        "sheetId": sheet_id,
                                        "startRowIndex": 0,  # 헤더 포함
                                        "endRowIndex": len(importance_df)+1,
                                        "startColumnIndex": 1,  # 중요도 값 열 (B열)
                                        "endColumnIndex": 2
                                    }]
                                }
                            },
                            "targetAxis": "BOTTOM_AXIS",
                            "colorStyle": {
                                "rgbColor": {"red": 0.2, "green": 0.4, "blue": 0.6}
                            }
                        }
                    ],
                    "headerCount": 1  # 헤더 행 포함
                }
            },
            "position": {
                "overlayPosition": {
                    "anchorCell": {
                        "sheetId": sheet_id,
                        "rowIndex": 0,
                        "columnIndex": 3  # D열에 차트 배치
                    }
                }
            }
        }
    }
}]

# 실행
response = service.spreadsheets().batchUpdate(
    spreadsheetId=spreadsheet_id,
    body={"requests": requests}
).execute()
