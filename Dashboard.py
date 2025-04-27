import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

import json
import gspread


json_file_path = 'service_account.json'
spreadsheet_url = "https://docs.google.com/spreadsheets/d/1DhfQFFR9gSV7plLLGgrqmNaohfbYW3Q9Fm_vuli8czI/edit?usp=sharing"

# 앱 제목 설정
st.title("학업중단 요인분석")


# 엑셀 파일 로드
@st.cache_data  # 성능 향상을 위한 캐싱
def load_data():
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

    worksheets = doc.worksheets()

    sheets = {}
    for worksheet in worksheets:
        name = worksheet.title
        if name != "시트1":  # "시트1"은 제외
            # worksheet 데이터를 DataFrame으로 변환
            data = worksheet.get_all_values()
            df = pd.DataFrame(data[1:], columns=data[0])  # 첫 줄을 컬럼 이름으로 사용
            sheets[name] = df

    return sheets



# 데이터 로드
sheets = load_data()
sheet_names = list(sheets.keys())

# 사이드바에 시트 선택 위젯 추가
selected_sheet = st.selectbox(
    "분석할 시트를 선택하세요",
    sheet_names,
    index=0
)

# 확인 버튼
if st.button("확인"):
    st.subheader(f"{selected_sheet} 분석 결과")

    # 선택된 시트 데이터 가져오기
    df = sheets[selected_sheet]

    # 데이터 표시
    st.write("### 데이터 테이블")
    st.dataframe(df)

    # 시트 유형에 따라 다른 시각화 제공
    if "SHAP" in selected_sheet or "Importance" in selected_sheet:
        st.write("### 중요도 시각화")

        # SHAP Importance 또는 Feature Importance 시트
        if selected_sheet == "SHAP Importance":
            fig = px.bar(df, x='feature', y='shap_importance',
                         title='SHAP Feature Importance',
                         labels={'feature': '요인', 'shap_importance': 'SHAP 중요도'})
        elif selected_sheet == "Feature Importance":
            fig = px.bar(df, x='Feature', y='Importance',
                         title='Feature Importance',
                         labels={'Feature': '요인', 'Importance': '중요도'})
        else:
            # 다른 중요도 시트 (첫 번째 열이 feature, 두 번째 열이 value라고 가정)
            col1, col2 = df.columns[0], df.columns[1]
            fig = px.bar(df, x=col1, y=col2,
                         title=f'{selected_sheet}',
                         labels={col1: '요인', col2: '중요도'})

        st.plotly_chart(fig)

    elif "Depend_" in selected_sheet:
        # 산점도 추가
        # 첫 번째 열이 x값, 두 번째 열이 y값이라고 가정
        x_col, y_col = df.columns[0], df.columns[1]
        fig_scatter = px.scatter(df, x=x_col, y=y_col,
                                 title=f'{selected_sheet} 산점도',
                                 labels={x_col: x_col, y_col: 'SHAP 값'})
        st.plotly_chart(fig_scatter)

    elif "RNN" in selected_sheet:
        st.write("### RNN 예측 결과")

        # 라인 차트 생성 (실제 값과 예측 값 비교)
        fig = px.line(df, x='Year', y=['Actual', 'Predicted'],
                      title='실제 vs 예측',
                      labels={'Year': '년도', 'value': '값'})

        st.plotly_chart(fig)

# 실행 방법 안내
st.sidebar.markdown("""
### 실행 방법
1. 분석할 시트를 선택하세요
2. '확인' 버튼을 클릭하세요
3. 결과가 메인 화면에 표시됩니다
""")

