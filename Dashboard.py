import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import google.generativeai as genai
import json
import gspread
import hashlib

# json_file_path = 'service_account.json'
spreadsheet_url = st.secrets["spreadsheet_url"]
apiKey = st.secrets["API_KEY"]

# 앱 제목 설정
st.title("학업중단 요인분석")


# 엑셀 파일 로드
@st.cache_data  # 성능 향상을 위한 캐싱
def load_data():
    # service_account.json 파일을 직접 읽기
    try:
        service_account_info = st.secrets["gcp_service_account"]
        gc = gspread.service_account_from_dict(service_account_info)
        doc = gc.open_by_url(spreadsheet_url)
    except Exception as e:
        st.error(f"Google Sheets 연결 실패: {str(e)}")
        st.stop()

    worksheets = doc.worksheets()

    sheets = {}
    for worksheet in worksheets:
        name = worksheet.title
        if name != "시트1" and not name.startswith("Cache_"):  # "시트1"과 캐시 시트는 제외
            # worksheet 데이터를 DataFrame으로 변환
            data = worksheet.get_all_values()
            df = pd.DataFrame(data[1:], columns=data[0])  # 첫 줄을 컬럼 이름으로 사용
            sheets[name] = df

    # 캐시 시트 확인
    cache_sheet_name = "Cache_Results"
    cache_exists = False
    cache_worksheet = None

    for worksheet in worksheets:
        if worksheet.title == cache_sheet_name:
            cache_exists = True
            cache_worksheet = worksheet
            break

    # 캐시 시트가 없으면 생성
    if not cache_exists:
        cache_worksheet = doc.add_worksheet(title=cache_sheet_name, rows=1000, cols=20)
        # 헤더 추가
        cache_worksheet.update('A1:E1',
                               [['sheet_name', 'data_hash', 'additional_question', 'chart_title', 'analysis_result']])

    # 캐시 데이터 불러오기
    cache_data = cache_worksheet.get_all_values()
    cache_df = pd.DataFrame(data=cache_data[1:], columns=cache_data[0]) if len(cache_data) > 1 else pd.DataFrame(
        columns=cache_data[0])

    genai.configure(api_key=apiKey)
    # 모델 선택 (일반 대화용)
    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
    return sheets, model, doc, cache_worksheet, cache_df


# 데이터 로드
sheets, model, doc, cache_worksheet, cache_df = load_data()
sheet_names = list(sheets.keys())

# 사이드바에 시트 선택 위젯 추가
selected_sheet = st.selectbox(
    "분석할 시트를 선택하세요",
    sheet_names,
    index=0
)

# 추가 질문 입력 필드
additional_question = st.text_input("추가 분석 질문 (선택사항)", "")


# 데이터 해시 생성 함수
def get_data_hash(df, additional_q=""):
    # 데이터프레임을 문자열로 변환하고 추가 질문과 함께 해시
    df_str = df.to_string()
    combined_str = df_str + additional_q
    return hashlib.md5(combined_str.encode()).hexdigest()


# 확인 버튼
if st.button("확인"):
    st.subheader(f"{selected_sheet} 분석 결과")

    # 선택된 시트 데이터 가져오기
    df = sheets[selected_sheet]

    # 데이터 표시
    st.write("### 데이터 테이블")
    st.dataframe(df)

    # 데이터 해시 계산
    data_hash = get_data_hash(df, additional_question)

    # 캐시에서 결과 검색
    cached_result = None
    chart_title = ""

    # 캐시 데이터프레임에서 일치하는 항목 찾기
    mask = (cache_df['sheet_name'] == selected_sheet) & (cache_df['data_hash'] == data_hash)
    if not cache_df.empty and mask.any():
        cached_row = cache_df.loc[mask].iloc[0]
        cached_result = cached_row['analysis_result']
        chart_title = cached_row['chart_title']
        st.success("캐시된 분석 결과를 불러왔습니다.")

    # 시트 유형에 따라 다른 시각화 제공
    if "SHAP" in selected_sheet or "Importance" in selected_sheet:
        st.write("### 중요도 시각화")

        # SHAP Importance 또는 Feature Importance 시트
        if selected_sheet == "SHAP Importance":
            fig = px.bar(df, x='feature', y='shap_importance',
                         title='SHAP Feature Importance',
                         labels={'feature': '요인', 'shap_importance': 'SHAP 중요도'})
            chart_title = "SHAP Feature Importance"

            # 중요도 순으로 정렬된 요인들 추출
            sorted_df = df.sort_values(by='shap_importance', ascending=False)
            top_features = sorted_df.head(5)  # 상위 5개 요인

            # 차트 데이터 설명 추가
            chart_data_desc = "차트 데이터 (SHAP 중요도 상위 5개 요인):\n"
            for idx, row in top_features.iterrows():
                chart_data_desc += f"- {row['feature']}: {row['shap_importance']}\n"

            prompt = f"아래는 학업중단 요인분석에 대한 '{selected_sheet}' 데이터입니다.\n\n"
            prompt += f"데이터 테이블:\n{df.to_string(index=False)}\n\n"
            prompt += "이 데이터는 SHAP Feature Importance 분석 결과입니다. "
            prompt += "SHAP 값이 높을수록 해당 요인이 학업중단 예측에 더 큰 영향을 미칩니다.\n\n"
            prompt += chart_data_desc + "\n"
            prompt += "1. 위 SHAP Feature Importance 결과가 의미하는 것은 무엇인가요?\n"
            prompt += "2. 가장 중요한 상위 3개 요인은 무엇이며, 이것이 학업중단에 어떤 영향을 미칠까요?\n"
            prompt += "3. 이 결과를 바탕으로 학업중단을 줄이기 위한 교육적 개입 방안을 제안해주세요.\n"

        elif selected_sheet == "Feature Importance":
            fig = px.bar(df, x='Feature', y='Importance',
                         title='Feature Importance',
                         labels={'Feature': '요인', 'Importance': '중요도'})
            chart_title = "Feature Importance"

            # 중요도 순으로 정렬된 요인들 추출
            sorted_df = df.sort_values(by='Importance', ascending=False)
            top_features = sorted_df.head(5)  # 상위 5개 요인

            # 차트 데이터 설명 추가
            chart_data_desc = "차트 데이터 (중요도 상위 5개 요인):\n"
            for idx, row in top_features.iterrows():
                chart_data_desc += f"- {row['Feature']}: {row['Importance']}\n"

            prompt = f"아래는 학업중단 요인분석에 대한 '{selected_sheet}' 데이터입니다.\n\n"
            prompt += f"데이터 테이블:\n{df.to_string(index=False)}\n\n"
            prompt += "이 데이터는 머신러닝 모델의 Feature Importance 분석 결과입니다. "
            prompt += "중요도 값이 높을수록 해당 요인이 학업중단 예측에 더 큰 영향을 미칩니다.\n\n"
            prompt += chart_data_desc + "\n"
            prompt += "1. 위 Feature Importance 결과가 말해주는 주요 인사이트는 무엇인가요?\n"
            prompt += "2. 상위 중요도를 가진 요인들이 학업중단에 미치는 영향을 설명해주세요.\n"
            prompt += "3. 이러한 요인들을 고려했을 때 학교나 교육기관이 취해야 할 조치는 무엇일까요?\n"

        else:
            # 다른 중요도 시트 (첫 번째 열이 feature, 두 번째 열이 value라고 가정)
            col1, col2 = df.columns[0], df.columns[1]
            fig = px.bar(df, x=col1, y=col2,
                         title=f'{selected_sheet}',
                         labels={col1: '요인', col2: '중요도'})
            chart_title = selected_sheet

            # 중요도 순으로 정렬된 요인들 추출
            try:
                sorted_df = df.sort_values(by=col2, ascending=False)
                top_features = sorted_df.head(5)  # 상위 5개 요인

                # 차트 데이터 설명 추가
                chart_data_desc = f"차트 데이터 ('{col2}' 기준 상위 5개 요인):\n"
                for idx, row in top_features.iterrows():
                    chart_data_desc += f"- {row[col1]}: {row[col2]}\n"
            except:
                chart_data_desc = "차트 데이터:\n"
                for idx, row in df.head(5).iterrows():
                    chart_data_desc += f"- {row[col1]}: {row[col2]}\n"

            prompt = f"아래는 학업중단 요인분석에 대한 '{selected_sheet}' 데이터입니다.\n\n"
            prompt += f"데이터 테이블:\n{df.to_string(index=False)}\n\n"
            prompt += f"이 데이터는 '{selected_sheet}' 분석 결과로, 학업중단에 관련된 다양한 요인들의 중요도를 보여줍니다.\n\n"
            prompt += chart_data_desc + "\n"
            prompt += f"1. 이 '{selected_sheet}' 분석 결과의 주요 발견점은 무엇인가요?\n"
            prompt += "2. 가장 중요한 요인들은 무엇이며, 이것이 학업중단과 어떤 관련이 있나요?\n"
            prompt += "3. 이 결과를 바탕으로 학생들의 학업유지를 위한 전략적 제안을 해주세요.\n"

        st.plotly_chart(fig)

    elif "Depend_" in selected_sheet:
        # 산점도 추가
        # 첫 번째 열이 x값, 두 번째 열이 y값이라고 가정
        x_col, y_col = df.columns[0], df.columns[1]
        fig_scatter = px.scatter(df, x=x_col, y=y_col,
                                 title=f'{selected_sheet} 산점도',
                                 labels={x_col: x_col, y_col: 'SHAP 값'})
        st.plotly_chart(fig_scatter)
        chart_title = f"{selected_sheet} 산점도"

        # 변수 이름에서 'Depend_' 제거하여 실제 변수명 추출
        feature_name = selected_sheet.replace('Depend_', '')

        # 데이터 패턴 설명
        # 상관관계 계산
        try:
            # 숫자형 데이터로 변환 시도
            df_numeric = df.copy()
            df_numeric[x_col] = pd.to_numeric(df_numeric[x_col], errors='coerce')
            df_numeric[y_col] = pd.to_numeric(df_numeric[y_col], errors='coerce')

            correlation = df_numeric[x_col].corr(df_numeric[y_col])
            correlation_desc = f"상관계수: {correlation:.4f}"

            # 데이터 포인트 몇 개 샘플링하여 설명
            data_points = df.head(5).to_dict('records')
            data_points_desc = "데이터 포인트 샘플:\n"
            for point in data_points:
                data_points_desc += f"- {x_col}: {point[x_col]}, {y_col}: {point[y_col]}\n"

            # 추세선 기울기 (양/음)
            trend = "양의 상관관계" if correlation > 0 else "음의 상관관계" if correlation < 0 else "뚜렷한 상관관계 없음"
            trend_desc = f"추세: {trend}"

            scatter_data_desc = f"{data_points_desc}\n{correlation_desc}\n{trend_desc}\n"
        except:
            # 숫자형 변환 실패 시 단순 데이터 포인트만 제공
            data_points = df.head(5).to_dict('records')
            scatter_data_desc = "데이터 포인트 샘플:\n"
            for point in data_points:
                scatter_data_desc += f"- {x_col}: {point[x_col]}, {y_col}: {point[y_col]}\n"

        prompt = f"아래는 학업중단 요인분석에 대한 '{selected_sheet}' 데이터입니다.\n\n"
        prompt += f"데이터 테이블:\n{df.to_string(index=False)}\n\n"
        prompt += f"이 데이터는 '{feature_name}' 요인과 학업중단 예측 간의 종속성(의존도) 분석 결과를 보여주는 산점도입니다.\n\n"
        prompt += f"차트 데이터 정보:\n{scatter_data_desc}\n"
        prompt += f"1. '{feature_name}' 요인과 학업중단 간의 관계는 어떤 패턴을 보이나요?\n"
        prompt += f"2. 이 산점도에서 {x_col}가 증가하거나 감소할 때 SHAP 값은 어떻게 변하며, 이것은 무엇을 의미하나요?\n"
        prompt += f"3. 이 결과를 바탕으로 '{feature_name}' 관련 학업중단 위험을 줄이기 위한 구체적인 전략을 제안해주세요.\n"

    elif "RNN" in selected_sheet:
        st.write("### RNN 예측 결과")

        # 라인 차트 생성 (실제 값과 예측 값 비교)
        fig = px.line(df, x='Year', y=['Actual', 'Predicted'],
                      title='실제 vs 예측',
                      labels={'Year': '년도', 'value': '값'})

        st.plotly_chart(fig)
        chart_title = "RNN 학업중단율 예측 결과"

        # 예측 성능 계산 시도
        try:
            # 숫자형으로 변환
            actual = pd.to_numeric(df['Actual'], errors='coerce')
            predicted = pd.to_numeric(df['Predicted'], errors='coerce')

            # 모델 성능 지표 계산
            mse = ((actual - predicted) ** 2).mean()
            mae = (actual - predicted).abs().mean()

            # 최근 추세 분석
            recent_years = df.tail(3)
            trend_desc = "최근 3년 데이터:\n"
            for idx, row in recent_years.iterrows():
                trend_desc += f"- {row['Year']}: 실제값 {row['Actual']}, 예측값 {row['Predicted']}\n"

            # 예측 정확도 정보
            accuracy_desc = f"모델 성능 지표:\n- MSE(평균제곱오차): {mse:.4f}\n- MAE(평균절대오차): {mae:.4f}\n"

            rnn_data_desc = trend_desc + "\n" + accuracy_desc
        except:
            # 계산 실패 시 원본 데이터만 제공
            rnn_data_desc = "연도별 실제값과 예측값:\n"
            for idx, row in df.iterrows():
                rnn_data_desc += f"- {row['Year']}: 실제값 {row['Actual']}, 예측값 {row['Predicted']}\n"

        prompt = f"아래는 학업중단 요인분석에 대한 '{selected_sheet}' 데이터입니다.\n\n"
        prompt += f"데이터 테이블:\n{df.to_string(index=False)}\n\n"
        prompt += "이 데이터는 RNN(순환신경망) 모델을 사용한 학업중단율 예측 결과입니다. 실제 값과 예측 값을 비교하고 있습니다.\n\n"
        prompt += f"차트 데이터 분석:\n{rnn_data_desc}\n"
        prompt += "1. 실제 값과 예측 값 사이의 일치도는 어떤가요? 모델의 예측 정확도를 평가해주세요.\n"
        prompt += "2. 이 예측 결과에서 볼 수 있는 학업중단율의 추세는 무엇인가요?\n"
        prompt += "3. 향후 학업중단율이 어떻게 변할 것으로 예상되며, 이에 대응하기 위한 교육정책적 제안은 무엇인가요?\n"

    # 추가 질문이 있으면 프롬프트에 추가
    if additional_question:
        prompt += f"\n추가 질문: {additional_question}\n"

    # 최종 프롬프트 완성
    prompt += "\n위 질문들에 대해 데이터를 기반으로 상세히 분석해주세요. 교육 전문가의 관점에서 인사이트를 제공해주세요."

    # 캐시된 결과가 없으면 Gemini API 호출
    if cached_result is None:
        # 로딩 표시
        with st.spinner('Gemini API로 분석 결과 생성 중...'):
            try:
                # Gemini API 요청을 위한 설정
                generation_config = {
                    "temperature": 0.3,  # 낮은 온도로 설정하여 일관된 응답 생성
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 2048,  # 충분히 긴 응답을 위한 설정
                }

                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ]

                # 응답 생성
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )

                # 응답이 여러 부분으로 나뉘어 있는 경우를 처리
                if hasattr(response, 'parts'):
                    response_text = ''.join([part.text for part in response.parts])
                else:
                    response_text = response.text

                # 캐시에 결과 저장
                try:
                    # 기존 데이터에 새 행 추가
                    new_row = [[selected_sheet, data_hash, additional_question, chart_title, response_text]]
                    cache_worksheet.append_rows(new_row)
                    st.success("분석 결과가 캐시에 저장되었습니다.")
                except Exception as e:
                    st.warning(f"캐시 저장 실패: {str(e)}")

                # 응답 표시
                st.write("### AI 분석 인사이트")
                st.markdown(response_text)

            except Exception as e:
                st.error(f"Gemini API 오류: {str(e)}")
                st.error("프롬프트가 너무 길거나 API 연결에 문제가 있을 수 있습니다.")
    else:
        # 캐시된 결과 표시
        st.write("### AI 분석 인사이트 (캐시에서 불러옴)")
        st.markdown(cached_result)

# 실행 방법 안내
st.sidebar.markdown("""
### 실행 방법
1. 분석할 시트를 선택하세요
2. 원하시면 추가 분석 질문을 입력하세요
3. '확인' 버튼을 클릭하세요
4. 데이터 시각화와 AI 분석 인사이트가 메인 화면에 표시됩니다
""")

# 앱 정보
st.sidebar.markdown("""
---
### 앱 정보
이 애플리케이션은 학업중단 요인을 분석하고 Gemini AI를 활용하여 
데이터 기반 인사이트를 제공합니다. 동일한 데이터 분석은 캐시되어 
API 호출을 최소화합니다.
""")