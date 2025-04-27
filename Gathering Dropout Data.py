import requests
from bs4 import BeautifulSoup
import pandas as pd
import xml.etree.ElementTree as ET
import re
import numpy as np
from scipy import stats
import gspread
import json
import os

json_file_path = 'service_account.json'
spreadsheet_url = "https://docs.google.com/spreadsheets/d/1DhfQFFR9gSV7plLLGgrqmNaohfbYW3Q9Fm_vuli8czI/edit?usp=sharing"

# 환경 변수 가져오기
key = os.getenv('API_KEY')
if key:
    print(f"API 키: {key}")
else:
    print("Error: API_KEY가 설정되지 않았습니다.")

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

def fill_missing_values(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        if df[col].isnull().sum() == 0:
            continue

        # 숫자형으로 변환 (NaN 생성 방지)
        df.index = pd.to_numeric(df.index, errors='coerce')

        # 1. 선형성 검정 (R² > 0.7이면 선형 보간)
        x = df.index[~df[col].isnull()]

        y = df[col].dropna()
        if len(y) >= 2:
            slope, intercept, r_value, _, _ = stats.linregress(x, y)
            if r_value ** 2 > 0.7:  # 선형성이 강하면 선형 보간
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
                continue

        # 2. 인접 값의 평균으로 채우기 (단일 NaN인 경우)
        df[col] = df[col].fillna(df[col].rolling(window=3, min_periods=1, center=True).mean())

        # 3. 남은 NaN은 지수 보간 (비선형 데이터)
        if df[col].isnull().sum() > 0:
            df[col] = df[col].interpolate(method='spline', order=2, limit_direction='both')

    return df

def extract_indicator_codes_and_names(xml_content):
    """
    XML 내용에서 모든 지표코드와 지표명을 추출하는 함수

    Args:
        xml_content (str): XML 파일 내용

    Returns:
        list: 지표코드와 지표명을 담은 딕셔너리 리스트. 예: [{'지표코드': '1060', '지표명': '소비자물가지수'}, ...]
    """
    try:
        # XML 파싱
        root = ET.fromstring(xml_content)

        # 결과를 저장할 리스트
        indicators = []

        # 모든 '지표' 요소 찾기
        for indicator in root.findall('.//통계표'):
            # 지표코드와 지표명 추출
            statisCode = indicator.find('통계표코드').text if indicator.find('통계표코드') is not None else None
            name = indicator.find('통계표명').text if indicator.find('통계표명') is not None else None
            code = statisCode[:5] if statisCode.startswith('F') else statisCode[:4]

            # 코드와 이름이 모두 있는 경우만 결과에 추가
            if code and name:
                indicators.append({
                    '통계명': name,
                    '통계코드': statisCode,
                    '지표코드': code
                })

        return indicators

    except ET.ParseError as e:
        print(f"XML 파싱 오류: {e}")
        return []
    except Exception as e:
        print(f"오류 발생: {e}")
        return []

def GetDataFromGroup( aTable_name, aItemGroup):
    data = []
    categories = aItemGroup.find_all('분류1')
    if not categories:
        categories = aItemGroup.find_all('항목')
    if categories:
        for category in categories:
            category_name = category.get('이름', '')
            category_code = category.get('코드', '')

            # 4. <열>에서 연도와 값 추출
            for col in category.find_all('열'):
                year = col.get('주기', '')
                value = col.text.strip() if col.text else None

                # 데이터 저장
                data.append({
                    '연도': year,
                    aTable_name + '_' + category_name + '_' + table_unit: value
                })
    else:
        print('카테고리를 찾지 못 했습니다.')
    return data


with open('통계데이터.xml', 'r', encoding='utf-8') as file:
    xml_content = file.read()

# 함수 호출
indicators = extract_indicator_codes_and_names(xml_content)

# 결과 출력
#    for idx, indicator in enumerate(indicators, 1):
#        print(f"{idx}. 통계명: {indicator['통계명']}, 통계코드: {indicator['통계코드']}, 지표코드: {indicator['지표코드']}")

#indicator = indicators[2]

merged_df = pd.DataFrame()  # 컬럼 이름을 원하는 대로 변경 가능

for indicator in indicators:
    base_url = f"https://www.index.go.kr/unity/openApi/xml_stts.do?idntfcId={key}&ixCode={indicator['지표코드']}&statsCode={indicator['통계코드']}&period=1999:2023"
    #print( indicator['통계명'] + '\t' + base_url)
    # URL에서 XML 데이터 가져오기
    response = requests.get(base_url)
    # 2. 응답 확인 (200이면 성공)
#        print(f"Status Code: {response.status_code}")  # 200이어야 함
#        print(f"Response Content:\n{response.text[:500]}")  # XML 앞부분 확인

    # 2. BeautifulSoup으로 XML 파싱 (반드시 'lxml' 지정)
    soup = BeautifulSoup(response.content, 'lxml-xml')  # 또는 'xml' (lxml이 설치된 경우)

    #print(response.text)
    # 데이터 저장을 위한 리스트
    data = []

    # 3. 데이터 추출 (태그 구조에 맞게 수정)
    for stat_table in soup.find_all('통계표'):
        table_name = stat_table.find('통계표명').text
        table_unit = stat_table.find('단위').text

        #for table in stat_table.find_all('표'):
        itemGroups = stat_table.find_all('항목그룹')
        if not itemGroups:
            data.extend(GetDataFromGroup( table_name, stat_table))
        else:
            for itemGroup in itemGroups:
                data.extend( GetDataFromGroup( table_name, itemGroup))


        # DataFrame으로 변환
        df = pd.DataFrame(data)

        # '연도'를 제외한 컬럼명 가져오기
        # columns_to_agg = df.columns.difference(['연도'])
        # 연도별로 그룹화하고, 각 컬럼의 첫 번째 유효한 값 선택
        df = df.groupby('연도').agg('first')

        # 결과 확인
        #output_path = "./data/" + table_name + '.xlsx'
        #df.to_excel(output_path)

        if table_name != '학교생활 만족도':
            df_filtered = df.filter(like='전체')
            if df_filtered.empty:
                # 검색할 단어 리스트
                keywords = ['원화_만', '고졸', '_실질', '백분율', '요일평균', '_조이혼율', '합계', '고등학교_소계', '고등학교_%', '소비자물가상승률(%)']
                # 정규식 패턴 생성 (단어들을 |로 연결)
                pattern = '|'.join(keywords)

                # 정규식으로 필터링
                filtered_cols = [col for col in df.columns if re.search(pattern, col)]
                if filtered_cols:
                    df_filtered = df[filtered_cols]
                else:
                    # print("일치하는 컬럼이 없습니다. " + table_name)
                    df[table_name] = df.astype(float).mean(axis=1)
                    # 2. 'table_name' 컬럼만 남기기
                    df_filtered = df[[table_name]]

            if len(df_filtered.columns) >= 2:
                df_filtered = df_filtered.filter(like='_고등학교')
        else:
            df_filtered = df
        if len(df_filtered.columns) >= 2:
            print(table_name + "   DataFrame에 컬럼이 2개 이상 있습니다.")

        if len(df_filtered.columns) < 1:
            print(table_name + "   DataFrame에 컬럼이 0개 있습니다.")

        # print(table_name + df_filtered.columns)


        # 결과 확인
        #output_path = "./data/" + table_name + '.xlsx'
        #df_filtered.to_excel(output_path)

        # 기존 인덱스와 새로운 인덱스 비교
        common_index = merged_df.index.intersection(df_filtered.index)
        new_index = df_filtered.index.difference(merged_df.index)

        # 공통 인덱스는 옆으로 병합
        if not common_index.empty:
            merged_df = pd.concat([merged_df, df_filtered.loc[common_index]], axis=1)

        # 새로운 인덱스는 아래로 추가
        if not new_index.empty:
            merged_df = pd.concat([merged_df, df_filtered.loc[new_index]], axis=0)


# 1. 인덱스를 초기화 (기존 인덱스를 열로 변환하지 않음)
merged_df.index.names = ['연도']
merged_df.reset_index(drop=False, inplace=True)
# 2. 'p'가 포함된 행 삭제
merged_df['연도'] = merged_df['연도'].astype(str)  # 문자열로 변환
merged_df = merged_df[~merged_df['연도'].str.contains('p', na=False)]
merged_df.set_index('연도', inplace=True)  # 연도 열을 인덱스로

#output_file = 'merged_df.xlsx'
#merged_df.to_excel(output_file)

merged_df = merged_df.astype(float, errors='ignore')  # errors='ignore'로 변환 불가 열은 유지
merged_df = fill_missing_values( merged_df)

# 엑셀 파일로 저장
#output_file = 'fill_missing_merged_df.xlsx'
#merged_df.to_excel(output_file)


# 데이터프레임의 인덱스 이름 추가 (선택 사항)
merged_df = merged_df.reset_index()
# 기존 컬럼명
old_columns = [
"연도", "1인당 GNI_명목/원화_만 원, 달러, %", "명목 및 실질 경제성장률 1)2)_실질 경제성장률_%", "학교급별 교원 1인당 학생 수 1)2)3)_고등학교_소계_명", "교육 수준별 임금수준 1)2)3)4)_고졸_%", "가구주연령 및 소득수준별 교육비 부담도 1)_전체_%", "성·연령·소득수준별 긍정적 정서경험 1)2)_전체_%", "기초학력 미달률 1)", "학교급별 다문화학생 1)2) 비율 3)_전체_%", "부패인식 지수와 부패인식 지수 국제순위_한국순위 백분율 4)_점, 위, 개국, %", "학교급별 사교육 참여율_고등학교_전체_%", "학교급별 학생 1인당 월평균 사교육비 1)_고등학교_전체_만 원", "성·연령·소득수준별 삶의 만족도 1)2)_전체_%", "범죄유형별 소년 범죄자율 1)2)_전체_명/10만 명", "연령 및 소득수준별 소득 만족도_전체_%", "실업률 1)_전체_%", "성 및 연령별 우울감 경험률 1)_전체_%", "성·연령·소득수준별 일의 가치 인식 1)2)_전체_%", "성·교육수준·직업별 전공직업일치도_전체_%", "이혼건수와 조이혼율_조이혼율 1)_건, 건/1,000명", "교육단계별 진학률_전체_%", "학교생활 만족도_교육내용_%", "학교생활 만족도_교육방법_%", "학교생활 만족도_교우관계_%", "학교생활 만족도_교사와의 관계_%", "학교생활 만족도_학교시설_%", "학교생활 만족도_학교주변 환경_%", "학교생활 만족도_전반적 학교생활_%", "학교폭력 피해율_전체_%", "학교급별 학급당 학생 수_고등학교_소계_명", "학교급별 학업중단율_고등학교_%", "여가시간 충분도_요일평균_%", "소비자물가총지수와 주요 품목별 소비자물가상승률"
]

# 수정 컬럼명
new_columns = [
"년도", "1인당 GNI", "실질 경제성장률", "교원 1인당 학생 수", "교육 수준별 임금수준", "교육비 부담도", "긍정적 정서경험", "기초학력 미달률", "다문화학생 비율", "부패인식", "사교육 참여율", "1인당 월평균 사교육비", "삶의 만족도", "소년 범죄자율", "소득 만족도", "실업률", "우울감 경험률", "일의 가치 인식", "전공직업일치도", "조이혼율", "진학률", "교육내용 만족도", "교육방법 만족도", "교우관계 만족도", "교사와의 관계 만족도", "학교시설 만족도", "학교주변 환경 만족도", "학교생활 만족도", "학교폭력 피해율", "고등학교 학생 수", "학업중단율", "여가시간 충분도", "소비자물가상승률"
]

# 컬럼 이름 매핑 딕셔너리 생성
column_mapping = dict(zip(old_columns, new_columns))
# 컬럼명 교체
merged_df.rename(columns=column_mapping, inplace=True)
merged_df['소년 범죄자율'] = merged_df['소년 범죄자율'] / 1000
#print(merged_df.head(5))

# 데이터프레임을 리스트로 변환 (인덱스 포함)
data = [merged_df.columns.tolist()] + merged_df.values.tolist()

worksheet = doc.get_worksheet(0)  # 0번 인덱스 = 첫 번째 시트
# 기존 데이터 삭제
worksheet.clear()

# 시트에 데이터 업로드
worksheet.update('A1', data)