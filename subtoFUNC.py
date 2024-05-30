import os
import re

def replace_functions(content):
    # 함수 호출 부분 변경
    content = re.sub(r'\bsub_\w+', 'FUNC', content)

    # 함수 정의 부분 변경
    content = re.sub(r'(__fastcall\s+)sub_\w+', r'\1FUNC', content)

    return content

# 디렉토리 경로 설정
source_path = "/LLama/functions"

# 디렉토리 내의 파일들을 순회하며 작업 수행
for filename in os.listdir(source_path):
    # 파일의 전체 경로 구성
    file_path = os.path.join(source_path, filename)
    
    # 파일인 경우에만 작업 수행
    if os.path.isfile(file_path):
        # 파일 읽기
        with open(file_path, "r") as file:
            file_content = file.read()

        # 함수 이름 및 호출 변경
        updated_content = replace_functions(file_content)

        # 변경된 내용 출력 (테스트용)
        print(updated_content)

        # 변경된 내용을 파일에 쓰기
        # 이 부분은 실제로 파일을 변경하고 싶을 때 사용하세요
        with open(file_path, "w") as file:
            file.write(updated_content)
