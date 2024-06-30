from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import os
import random
import json

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

app = Flask(__name__)
CORS(app)

# 이미지 디렉토리 경로
IMAGE_DIRECTORY = 'C:\\back-end-app\\images'

exclude_indices = set()  # 제외할 메뉴 인덱스 목록 초기화



# JSON 파일 읽기
# 메뉴 데이터가 담긴 JSON 파일을 읽어옵니다.
with open('menu.json', 'r', encoding='utf-8') as f:
    menu_data = json.load(f)
############################################################################################################
#0과 1의 가중치 계산을 각 0과 1의 횟수로 계산하여 값이 날라가지 않게 수정. 다만 여러 문제가 있어 다음으로 변경

# JSON 파일 읽기
# 메뉴 데이터가 담긴 JSON 파일을 읽어옵니다.
with open('menu.json', 'r', encoding='utf-8') as f:
    menu_data = json.load(f)

# 메뉴 속성 벡터화 함수
# 각 메뉴의 속성을 벡터(숫자의 배열)로 변환하는 함수입니다.
def get_menu_vector(menu):
    vector = []
    vector.append(menu["spiciness"])  # 매운 정도
    vector.append(menu["temperature"])  # 온도
    vector.append(menu["price"])  # 가격
    vector.extend([menu["taste"]["sweet"], menu["taste"]["salty"], menu["taste"]["sour"],
                   menu["taste"]["bitter"], menu["taste"]["savory"], menu["taste"]["pungent"],
                   menu["taste"]["bland"]])  # 맛 속성들 추가
    vector.extend([menu["country"]["korean"], menu["country"]["schoolFood"], menu["country"]["western"],
                   menu["country"]["chinese"], menu["country"]["japanese"], menu["country"]["etc"]])  # 국가 속성들 추가
    vector.extend([menu["mainIngredient"]["grain"]["bread"], menu["mainIngredient"]["grain"]["rice"],
                   menu["mainIngredient"]["grain"]["noodle"], menu["mainIngredient"]["meat"]["meat"],
                   menu["mainIngredient"]["meat"]["poultry"], menu["mainIngredient"]["seafood"]["fish"],
                   menu["mainIngredient"]["seafood"]["shellfish"], menu["mainIngredient"]["seafood"]["crustacean"],
                   menu["mainIngredient"]["vegetable"]["leafyVegetable"], menu["mainIngredient"]["vegetable"]["rootVegetable"],
                   menu["mainIngredient"]["dairy"], menu["mainIngredient"]["egg"],
                   menu["mainIngredient"]["bean"], menu["mainIngredient"]["etc"]])  # 주재료 속성들 추가
    vector.extend([menu["cookingMethod"]["grilled"], menu["cookingMethod"]["fried"], menu["cookingMethod"]["steamed"],
                   menu["cookingMethod"]["soup"], menu["cookingMethod"]["raw"],
                   menu["cookingMethod"]["stirFried"], menu["cookingMethod"]["etc"]])  # 조리 방법 속성들 추가
    vector.extend([menu["mealTime"]["breakfast"], menu["mealTime"]["lunch"], menu["mealTime"]["dinner"],
                   menu["mealTime"]["midnightSnack"]])  # 식사 시간 속성들 추가
    return np.array(vector).reshape(1, -1)  # 벡터를 2차원 배열로 변환 (1행, n열)


# 유사한 메뉴 찾기 함수
# 주어진 선호 벡터와 유사한 메뉴를 찾아주는 함수입니다.
#이게 사용자한테
def find_similar_menus(target_vector, menu_data, exclude_indices, count):
    similarities = []
    for index, menu in enumerate(menu_data):
        if index not in exclude_indices:  # 이미 추천된 메뉴는 제외합니다.
            menu_vector = get_menu_vector(menu)  # 메뉴를 벡터로 변환
            valid_indices = ~np.isnan(target_vector) & ~np.isnan(menu_vector)  # NaN이 아닌 인덱스 찾기
            if np.any(valid_indices):
                similarity = cosine_similarity(target_vector[valid_indices].reshape(1, -1), menu_vector[valid_indices].reshape(1, -1))[0][0]  # 코사인 유사도 계산
                similarities.append((similarity, index))
    similarities.sort(reverse=True, key=lambda x: x[0])  # 유사도 기준으로 정렬
    return [menu_data[i] for _, i in similarities[:count]], [i for _, i in similarities[:count]]  # 유사한 상위 count개수 메뉴 반환

# 선호 벡터 계산 함수
# 사용자가 좋아요/싫어요 한 메뉴들을 기반으로 선호도를 계산하는 함수입니다.
def calculate_preference_vector(liked_menus, disliked_menus):
    global preference_vector
    preference_vector = np.full(len(get_menu_vector(menu_data[0])[0]), np.nan)  # 선호 벡터를 NaN으로 초기화
    ones_count = np.zeros_like(preference_vector)  # 좋아요 횟수를 저장할 배열
    zeros_count = np.zeros_like(preference_vector)  # 싫어요 횟수를 저장할 배열
    total_count = np.zeros_like(preference_vector)  # 전체 횟수를 저장할 배열

    for menu in liked_menus:
        menu_vector = get_menu_vector(menu)[0]
        for i, value in enumerate(menu_vector):
            if i < 3:  # spiciness, temperature, price는 기존 방식 유지
                ones_count[i] += value
                total_count[i] += 1
            else:
                if value == 1:
                    ones_count[i] += 1
                    total_count[i] += 1

    for menu in disliked_menus:
        menu_vector = get_menu_vector(menu)[0]
        for i, value in enumerate(menu_vector):
            if i < 3:  # spiciness, temperature, price는 기존 방식 유지
                zeros_count[i] += value
                total_count[i] += 1
            else:
                if value == 1:
                    zeros_count[i] += 1
                    total_count[i] += 1

    for i in range(len(preference_vector)):
        if i < 3:
            preference_vector[i] = ones_count[i] / total_count[i] if total_count[i] != 0 else np.nan
        else:
            preference_vector[i] = ones_count[i] / total_count[i] if total_count[i] != 0 else np.nan

    return preference_vector.reshape(1, -1)  # 선호 벡터 반환


# 추천 로직
# 선호 벡터를 기반으로 메뉴를 추천하는 함수입니다.
def recommend_menu(preference_vector, exclude_indices):
    recommended_menus, recommended_indices = find_similar_menus(preference_vector, menu_data, exclude_indices)
    return recommended_menus, recommended_indices

# 좋아요 목록 보여주기 함수
# 사용자가 좋아요 한 메뉴 목록을 보여줍니다.
def show_liked_menus():
    if liked_menus:
        liked_list = "\n".join([f"{idx + 1}. {menu['name']}" for idx, menu in enumerate(liked_menus)])
        messagebox.showinfo("좋아요한 메뉴 목록", liked_list)
    else:
        messagebox.showinfo("좋아요한 메뉴 목록", "좋아요한 메뉴가 없습니다.")


def update_preference_vector():
    global preference_vector
    preference_vector = calculate_preference_vector(liked_menus, disliked_menus)
    preference_vector_list = preference_vector.tolist()[0]

    # 선호 벡터를 표 형식으로 변환
    preference_vector_table = "속성\t값\n"
    attributes = ["매운 정도", "온도", "달콤한", "짠", "신", "쓴", "고소한", "자극적인",
                  "한식", "분식", "양식", "중식", "일식", "기타",
                  "빵", "밥", "면", "육류", "가금류", "생선", "조개류", "갑각류",
                  "잎채소", "뿌리채소", "유제품", "계란", "콩", "기타",
                  "구이", "튀김", "찜", "국", "생", "볶음", "기타",
                  "아침", "점심", "저녁", "야식",
                  "가격"]
    for attr, value in zip(attributes, preference_vector_list):
        preference_vector_table += f"{attr}\t{value:.2f}\n"



#무작위로 메뉴보여주는것.
def get_next_menu():
    global recommendation_count, menu, index

    random_menu_index = random.randint(0, len(menu_data) - 1)  # 무작위로 인덱스 선택
    while random_menu_index in exclude_indices:
        random_menu_index = random.randint(0, len(menu_data) - 1)
    menu = menu_data[random_menu_index]
    index = random_menu_index

    # update_gui(menu, recommendation_count)

# 초기 설정
recommendation_count = 1  # 추천 횟수 초기화
exclude_indices = set()  # 제외할 메뉴 인덱스 목록 초기화
liked_menus = []  # 좋아요 한 메뉴 목록 초기화
disliked_menus = []  # 싫어요 한 메뉴 목록 초기화
preference_vector = np.zeros(len(get_menu_vector(menu_data[0])[0]))  # 선호 벡터 초기화
phase_count = 0  # 추천 단계 초기화











############################################################################################################

@app.route('/data', methods=['GET'])
def get_random5_menu():
    global recommendation_count, menu, index
    menu = []
    for _ in range(5):
        random_menu_index = random.randint(0, len(menu_data) - 1)  # 무작위로 인덱스 선택
        while random_menu_index in exclude_indices:
            random_menu_index = random.randint(0, len(menu_data) - 1)
        menu.append(menu_data[random_menu_index])
        index = random_menu_index
        exclude_indices.add(index)
    # print(exclude_indices)
    # print(menu)
    print(len(menu))
    return jsonify(menu, len(menu))

@app.route('/recommendation1', methods=['POST']) #5개중 3개 피드백해서 추천해주고 2개 무작위로
def get_recommendation1():
    content = request.json
    like_foods = content.get('likeFoods', [])
    dislike_foods = content.get('dislikeFoods', [])
    menu = []
    for _ in range(2):
        random_menu_index = random.randint(0, len(menu_data) - 1)  # 무작위로 인덱스 선택
        while random_menu_index in exclude_indices:
            random_menu_index = random.randint(0, len(menu_data) - 1)
        menu.append(menu_data[random_menu_index])
        index = random_menu_index
        exclude_indices.add(index)
    preference_vector = calculate_preference_vector(like_foods, dislike_foods)
    recommended_menus, recommended_indices = find_similar_menus(preference_vector,menu_data, exclude_indices,3)
    for i in recommended_menus:
        menu.append(i)
    for i in recommended_indices:
        exclude_indices.add(i)

    # print("recommendation1")
    # print("feedback된 메뉴개수", len(recommended_indices))
    # print("추천된 메뉴index", recommended_indices)
    # print("선택된 인덱스들",exclude_indices)

    # print("지금 줄 메뉴들")
    # for i in menu:
    #     print(i['name'])
    print(len(menu))

    return jsonify(menu, len(menu))


@app.route('/recommendation2', methods=['POST']) #5개중 4개 피드백해서 추천해주고 1개 무작위로
def get_recommendation2():
    content = request.json
    like_foods = content.get('likeFoods', [])
    dislike_foods = content.get('dislikeFoods', [])
    menu = []
    for _ in range(1):
        random_menu_index = random.randint(0, len(menu_data) - 1)  # 무작위로 인덱스 선택
        while random_menu_index in exclude_indices:
            random_menu_index = random.randint(0, len(menu_data) - 1)
        menu.append(menu_data[random_menu_index])
        index = random_menu_index
        exclude_indices.add(index)
    preference_vector = calculate_preference_vector(like_foods, dislike_foods)
    recommended_menus, recommended_indices = find_similar_menus(preference_vector,menu_data, exclude_indices,4)
    for i in recommended_menus:
        menu.append(i)
    for i in recommended_indices:
        exclude_indices.add(i)

    # print("recommendation2")
    # print("feedback된 메뉴개수", len(recommended_indices))
    # print("추천된 메뉴index", recommended_indices)
    # print("선택된 인덱스들",exclude_indices)

    # print("지금 줄 메뉴들")
    # for i in menu:
    #     print(i['name'])
    print(len(menu))

    return jsonify(menu, len(menu))

@app.route('/recommendation3', methods=['POST']) #5개중 5개 모두 피드백해서 추천해주기
def get_recommendation3():
    content = request.json
    like_foods = content.get('likeFoods', [])
    dislike_foods = content.get('dislikeFoods', [])
    menu = []
    preference_vector = calculate_preference_vector(like_foods, dislike_foods)
    recommended_menus, recommended_indices = find_similar_menus(preference_vector,menu_data, exclude_indices,5)
    for i in recommended_menus:
        menu.append(i)
    for i in recommended_indices:
        exclude_indices.add(i)
    
    # print("recommendation3")
    # print("feedback된 메뉴개수", len(recommended_indices))
    # print("추천된 메뉴index", recommended_indices)
    # print("선택된 인덱스들",exclude_indices)

    # print("지금 줄 메뉴들")
    # for i in menu:
    #     print(i['name'])
    print(len(menu))

    return jsonify(menu, len(menu))
    



@app.route('/images/<path:filename>', methods=['GET'])
def get_picture(filename):
    image_path = os.path.join(IMAGE_DIRECTORY, filename)
    if os.path.exists(image_path):
        return send_file(image_path)
    else:
        return jsonify({'error': 'Image not found'}), 404
# def get_picture(filename):
#     return (os.path.join(IMAGE_DIRECTORY, filename))
send_file
# 클라이언트에서 보낸 데이터 받는 엔드포인트
@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        content = request.json
        like_foods = content.get('likeFoods', [])
        dislike_foods = content.get('dislikeFoods', [])
        for food in like_foods:
            print("like:",food.get('name'))
        for food in dislike_foods:
            print("dislike:" ,food.get('name'))
        return jsonify({'status': 'success', 'message': 'Data received successfully'}), 200
    except Exception as e:
        print('Error:', e)
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/excludedListInit', methods=['POST'])
def excludedListInit():
    global exclude_indices
    exclude_indices = set()
    return jsonify({'status': 'success', 'message': 'Data received successfully'}), 200



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)



############################################################################################################

