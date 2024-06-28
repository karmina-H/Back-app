import random
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk


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
def find_similar_menus(target_vector, menu_data, exclude_indices):
    similarities = []
    for index, menu in enumerate(menu_data):
        if index not in exclude_indices:  # 이미 추천된 메뉴는 제외합니다.
            menu_vector = get_menu_vector(menu)  # 메뉴를 벡터로 변환
            valid_indices = ~np.isnan(target_vector) & ~np.isnan(menu_vector)  # NaN이 아닌 인덱스 찾기
            if np.any(valid_indices):
                similarity = cosine_similarity(target_vector[valid_indices].reshape(1, -1), menu_vector[valid_indices].reshape(1, -1))[0][0]  # 코사인 유사도 계산
                similarities.append((similarity, index))
    similarities.sort(reverse=True, key=lambda x: x[0])  # 유사도 기준으로 정렬
    return [menu_data[i] for _, i in similarities[:3]], [i for _, i in similarities[:3]]  # 유사한 상위 3개 메뉴 반환

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

# GUI 업데이트 함수
# 추천 메뉴를 GUI에 업데이트합니다.
def update_gui(menu, recommendation_count):
    menu_label.config(text=f"추천 번호 {recommendation_count}: 메뉴 - {menu['name']}")
    image_path = menu['image']
    try:
        img = Image.open(image_path)
        max_width, max_height = 600, 400
        img_ratio = img.width / img.height
        if img.width > max_width or img.height > max_height:
            if img_ratio > 1:
                new_width = max_width
                new_height = int(max_width / img_ratio)
            else:
                new_height = max_height
                new_width = int(max_height * img_ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)  # 이미지 크기 조정
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
    except Exception as e:
        image_label.config(text="이미지를 찾을 수 없습니다.", image='')
        print(e)

# 선호 벡터 업데이트 및 GUI 표시 함수
# 선호 벡터를 계산하고 GUI에 업데이트합니다.
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

    preference_vector_label.config(text=preference_vector_table, justify=tk.LEFT, font=("Courier", 10))

# 피드백 처리 함수
# 사용자의 피드백(좋아요, 싫어요 등)을 처리하고 다음 추천을 준비합니다.
def process_feedback(feedback):
    global recommendation_count, menu, index, exclude_indices, liked_menus, disliked_menus, preference_vector, phase_count

    if feedback == 0:  # 최종 선택 버튼 클릭
        messagebox.showinfo("최종 선택", f"최종 선택한 메뉴: {menu['name']}")
        root.quit()
    elif feedback == 5:  # 좋아요 목록 보기 버튼 클릭
        show_liked_menus()
        return
    else:
        if feedback == 3:  # 좋아요 버튼 클릭
            liked_menus.append(menu)
        elif feedback == 1:  # 싫어요 버튼 클릭
            disliked_menus.append(menu)

        exclude_indices.add(index)  # 현재 메뉴를 제외 목록에 추가
        recommendation_count += 1

        if recommendation_count in [6, 9, 13, 16, 18, 20, 21]:  # 유사도 기반 추천 단계
            preference_vector = calculate_preference_vector(liked_menus, disliked_menus)
            recommended_menus, recommended_indices = recommend_menu(preference_vector, exclude_indices)
            if recommended_menus:
                menu = recommended_menus[0]
                index = recommended_indices[0]
                update_gui(menu, recommendation_count)
            else:
                messagebox.showinfo("알림", "추천할 메뉴가 없습니다.")
                root.quit()
        else:  # 무작위 추천 단계
            get_next_menu()

        update_preference_vector()

# 다음 메뉴 가져오기 함수(무작위)
# 무작위로 다음 메뉴를 선택하여 추천합니다.

#무작위로 메뉴보여주는것.
def get_next_menu():
    global recommendation_count, menu, index

    random_menu_index = random.randint(0, len(menu_data) - 1)  # 무작위로 인덱스 선택
    while random_menu_index in exclude_indices:
        random_menu_index = random.randint(0, len(menu_data) - 1)
    menu = menu_data[random_menu_index]
    index = random_menu_index

    update_gui(menu, recommendation_count)

# 초기 설정
recommendation_count = 1  # 추천 횟수 초기화
exclude_indices = set()  # 제외할 메뉴 인덱스 목록 초기화
liked_menus = []  # 좋아요 한 메뉴 목록 초기화
disliked_menus = []  # 싫어요 한 메뉴 목록 초기화
preference_vector = np.zeros(len(get_menu_vector(menu_data[0])[0]))  # 선호 벡터 초기화
phase_count = 0  # 추천 단계 초기화

# GUI 설정
root = tk.Tk()
root.title("메뉴 추천 시스템")

left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, padx=10)

right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT, padx=10)

menu_label = tk.Label(left_frame, text="메뉴 추천 시스템", font=("Arial", 16))
menu_label.pack()

image_label = tk.Label(left_frame)
image_label.pack()

preference_vector_label = tk.Label(right_frame, text="선호 벡터:")
preference_vector_label.pack()

button_frame = tk.Frame(left_frame)
button_frame.pack()

# dislike_button = tk.Button(button_frame, text="싫어요", command=lambda: process_feedback(1))
# dislike_button.grid(row=0, column=0, padx=15)

# unknown_button = tk.Button(button_frame, text="몰라요", command=lambda: process_feedback(2))
# unknown_button.grid(row=0, column=1, padx=15)

# like_button = tk.Button(button_frame, text="좋아요", command=lambda: process_feedback(3))
# like_button.grid(row=0, column=2, padx=15)

# final_select_button = tk.Button(button_frame, text="최종선택", command=lambda: process_feedback(0))
# final_select_button.grid(row=0, column=3, padx=15)

# show_liked_button = tk.Button(button_frame, text="좋아요 목록", command=lambda: process_feedback(5))
# show_liked_button.grid(row=0, column=4, padx=15)

get_next_menu()

root.mainloop()
