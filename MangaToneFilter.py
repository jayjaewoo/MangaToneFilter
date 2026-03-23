import cv2
import numpy as np
import os


def create_manga_filter(img):
    # 주어진 이미지를 5단계 명암의 망가(Manga) 스타일로 변환합니다.
    h, w = img.shape[:2]

    # 1. 흑백 변환 및 강력한 노이즈 제거 (Bilateral Filter 활용)
    # 피부 등의 미세한 굴곡을 더 뭉개서 스크린톤이 깔끔하게 입혀지도록 필터 강도 조절 (75 -> 150)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 9, 150, 150)

    # 2. 외곽선 검출 및 강력한 팽창 (Thick Black Edges)
    edges = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 9)

    # 외곽선을 더 강조하기 위해 3x3 커널로 2회 팽창 (iterations=2)
    kernel = np.ones((3, 3), np.uint8)
    thick_edges_inv = cv2.dilate(edges, kernel, iterations=2)
    thick_edges = cv2.bitwise_not(thick_edges_inv)

    # 3. 5단계 명암 분할용 마스크 생성 및 노이즈 정리 (Median Filter 활용)
    # 단계 1: 0 ~ 40 (완전 검정)
    _, m1 = cv2.threshold(smooth, 40, 255, cv2.THRESH_BINARY_INV)
    mask1 = cv2.medianBlur(m1, 3)  # 파편 노이즈 제거

    # 단계 2: 41 ~ 94 (사선 빗금)
    _, m2_t = cv2.threshold(smooth, 94, 255, cv2.THRESH_BINARY_INV)
    mask2 = cv2.bitwise_xor(mask1, cv2.medianBlur(m2_t, 3))

    # 단계 3: 95 ~ 148 (굵은 점)
    _, m3_t = cv2.threshold(smooth, 148, 255, cv2.THRESH_BINARY_INV)
    mask3 = cv2.bitwise_xor(cv2.medianBlur(m2_t, 3), cv2.medianBlur(m3_t, 3))

    # 단계 4: 149 ~ 202 (얇은 점)
    _, m4_t = cv2.threshold(smooth, 202, 255, cv2.THRESH_BINARY_INV)
    mask4 = cv2.bitwise_xor(cv2.medianBlur(m3_t, 3), cv2.medianBlur(m4_t, 3))

    # 단계 5: 203 ~ 255 (완전 하양 - 배경)

    # 4. 스크린톤 패턴 직접 생성 (Numpy 배열 기반 타일링)
    # 패턴 1: 사선 빗금 (Hatching)
    hatch = np.ones((8, 8), dtype=np.uint8) * 255
    for i in range(8): hatch[i, i] = 0
    hatch_bg = np.tile(hatch, (h // 8 + 1, w // 8 + 1))[:h, :w]

    # 패턴 2: 굵은 점 (Dots)
    dots = np.ones((8, 8), dtype=np.uint8) * 255
    cv2.circle(dots, (4, 4), 2, 0, -1)
    dots_bg = np.tile(dots, (h // 8 + 1, w // 8 + 1))[:h, :w]

    # 패턴 3: 얇은 점 (Sparse Dots)
    sparse_dots = np.ones((8, 8), dtype=np.uint8) * 255
    sparse_dots[4, 4] = 0
    sparse_dots_bg = np.tile(sparse_dots, (h // 8 + 1, w // 8 + 1))[:h, :w]

    # 5. 패턴 마스킹 및 합성
    canvas = np.ones_like(gray) * 255

    canvas[mask1 == 255] = 0
    canvas[mask2 == 255] = hatch_bg[mask2 == 255]
    canvas[mask3 == 255] = dots_bg[mask3 == 255]
    canvas[mask4 == 255] = sparse_dots_bg[mask4 == 255]

    # 6. 최종 합성
    final_manga = cv2.bitwise_and(canvas, thick_edges)

    return final_manga


if __name__ == '__main__':
    # 이미지 불러오기 (test.jpg)
    test_img = cv2.imread('test.jpg')
    # test_img = cv2.imread('test2.jpg')

    if test_img is None:
        print("이미지를 불러올 수 없습니다. 파일 이름을 확인해주세요.")
        exit()

    # 필터 적용
    result = create_manga_filter(test_img)

    # 결과 저장
    cv2.imwrite('manga_result.jpg', result)
    # cv2.imwrite('manga_result2.jpg', result)

    # 화면 출력 및 크기 조절 (락 효과)
    display_img = result.copy()
    max_h, max_w = 800, 1200
    img_h, img_w = display_img.shape[:2]

    if img_h > max_h or img_w > max_w:
        ratio = min(max_h / img_h, max_w / img_w)
        display_img = cv2.resize(display_img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)

    cv2.imshow('Manga Renderer: Result', display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
