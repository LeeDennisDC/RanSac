import cv2
import numpy as np

# 1. 이미지 로드
img_l = cv2.imread('Left.jpg')
img_r = cv2.imread('Right.jpg')

if img_l is None or img_r is None:
    print("[오류] Left.jpg 또는 Right.jpg 파일을 찾을 수 없습니다.")
    exit()

gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

# ==========================================================
# [알고리즘 1] ORB 특징점 추출 및 기술자 계산 (Lecture 8 수록)
# ==========================================================
orb = cv2.ORB_create(nfeatures=4000)
kp_l, desc_l = orb.detectAndCompute(gray_l, None)
kp_r, desc_r = orb.detectAndCompute(gray_r, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(desc_l, desc_r)
matches = sorted(matches, key=lambda x: x.distance)

# ==========================================================
# [알고리즘 2] RANSAC 기반 Homography 계산 (Lecture 8~9 수록)
# ==========================================================
pts_l = np.float32([kp_l[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts_r = np.float32([kp_r[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
H, mask = cv2.findHomography(pts_l, pts_r, cv2.RANSAC, 5.0)

# 매칭 결과 시각화 저장
img_matches = cv2.drawMatches(img_l, kp_l, img_r, kp_r, matches[:50], None, 
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('step1_ransac_matches.jpg', img_matches)

# ==========================================================
# [알고리즘 3] 가상 도화지 투영 및 겹침 영역 분할 합성 (Lecture 9 수록)
# ==========================================================
h_l, w_l = img_l.shape[:2]
h_r, w_r = img_r.shape[:2]

pts_corners_l = np.float32([[0, 0], [0, h_l], [w_l, h_l], [w_l, 0]]).reshape(-1, 1, 2)
pts_transformed_l = cv2.perspectiveTransform(pts_corners_l, H)
pts_corners_r = np.float32([[0, 0], [0, h_r], [w_r, h_r], [w_r, 0]]).reshape(-1, 1, 2)
all_pts = np.concatenate((pts_transformed_l, pts_corners_r), axis=0)

[x_min, y_min] = np.int32(all_pts.min(axis=0).flatten())
[x_max, y_max] = np.int32(all_pts.max(axis=0).flatten())

T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
H_final = T.dot(H)

out_w, out_h = x_max - x_min, y_max - y_min

# 각각 가상 도화지에 사진 배치
warped_l = cv2.warpPerspective(img_l, H_final, (out_w, out_h))
warped_r = np.zeros_like(warped_l)
warped_r[-y_min:-y_min+h_r, -x_min:-x_min+w_r] = img_r

# 단일 채널(그레이스케일) 마스크 생성
mask_l = (cv2.cvtColor(warped_l, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255
mask_r = (cv2.cvtColor(warped_r, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255

# ★ 수정 포인트: cv2.bitwiseAnd -> cv2.bitwise_and 오타 수정 완료
overlap_mask = cv2.bitwise_and(mask_l, mask_r)

contours, _ = cv2.findContours(overlap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
result_stitched = warped_r.copy()

if contours:
    # 겹치는 구간의 중심을 계산하여 절삭선(Seam) 설정
    x, y, w, h = cv2.boundingRect(contours[0])
    seam_x = x + w // 2

    # 절삭 마스크 생성 (왼쪽은 1, 오른쪽은 0)
    blend_mask = np.zeros((out_h, out_w), dtype=np.float32)
    blend_mask[:, :seam_x] = 1.0

    # 경계선을 부드럽게 융합하기 위한 가우시안 블러 적용
    blend_mask = cv2.GaussianBlur(blend_mask, (31, 31), 0)
    blend_mask = np.repeat(blend_mask[:, :, np.newaxis], 3, axis=2)

    # 절삭면 기준으로 가중치 합성 처리
    result_stitched = (warped_l * blend_mask + warped_r * (1.0 - blend_mask)).astype(np.uint8)
else:
    non_zero_l = (warped_l > 0)
    result_stitched[non_zero_l] = warped_l[non_zero_l]

# 최종 파노라마 저장
cv2.imwrite('step2_stitched_panorama.jpg', result_stitched)

# ==========================================================
# [알고리즘 4] 템플릿 매칭을 통한 최종 객체 탐지 (Lecture 9 수록)
# ==========================================================
# 자동 크롭 타겟 지정 및 검출
h_center, w_center = img_r.shape[0] // 2, img_r.shape[1] // 2
template = img_r[h_center-40:h_center+60, w_center-100:w_center]
cv2.imwrite('template_auto_cropped.jpg', template)

th, tw = template.shape[:2]
res = cv2.matchTemplate(result_stitched, template, cv2.TM_CCOEFF_NORMED)
_, _, _, max_loc = cv2.minMaxLoc(res)

top_left = max_loc
bottom_right = (top_left[0] + tw, top_left[1] + th)

final_output = result_stitched.copy()
cv2.rectangle(final_output, top_left, bottom_right, (0, 0, 255), 4)
cv2.putText(final_output, "Target Found", (top_left[0], top_left[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

cv2.imwrite('step3_final_detection.jpg', final_output)
