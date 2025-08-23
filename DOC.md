# ASSIGNMENT 1 – PHẦN LÝ THUYẾT (1.1 → 1.6)

**Môn:** Intelligence System Development – **Học kỳ I/2025**
**Sinh viên:** \[Điền tên] – **Lớp/Nhóm:** \[Điền lớp/nhóm]

> Lưu ý: Nội dung tóm lược và phân tích dựa trên kiến thức nền tảng về Hệ thống thông minh/AI và các tài liệu được gợi ý trong đề bài (Forbes, Apple Siri, Simplilearn, Edureka, v.v.). Với mục 1.5, vì không truy cập trực tiếp Figure 7 của bài báo arXiv (2009.09083) trong điều kiện offline, em trình bày **bản mô phỏng/diễn giải lại** theo cấu trúc phổ biến: lĩnh vực ứng dụng ↔ kỹ thuật AI, để phù hợp yêu cầu trình bày ứng dụng qua sơ đồ/ma trận.

---

## 1.1. Investigate, discover and write (3 trang)

### 1) 10 ví dụ tiêu biểu về AI trong thực tế

1. **Trợ lý ảo & nhận dạng giọng nói**: Siri, Google Assistant nhận lệnh thoại, chuyển giọng nói thành văn bản, điều khiển thiết bị.
2. **Hệ khuyến nghị**: Netflix/YouTube/Spotify/Amazon đề xuất nội dung/sản phẩm dựa trên lịch sử tương tác (CF, deep learning).
3. **Phát hiện gian lận**: Ngân hàng và cổng thanh toán phát hiện giao dịch bất thường bằng học máy giám sát/không giám sát.
4. **Thị giác máy tính trong y tế**: Phát hiện khối u trên ảnh X‑quang/CT/MRI (CNN).
5. **Xe tự hành & ADAS**: Nhận biết làn đường, vật cản, người đi bộ; lập kế hoạch đường đi (CV + RL).
6. **Chatbot chăm sóc khách hàng**: NLP + LLM trả lời tự động 24/7, định tuyến case.
7. **Dịch máy & tóm tắt văn bản**: Transformer/LLM dịch nhanh đa ngôn ngữ, tóm tắt báo cáo dài.
8. **Nhà thông minh/IoT**: Điều chỉnh nhiệt độ/chiếu sáng tối ưu theo thói quen người dùng (học tăng cường + dự báo chuỗi thời gian).
9. **Bảo mật & giám sát**: Phát hiện xâm nhập mạng (IDS) và bất thường log; nhận diện khuôn mặt trong kiểm soát ra vào.
10. **Bảo trì dự đoán trong công nghiệp**: Phân tích rung/âm/điện năng để dự đoán hỏng hóc thiết bị, giảm downtime.

### 2) Chuỗi giá trị dữ liệu/AI (Data → Value)

- **Thu thập & gắn nhãn** (sensor, log, clickstream) → **Làm sạch/tiền xử lý** (lọc nhiễu, chuẩn hóa) → **Đặc trưng** (feature engineering, embedding) → **Huấn luyện/điều chỉnh** (supervised/unsupervised/RL) → **Triển khai** (API, edge) → **Giám sát** (drift, fairness, an toàn) → **Vòng lặp cải tiến**.
- **Thách thức**: chất lượng dữ liệu, thiên lệch (bias), quyền riêng tư, khả năng giải thích (XAI), chi phí hạ tầng, an toàn khi triển khai trong môi trường vật lý (robot/xe).

### 3) Chỉ số & thực hành tốt

- **Chỉ số mô hình**: Accuracy/F1/AUC (phân loại), RMSE/MAE (hồi quy), mAP (CV), BLEU/ROUGE (NLP).
- **MLOps**: versioning dữ liệu/mô hình, pipeline CI/CD, monitoring drift, kiểm thử an toàn (adversarial).
- **Đạo đức & tuân thủ**: minh bạch, công bằng, bảo vệ dữ liệu cá nhân.

---

## 1.2. Intelligent System là gì? Định nghĩa ấn tượng nhất & ví dụ (3 trang)

### 1) Tổng hợp định nghĩa

- **Định nghĩa khái quát**: _Hệ thống thông minh_ là **hệ thống phần mềm/ phần cứng** có khả năng **nhận thức (perception)**, **lý luận/quyết định (reasoning/decision‑making)** và **học hỏi (learning)** từ dữ liệu & tương tác môi trường để **đạt mục tiêu** với hiệu quả vượt trội so với thủ tục cố định.
- **Các góc nhìn phổ biến**:

  - **Theo chức năng**: Cảm nhận → Hiểu → Lập kế hoạch → Hành động → Học (vòng kín).
  - **Theo tác tử (agent)**: tác tử thông minh tối đa hóa **hàm lợi ích** dựa trên trạng thái/quan sát.
  - **Theo kỹ thuật**: dùng **kiến thức + suy diễn** (KBS) / **học máy** / **lai**.

### 2) Định nghĩa ấn tượng nhất (lý do)

> **“Intelligent system là tác tử tự trị có khả năng cảm nhận môi trường, xây dựng mô hình nội tại, học hỏi để cập nhật mô hình, và hành động nhằm tối đa hóa mục tiêu trong điều kiện không chắc chắn.”** > **Lý do**: ngắn gọn, bao quát 4 đặc tính cốt lõi (nhận thức–mô hình hóa–học–hành động), nhấn mạnh **bất định** (uncertainty) – tình huống thật.

### 3) Ví dụ hệ thống thông minh

- **Phần mềm**: Hệ khuyến nghị, chatbot đa ngôn ngữ, hệ hỗ trợ chẩn đoán y khoa (CDSS).
- **Cyber‑Physical**: robot di động kho hàng (SLAM + RL), UAV giám sát rừng (CV + path planning).
- **Edge/IoT**: camera thông minh phát hiện ngã đột quỵ, đồng hồ sức khỏe dự báo rối loạn nhịp.
- **Doanh nghiệp**: tối ưu chuỗi cung ứng (forecast + MILP), phát hiện gian lận giao dịch (anomaly/graph ML).

---

## 1.3. Ứng dụng hệ thống thông minh: lĩnh vực & kỹ thuật AI (3 trang)

### 1) Lĩnh vực chính

- **Y tế** (chẩn đoán hình ảnh, phân loại bệnh, dự báo tái nhập viện, trợ lý bệnh án).
- **Tài chính** (phát hiện gian lận, xếp hạng tín dụng, định giá rủi ro, giao dịch thuật toán).
- **Sản xuất/IIoT** (bảo trì dự đoán, thị giác kiểm lỗi, tối ưu lịch sản xuất).
- **Giao thông** (ITS, định tuyến, quản lý tín hiệu, ADAS/xe tự hành).
- **Năng lượng** (dự báo phụ tải/giá, tối ưu vận hành lưới, phát hiện sự cố).
- **Bán lẻ/Marketing** (khuyến nghị, phân khúc khách hàng, định giá động).
- **An ninh mạng** (IDS, phát hiện botnet, phân tích log).
- **Giáo dục** (tutor thông minh, đánh giá thích ứng, proctoring).
- **Nông nghiệp** (dự báo năng suất, nhận diện sâu bệnh, drone phun thuốc chính xác).
- **Chính phủ/Smart city** (quan trắc môi trường, phân tích lưu lượng, dịch vụ công số).

### 2) Kỹ thuật AI tiêu biểu theo bài toán

- **Thị giác máy tính (CV)**: CNN/ViT cho nhận dạng, phát hiện, phân đoạn.
- **Xử lý ngôn ngữ (NLP/LLM)**: Transformer, fine‑tuning/PEFT, RAG cho hỏi‑đáp, tóm tắt.
- **Học có giám sát**: hồi quy, cây quyết định, SVM, gradient boosting cho dự báo/chấm điểm.
- **Học không giám sát**: clustering (K‑means, DBSCAN), PCA/UMAP, phát hiện bất thường.
- **Học tăng cường (RL)**: kiểm soát/điều khiển, tối ưu chính sách.
- **Tri thức & suy diễn**: ontology, knowledge graph, logic.
- **Tối ưu/OR**: quy hoạch tuyến tính, ràng buộc, meta‑heuristics (GA/PSO) cho lịch/định tuyến.
- **Chuỗi thời gian**: ARIMA/Prophet/RNN/Transformer thời gian cho dự báo.
- **XAI & an toàn**: SHAP/LIME, kiểm soát drift, fairness.

---

## 1.4. Các loại hệ thống thông minh (3 trang)

### A) Phân loại theo **mức năng lực** (thường thấy trong Simplilearn/Edureka)

1. **Reactive Machines**: chỉ phản ứng hiện tại, không nhớ quá khứ (ví dụ: Deep Blue).
2. **Limited Memory**: dùng dữ liệu gần đây để quyết định (đa số hệ thống ML hiện nay).
3. **Theory of Mind** _(mục tiêu nghiên cứu)_: mô hình hóa trạng thái tinh thần người khác.
4. **Self‑aware** _(giả thuyết xa)_: có ý thức về bản thân.

### B) Phân loại theo **phạm vi trí tuệ**

- **ANI (Narrow/Weak AI)**: làm tốt một nhiệm vụ hẹp (hệ khuyến nghị, phân loại ảnh).
- **AGI (General AI)**: năng lực rộng như người (đang nghiên cứu).
- **ASI (Superintelligence)**: vượt xa con người (mang tính giả thuyết).

### C) Phân loại theo **kiến trúc/chiến lược tác tử**

- **Reactive** vs **Deliberative (lập kế hoạch)** vs **Hybrid**.
- **Rule‑based/KBS** (hệ luật + suy diễn) vs **Learning‑based** (ML/DL) vs **Neuro‑symbolic** (lai).
- **Centralized** vs **Multi‑Agent** (MAS: phối hợp/đấu giá/đồng thuận).
- **On‑cloud** vs **Edge/On‑device**.

**Nhận xét**: Hệ thống thực tế thường **lai**: perception (DL) + planning/optimization (OR) + rule/constraint để bảo đảm an toàn/tuân thủ.

---

## 1.5. Ứng dụng hệ thống thông minh **qua Figure 7** của arXiv:2009.09083 (3 trang)

> Do không xem được trực tiếp Figure 7 trong điều kiện offline, em trình bày **ma trận ứng dụng–kỹ thuật** mô phỏng tinh thần tổng quan: mỗi ô gợi ý kỹ thuật AI phù hợp cho một lĩnh vực. Khi có hình gốc, em sẽ thay thế/đối chiếu lại cho khớp chú thích.

### 1) Ma trận mô phỏng lĩnh vực ↔ kỹ thuật

| Lĩnh vực    | Nhận thức (CV/ASR)                  | NLP/LLM                       | Dự báo (TS)          | Tối ưu/OR              | RL/Control                        | KBS/Graph                |
| ----------- | ----------------------------------- | ----------------------------- | -------------------- | ---------------------- | --------------------------------- | ------------------------ |
| Y tế        | Phân đoạn ảnh, phát hiện tổn thương | Tóm tắt hồ sơ, trích thực thể | Dự báo tái nhập viện | Lập lịch phòng mổ      | Chính sách dùng thuốc cá nhân hóa | Ontology y khoa          |
| Tài chính   | OCR chứng từ, phát hiện giả mạo     | Phân tích văn bản tin tức     | Dự báo rủi ro/giá    | Tối ưu danh mục        | RL cho giao dịch                  | Đồ thị quan hệ gian lận  |
| Sản xuất    | Kiểm lỗi thị giác                   | QA bằng ngôn ngữ tự nhiên     | Bảo trì dự đoán      | Lịch sản xuất          | Tối ưu tham số                    | Tri thức quy trình       |
| Giao thông  | Nhận diện vật thể                   | Giao tiếp V2X ngôn ngữ        | Dự báo lưu lượng     | Định tuyến đa mục tiêu | Điều khiển tín hiệu               | Bản đồ tri thức lộ trình |
| Năng lượng  | Nhận diện sự cố                     | Tóm tắt sự cố                 | Dự báo phụ tải/giá   | Tối ưu vận hành        | Điều khiển lưới                   | Tri thức thiết bị        |
| Bán lẻ      | Nhận diện sản phẩm                  | Chatbot                       | Dự báo nhu cầu       | Tối ưu tồn kho         | Định giá động                     | KG sản phẩm              |
| Nông nghiệp | Phát hiện sâu bệnh                  | Nhật ký nông vụ               | Dự báo mùa vụ        | Tối ưu tưới/bón        | Robot nông nghiệp                 | Tri thức cây trồng       |

### 2) Quy trình triển khai tham chiếu (theo Fig. tổng quan)

- **Tầng cảm biến/dữ liệu** → **Xử lý & học** (DL/ML) → **Lập kế hoạch/tối ưu** → **Hành động** (robotics/actuator) → **Giám sát an toàn/XAI**.
- **Chỉ số & quản trị**: chất lượng dữ liệu, audit, bảo mật, tuân thủ.

> _Khi có Figure 7 gốc, em sẽ cập nhật: tên lớp/nhóm kỹ thuật, chú thích, và ví dụ minh họa đúng như hình._

---

## 1.6. Numpy, Pandas, Matplotlib, Scikit‑learn – tính năng, mục đích & ví dụ

### 1) NumPy

- **Mục đích**: mảng ND hiệu năng cao, phép toán vector hóa, tuyến tính đại số.
- **Tính năng**: `ndarray`, broadcasting, ufuncs, random, linalg.
- **Ví dụ**:

```python
import numpy as np
x = np.array([1,2,3], dtype=float)
y = np.array([4,5,6], dtype=float)
cos_sim = np.dot(x,y) / (np.linalg.norm(x) * np.linalg.norm(y))
```

### 2) Pandas

- **Mục đích**: thao tác dữ liệu dạng bảng (DataFrame), làm sạch, tổng hợp.
- **Tính năng**: `read_csv`, indexing, `groupby`, `merge`, xử lý thiếu/chuỗi thời gian.
- **Ví dụ**:

```python
import pandas as pd
df = pd.DataFrame({"student":["Ann","Joe"], "math":[8.5,7.0], "eng":[7.5,8.0]})
df["avg"] = df[["math","eng"]].mean(axis=1)
by = df.groupby("student")["avg"].mean().reset_index()
```

### 3) Matplotlib

- **Mục đích**: vẽ biểu đồ 2D cơ bản; tùy biến cao.
- **Tính năng**: line/bar/scatter/hist, chú thích, style.
- **Ví dụ**:

```python
import matplotlib.pyplot as plt
subjects = ["Math","Phys","Chem"]
marks = [8.0, 7.5, 8.8]
plt.figure(figsize=(5,3))
plt.bar(subjects, marks)
plt.title("Marks by Subject"); plt.ylim(0,10)
plt.show()
```

### 4) Scikit‑learn

- **Mục đích**: thư viện ML cổ điển (supervised/unsupervised), pipeline, đánh giá.
- **Thành phần**: `LinearRegression`, `LogisticRegression`, SVM, tree/ensemble (RF, GBM), KMeans, PCA, `train_test_split`, `Pipeline`, `GridSearchCV`.
- **Ví dụ** (hồi quy tuyến tính):

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
X = np.array([[50],[60],[70],[80],[90]], dtype=float)
y = np.array([2.5,3.0,3.8,4.5,5.4])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(Xtr, ytr)
rmse = mean_squared_error(yte, model.predict(Xte), squared=False)
```

### 5) Gợi ý chuẩn hóa trình bày & tái sử dụng mã

- Tách **hàm xử lý dữ liệu**, **hàm huấn luyện**, **hàm vẽ**; dùng `requirements.txt`/`pipenv` để cố định phiên bản.
- Ghi chú **seed** để tái lập (reproducibility).
- Khi trực quan: ghi rõ **đơn vị**, **trú thích (legend)**, **nguồn dữ liệu**.

---

## Kết luận ngắn

Các hệ thống thông minh kết hợp **nhận thức–học–lý luận–hành động**; triển khai thực tế thường **lai** giữa DL/ML, tối ưu và tri thức. Bộ công cụ **NumPy/Pandas/Matplotlib/Scikit‑learn** là nền tảng để hiện thực hóa pipeline từ xử lý dữ liệu đến mô hình và trực quan hóa. Với mục 1.5, em sẽ cập nhật lại khi có Figure 7 gốc để nội dung trùng khớp 100% theo sơ đồ của bài báo.
