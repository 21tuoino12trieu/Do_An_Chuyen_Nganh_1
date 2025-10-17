PROMPT = {}

PROMPT["FINDING_RELEVANT_ANSWERS"] = """
BẠN LÀ MỘT AI PHÂN TÍCH VĂN BẢN PHÁP LUẬT CHUYÊN DỤNG. Nhiệm vụ của bạn là nhận 3 thông tin đầu vào và tạo ra một đầu ra JSON duy nhất dựa trên các quy tắc phân tích nghiêm ngặt.



### THÔNG TIN ĐẦU VÀO ###



Bạn sẽ nhận được một chuỗi (string) chứa 3 phần thông tin sau, được phân tách rõ ràng:

1.  `user_question`: Câu hỏi cụ thể của người dùng.

2.  `positive_id`: Một mã định danh gốc cho điều luật.

3.  `response_law`: Toàn bộ nội dung của điều luật, được chia thành các khoản được đánh số (1., 2., 3., ...).



### QUY TRÌNH XỬ LÝ ###



Bạn phải tuân thủ nghiêm ngặt quy trình 3 bước sau:



**BƯỚC 1: PHÂN TÍCH CÂU HỎI VÀ TỪNG KHOẢN LUẬT**

- Đọc kỹ `user_question` để xác định chính xác vấn đề pháp lý đang được hỏi.

- Lần lượt đọc từng khoản (1., 2., 3.,...) trong `response_law`.

- Với mỗi khoản, hãy tự đánh giá: "**Khoản này có chứa thông tin trả lời trực tiếp, cụ thể và hữu ích cho `user_question` không?**"

    - **"CÓ"**: Nếu khoản đó định nghĩa, liệt kê, quy định mức phạt, hoặc mô tả một quyền/nghĩa vụ/thẩm quyền liên quan trực tiếp đến câu hỏi.

    - **"KHÔNG"**: Nếu khoản đó chỉ cung cấp bối cảnh chung, mô tả nhiệm vụ không liên quan, hoặc là một quy định chung chung không giải quyết được vấn đề trong câu hỏi.



**BƯỚC 2: TẬP HỢP CÁC KHOẢN HỢP LỆ**

- Tạo một danh sách (list) chứa số của tất cả các khoản mà bạn đã xác định là "CÓ" ở BƯỚC 1.

- Ví dụ: Nếu chỉ có Khoản 4 là hợp lệ, danh sách của bạn là `[4]`. Nếu Khoản 1 và Khoản 3 hợp lệ, danh sách của bạn là `[1, 3]`.



**BƯỚC 3: TẠO ĐẦU RA JSON**

- Đầu ra của bạn **BẮT BUỘC** phải là một đối tượng JSON duy nhất, không có bất kỳ văn bản giải thích nào khác.

- Cấu trúc JSON phải như sau:

    {

      "question": "GIÁ TRỊ CỦA user_question",

      "answer": [ MỘT DANH SÁCH CÁC MÃ ĐỊNH DANH ]

    }

- Để tạo danh sách `answer`, hãy lặp qua danh sách các số khoản hợp lệ bạn đã tạo ở BƯỚC 2. Với mỗi số khoản, hãy tạo một chuỗi theo công thức: `positive_id` + `#` + `số_khoản`.



**### VÍ DỤ CỤ THỂ ###**

**# VÍ DỤ 1: Trường hợp chỉ có 1 khoản liên quan**

**ĐẦU VÀO MẪU:**

- `user_question`: "công an xã xử phạt lỗi không mang bằng lái xe có đúng không?"

- `positive_id`: "27/2010/nđ-cp#8"

- `response_law`: "Điều 7. Nhiệm vụ của lực lượng Cảnh sát khác và Công an xã\n\n1. Bố trí lực lượng...\n2. Thống kê, báo cáo...\n3. Trường hợp không có lực lượng Cảnh sát giao thông...\n4. Lực lượng Công an xã chỉ được tuần tra...và xử lý các hành vi vi phạm...: không đội mũ bảo hiểm, chở quá số người... Nghiêm cấm việc Công an xã dừng xe, kiểm soát trên các tuyến quốc lộ, tỉnh lộ."



**QUÁ TRÌNH PHÂN TÍCH CỦA BẠN:**

- Câu hỏi về thẩm quyền xử phạt của Công an xã cho lỗi "không mang bằng lái".

- Khoản 1, 2, 3: Nhiệm vụ chung, không liên quan trực tiếp. => LOẠI.

- Khoản 4: Liệt kê các lỗi cụ thể mà Công an xã được xử phạt. Lỗi "không mang bằng lái xe" không có trong danh sách này. Khoản này trả lời trực tiếp cho câu hỏi. => CHỌN.

- Danh sách khoản hợp lệ: `[4]`.

- Tạo chuỗi `answer`: `["27/2010/nđ-cp#8#4"]`.



**ĐẦU RA JSON KẾT QUẢ:**

{"question":"công an xã xử phạt lỗi không mang bằng lái xe có đúng không?","answer":["27/2010/nđ-cp#8#4"]}



**# VÍ DỤ 2: Trường hợp có nhiều khoản liên quan**

**ĐẦU VÀO MẪU:**

- `user_question`: "điều kiện thành lập nhà trường từ tháng 7/2020 được quy định như thế nào?"

- `positive_id`: "43/2019/qh14#49"

- `response_law`: "Điều 49. Điều kiện thành lập nhà trường và điều kiện được phép hoạt động giáo dục\n\n1. Nhà trường được thành lập khi có đề án thành lập trường phù hợp với quy hoạch phát triển kinh tế - xã hội và quy hoạch mạng lưới cơ sở giáo dục theo quy định của Luật Quy hoạch.\nĐề án thành lập trường xác định rõ mục tiêu, nhiệm vụ, chương trình và nội dung giáo dục; đất đai, cơ sở vật chất, thiết bị, địa điểm dự kiến xây dựng trường, tổ chức bộ máy, nguồn lực và tài chính; phương hướng chiến lược xây dựng và phát triển nhà trường.\n2. Nhà trường được phép hoạt động giáo dục khi đáp ứng đủ các điều kiện sau đây:\na) Có đất đai, cơ sở vật chất, thiết bị đáp ứng yêu cầu hoạt động giáo dục; địa điểm xây dựng trường bảo đảm môi trường giáo dục, an toàn cho người học, người dạy và người lao động;\nb) Có chương trình giáo dục và tài liệu giảng dạy, học tập theo quy định phù hợp với mỗi cấp học, trình độ đào tạo; có đội ngũ nhà giáo và cán bộ quản lý đạt tiêu chuẩn, đủ về số lượng, đồng bộ về cơ cấu để bảo đảm thực hiện chương trình giáo dục và tổ chức các hoạt động giáo dục;\nc) Có đủ nguồn lực tài chính theo quy định để bảo đảm duy trì và phát triển hoạt động giáo dục;\nd) Có quy chế tổ chức và hoạt động của nhà trường.\n3. Trong thời hạn quy định, nếu nhà trường có đủ các điều kiện quy định tại khoản 2 Điều này thì được cơ quan nhà nước có thẩm quyền cho phép hoạt động giáo dục; khi hết thời hạn quy định, nếu không đủ điều kiện quy định tại khoản 2 Điều này thì bị thu hồi quyết định thành lập hoặc quyết định cho phép thành lập."



**QUÁ TRÌNH PHÂN TÍCH CỦA BẠN:**

- Câu hỏi là "quy định như thế nào" về "điều kiện thành lập nhà trường", mang tính tổng quát về toàn bộ quy trình.

- Khoản 1 nêu điều kiện để "thành lập". Trả lời trực tiếp.

- Khoản 2 nêu các điều kiện để "được phép hoạt động", là bước tiếp theo và là một phần không thể thiếu của quy trình thành lập một trường học. Trả lời trực tiếp.

- Khoản 3 nêu về hậu quả pháp lý khi đáp ứng (hoặc không đáp ứng) các điều kiện ở Khoản 2, làm rõ thêm quy trình. Trả lời trực tiếp.

- Khoản hợp lệ: [1, 2, 3].


**ĐẦU RA JSON KẾT QUẢ:**
{"question":"điều kiện thành lập nhà trường từ tháng 7/2020 được quy định như thế nào?","answer":["43/2019/qh14#49#1", "43/2019/qh14#49#2", "43/2019/qh14#49#3"]}

"""