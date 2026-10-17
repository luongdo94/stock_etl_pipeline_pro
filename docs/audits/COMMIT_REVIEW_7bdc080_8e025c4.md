# Review hai commit: tối ưu screener/valuation và SEC RAG/assistant

Ngày ghi nhận: 2026-10-17.

## Phạm vi và kết luận

Hai commit được review riêng theo thay đổi so với parent của từng commit:

- `7bdc080bbd142106ab2bbb9ee2a972cac4e78f07`: partition dữ liệu theo ticker trong screener và memo dữ liệu cơ sở trong mỗi lần định giá.
- `8e025c4fd6e896884970aeceb86aa65d3b19d957`: SEC filings, trích xuất finance-lease principal bằng LLM, RAG và research assistant.

HEAD khi kiểm tra là `8e025c4fd6e896884970aeceb86aa65d3b19d957`. Có **8 phát hiện cần sửa trong commit thứ hai**. Chưa thấy lỗi hồi quy được xác nhận trong phần partition screener và memo valuation của commit đầu qua kiểm tra code; đây không phải kết luận rằng commit đó đã được kiểm chứng đầy đủ bằng thực thi.

Đánh giá được chia thành hai nhóm: kỹ thuật/bảo mật và logic tài chính/hành vi. P1 là vấn đề cần xử lý trước khi tin cậy tính năng cho dữ liệu nhạy cảm hoặc quyết định đầu tư; P2 là lỗi đúng đắn hoặc độ tin cậy cần sửa tiếp theo.

## Giới hạn bằng chứng

- Các phát hiện dưới đây dựa trên truy vết implementation và đối chiếu fixtures/tests hiện có, không phải regression tests đã chạy thành công.
- Python launcher tại `%TEMP%\hqv\Scripts\python.exe` báo `Access is denied` với Python313 nền, nên chưa chạy lại pytest hoặc các testcase tái hiện.
- `git diff --check 7bdc080^ 8e025c4` không báo lỗi whitespace. Kết quả này không kiểm chứng logic tài chính hoặc hành vi ứng dụng.
- Không gọi SEC/Cohere, không thực hiện giao dịch, không sửa production code hay dữ liệu lưu trữ trong quá trình review.
- Các test nêu trong từng phát hiện là **test đề xuất**, chưa được thêm hoặc chạy bởi reviewer.

## I. Kỹ thuật và bảo mật — Standards

### REV-01 — [P1] Lịch sử chat có thể lộ giữa hai tài khoản dùng cùng phiên trình duyệt

**Phân loại:** lỗi cô lập dữ liệu người dùng; độ tin cậy cao qua truy vết code, chưa tái hiện trên trình duyệt.

**Vị trí:** [ui/assistant.py:113](../../ui/assistant.py#L113), `render()`; đường logout tại [services/auth.py:320](../../services/auth.py#L320).

**Hành vi và bằng chứng:** mọi hội thoại được lưu vào cùng khóa `assistant_history` trong `st.session_state`, không gắn với `user_id`. Kết quả lưu cả tài liệu trả về từ `my_portfolio`. Logout chỉ xóa trạng thái xác thực, không xóa history. Vòng render tiếp tục hiển thị các lượt chat cũ; lời gọi tại `ui/assistant.py:124` còn gửi phần câu hỏi/câu trả lời cũ làm context cho Cohere.

**Tác động:** A hỏi về danh mục, logout rồi B đăng nhập trong cùng Streamlit session thì B có thể xem câu trả lời và tài liệu danh mục của A. Câu trả lời cũ của A cũng có thể được gửi lại cho model khi B hỏi tiếp. Phát hiện này nói về việc đổi tài khoản trong cùng session, không khẳng định rò rỉ giữa mọi session độc lập.

**Đề xuất sửa:** tách hội thoại theo authenticated user ID và xóa trạng thái nhạy cảm khi đổi danh tính/logout.

**Test nghiệm thu:** A → hỏi danh mục → logout → B đăng nhập; không có nội dung/tài liệu của A trong giao diện hoặc messages gửi cho model của B.

### REV-02 — [P2] Cache Decision thiếu các đầu vào quyết định kết quả

**Phân loại:** lỗi cache invalidation; độ tin cậy cao qua truy vết code, chưa thực thi testcase.

**Vị trí:** [ui/assistant.py:45](../../ui/assistant.py#L45), `_toolbox.deep_decision()`; đầu vào ngày/giá tại [views/stock_analysis/context.py:70](../../views/stock_analysis/context.py#L70).

**Hành vi và bằng chứng:** khóa cache chỉ chứa ticker, số dòng `prices_full` và ngày cuối của toàn bộ lịch sử. `context.build()` còn phụ thuộc giá đã lọc theo horizon, fundamentals, macro, cấu hình, earnings calendar và Track Record. Các thay đổi này không nhất thiết làm khóa cache đổi. Tool trả lời lại ghép kết quả cache với giá screener hiện tại và nguồn ghi ngày tính hiện tại.

**Tác động:** trợ lý có thể tiếp tục cung cấp stance/valuation cũ hoặc trộn dữ liệu khác ngày, dù người dùng đã đổi Custom Range hoặc cập nhật dữ liệu/cấu hình.

**Đề xuất sửa:** bỏ cache session bổ sung hoặc dùng fingerprint đầy đủ của dữ liệu/model, ngày đánh giá và khoảng ngày có hiệu lực.

**Test nghiệm thu:** hỏi một mã với ngày kết thúc lịch sử; đổi Custom Range sang hiện tại nhưng giữ nguyên `prices_full`; phải rebuild Decision. Thêm trường hợp sửa fundamentals/config mà số dòng giá và ngày cuối không đổi.

### REV-03 — [P2] Thiếu giá vị thế nhưng vẫn công bố NAV và rủi ro toàn danh mục

**Phân loại:** lỗi tính toán/báo cáo danh mục; độ tin cậy cao qua code và test hiện có, chưa thực thi.

**Vị trí:** [services/assistant_tools.py:284](../../services/assistant_tools.py#L284), `tool_my_portfolio()`.

**Hành vi và bằng chứng:** vị thế không có giá bị bỏ qua, rồi tổng giá trị phần định giá được cộng cash vẫn được đặt tên `nav_eur`. Tỷ trọng, stress loss và kiểm tra mandate dùng mẫu số này. Kiểm tra giá chỉ xét `None`, không chặn `NaN`. Test danh mục mới còn chấp nhận holding thiếu giá `ZZZ` trong khi vẫn kiểm tra một NAV hoàn chỉnh.

**Tác động:** có thể tạo vi phạm concentration giả, bỏ sót exposure/stress loss hoặc trình bày tỷ trọng không phản ánh toàn danh mục. Danh sách `unpriced_holdings` không làm NAV và các tỷ lệ đó trở thành số đầy đủ.

**Đề xuất sửa:** khi thiếu giá hữu hạn, hợp lệ của một vị thế khác zero, chỉ công bố subtotal phần định giá được; NAV toàn danh mục và các chỉ số phụ thuộc NAV phải là unknown hoặc được ghi rõ là phân tích phần danh mục đã định giá.

**Test nghiệm thu:** holding thiếu giá/NaN không được tạo NAV hoàn chỉnh, tỷ trọng toàn NAV hoặc kết luận rủi ro toàn danh mục như thể dữ liệu đã đủ.

## II. Logic tài chính và hành vi — Spec

### REV-04 — [P1] Bộ kiểm tra chấp nhận sai đơn vị tới một triệu lần

**Phân loại:** lỗi validation dữ liệu tài chính; độ tin cậy cao qua đường điều kiện, chưa thực thi testcase.

**Vị trí:** [core/filing_facts.py:20](../../core/filing_facts.py#L20), `_UNIT_EVIDENCE`; `unit_evidenced()` và [validate_lease_extraction():149](../../core/filing_facts.py#L149).

**Hành vi và bằng chứng:** `"units": r"."` coi mọi passage không rỗng là bằng chứng cho đơn vị nguyên. Với fixture `_amount()` có passage ghi rõ “in millions”, chỉ đổi `unit` thành `"units"` vẫn đáp ứng các kiểm tra hiện tại. `1,234` được nhân với 1 thay vì một triệu. Chặn số vượt total debt không phát hiện được việc ghi thiếu số tiền này.

**Tác động:** một khoản 1.234 tỷ USD có thể trở thành 1.234 USD khi ghi vào cash-flow deduction, làm tăng FCFE và định giá vì khấu trừ quá ít.

**Đề xuất sửa:** lấy đơn vị từ tiêu đề bảng hoặc bằng chứng gắn với dòng số liệu; yêu cầu đơn vị model trả về khớp bằng chứng. Từ chối đơn vị chưa rõ hoặc mâu thuẫn.

**Test nghiệm thu:** fixture có “in millions” phải từ chối `unit="units"`; thêm kiểm tra đơn vị xung đột và thiếu tiêu đề đơn vị.

### REV-05 — [P1] Con số đúng trong trích dẫn có thể bị gán nhầm năm

**Phân loại:** lỗi liên kết kỳ báo cáo với giá trị; độ tin cậy cao qua đường điều kiện, chưa thực thi testcase.

**Vị trí:** [core/filing_facts.py:139](../../core/filing_facts.py#L139), `validate_lease_extraction()`.

**Hành vi và bằng chứng:** validator chỉ xác nhận amount xuất hiện ở đâu đó trong quote và năm được khai nằm trong cửa sổ ba năm. Nó không gắn vị trí con số với tiêu đề cột năm. Với dòng fixture `Financing cash flows from finance leases 1,234 1,100 987`, khai FY2025 = 987 vẫn qua các điều kiện, dù 987 là số ở cột thứ ba/FY2023 trong ví dụ.

**Tác động:** quote có thật vẫn có thể xác nhận một annual fact sai kỳ, làm sai normalized FCFE và các phép định giá dùng chuỗi ba năm.

**Đề xuất sửa:** giữ cấu trúc bảng và yêu cầu bằng chứng tiêu đề năm/vị trí cột; nếu không khôi phục được liên kết đó thì chỉ chấp nhận câu văn ghi rõ năm và số tiền.

**Test nghiệm thu:** hoán đổi năm gán cho 1,234 và 987 phải bị từ chối; kiểm tra cả bảng đổi thứ tự cột và fiscal-year labels khác năm kết thúc kỳ.

### REV-06 — [P1] Không còn thuê tài chính cuối năm bị chuyển thành trả gốc cả năm bằng 0

**Phân loại:** lỗi diễn giải stock so với flow; độ tin cậy cao qua code và fixture hiện có, chưa thực thi.

**Vị trí:** [core/filing_facts.py:94](../../core/filing_facts.py#L94), nhánh `NONE` của `validate_lease_extraction()`; bước nâng trạng thái tại [core/valuation.py:225](../../core/valuation.py#L225), `resolve_deduction()`.

**Hành vi và bằng chứng:** fixture hiện có nói “does not have any finance leases as of year-end”. Validator biến câu nói về trạng thái cuối năm thành annual principal payment bằng 0. Khi ghi vào cash flow, valuation coi số 0 đó là `CONFIRMED_ZERO`.

**Tác động:** công ty có thể trả hết thuê tài chính trong năm rồi không còn nghĩa vụ cuối năm. Zero bị suy ra sẽ bỏ mất dòng tiền trả gốc thực tế, tăng FCFE và loại cảnh báo thiếu bằng chứng, kể cả khi tắt việc dùng giả định.

**Đề xuất sửa:** chỉ xác nhận zero khi có bằng chứng về khoản trả gốc bằng 0 trong kỳ báo cáo. Câu nói về dư nợ cuối kỳ phải giữ UNKNOWN/ASSUMED theo quy tắc có hiệu lực.

**Test nghiệm thu:** công ty có dư nợ thuê cuối năm bằng 0 nhưng tiền trả gốc trong năm dương không được nhận annual principal = `CONFIRMED_ZERO`.

### REV-07 — [P2] Một lần trích xuất thành công chặn đọc 10-K tương lai

**Phân loại:** lỗi vòng đời incremental extraction; độ tin cậy cao qua đường điều kiện, chưa thực thi testcase.

**Vị trí:** [etl/filings.py:309](../../etl/filings.py#L309), `extract_leases()`; cập nhật coverage tại [etl/filings.py:342](../../etl/filings.py#L342).

**Hành vi và bằng chứng:** `covered` chứa ticker có bất kỳ `AMOUNT` được chấp nhận trong toàn bộ lịch sử `lease_extractions`. Khi ticker nằm trong tập này, vòng đọc filings dừng trước khi xét accession mới. Một extraction chỉ chấp nhận được một năm cũng đủ chặn đọc báo cáo khác để bổ sung các năm còn thiếu.

**Tác động:** 10-K năm tiếp theo hoặc dữ liệu restatement không được trích xuất; dữ liệu cash-flow deduction mất khả năng cập nhật dù ingestion đã tải filing mới. Điều này cũng làm logic “latest filing wins” phía sau không nhận được facts mới.

**Đề xuất sửa:** quản lý coverage theo accession/version và từng fiscal year cần thiết; tiếp tục xử lý accession mới và các năm còn thiếu.

**Test nghiệm thu:** extract báo cáo cũ, thêm 10-K năm kế tiếp, chạy lại và xác nhận extractor được gọi cho báo cáo mới. Thêm trường hợp chỉ một năm trong ba năm được chấp nhận.

### REV-08 — [P2] Thiếu FX bị ngầm coi USD bằng EUR

**Phân loại:** lỗi chuyển đổi tiền tệ khi thiếu dữ liệu; độ tin cậy cao qua SQL, chưa thực thi testcase.

**Vị trí:** [etl/filings.py:408](../../etl/filings.py#L408), `apply_filing_facts()`.

**Hành vi và bằng chứng:** `s.value_usd * COALESCE(fx.rate, 1.0)` ghi nguyên số USD vào bảng cash flow EUR nếu không có tỷ giá của ticker/năm tương ứng hoặc rate là NULL. Đây là đường chuyển đổi mới trong commit được review; việc đường bổ sung XBRL cũ có cùng mẫu không làm fallback này đúng.

**Tác động:** số tiền được đánh dấu là deduction đã quan sát nhưng sai currency basis, làm sai FCFE và định giá. Tỷ giá thiếu không phải bằng chứng USD/EUR = 1.

**Đề xuất sửa:** chỉ cập nhật khi có tỷ giá USD/EUR dương, hữu hạn và truy vết được. Nếu thiếu thì giữ khoản khấu trừ unresolved và ghi nguyên nhân.

**Test nghiệm thu:** fact USD không có annual FX tương ứng phải không được điền vào trường EUR; thêm rate NULL, zero và rate không hợp lệ.

## Trạng thái bàn giao

- Standards: 3 phát hiện; nghiêm trọng nhất là cô lập lịch sử chat giữa tài khoản.
- Spec: 5 phát hiện; các lỗi đơn vị, gán năm và suy diễn zero có thể nâng dữ liệu sai thành đầu vào tài chính được coi là đã xác minh.
- Chưa thực hiện bản sửa hoặc xác nhận regression tests cho các phát hiện này.

## Kiểm tra lại bản sửa — 2026-10-17

Phần này đối chiếu working tree sau khi người sửa cập nhật [REMEDIATION_STATUS.md](REMEDIATION_STATUS.md). HEAD vẫn là `8e025c4fd6e896884970aeceb86aa65d3b19d957`; các bản sửa được kiểm tra là thay đổi chưa commit. Các mục REV-01 đến REV-08 phía trên là lịch sử review, không phải tuyên bố rằng code hiện tại vẫn giữ nguyên mọi lỗi ban đầu.

### Trạng thái đối chiếu

| Mục | Kết quả kiểm tra implementation | Giới hạn / việc còn lại |
|---|---|---|
| REV-01 | Logout xóa session state; `_own_session()` xóa các khóa assistant khi đổi user ID. | Chưa chạy kiểm thử Streamlit/logout thực tế. |
| REV-02 | Bỏ cache Decision xuyên câu hỏi; `built` chỉ sống trong một toolbox của một câu hỏi. | Đã đọc test bảo đảm rebuild; chưa chạy test. |
| REV-03 | Giá thiếu/NaN/inf/không dương của vị thế khác zero làm NAV unknown; không công bố weights/exposures/breaches toàn danh mục. | Đã đọc test với bốn dạng giá không hợp lệ; chưa chạy test. |
| REV-04 | Đơn vị model phải khớp scale tìm thấy trong passage/context; trường hợp `units` đối lập `in millions` ban đầu đã có nhánh từ chối. | Chưa xác nhận bằng thực thi. Code vẫn coi không tìm thấy scale là điều kiện chấp nhận `units`; không nên diễn giải đó là bằng chứng đơn vị đã được báo cáo. |
| REV-05 | Có kiểm tra vị trí số/cột năm, kể cả dấu gạch và heading trước passage. | **Chưa đóng hoàn toàn:** fallback theo quy ước toàn filing vẫn có thể gán nhầm năm; REV-F01. |
| REV-06 | Câu chỉ nói không có lease tại cuối kỳ được chuyển thành `NONE_AT_PERIOD_END`, không ghi zero. | **Chưa đóng hoàn toàn:** keyword chỉ kỳ có thể thuộc mệnh đề khác; REV-F02. |
| REV-07 | Coverage chuyển từ ticker sang accession; 10-K mới không còn bị ticker cũ chặn vĩnh viễn. | **Chưa đóng hoàn toàn:** chỉ chấp nhận một năm vẫn chặn bổ sung năm thiếu; REV-F03. |
| REV-08 | Cả text và XBRL đều yêu cầu FX hữu hạn, dương; bỏ fallback tỷ giá bằng 1. | Chưa chạy SQL/tests; cần xem xét dữ liệu đã điền bằng logic cũ, REV-F04. |

Các thay đổi trên là bằng chứng ở mức source. Reviewer **không xác nhận tests đã pass** hoặc cả tám mục đã được sửa hoàn toàn khi chạy ứng dụng.

### REV-F01 — [P1] Fallback quy ước toàn filing vẫn xác nhận sai năm của bảng cụ thể

**Liên quan:** REV-05. **Phân loại:** lỗi liên kết kỳ báo cáo trong nhánh fallback; độ tin cậy cao qua đường điều kiện, chưa chạy Python.

**Vị trí:** [core/filing_facts.py:125](../../core/filing_facts.py#L125), `column_years()` giới hạn tìm heading trong 40 dòng; [core/filing_facts.py:183](../../core/filing_facts.py#L183), `year_evidenced()` dựng heading từ `doc_columns` khi không tìm thấy heading tại chỗ.

**Bằng chứng:** `document_columns()` chỉ suy ra thứ tự và năm mới nhất chiếm đa số của các heading ba năm trong filing. Điều đó không chứng minh một bảng thuê cụ thể dùng cùng năm. Khi heading thực tế cách dòng số hơn 40 dòng, `column_years()` không tìm thấy nó. Fallback sau đó gán các cột từ năm phổ biến của toàn filing và trả về hợp lệ nếu model khai đúng năm suy ra đó.

**Phản ví dụ cần test:** filing năm 2025 có ít nhất ba heading `2025 2024 2023` trong các bảng khác; bảng lease có heading `2024 2023`, rồi 50 dòng mô tả/bảng trước dòng `Principal payments on finance leases 30 20`. Heading lease vẫn có trong text nhưng ngoài cửa sổ 40 dòng. Với `doc_columns=('desc', 2025)`, model khai FY2025 = 30 sẽ được nhánh fallback xem là đúng, dù số 30 thuộc FY2024. Ví dụ này được truy vết theo code, chưa chạy validator.

**Tác động:** số tiền có quote thật vẫn được lưu thành annual fact sai năm, giống loại rủi ro REV-05 ban đầu.

**Đề xuất:** giữ liên kết cấu trúc giữa heading và từng bảng/row. Khi không có bằng chứng năm của chính bảng, không nâng suy đoán từ quy ước toàn tài liệu thành fact đã xác minh; từ chối hoặc phân loại rõ là ước tính, không dùng như khoản khấu trừ đã quan sát.

**Test nghiệm thu:** fixture trên phải không tạo fact FY2025 = 30; kiểm tra thêm bảng hai năm khác năm hiện tại và bảng có thứ tự cột khác các bảng chủ đạo.

### REV-F02 — [P2; P1 khi bắt buộc deductions đã xác minh] Keyword thời gian không gắn với mệnh đề nói không có lease

**Liên quan:** REV-06. **Phân loại:** lỗi diễn giải phạm vi phủ định; độ tin cậy cao qua source và probe regex độc lập, chưa chạy Python validator.

**Vị trí:** [core/filing_facts.py:235](../../core/filing_facts.py#L235), `validate_lease_extraction()` kiểm tra `_NO_FINANCE_LEASE` và `_WHOLE_PERIOD` riêng trên toàn quote; nhánh ghi zero tại dòng 244.

**Phản ví dụ:**

> The Company has no finance leases as of year-end; principal payments during the year were $30 million.

Cụm `no finance leases` khớp regex về sự vắng mặt; cụm `during the year` ở mệnh đề nói về tiền trả gốc khớp regex về cả kỳ. Nếu model trả `kind=NONE` với quote này, code không chuyển thành `NONE_AT_PERIOD_END` mà đi vào nhánh tạo zero cho cả năm. Hai điều kiện không chứng minh rằng lease vắng mặt trong suốt kỳ.

**Chẩn đoán đã chạy:** probe PowerShell dùng hai regex tương ứng trả `True` cho cả hai. Đây chỉ là probe regex với câu tiếng Anh trên, **không phải thực thi module Python hoặc regression test**.

**Tác động:** vẫn có thể ghi zero sai và nâng thành `CONFIRMED_ZERO`. Với cấu hình cho phép giả định zero, lỗi chủ yếu làm sai trạng thái bằng chứng/confidence; khi yêu cầu deduction đã xác minh, nó còn có thể vượt qua cổng dữ liệu.

**Đề xuất:** bằng chứng vắng mặt trong kỳ phải gắn trực tiếp với cùng mệnh đề phủ định. Câu có điều kiện `as of`/`at year-end`, mệnh đề mâu thuẫn hoặc tiền trả gốc dương phải không xác nhận annual zero.

**Test nghiệm thu:** `kind=NONE` trên quote này không được sinh value zero; kiểm tra thêm hai câu tách biệt và trường hợp vắng mặt thật sự trong suốt kỳ.

### REV-F03 — [P2] Coverage vẫn chưa theo từng năm cần bổ sung

**Liên quan:** REV-07. **Phân loại:** lỗi coverage extraction còn lại; độ tin cậy cao qua code, chưa chạy testcase.

**Vị trí:** [etl/filings.py:327](../../etl/filings.py#L327), `extract_leases()` dừng khi accession có bất kỳ AMOUNT được chấp nhận; [etl/filings.py:358](../../etl/filings.py#L358), dừng ngay sau một extraction có AMOUNT.

**Bằng chứng:** sửa theo accession giải quyết việc chặn mọi 10-K tương lai. Tuy nhiên, một AMOUNT cho duy nhất FY2025 vẫn đủ để không đọc 10-K cũ có thể cung cấp FY2024/FY2023. Lần chạy sau gặp accession mới nhất trong `with_amount` cũng dừng ngay. Test mới kiểm tra có đọc 10-K mới, chưa kiểm tra bổ sung đầy đủ các năm còn thiếu.

**Tác động:** normalized cash flow vẫn có thể dùng khoản khấu trừ assumed/unknown ở các năm cơ sở dù tài liệu cần thiết đã lưu.

**Đề xuất:** dừng theo tập fiscal years cần thiết đã được phủ, không theo việc accession có bất kỳ amount. Cho phép báo cáo cũ bổ sung năm thiếu và giữ quy tắc filing mới ưu tiên khi cùng năm.

**Test nghiệm thu:** 10-K mới chỉ cho FY2025; 10-K cũ có FY2024/FY2023. Cùng một run hoặc run kế tiếp phải bổ sung các năm thiếu, không đọc lại vô hạn những năm đã đủ.

### REV-F04 — [P1 nếu có legacy facts bị ảnh hưởng] Validator mới không xử lý lại facts và deductions đã lưu

**Liên quan:** REV-04/05/06/08. **Phân loại:** thiếu cơ chế migration/revalidation; trạng thái dữ liệu thực tế **UNVERIFIED**.

**Vị trí:** [etl/filings.py:198](../../etl/filings.py#L198), ingestion bỏ qua accession đã có; [etl/filings.py:317](../../etl/filings.py#L317), `done` không gắn với phiên bản validator; [etl/filings.py:413](../../etl/filings.py#L413), `lease_facts()` chỉ lọc cờ `accepted`; [etl/filings.py:438](../../etl/filings.py#L438), apply không thay khoản deduction đã điền.

**Bằng chứng:** bản sửa thay parser, validation và FX cho dữ liệu xử lý mới, nhưng chưa có version của parser/validator hoặc đường kiểm tra lại kết quả cũ. Một row legacy được chấp nhận sai đơn vị, sai năm hoặc `NONE` từ câu chỉ nói về cuối năm vẫn có thể được lấy lại bởi `lease_facts()` mà không đi qua validator mới. Một deduction sai đã điền cũng không được sửa vì điều kiện `IS NULL`. Fast mode vẫn re-apply facts đã lưu.

**Giới hạn:** `warehouse/filings.duckdb` tồn tại khi kiểm tra, nhưng reviewer chưa đọc nội dung và không khẳng định kho có facts sai. Pilot fresh ingestion trong tài liệu không chứng minh việc nâng cấp kho cũ an toàn.

**Đề xuất:** version hóa ingestion/validation, quarantine hoặc kiểm tra lại legacy extractions; có quy trình sửa riêng các deduction có provenance từ kết quả cũ bị ảnh hưởng. Giữ nguyên giá trị báo cáo hợp lệ từ Yahoo/SEC; không xóa hoặc ghi đè hàng loạt thiếu provenance.

**Test nghiệm thu:** seed kho theo schema cũ với một accepted NONE từ câu `as of year-end`, một amount sai scale và deduction đã điền từ các facts đó; chạy quy trình nâng cấp và xác nhận chúng không còn là verified facts/deductions. Đồng thời bảo đảm facts hợp lệ và giá trị có nguồn độc lập không bị thay đổi.

### Kiểm chứng của vòng kiểm tra lại

- Đã đọc bản sửa của tám mục, các thay đổi trong `tests/test_commit_review.py`, và các đường gọi liên quan.
- Đã chạy probe regex PowerShell cho REV-F02: cả hai predicate khớp câu phản ví dụ.
- Đã chạy `git diff --check`: không báo lỗi whitespace; Git có cảnh báo chuyển LF/CRLF cho một số file.
- Python vẫn báo `Access is denied`; **0 lần chạy pytest/Python validator/SQL regression** trong vòng này. Không dùng các kết quả người sửa báo cáo như kết quả kiểm chứng độc lập của reviewer.
- Chỉ bổ sung phần tài liệu review này; không sửa production code, tests hoặc database.

## Kiểm tra lại lần hai — cập nhật REV-F01 đến REV-F04, 2026-10-17

Đã đọc phần follow-up mới trong [REMEDIATION_STATUS.md](REMEDIATION_STATUS.md), [AI_INTELLIGENCE.md](../en/AI_INTELLIGENCE.md), implementation hiện tại và các acceptance tests tương ứng. Phần này cập nhật trạng thái các phản ví dụ của vòng trước; các phần phía trên được giữ làm lịch sử.

### Đối chiếu bốn bản sửa

| Mục | Bằng chứng implementation hiện tại | Trạng thái reviewer |
|---|---|---|
| REV-F01 | `column_years()` đã bỏ cửa sổ 40 dòng. `year_evidenced()` chỉ suy theo convention cho row ba cột, ghi `basis='convention'`; source được truyền qua `lease_facts()` và `resolve_deduction()` phân loại `ESTIMATED`. | Đã có xử lý tương ứng với phản ví dụ cũ trong source; chưa chạy test độc lập. |
| REV-F02 | `absent_all_period()` yêu cầu cùng clause có phủ định và kỳ, loại clause có `as of`/ngày cuối kỳ, và loại quote có amount dương không phải year/date. | Hai mệnh đề trong phản ví dụ cũ không còn đủ điều kiện ghi zero theo đường điều kiện đã đọc; chưa chạy validator độc lập. |
| REV-F03 | `extract_leases()` dùng `needed={newest,newest-1,newest-2}` và coverage theo năm, tiếp tục đọc báo cáo cũ khi còn thiếu năm. | Có xử lý trường hợp mới chỉ lấy được một năm; đã đọc acceptance test, chưa chạy. |
| REV-F04 | Schema thêm `validator_version`, `basis`, `raw_json`; `refresh_readings()` re-judge hoặc drop phiên bản cũ; `lease_facts()` chỉ trả phiên bản hiện tại; `apply_filing_facts()` xóa deduction/note có provenance text rồi dẫn xuất lại, giữ giá trị Yahoo/XBRL. | Có cơ chế migration cho text-derived facts/deductions đã nêu; chưa chạy trên DB thật hoặc DB fixture. |

Vị trí chính: [core/filing_facts.py:130](../../core/filing_facts.py#L130), [etl/filings.py:352](../../etl/filings.py#L352), [etl/filings.py:375](../../etl/filings.py#L375), [etl/filings.py:475](../../etl/filings.py#L475), [core/valuation.py:225](../../core/valuation.py#L225); tests tại [tests/test_commit_review.py:227](../../tests/test_commit_review.py#L227).

**Giới hạn migration:** `refresh_readings()` được gọi trong `extract_leases()`. `run_step()` chỉ vào đường đó khi full mode, có SEC user agent, LLM khả dụng và có targets. Fast mode không re-judge; nó chỉ lấy current-version facts và re-derive text deductions. Do đó không nên mô tả mọi chế độ chạy là đã tự động hoàn tất revalidation. Cơ chế mới xử lý provenance text; chưa kiểm chứng dữ liệu SEC XBRL có thể đã được điền bằng fallback FX cũ. Không khẳng định có dữ liệu sai như vậy trong kho thực tế.

### REV-F05 — [P1] Phép đối chiếu con số làm tròn cho phép xác nhận amount không xuất hiện trong quote

**Phân loại:** lỗi validation số liệu mới phát hiện trong logic đã tồn tại; không quy kết là lỗi mới sinh từ bản sửa follow-up. Độ tin cậy cao qua đường điều kiện; chưa thực thi Python testcase.

**Vị trí:** [core/filing_facts.py:77](../../core/filing_facts.py#L77), `number_forms()`; các nơi dùng là `number_in()`, `descriptor()` và `year_evidenced()`.

**Hành vi:** `number_forms()` tạo cách viết amount với 0, 1 và 2 chữ số sau dấu thập phân bằng f-string. Việc này thay đổi giá trị số qua làm tròn, thay vì chỉ chuẩn hóa dấu phân cách và zero cuối. Ví dụ `number_forms(1.4)` chứa chuỗi `"1"`. Vì cả ba bước kiểm tra cùng dùng tập này, chúng đều có thể coi token `1` trong quote là bằng chứng cho amount `1.4`.

**Phản ví dụ dẫn xuất từ code:**

```python
row = "Principal payments on finance leases 1 2 3"
passage = "(in billions)\nYear Ended December 31, 2025 2024 2023\n" + row
ext = {
    "kind": "AMOUNT", "unit": "billions",
    "values": [{"fiscal_year": 2025, "amount": 1.4,
                "quote": row, "chunk_id": "c"}],
}
# Test đề xuất; reviewer CHƯA chạy:
res = validate_lease_extraction(ext, {"c": passage}, fiscal_year_end=2025)
assert not res["accepted"]
```

Theo các điều kiện hiện tại, `1.4` khớp token `1` qua định dạng 0 decimal, descriptor đúng dòng principal, column đầu đúng FY2025, đơn vị billions hợp lệ. Số tiền ghi ra được tính từ amount của model, tức 1.4 tỷ USD, trong khi quote chỉ có 1 tỷ USD. Tương tự, amount `0.6` cũng có định dạng 0 decimal là `"1"`, có thể ghi thiếu 400 triệu USD tiền trả gốc.

**Tác động:** một fact sai số tiền có thể được chấp nhận với `basis='heading'` và vào valuation như `OBSERVED`; đây không chỉ là lỗi định dạng hiển thị. Sai khấu trừ có thể làm sai FCFE, intrinsic value và quyết định mua/tránh.

**Đề xuất:** parse token và amount thành giá trị số chính xác, ví dụ `Decimal` từ biểu diễn chuỗi, chỉ chuẩn hóa dấu hàng nghìn/zero thừa; không chấp nhận khớp bằng cách làm tròn amount. Dùng cùng quy tắc cho kiểm tra số, descriptor và liên kết cột năm.

**Test nghiệm thu:** quote token `1` phải từ chối amount `0.6` và `1.4`; token `1.4` phải từ chối `1.44`. Các cách viết cùng giá trị như `1`, `1.0`, `1.00` và `1,000`/`1000` phải tiếp tục khớp. Chạy end-to-end validator với stated scale và heading, không chỉ test helper.

### Hai điểm tài liệu cần làm rõ

- [AI_INTELLIGENCE.md:339](../en/AI_INTELLIGENCE.md#L339) vẫn nói báo cáo cũ chỉ được đọc khi chưa có báo cáo mới hơn cho amount. Code hiện tại đọc theo các năm còn thiếu; cần cập nhật câu này để không mô tả lại hành vi của REV-F03.
- [REMEDIATION_STATUS.md:125](REMEDIATION_STATUS.md#L125) nêu 30/30 accepted fiscal years, nhưng breakdown 25 heading + 6 convention là 31. Có thể đang đếm các đơn vị khác nhau hoặc có overlap; cần ghi rõ sample/grain và cách deduplicate. Reviewer chưa có bảng kết quả pilot để xác định nguyên nhân, không kết luận pilot sai.

### Bằng chứng và giới hạn lần hai

- Đã kiểm tra source và nội dung các tests follow-up; chưa chạy chúng.
- Thử lại Python launcher vẫn trả `Access is denied`; không cài môi trường hoặc vượt hạn chế truy cập.
- `git diff --check` không báo lỗi whitespace; có cảnh báo LF/CRLF.
- Kết quả **716 passed** và pilot **30/30** là kết quả người sửa ghi trong tài liệu, chưa được reviewer xác nhận độc lập.
- Không chạy ETL/migration với database thật, không gọi SEC/Cohere, không sửa production hoặc tests. Chỉ bổ sung tài liệu review.
