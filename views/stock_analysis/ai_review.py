"""Overview: the dated external (ChatGPT) review of a BUY CANDIDATE, when one exists. It never changes the Decision."""
import html

import streamlit as st

from core import ai_review as ar


LABELS = {
    "English": {
        "business": "Business", "bull_case": "Bull case", "bear_case": "Bear case / risks",
        "financial_health": "Financial health", "valuation": "Valuation", "upcoming_events": "Next 90 days",
        "data_disagreements": "Where the screener numbers may mislead",
        "label_would_change_if": "The label would change if", "verify_yourself": "Verify yourself",
        "caption": "A language model's opinion on the {as_of} snapshot (written {generated_at}, facts as of {facts_as_of}). "
                   "It is NOT the Decision and does not change it; it can be wrong or out of date — "
                   "check the items under “Verify yourself” in official filings.",
    },
    "Vietnamese": {
        "business": "Hoạt động kinh doanh", "bull_case": "Luận điểm tích cực", "bear_case": "Luận điểm tiêu cực / rủi ro",
        "financial_health": "Sức khỏe tài chính", "valuation": "Định giá", "upcoming_events": "90 ngày tới",
        "data_disagreements": "Những số liệu sàng lọc có thể gây hiểu nhầm",
        "label_would_change_if": "Nhãn đánh giá sẽ thay đổi nếu", "verify_yourself": "Tự kiểm chứng",
        "caption": "Ý kiến của mô hình ngôn ngữ về dữ liệu ngày {as_of} (viết lúc {generated_at}, thông tin đến {facts_as_of}). "
                   "Đây KHÔNG phải Decision và không làm thay đổi Decision; nội dung có thể sai hoặc lỗi thời — "
                   "hãy kiểm tra các mục “Tự kiểm chứng” trong hồ sơ công bố chính thức.",
    },
}


def _bullets(items):
    for x in items:
        st.markdown(f"- {x}")


def render(dd, ctx):
    got = ar.load_latest(dd.ticker)
    if not got:
        return
    r = got["review"]
    labels = LABELS.get(got.get("language"), LABELS["Vietnamese"])
    badge = {"agrees": "Agrees", "cautious": "Cautious", "disagrees": "Disagrees"}[ar.agreement(got["decision"], r["label"])]
    with st.expander(f"{badge} · External AI review (ChatGPT): {r['label']} · confidence {r['confidence']} "
                     f"· snapshot {got['as_of']}"):
        st.caption(labels["caption"].format(as_of=got["as_of"], generated_at=str(got["generated_at"])[:16],
                                          facts_as_of=html.escape(r["facts_as_of"])))
        st.markdown(f"**{labels['business']}** — {r['business']}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{labels['bull_case']}**")
            _bullets(r["bull_case"])
        with c2:
            st.markdown(f"**{labels['bear_case']}**")
            _bullets(r["bear_case"])
        st.markdown(f"**{labels['financial_health']}** — {r['financial_health']}")
        st.markdown(f"**{labels['valuation']}** — {r['valuation']}")
        if r["upcoming_events"]:
            st.markdown(f"**{labels['upcoming_events']}**")
            _bullets(r["upcoming_events"])
        if r["data_disagreements"]:
            st.markdown(f"**{labels['data_disagreements']}**")
            _bullets(r["data_disagreements"])
        st.markdown(f"**{labels['label_would_change_if']}**")
        _bullets(r["label_would_change_if"])
        st.markdown(f"**{labels['verify_yourself']}**")
        _bullets(r["verify_yourself"])
