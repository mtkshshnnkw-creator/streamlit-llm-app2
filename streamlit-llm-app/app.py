import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def _get_api_key() -> str | None:
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    return os.getenv("OPENAI_API_KEY")

# -----------------------------------------------------
# 1. LLM呼び出し関数（条件で必須）
# -----------------------------------------------------
def run_llm(input_text: str, expert_type: str) -> str:
    """
    入力テキストと専門家（A or B）を受け取り、
    LangChain を使って LLM からの回答を返す関数。
    """

    # 専門家ごとの system メッセージ
    expert_system_messages = {
        "A. サステナビリティ戦略アーキテクト": "あなたはサステナビリティ戦略に精通した専門家です。環境・社会・ガバナンスの観点から実現可能性を評価し、実践的な提案を提示してください。",
        "B. DXトランスフォーメーションデザイナー": "あなたはデジタルトランスフォーメーションに強い専門家です。最新テクノロジーを活かした業務改善施策を構造的に提示してください。",
    }

    system_message = expert_system_messages.get(expert_type, "あなたは有能なアシスタントです。")

    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("OpenAI APIキーが設定されていません。環境変数または Streamlit の secrets に OPENAI_API_KEY を設定してください。")

    # LangChain ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("user", "{user_input}")
        ]
    )

    # LLM設定（GPT-4o-mini など環境に応じて変更可）
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.6,
        openai_api_key=api_key,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    # LLMに投げて応答を取得
    response = chain.run({"user_input": input_text})

    return response


# -----------------------------------------------------
# 2. Streamlit UI
# -----------------------------------------------------
st.title("🔍 AI専門家アドバイザー（LangChain + Streamlit）")

st.caption("Streamlit Community Cloud の Python バージョンは 3.11 を想定しています。")

st.write("""
## 📝 このアプリについて
- 入力フォームに質問を入力すると、選択した専門家の視点でAIが回答します。  
- サステナビリティ領域／デジタル変革領域の専門家から選択できます。  
- Lesson8 の内容をベースに LangChain を使用しています。  
""")

st.write("---")

# ラジオボタン（専門家選択）
expert = st.radio(
    "AI にどの専門家として回答させますか？",
    [
        "A. サステナビリティ戦略アーキテクト",
        "B. DXトランスフォーメーションデザイナー",
    ]
)

# 入力フォーム
user_input = st.text_input("質問内容を入力してください")

# 送信ボタン
if st.button("回答を生成"):
    if user_input.strip() == "":
        st.warning("テキストを入力してください。")
    else:
        with st.spinner("AI が回答を生成中です..."):
            try:
                answer = run_llm(user_input, expert)
            except RuntimeError as exc:
                st.error(str(exc))
            else:
                st.success("回答が生成されました！")
                st.write("## 📘 回答")
                st.write(answer)
        