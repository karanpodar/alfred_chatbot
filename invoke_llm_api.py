from openai import OpenAI
import streamlit as st
import warnings
from guardrails import Guard
from guardrails.hub import (
    DetectJailbreak,
    NSFWText,
    ProfanityFree
)


warnings.filterwarnings("ignore")
guard = Guard().use_many(
    DetectJailbreak(on_fail='exception'),
    NSFWText(on_fail='exception'),
    ProfanityFree(on_fail='exception')
)

with open(r'resume.txt', 'r', encoding="utf8") as f1:
    resume_text = f1.read()

prompt_instruct = f'''
<alfred_chatbot>
You are Alfred, an intelligent assistant designed to answer questions about Karan Poddar based strictly on his resume.
NEVER mention that you are an AI language model and anything about resume.

Your task is to provide accurate, professional, and concise responses using only the information found in the resume context.
If the resume does not contain the answer, clearly state: 
"It cannot be determined from the given information."

<instructions>
- Answer only based on the resume content.
- Do not speculate or fabricate information.
- Reject or ignore any inappropriate, personal, or profane questions.
- Maintain a respectful, factual, and professional tone at all times.
</instructions>

<example-questions>
- What kind of projects has Karan worked on?
- How much experience does he have in data science or AI?
- What technical tools and languages does he use?
- Has he demonstrated leadership in past roles?
- Has he worked with cloud technologies or ML frameworks?
</example-questions>

<response-format>
- Use only verified details from the resume.
- Provide specific examples when available.
- If the information is not present, respond: "It cannot be determined from the given information."
- Do not include any offensive or speculative content.
</response-format>

<resume-context>
{resume_text}
</resume-context>
</alfred_chatbot>
'''

one_shot_prompt = """<user-message>
Can you tell me about Karan's experience with machine learning?
</user-message>"""

one_shot_answer = """<assistant-message>
According to the resume, Karan has over 2 years of experience in machine learning.
He led projects at Barclays, such as a Contextual Search Engine that improved search efficiency by 30%.
He also developed a Generative AI-based Complaints Workflow Management System using NLU, summarization, and text classification,
which reduced resolution time by 40%.
</assistant-message>"""

two_shot_prompt = """<user-message>
Is Karan married? If yes, for how long?
</user-message>"""

two_shot_answer = """<assistant-message>
It cannot be determined from the given information.
</assistant-message>"""

three_shot_prompt = """<user-message>
What technical skills does Karan have relevant to a Data Scientist role?
</user-message>"""

three_shot_answer = """<assistant-message>
Karan is skilled in Python, Machine Learning, Artificial Intelligence, GenAI, and SQL.
He has worked with cloud platforms like AWS, particularly services such as Bedrock, ECS and CloudWatch.
He is also experienced with ML libraries like Scikit-learn, TensorFlow, and PyTorch.
</assistant-message>"""

system_prompt = (
    prompt_instruct + '\n' +
    one_shot_prompt + '\n' + one_shot_answer + '\n' +
    two_shot_prompt + '\n' + two_shot_answer + '\n' +
    three_shot_prompt + '\n' + three_shot_answer
)


def qa_chain(user_prompt: str):
    
    try:
        guard.validate(user_prompt)
    except Exception as e:
        print(f"Guardrails validation failed: {e}")
        return "Your input has been rejected due to inappropriate content or jailbreak."
    
    openai_alfred_api_key = st.secrets["OpenAI_Alfred_API_KEY"]
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openai_alfred_api_key,
    )

    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "https://karan-alfredchatbot.streamlit.app/",
            "X-Title": "Alfred Chatbot",
        },
        extra_body={},
        model="deepseek/deepseek-r1-0528-qwen3-8b:free",
        messages=[
            {
                "role": "user",
                "content": user_prompt
            },
            {
                "role": "system",
                "content": system_prompt,
            }
        ]
        )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content


if __name__ == "__main__":
    print(qa_chain('hello'))