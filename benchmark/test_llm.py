from openai import OpenAI
client = OpenAI(
    base_url="http://test-k8s.misa.local/llm-router/v1",  # hoặc URL custom
    api_key="ml-BRLNIyva65v4ltx1diADpn5mgY5ka9W9jpUX2DSy00iECWTiYe-AU7900zpWC0oJJDmI5qDFDXQxD5Ccc0m2lIQDqulCsPxlxfjRs"
)
 
response = client.chat.completions.create(
    model="misa-gpt-oss-120b",   # hoặc model bạn muốn
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Việc bạn làm giỏi nhất là gì ?"}
    ]
)
 
print(response.choices[0].message.content)
 