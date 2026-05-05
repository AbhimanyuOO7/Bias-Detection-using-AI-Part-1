import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="MY_API_KEY")

def generate_bias_summary(dp, eo, pe, avg, level):

    prompt = f"""
    You are an AI fairness analyst.

    Metrics:
    DP = {dp}
    EO = {eo}
    PE = {pe}
    Avg Bias = {avg}%
    Level = {level}

    STRICT RULES:
    - Each section MUST be on a NEW LINE
    - Add a BLANK LINE between sections
    - Do NOT merge lines
    - Keep it short and specific to values

    FORMAT EXACTLY LIKE THIS:

    Summary:
    <text>

    Key Issue:
    <text>

    Recommendation:
    <text>
    """

    model = genai.GenerativeModel("gemini-flash-lite-latest")

    response = model.generate_content(prompt)

    return response.text