prompt = """
Format BOTH the question and answer in this TOON schema:

question_id=<short_id>|
question_text=<compressed_question>|
topic=<topic>|
answer=<short_answer>|
detail=<extra>

Convert this question:
"Who discovered gravity?"
"""

response = client.responses.create(
    model="gpt-4.1",
    input=prompt
)

print(response.output_text)


question_id=q_gravity|question_text=discoverer_of_gravity|topic=physics|answer=Newton|detail=17th_century
