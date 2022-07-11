from transformers import pipeline
classifier = pipeline("text-classification")

import pandas as pd

text = """Dear Amazon, last week I ordered an Optimus Prime action figure
from your online store in Germany. Unfortunately, when I opened the package,
I discovered to my horror that I had been sent an action figure of Megatron
instead! As a lifelong enemy of the Decepticons, I hope you can understand my
dilemma. To resolve the issue, I demand an exchange of Megatron for the
Optimus Prime figure I ordered. Enclosed are copies of my records concerning
this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

outputs = classifier(text)

print (pd.DataFrame(outputs))
###############################################################
ner_tagger = pipeline("ner", aggregation_strategy="simple")
outputs_1 = ner_tagger(text)
print (pd.DataFrame(outputs_1))
###############################################################
reader = pipeline("question-answering")
question = "what does the answer want?"
outputs_2 = reader(question = question, context = text)
print (outputs_2)
##############################################################
summarizer = pipeline("summarization")
outputs_3 = summarizer(text, max_length = 45, clean_up_tokenization_spaces = True)
print (outputs_3[0]['summary_text'])
##############################################################
#translator = pipeline("translation_en_to_de",model="Helsinki-NLP/opus-mt-en-de")
#outputs_4 = translator(text, clean_up_tokenization_spaces=True, min_length=100)
#print(outputs_4[0]['translation_text'])
##############################################################
generator = pipeline("text-generation")
response = "Dear Bumblebee, I am sorry to hear that your order was mixed up."
prompt = text + "\n\nCustomer service response:\n" + response
outputs_5 = generator(prompt, max_length=200)
print(outputs_5[0]['generated_text'])