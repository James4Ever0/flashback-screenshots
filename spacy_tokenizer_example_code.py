import spacy
# Load English model
# python -m spacy download en_core_web_sm

# possible models:
# zh_core_web_sm
# en_core_web_sm

nlp_model = "zh_core_web_sm"
# nlp_model = "en_core_web_sm"

try:
    nlp = spacy.load(nlp_model)
except OSError:
    print("Model %s not found, downloading" % nlp_model)
    import spacy.cli
    spacy.cli.download(nlp_model)
    nlp = spacy.load(nlp_model)

# Process text
doc = nlp("I love natural language processing!")
# Iterate over tokens
for token in doc:
   print(token.text)