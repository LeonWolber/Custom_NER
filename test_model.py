# test the saved model
import spacy
output_dir = 'models'


print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)
# Check the classes have loaded back consistently
# assert nlp2.get_pipe("ner").move_names == list(ner.move_names)
test_text = 'Vertriebsleiter'


doc2 = nlp2(test_text)

for ent in doc2.ents:
    print(ent.label_, ent.text)