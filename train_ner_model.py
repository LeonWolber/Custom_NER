import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example
from pathlib import Path
import warnings

import random
from spacy.pipeline import EntityRuler




from prepare import *



def main(train_df, encoder, model=None, new_model_name="label_finder", output_dir="models_single_input", n_iter=15):

    targets_fine = pd.Series([i.split('->')[1].strip() if '>' in i else i.strip() for i in train_df['target']]).unique()

    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe('ner')
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    for i in targets_fine:
        ner.add_label(i)  # add new entity label to entity recognizer

    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    # print('move names:', move_names)
    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # only train NER
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        for itn in range(n_iter):
            TRAIN_DATA = create_train_data_single(train_df, encoder)
            #             print(TRAIN_DATA)
            random.shuffle(TRAIN_DATA)
            batches = minibatch(TRAIN_DATA, size=2)
            losses = {}
            for batch in batches:
                for text, annotations in batch:
                    # print('text = ', text)
                    # print('annotation = ', annotations)

                    # create Example
                    doc = nlp.make_doc(text)
                    # print('doc = ', doc)
                    example = Example.from_dict(doc, annotations)
                    # Update the model
                    nlp.update([example], losses=losses, drop=0.3, sgd=optimizer)

            print("Losses", losses)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

if __name__ == '__main__':
    df, df_unknown, le = prepare_data(high_level_pred=False, set_=False)

    main(df, encoder=le)